"""Hybrid retrieval combining ChromaDB semantic search with BM25 keywords.

Semantic search catches paraphrases; BM25 catches exact terms like a fee
amount or a policy name. Blending both is more reliable than either alone.
"""

from __future__ import annotations

import logging
import pickle

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from ..config import (
    BM25_INDEX_PATH,
    BM25_WEIGHT,
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    SEMANTIC_WEIGHT,
)
from .ingest import tokenize

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Searches the knowledge base semantically and by keyword at once."""

    def __init__(self) -> None:
        """Load the Chroma collection, the BM25 index, and the embedder.

        Raises FileNotFoundError when either index is missing, which means
        ingestion has not been run yet.
        """
        if not CHROMA_PERSIST_DIR.is_dir():
            raise FileNotFoundError(
                f"No ChromaDB at {CHROMA_PERSIST_DIR}. Run: python -m backend.retrieval.ingest"
            )

        client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
        try:
            self.collection = client.get_collection(COLLECTION_NAME)
        except Exception as exc:
            raise FileNotFoundError(
                f"Collection '{COLLECTION_NAME}' not found. "
                "Run: python -m backend.retrieval.ingest"
            ) from exc

        if not BM25_INDEX_PATH.is_file():
            raise FileNotFoundError(
                f"No BM25 index at {BM25_INDEX_PATH}. Run: python -m backend.retrieval.ingest"
            )
        with BM25_INDEX_PATH.open("rb") as handle:
            payload = pickle.load(handle)

        # Rebuild from the stored corpus so the index does not depend on
        # the exact rank_bm25 version that pickled it.
        self.bm25_texts: list[str] = payload["texts"]
        self.bm25_sources: list[str] = payload["sources"]
        self.bm25 = BM25Okapi(payload["tokenized_corpus"])

        self.model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info(
            "HybridRetriever ready: %d chunk(s) in Chroma, %d in BM25",
            self.collection.count(),
            len(self.bm25_texts),
        )

    def embed(self, text: str) -> np.ndarray:
        """Embed a single string into a normalized vector."""
        vector = self.model.encode(
            [text],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )[0]
        return np.asarray(vector, dtype=np.float32)

    def compute_confidence(self, query: str, top_chunk: str) -> float:
        """Return cosine similarity between a query and a chunk, 0 to 1.

        Both vectors are unit length, so their dot product is the cosine.
        Negative values are clamped to 0 because the agent treats this
        number as a confidence score.
        """
        if not query.strip() or not top_chunk.strip():
            return 0.0
        try:
            similarity = float(np.dot(self.embed(query), self.embed(top_chunk)))
        except Exception as exc:
            logger.error("Confidence computation failed: %s", exc)
            return 0.0
        return max(0.0, min(1.0, similarity))

    def _semantic_search(self, query: str, n_results: int) -> list[dict[str, object]]:
        """Return the nearest chunks from ChromaDB with scores in 0 to 1."""
        count = self.collection.count()
        if count == 0:
            return []

        result = self.collection.query(
            query_embeddings=[self.embed(query).tolist()],
            n_results=min(n_results, count),
            include=["documents", "metadatas", "distances"],
        )
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        hits: list[dict[str, object]] = []
        for text, metadata, distance in zip(documents, metadatas, distances):
            # The collection uses cosine space, so distance = 1 - similarity.
            score = max(0.0, min(1.0, 1.0 - float(distance)))
            hits.append(
                {
                    "text": str(text),
                    "source": str((metadata or {}).get("source", "unknown")),
                    "semantic_score": score,
                    "bm25_score": 0.0,
                }
            )
        return hits

    def _keyword_search(self, query: str, n_results: int) -> list[dict[str, object]]:
        """Return the best BM25 matches with scores normalized to 0 to 1."""
        tokens = tokenize(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        best = float(np.max(scores)) if len(scores) else 0.0
        if best <= 0:
            return []

        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        hits: list[dict[str, object]] = []
        for position in ranked[:n_results]:
            raw = float(scores[position])
            if raw <= 0:
                continue
            hits.append(
                {
                    "text": self.bm25_texts[position],
                    "source": self.bm25_sources[position],
                    "semantic_score": 0.0,
                    # Normalized against the best hit for this query, so the
                    # weighted blend compares like with like.
                    "bm25_score": raw / best,
                }
            )
        return hits

    def retrieve(self, query: str, n_results: int = 5) -> dict[str, object]:
        """Search both indexes, merge the hits, and rank them.

        Chunks found by both methods keep the higher score from each, so
        agreement between semantic and keyword search pushes a chunk up.
        Confidence is the cosine similarity of the best merged chunk.
        """
        empty: dict[str, object] = {
            "chunks": [],
            "sources": [],
            "confidence": 0.0,
            "query": query,
        }
        if not query or not query.strip():
            return empty

        try:
            semantic_hits = self._semantic_search(query, n_results)
            keyword_hits = self._keyword_search(query, n_results)
        except Exception as exc:
            logger.error("Retrieval failed for %r: %s", query, exc)
            return empty

        # Deduplicate on the chunk text itself, since the same passage can
        # surface from both indexes.
        merged: dict[str, dict[str, object]] = {}
        for hit in semantic_hits + keyword_hits:
            key = str(hit["text"])
            existing = merged.get(key)
            if existing is None:
                merged[key] = dict(hit)
                continue
            existing["semantic_score"] = max(
                float(existing["semantic_score"]), float(hit["semantic_score"])
            )
            existing["bm25_score"] = max(
                float(existing["bm25_score"]), float(hit["bm25_score"])
            )

        if not merged:
            return empty

        for hit in merged.values():
            hit["combined_score"] = (
                SEMANTIC_WEIGHT * float(hit["semantic_score"])
                + BM25_WEIGHT * float(hit["bm25_score"])
            )

        ranked = sorted(
            merged.values(),
            key=lambda hit: float(hit["combined_score"]),
            reverse=True,
        )[:5]

        chunks = [str(hit["text"]) for hit in ranked]
        sources = [str(hit["source"]) for hit in ranked]
        confidence = self.compute_confidence(query, chunks[0])

        logger.info(
            "Retrieved %d chunk(s) for %r (confidence=%.3f)",
            len(chunks),
            query[:60],
            confidence,
        )
        return {
            "chunks": chunks,
            "sources": sources,
            "confidence": confidence,
            "query": query,
        }

    def format_context(self, chunks: list[str], sources: list[str]) -> str:
        """Render retrieved chunks into one labelled block for the LLM.

        Each passage is numbered and tagged with its source file so the
        model can ground its answer and the log shows where it came from.
        """
        if not chunks:
            return "No relevant information found in the knowledge base."

        sections = []
        for position, chunk in enumerate(chunks, start=1):
            source = sources[position - 1] if position <= len(sources) else "unknown"
            sections.append(f"[{position}] (source: {source})\n{chunk}")
        return "\n\n".join(sections)


_retriever: HybridRetriever | None = None


def get_retriever() -> HybridRetriever:
    """Return a shared HybridRetriever, building it on first use.

    Loading the embedding model takes a few seconds, so the instance is
    cached rather than rebuilt for every request.
    """
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


def reset_retriever() -> None:
    """Drop the cached retriever so the next call reloads both indexes.

    Called after re-ingestion, which replaces the files underneath it.
    """
    global _retriever
    _retriever = None
