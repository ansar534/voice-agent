"""Hybrid retrieval over the local Chroma and BM25 indexes.

A query is embedded with the same all-MiniLM-L6-v2 model used at ingest
time, then matched two ways:

* Chroma semantic search, ranked by cosine similarity
* BM25 keyword search, ranked by term overlap

The two lists are merged with reciprocal rank fusion and de-duplicated by
chunk id. Confidence is the cosine similarity of the top merged chunk,
clamped to the range 0 to 1.
"""

from __future__ import annotations

import logging
import pickle
import time
from dataclasses import dataclass

import chromadb
import numpy as np
from chromadb.api.models.Collection import Collection
from chromadb.errors import NotFoundError
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from backend.config import settings
from backend.retrieval.ingest import tokenize

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 5
# Standard reciprocal-rank-fusion constant. Higher k makes ranks flatter.
RRF_K = 60

_model: SentenceTransformer | None = None
_chroma_client: chromadb.PersistentClient | None = None
_bm25_cache: tuple[float, BM25Okapi, list[dict[str, str | int]]] | None = None


@dataclass(frozen=True)
class RetrievedChunk:
    """One knowledge-base chunk chosen by hybrid search."""

    id: str
    text: str
    source: str
    chunk_index: int
    cosine_similarity: float
    bm25_score: float
    fusion_score: float


@dataclass(frozen=True)
class HybridSearchResult:
    """Merged chunks plus how well the top chunk matches the query."""

    chunks: list[RetrievedChunk]
    confidence: float


@dataclass
class _Candidate:
    """Working record while semantic and keyword hits are combined."""

    id: str
    text: str
    source: str
    chunk_index: int
    cosine_similarity: float = 0.0
    bm25_score: float = 0.0
    semantic_rank: int | None = None
    bm25_rank: int | None = None


def _get_model() -> SentenceTransformer:
    """Load all-MiniLM-L6-v2 once and reuse it for later queries."""
    global _model
    if _model is None:
        try:
            _model = SentenceTransformer(settings.embedding_model_name)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load embedding model {settings.embedding_model_name}: {exc}"
            ) from exc
        logger.info("Loaded embedding model %s", settings.embedding_model_name)
    return _model


def _embed_query(query: str) -> np.ndarray:
    """Embed one query with the same normalization used at ingest time."""
    try:
        vector = _get_model().encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )[0]
    except Exception as exc:
        raise RuntimeError(f"Failed to embed query: {exc}") from exc
    return np.asarray(vector, dtype=np.float32)


def _get_collection() -> Collection:
    """Open the persisted Chroma collection created by ingestion."""
    global _chroma_client
    if not settings.chroma_persist_dir.is_dir():
        raise FileNotFoundError(
            f"Chroma directory not found at {settings.chroma_persist_dir}. "
            "Run ingestion first."
        )
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=str(settings.chroma_persist_dir))
    try:
        collection = _chroma_client.get_collection(settings.collection_name)
    except NotFoundError as exc:
        raise FileNotFoundError(
            f"Chroma collection '{settings.collection_name}' does not exist. "
            "Run ingestion first."
        ) from exc
    if collection.count() == 0:
        raise ValueError(
            f"Chroma collection '{settings.collection_name}' is empty. "
            "Run ingestion first."
        )
    return collection


def _load_bm25() -> tuple[BM25Okapi, list[dict[str, str | int]]]:
    """Load the BM25 corpus, rebuilding the index when the pickle changes."""
    global _bm25_cache
    path = settings.bm25_index_path
    if not path.is_file():
        raise FileNotFoundError(
            f"BM25 index not found at {path}. Run ingestion first."
        )
    mtime = path.stat().st_mtime
    if _bm25_cache is not None and _bm25_cache[0] == mtime:
        return _bm25_cache[1], _bm25_cache[2]

    with path.open("rb") as handle:
        payload = pickle.load(handle)
    chunks = payload.get("chunks") or []
    tokenized = payload.get("tokenized_corpus") or []
    if not chunks or not tokenized or len(chunks) != len(tokenized):
        raise ValueError(f"BM25 index at {path} is empty or malformed.")
    index = BM25Okapi(tokenized)
    _bm25_cache = (mtime, index, chunks)
    logger.info("Loaded BM25 index with %d chunk(s) from %s", len(chunks), path)
    return index, chunks


def _clamp_unit(value: float) -> float:
    """Clamp a similarity into the 0 to 1 range the agent treats as confidence."""
    return max(0.0, min(1.0, float(value)))


def _cosine_by_id(
    collection: Collection,
    query_vector: np.ndarray,
    chunk_ids: list[str],
) -> dict[str, float]:
    """Return cosine similarity between the query and each stored chunk.

    Ingest stores L2-normalized embeddings, so the dot product is cosine
    similarity. Negative values are clamped to 0.
    """
    if not chunk_ids:
        return {}
    stored = collection.get(ids=chunk_ids, include=["embeddings"])
    similarities: dict[str, float] = {}
    embeddings = stored.get("embeddings")
    if embeddings is None:
        return similarities
    for chunk_id, embedding in zip(stored.get("ids") or [], embeddings):
        if embedding is None:
            continue
        vector = np.asarray(embedding, dtype=np.float32)
        similarities[str(chunk_id)] = _clamp_unit(float(np.dot(query_vector, vector)))
    return similarities


def semantic_search(
    query_vector: np.ndarray,
    collection: Collection,
    top_k: int,
) -> list[_Candidate]:
    """Return the nearest Chroma chunks for an already embedded query."""
    result_count = min(top_k, collection.count())
    try:
        result = collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=result_count,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        raise RuntimeError(f"Chroma semantic search failed: {exc}") from exc

    ids = (result.get("ids") or [[]])[0]
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    if not ids:
        return []

    cosine = _cosine_by_id(collection, query_vector, [str(chunk_id) for chunk_id in ids])
    candidates: list[_Candidate] = []
    for rank, chunk_id in enumerate(ids, start=1):
        position = rank - 1
        metadata = metadatas[position] or {}
        candidates.append(
            _Candidate(
                id=str(chunk_id),
                text=str(documents[position] or ""),
                source=str(metadata.get("source", "")),
                chunk_index=int(metadata.get("chunk_index", 0)),
                cosine_similarity=cosine.get(str(chunk_id), 0.0),
                semantic_rank=rank,
            )
        )
    return candidates


def keyword_search(query: str, top_k: int) -> list[_Candidate]:
    """Return the highest-scoring BM25 chunks for a query.

    Chunks with a zero score are omitted. They share no query terms, so
    they are not keyword matches.
    """
    index, chunks = _load_bm25()
    tokens = tokenize(query)
    if not tokens:
        return []
    scores = index.get_scores(tokens)
    ranked = sorted(
        ((float(score), position) for position, score in enumerate(scores) if score > 0),
        key=lambda item: item[0],
        reverse=True,
    )
    candidates: list[_Candidate] = []
    for rank, (score, position) in enumerate(ranked[:top_k], start=1):
        chunk = chunks[position]
        candidates.append(
            _Candidate(
                id=str(chunk["id"]),
                text=str(chunk["text"]),
                source=str(chunk["source"]),
                chunk_index=int(chunk["chunk_index"]),
                bm25_score=score,
                bm25_rank=rank,
            )
        )
    return candidates


def _fusion_score(candidate: _Candidate) -> float:
    """Score a chunk by reciprocal rank fusion across the two lists."""
    score = 0.0
    if candidate.semantic_rank is not None:
        score += 1.0 / (RRF_K + candidate.semantic_rank)
    if candidate.bm25_rank is not None:
        score += 1.0 / (RRF_K + candidate.bm25_rank)
    return score


def _merge_candidates(
    semantic_hits: list[_Candidate],
    keyword_hits: list[_Candidate],
    cosine_by_id: dict[str, float],
    top_k: int,
) -> list[RetrievedChunk]:
    """De-duplicate both hit lists and keep the top fused chunks."""
    merged: dict[str, _Candidate] = {}
    for hit in semantic_hits:
        merged[hit.id] = hit
    for hit in keyword_hits:
        existing = merged.get(hit.id)
        if existing is None:
            hit.cosine_similarity = cosine_by_id.get(hit.id, 0.0)
            merged[hit.id] = hit
            continue
        existing.bm25_score = hit.bm25_score
        existing.bm25_rank = hit.bm25_rank

    ranked = sorted(
        merged.values(),
        key=lambda hit: (_fusion_score(hit), hit.cosine_similarity),
        reverse=True,
    )
    return [
        RetrievedChunk(
            id=hit.id,
            text=hit.text,
            source=hit.source,
            chunk_index=hit.chunk_index,
            cosine_similarity=hit.cosine_similarity,
            bm25_score=hit.bm25_score,
            fusion_score=_fusion_score(hit),
        )
        for hit in ranked[:top_k]
    ]


def hybrid_search(query: str, top_k: int = DEFAULT_TOP_K) -> HybridSearchResult:
    """Search the knowledge base with semantic and keyword retrieval.

    Returns up to ``top_k`` unique chunks. ``confidence`` is the cosine
    similarity of the highest-ranked chunk after fusion.
    """
    cleaned = query.strip()
    if not cleaned:
        raise ValueError("Query must not be empty.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    started = time.perf_counter()
    query_vector = _embed_query(cleaned)
    collection = _get_collection()
    semantic_hits = semantic_search(query_vector, collection, top_k)
    keyword_hits = keyword_search(cleaned, top_k)

    keyword_only_ids = [
        hit.id for hit in keyword_hits if all(hit.id != semantic.id for semantic in semantic_hits)
    ]
    cosine_for_keyword_only = _cosine_by_id(collection, query_vector, keyword_only_ids)
    chunks = _merge_candidates(
        semantic_hits,
        keyword_hits,
        cosine_for_keyword_only,
        top_k,
    )
    confidence = chunks[0].cosine_similarity if chunks else 0.0
    elapsed_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "Hybrid search returned %d chunk(s) in %.0f ms (confidence=%.3f)",
        len(chunks),
        elapsed_ms,
        confidence,
    )
    return HybridSearchResult(chunks=chunks, confidence=confidence)


if __name__ == "__main__":
    settings.configure_logging()
    samples = [
        "What are your Saturday hours?",
        "How much is the late cancellation fee?",
        "Do you set broken bones?",
        "Who won the 1994 world series?",
    ]
    for sample in samples:
        found = hybrid_search(sample)
        top = found.chunks[0]
        print(
            f"\nQ: {sample}\n"
            f"confidence={found.confidence:.3f} source={top.source} "
            f"cosine={top.cosine_similarity:.3f} bm25={top.bm25_score:.3f}\n"
            f"{top.text[:180]}"
        )
