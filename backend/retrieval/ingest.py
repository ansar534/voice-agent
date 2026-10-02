"""Knowledge base ingestion: text files -> chunks -> ChromaDB + BM25.

Run it directly to rebuild both indexes from scratch:
    python -m backend.retrieval.ingest
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from ..config import (
    BM25_INDEX_PATH,
    CHROMA_PERSIST_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    KNOWLEDGE_BASE_DIR,
)

_TOKEN_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lowercase text and split it into word tokens for BM25 scoring."""
    return _TOKEN_PATTERN.findall(text.lower())


def _split_oversized(piece: str, chunk_size: int, overlap: int) -> list[str]:
    """Hard-split a single block that is longer than one chunk.

    Used only when a paragraph cannot fit, so the sliding window is a
    last resort rather than the normal path.
    """
    pieces: list[str] = []
    start = 0
    while start < len(piece):
        end = start + chunk_size
        window = piece[start:end].strip()
        if window:
            pieces.append(window)
        if end >= len(piece):
            break
        start = end - overlap
    return pieces


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks of at most ``chunk_size`` characters.

    Paragraphs are packed together up to the size limit rather than the
    text being sliced at blind character offsets. Cutting mid-sentence
    measurably hurts retrieval: it strips the subject from the sentence
    that answers the question. Consecutive chunks share the tail of the
    previous one so a fact near a boundary is still findable.
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be larger than overlap.")

    paragraphs = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_split_oversized(paragraph, chunk_size, overlap))
            continue

        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        chunks.append(current)
        # Carry the tail of the finished chunk into the next one so the
        # boundary between them is not a hard information cut.
        tail = current[-overlap:].lstrip()
        current = f"{tail}\n\n{paragraph}" if tail else paragraph

    if current:
        chunks.append(current)
    return chunks


def load_documents(knowledge_base_dir: Path = KNOWLEDGE_BASE_DIR) -> list[dict[str, str]]:
    """Read every .txt file in the knowledge base directory.

    Returns one record per file as {"source": filename, "text": contents}.
    Empty files are skipped.
    """
    if not knowledge_base_dir.is_dir():
        raise FileNotFoundError(f"Knowledge base directory not found: {knowledge_base_dir}")

    documents: list[dict[str, str]] = []
    for path in sorted(knowledge_base_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            print(f"  ! Skipping empty file: {path.name}")
            continue
        documents.append({"source": path.name, "text": text})
        print(f"  - Loaded {path.name} ({len(text):,} characters)")

    if not documents:
        raise ValueError(f"No .txt documents found in {knowledge_base_dir}")
    return documents


def build_chunks(documents: list[dict[str, str]]) -> list[dict[str, object]]:
    """Chunk every document and attach source metadata to each piece.

    Each chunk records which file it came from, its position in that
    file, and how many chunks that file produced.
    """
    chunks: list[dict[str, object]] = []
    for document in documents:
        pieces = chunk_text(document["text"])
        total = len(pieces)
        for index, piece in enumerate(pieces):
            chunks.append(
                {
                    "id": f"{document['source']}::{index}",
                    "text": piece,
                    "source": document["source"],
                    "chunk_index": index,
                    "total_chunks": total,
                }
            )
        print(f"  - {document['source']}: {total} chunk(s)")

    if not chunks:
        raise ValueError("Chunking produced no chunks.")
    return chunks


def embed_chunks(chunks: list[dict[str, object]]) -> list[list[float]]:
    """Embed chunk texts locally with all-MiniLM-L6-v2.

    Embeddings are L2-normalized so a dot product between any two of
    them is their cosine similarity.
    """
    texts = [str(chunk["text"]) for chunk in chunks]
    model = SentenceTransformer(EMBEDDING_MODEL)
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return vectors.tolist()


def store_in_chroma(chunks: list[dict[str, object]], embeddings: list[list[float]]) -> int:
    """Write chunks and embeddings into a fresh ChromaDB collection.

    Any existing collection is deleted first so a re-ingest can never
    leave stale chunks from an older version of the documents behind.
    """
    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))

    existing = [collection.name for collection in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        print(f"  - Deleted existing collection '{COLLECTION_NAME}'")

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(
        ids=[str(chunk["id"]) for chunk in chunks],
        documents=[str(chunk["text"]) for chunk in chunks],
        embeddings=embeddings,
        metadatas=[
            {
                "source": str(chunk["source"]),
                "chunk_index": int(chunk["chunk_index"]),
                "total_chunks": int(chunk["total_chunks"]),
            }
            for chunk in chunks
        ],
    )
    return collection.count()


def build_bm25_index(chunks: list[dict[str, object]]) -> None:
    """Fit a BM25 index over the same chunks and pickle it to disk.

    The tokenized corpus is stored rather than the fitted object alone,
    so the index can be rebuilt on load and stays portable across
    rank_bm25 versions.
    """
    tokenized_corpus = [tokenize(str(chunk["text"])) for chunk in chunks]
    if not any(tokenized_corpus):
        raise ValueError("BM25 corpus is empty after tokenization.")

    index = BM25Okapi(tokenized_corpus)
    payload = {
        "index": index,
        "tokenized_corpus": tokenized_corpus,
        "texts": [str(chunk["text"]) for chunk in chunks],
        "sources": [str(chunk["source"]) for chunk in chunks],
        "ids": [str(chunk["id"]) for chunk in chunks],
    }
    BM25_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with BM25_INDEX_PATH.open("wb") as handle:
        pickle.dump(payload, handle)


def run_ingestion() -> dict[str, object]:
    """Run the full pipeline and return a summary of what was indexed."""
    print("=" * 60)
    print("KNOWLEDGE BASE INGESTION")
    print("=" * 60)

    print(f"\n[1/5] Loading documents from {KNOWLEDGE_BASE_DIR}")
    documents = load_documents()

    print(f"\n[2/5] Chunking (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    chunks = build_chunks(documents)
    print(f"  = {len(chunks)} chunk(s) total")

    print(f"\n[3/5] Embedding with {EMBEDDING_MODEL} (local, no API key)")
    embeddings = embed_chunks(chunks)
    print(f"  = {len(embeddings)} vector(s) of dimension {len(embeddings[0])}")

    print(f"\n[4/5] Storing in ChromaDB at {CHROMA_PERSIST_DIR}")
    stored = store_in_chroma(chunks, embeddings)
    print(f"  = collection '{COLLECTION_NAME}' holds {stored} chunk(s)")

    print(f"\n[5/5] Building BM25 keyword index at {BM25_INDEX_PATH}")
    build_bm25_index(chunks)
    print("  = BM25 index saved")

    return {
        "documents": len(documents),
        "chunks_stored": stored,
        "embedding_model": EMBEDDING_MODEL,
        "chroma_persist_dir": str(CHROMA_PERSIST_DIR),
        "bm25_index_path": str(BM25_INDEX_PATH),
    }


if __name__ == "__main__":
    summary = run_ingestion()
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for key, value in summary.items():
        print(f"  {key:22} {value}")
    print("\nIngestion complete.\n")
