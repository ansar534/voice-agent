"""Load knowledge-base documents, chunk them, and index them for hybrid search.

Text and PDF files under data/knowledge_base/ are split into overlapping
chunks, embedded locally with all-MiniLM-L6-v2, stored in ChromaDB, and
written to a BM25 keyword index beside the vector store.
"""

from __future__ import annotations

import logging
import pickle
import re
import time
from pathlib import Path

import chromadb
from chromadb.errors import NotFoundError
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from backend.config import settings

logger = logging.getLogger(__name__)

TEXT_EXTENSIONS = {".txt", ".md"}
PDF_EXTENSIONS = {".pdf"}
_TOKEN_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lowercase a string and return its word tokens for BM25."""
    return _TOKEN_PATTERN.findall(text.lower())


def load_text_file(path: Path) -> str:
    """Read a UTF-8 text file. Undecodable bytes are replaced, not dropped."""
    return path.read_text(encoding="utf-8", errors="replace")


def load_pdf_file(path: Path) -> str:
    """Extract text from every page of a PDF and join the pages with newlines."""
    reader = PdfReader(str(path))
    pages: list[str] = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def load_documents(knowledge_base_dir: Path) -> list[dict[str, str]]:
    """Load every supported document from the knowledge-base directory.

    Each item is ``{"source": filename, "text": full document text}``.
    Empty files are skipped. Unsupported extensions are logged and ignored.
    """
    if not knowledge_base_dir.is_dir():
        raise FileNotFoundError(
            f"Knowledge base directory does not exist: {knowledge_base_dir}"
        )

    documents: list[dict[str, str]] = []
    for path in sorted(knowledge_base_dir.iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue
        suffix = path.suffix.lower()
        if suffix in TEXT_EXTENSIONS:
            text = load_text_file(path)
        elif suffix in PDF_EXTENSIONS:
            text = load_pdf_file(path)
        else:
            logger.warning("Skipping unsupported file: %s", path.name)
            continue

        text = text.strip()
        if not text:
            logger.warning("Skipping empty document: %s", path.name)
            continue
        documents.append({"source": path.name, "text": text})
        logger.info("Loaded %s (%d characters)", path.name, len(text))

    if not documents:
        raise ValueError(
            f"No text or PDF documents found in {knowledge_base_dir}"
        )
    return documents


def split_documents(documents: list[dict[str, str]]) -> list[dict[str, str | int]]:
    """Split each document into overlapping character chunks.

    Chunk size and overlap come from config (500 / 50). Metadata keeps the
    source filename and the chunk's position inside that file.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
    )
    chunks: list[dict[str, str | int]] = []
    for document in documents:
        pieces = splitter.split_text(document["text"])
        for index, piece in enumerate(pieces):
            piece = piece.strip()
            if not piece:
                continue
            chunks.append(
                {
                    "id": f"{document['source']}::{index}",
                    "text": piece,
                    "source": document["source"],
                    "chunk_index": index,
                }
            )
    if not chunks:
        raise ValueError("Document splitting produced zero chunks.")
    logger.info(
        "Split %d document(s) into %d chunk(s) (size=%d, overlap=%d)",
        len(documents),
        len(chunks),
        settings.chunk_size,
        settings.chunk_overlap,
    )
    return chunks


def embed_chunks(chunks: list[dict[str, str | int]]) -> list[list[float]]:
    """Embed chunk texts locally with all-MiniLM-L6-v2.

    Vectors are L2-normalized so later cosine similarity is a dot product
    in the range 0 to 1 for typical queries. The model is downloaded on
    first use and then loaded from the local Hugging Face cache.
    """
    texts = [str(chunk["text"]) for chunk in chunks]
    started = time.perf_counter()
    try:
        model = SentenceTransformer(settings.embedding_model_name)
        vectors = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to embed {len(texts)} chunk(s) with "
            f"{settings.embedding_model_name}: {exc}"
        ) from exc
    elapsed_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "Embedded %d chunk(s) with %s in %.0f ms",
        len(texts),
        settings.embedding_model_name,
        elapsed_ms,
    )
    return vectors.tolist()


def store_in_chroma(
    chunks: list[dict[str, str | int]],
    embeddings: list[list[float]],
) -> int:
    """Replace the Chroma collection with the current chunks and embeddings.

    Re-ingestion deletes the previous collection first so stale chunks
    from an older knowledge base cannot be retrieved.
    """
    settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(settings.chroma_persist_dir))
    try:
        client.delete_collection(settings.collection_name)
        logger.info("Removed existing collection '%s'", settings.collection_name)
    except NotFoundError:
        logger.info("No existing collection '%s' to replace", settings.collection_name)

    collection = client.create_collection(
        name=settings.collection_name,
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
            }
            for chunk in chunks
        ],
    )
    logger.info(
        "Stored %d chunk(s) in Chroma at %s",
        len(chunks),
        settings.chroma_persist_dir,
    )
    return len(chunks)


def build_bm25_index(chunks: list[dict[str, str | int]]) -> BM25Okapi:
    """Fit a BM25 index on the same chunks stored in Chroma and save it.

    The pickle holds the chunk records and the tokenized corpus. Hybrid
    retrieval rebuilds BM25Okapi from that corpus so the index stays
    loadable across rank_bm25 versions.
    """
    tokenized_corpus = [tokenize(str(chunk["text"])) for chunk in chunks]
    if not any(tokenized_corpus):
        raise ValueError("BM25 corpus is empty after tokenization.")

    index = BM25Okapi(tokenized_corpus)
    settings.bm25_index_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "chunks": chunks,
        "tokenized_corpus": tokenized_corpus,
    }
    with settings.bm25_index_path.open("wb") as handle:
        pickle.dump(payload, handle)
    logger.info("Wrote BM25 index to %s", settings.bm25_index_path)
    return index


def ingest(knowledge_base_dir: Path | None = None) -> dict[str, int | str]:
    """Run the full ingestion pipeline and return a short summary.

    The summary is what the future POST /ingest endpoint will return:
    how many documents and chunks were indexed, and where they were stored.
    """
    settings.configure_logging()
    directory = knowledge_base_dir or settings.knowledge_base_dir
    documents = load_documents(directory)
    chunks = split_documents(documents)
    embeddings = embed_chunks(chunks)
    stored = store_in_chroma(chunks, embeddings)
    build_bm25_index(chunks)
    summary: dict[str, int | str] = {
        "documents": len(documents),
        "chunks": stored,
        "chroma_persist_dir": str(settings.chroma_persist_dir),
        "bm25_index_path": str(settings.bm25_index_path),
        "embedding_model": settings.embedding_model_name,
    }
    logger.info("Ingestion complete: %s", summary)
    return summary


if __name__ == "__main__":
    result = ingest()
    print(result)
