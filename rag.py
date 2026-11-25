from pathlib import Path
import hashlib

import chromadb
from pypdf import PdfReader


DATA_DIR = Path("data")
DB_DIR = DATA_DIR / "chroma"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
SUPPORTED_TYPES = {".pdf", ".md", ".txt"}

client = chromadb.PersistentClient(path=str(DB_DIR))
collection = client.get_or_create_collection(
    name="knowledge_retrieval",
    metadata={"hnsw:space": "cosine"},
)


def read_file(path: Path) -> str:
    """Read text from a PDF, Markdown, or plain text file."""
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    return path.read_text(encoding="utf-8", errors="replace")


def split_into_chunks(text: str):
    """Split text into overlapping chunks."""
    text = " ".join(text.split())
    start = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))

        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary

        chunk = text[start:end].strip()
        if chunk:
            yield chunk

        if end >= len(text):
            break

        start = max(end - CHUNK_OVERLAP, start + 1)


def ingest_path(path: Path) -> int:
    """Add one supported file, or all supported files in a folder, to Chroma."""
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(
            file_path
            for file_path in path.rglob("*")
            if file_path.suffix.lower() in SUPPORTED_TYPES
        )
    else:
        raise FileNotFoundError(f"Could not find: {path}")

    indexed_chunks = 0

    for file_path in files:
        if file_path.suffix.lower() not in SUPPORTED_TYPES:
            continue

        chunks = list(split_into_chunks(read_file(file_path)))
        if not chunks:
            continue

        ids = [
            hashlib.sha256(
                f"{file_path.resolve()}:{index}:{chunk}".encode("utf-8")
            ).hexdigest()
            for index, chunk in enumerate(chunks)
        ]

        collection.upsert(
            ids=ids,
            documents=chunks,
            metadatas=[
                {
                    "source": file_path.name,
                    "path": str(file_path),
                }
                for _ in chunks
            ],
        )

        indexed_chunks += len(chunks)

    return indexed_chunks


def collection_stats() -> int:
    """Return the number of indexed text chunks."""
    return collection.count()


def search_documents(question: str, number_of_results: int = 4) -> list[dict]:
    """Find the most relevant passages for a question."""
    total_chunks = collection_stats()
    if total_chunks == 0:
        raise ValueError(
            "The knowledge base is empty. Index the sample documents first."
        )

    results = collection.query(
        query_texts=[question],
        n_results=min(number_of_results, total_chunks),
        include=["documents", "metadatas", "distances"],
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    matches = []
    for document, metadata, distance in zip(documents, metadatas, distances):
        matches.append(
            {
                "text": document,
                "source": metadata["source"],
                "score": max(0.0, 1.0 - float(distance)),
            }
        )

    return matches