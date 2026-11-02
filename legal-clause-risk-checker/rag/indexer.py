"""ChromaDB + BM25 indexing for CUAD clause data.

Builds a clause-level retrieval corpus from processed JSONL files.
Each document is one clause with metadata:
    {"clause_type": <label>, "contract_id": <source>}

Uses ChromaDB for vector storage and a pickled BM25 corpus for keyword matching.

Default embeddings: all-MiniLM-L6-v2.
To swap in bge-m3 or bge-large later, replace the model name in index_cuad():

    # model = SentenceTransformer("BAAI/bge-m3")
    # model = SentenceTransformer("BAAI/bge-large-en-v1.5")
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
CHROMA_DIR = PROJECT_ROOT / ".chroma"
BM25_PATH = PROJECT_ROOT / ".bm25" / "bm25_corpus.pkl"


def _iter_jsonl(path: Path) -> Iterable[dict]:
    """Yield parsed JSON objects from a JSONL file, skipping blank lines."""
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def index_cuad(collection_name: str = "clauses", reset: bool = False) -> int:
    """Index clause JSONL files into ChromaDB + BM25.

    Reads ``data/processed/clause_classification_*.jsonl`` and indexes each
    clause as a separate document with metadata ``{"clause_type", "contract_id"}``.

    Args:
        collection_name: Name of the ChromaDB collection.
        reset: If True, delete and recreate the collection before indexing.

    Returns:
        Number of clauses indexed.

    Raises:
        RuntimeError: If dependencies are missing or no JSONL files are found.
    """
    try:
        import chromadb  # type: ignore
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "ChromaDB and sentence-transformers are required for indexing. "
            "Install them with: pip install chromadb sentence-transformers"
        ) from exc

    jsonl_files = sorted(DATA_PROCESSED.glob("clause_classification_*.jsonl"))
    if not jsonl_files:
        raise RuntimeError(
            f"No clause_classification JSONL files found in {DATA_PROCESSED}. "
            "Run the data preparation script first."
        )

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if reset:
        try:
            client.delete_collection(collection_name)
        except Exception:  # noqa: BLE001 - collection may not exist
            pass
    collection = client.get_or_create_collection(name=collection_name)

    # Default embedding model. To swap in bge-m3 or bge-large later,
    # replace the model name below:
    #     model = SentenceTransformer("BAAI/bge-m3")
    #     model = SentenceTransformer("BAAI/bge-large-en-v1.5")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    documents: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []
    embeddings: list[list[float]] = []
    bm25_corpus: list[tuple[str, str, str, str]] = []  # (id, text, clause_type, contract_id)

    count = 0
    for jsonl_file in jsonl_files:
        contract_id = jsonl_file.stem  # e.g. "clause_classification_train"
        for record in _iter_jsonl(jsonl_file):
            text = record.get("text") or record.get("clause_text") or ""
            if not text:
                continue
            clause_type = record.get("label") or record.get("clause_type", "")
            clause_id = f"{contract_id}:{count}"

            embedding = model.encode(text, normalize_embeddings=True).tolist()

            documents.append(text)
            metadatas.append(
                {"clause_type": clause_type, "contract_id": contract_id}
            )
            ids.append(clause_id)
            embeddings.append(embedding)
            bm25_corpus.append((clause_id, text, clause_type, contract_id))
            count += 1

    if documents:
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings,
        )

    # Persist the BM25 corpus for the retriever
    BM25_PATH.parent.mkdir(parents=True, exist_ok=True)
    with BM25_PATH.open("wb") as fh:
        pickle.dump(bm25_corpus, fh)

    return count


if __name__ == "__main__":
    indexed = index_cuad()
    print(f"Indexed {indexed} clauses into ChromaDB at {CHROMA_DIR}")
