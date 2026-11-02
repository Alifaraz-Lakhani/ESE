"""Hybrid retrieval + clause-type filtering over ChromaDB + BM25.

Combines vector similarity (ChromaDB + sentence embeddings) with BM25 keyword
matching. Optionally filters results by predicted clause_type before retrieval.
Returns top-k clauses with source metadata (clause_type, contract_id, id).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHROMA_DIR = PROJECT_ROOT / ".chroma"
BM25_PATH = PROJECT_ROOT / ".bm25" / "bm25_corpus.pkl"


def retrieve_clauses(
    query: str,
    clause_type: Optional[str] = None,
    top_k: int = 5,
    collection_name: str = "clauses",
    vector_weight: float = 0.7,
    bm25_weight: float = 0.3,
) -> list[dict]:
    """Retrieve semantically and lexically similar clauses.

    Args:
        query: Natural-language clause text to match against.
        clause_type: If set, only return clauses matching this type.
        top_k: Maximum number of results to return.
        collection_name: ChromaDB collection to query.
        vector_weight: Weight for vector similarity score (0-1).
        bm25_weight: Weight for BM25 keyword score (0-1).

    Returns:
        A list of dicts with keys: text, score, clause_type, contract_id, id.
    """
    try:
        import chromadb  # type: ignore
        from sentence_transformers import SentenceTransformer  # type: ignore
        from rank_bm25 import BM25Okapi  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency
        return []

    if not CHROMA_DIR.exists():
        return []

    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collection = client.get_collection(name=collection_name)
    except Exception:  # noqa: BLE001 - collection missing or corrupt
        return []

    if collection.count() == 0:
        return []

    # ---- Vector search ----------------------------------------------------
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = model.encode(query, normalize_embeddings=True).tolist()

    where: dict = {}
    if clause_type:
        where["clause_type"] = clause_type

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k * 3,
        where=where or None,
        include=["documents", "metadatas", "distances"],
    )

    vector_scores: dict[str, float] = {}
    vector_meta: dict[str, dict] = {}
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for text, metadata, distance in zip(documents, metadatas, distances):
        cid = metadata.get("id", text)
        vector_scores[cid] = 1.0 - float(distance)  # distance → similarity
        vector_meta[cid] = {
            "text": text,
            "clause_type": metadata.get("clause_type", ""),
            "contract_id": metadata.get("contract_id", "unknown"),
            "id": cid,
        }

    # ---- BM25 keyword search ----------------------------------------------
    bm25_scores: dict[str, float] = {}
    if BM25_PATH.exists():
        with BM25_PATH.open("rb") as fh:
            bm25_corpus = pickle.load(fh)  # list of (id, text, clause_type, contract_id)

        corpus_ids = [c[0] for c in bm25_corpus]
        corpus_texts = [c[1] for c in bm25_corpus]
        corpus_types = [c[2] for c in bm25_corpus]
        corpus_contracts = [c[3] for c in bm25_corpus]

        tokenized_corpus = [text.lower().split() for text in corpus_texts]
        bm25 = BM25Okapi(tokenized_corpus)

        tokenized_query = query.lower().split()
        raw_scores = bm25.get_scores(tokenized_query)
        max_score = max(raw_scores) if len(raw_scores) > 0 else 1.0

        for i, cid in enumerate(corpus_ids):
            if clause_type and corpus_types[i] != clause_type:
                continue
            bm25_scores[cid] = raw_scores[i] / max_score if max_score > 0 else 0.0
            if cid not in vector_meta:
                vector_meta[cid] = {
                    "text": corpus_texts[i],
                    "clause_type": corpus_types[i],
                    "contract_id": corpus_contracts[i],
                    "id": cid,
                }

    # ---- Hybrid fusion ----------------------------------------------------
    hybrid: dict[str, dict] = {}
    all_ids = set(vector_scores) | set(bm25_scores)
    for cid in all_ids:
        vs = vector_scores.get(cid, 0.0)
        bs = bm25_scores.get(cid, 0.0)
        score = vector_weight * vs + bm25_weight * bs
        meta = vector_meta.get(cid, {})
        hybrid[cid] = {
            "text": meta.get("text", ""),
            "score": round(score, 4),
            "clause_type": meta.get("clause_type", ""),
            "contract_id": meta.get("contract_id", "unknown"),
            "id": cid,
        }

    # Sort by hybrid score descending, return top_k
    ranked = sorted(hybrid.values(), key=lambda h: h["score"], reverse=True)
    return ranked[:top_k]


if __name__ == "__main__":
    hits = retrieve_clauses("indemnification for third-party claims", top_k=3)
    for hit in hits:
        print(f"{hit['score']:.3f} | {hit['clause_type']} | {hit['text'][:120]}")
