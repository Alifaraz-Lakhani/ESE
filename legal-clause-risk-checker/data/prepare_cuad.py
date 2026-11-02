"""Prepare CUAD clause data for training and evaluation.

Loads ``data/raw/CUAD_v1.json`` (SQuAD-style format) and produces:

(a) Clause classification examples:
    {"text": <clause_text>, "label": <clause_type>}
    -> data/processed/clause_classification_train.jsonl (90%)
    -> data/processed/clause_classification_val.jsonl   (10%)

(b) Risk Q&A examples (for RAG evaluation, not training):
    {"contract_context": ..., "question": ..., "answer": ..., "supporting_span": ...}
    -> data/processed/risk_qa_eval.jsonl

Empty or whitespace-only clauses are filtered out. Impossible QA pairs are skipped.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "CUAD_v1.json"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

CLASSIFICATION_TRAIN = PROCESSED_DIR / "clause_classification_train.jsonl"
CLASSIFICATION_VAL = PROCESSED_DIR / "clause_classification_val.jsonl"
RISK_QA_EVAL = PROCESSED_DIR / "risk_qa_eval.jsonl"

SPLIT_SEED = 42
TRAIN_FRAC = 0.9


def _clause_type_from_id(qa_id: str) -> str:
    """Extract the clause type from a CUAD QA id.

    CUAD ids look like ``{document_id}__{clause_type}``, e.g.
    ``LIMEENERGYCO_...__Document Name``. The clause type is everything after
    the first ``__``.
    """
    return qa_id.split("__", 1)[1] if "__" in qa_id else "Other"


def load_cuad(path: Path) -> list[dict]:
    """Load CUAD_v1.json and return the list of document dicts."""
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data.get("data", [])


def extract_examples(documents: list[dict]) -> tuple[list[dict], list[dict]]:
    """Extract classification examples and risk Q&A examples from CUAD documents.

    Returns:
        (classification_examples, qa_examples)
    """
    classification: list[dict] = []
    qa_examples: list[dict] = []

    for doc in documents:
        contract_title = doc.get("title", "unknown")
        for paragraph in doc.get("paragraphs", []):
            context = paragraph.get("context", "")
            for qa in paragraph.get("qas", []):
                # Skip impossible / unanswerable QA pairs
                if qa.get("is_impossible", False):
                    continue

                answers = qa.get("answers") or []
                if not answers:
                    continue

                answer_text = (answers[0].get("text") or "").strip()
                if not answer_text:
                    continue

                clause_type = _clause_type_from_id(qa.get("id", ""))
                question = qa.get("question", "").strip()

                classification.append(
                    {"text": answer_text, "label": clause_type}
                )

                qa_examples.append(
                    {
                        "contract_context": context,
                        "question": question,
                        "answer": answer_text,
                        "supporting_span": answer_text,
                    }
                )

    return classification, qa_examples


def split_train_val(
    examples: list[dict], train_frac: float = TRAIN_FRAC, seed: int = SPLIT_SEED
) -> tuple[list[dict], list[dict]]:
    """Shuffle and split examples into train / validation sets."""
    shuffled = list(examples)
    random.seed(seed)
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * train_frac)
    return shuffled[:split_idx], shuffled[split_idx:]


def save_jsonl(records: list[dict], path: Path) -> None:
    """Save a list of dicts as JSONL (one JSON object per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"CUAD data not found at {RAW_PATH}")

    documents = load_cuad(RAW_PATH)
    print(f"Loaded {len(documents)} documents from {RAW_PATH}")

    classification, qa_examples = extract_examples(documents)
    print(
        f"Extracted {len(classification)} classification examples and "
        f"{len(qa_examples)} Q&A examples"
    )

    train_examples, val_examples = split_train_val(classification)
    print(
        f"Split: {len(train_examples)} train / {len(val_examples)} val "
        f"({TRAIN_FRAC:.0%} / {1 - TRAIN_FRAC:.0%})"
    )

    save_jsonl(train_examples, CLASSIFICATION_TRAIN)
    save_jsonl(val_examples, CLASSIFICATION_VAL)
    save_jsonl(qa_examples, RISK_QA_EVAL)

    print(f"Saved {CLASSIFICATION_TRAIN}")
    print(f"Saved {CLASSIFICATION_VAL}")
    print(f"Saved {RISK_QA_EVAL}")


if __name__ == "__main__":
    main()
