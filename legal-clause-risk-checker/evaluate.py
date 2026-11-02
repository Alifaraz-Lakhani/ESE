"""Evaluation script for the Legal Clause Risk Checker pipeline.

Evaluates the complete pipeline against data/processed/risk_qa_eval.jsonl:
  1. Classification Accuracy: Does the QLoRA model predict the correct clause type?
  2. Retrieval Hit Rate: Does the supporting span appear in top-k RAG results?
  3. Refusal Rate: Does citation validation correctly refuse on fake/invalid clauses?

Usage:
    python evaluate.py [--max-samples N] [--use-adapter]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from langgraph_agent.graph import build_graph
from langgraph_agent.state import ClauseState
from langgraph_agent.nodes import clear_model_cache

PROJECT_ROOT = Path(__file__).resolve().parent
EVAL_DATA = PROJECT_ROOT / "data" / "processed" / "risk_qa_eval.jsonl"


def load_eval_data(path: Path, max_samples: Optional[int] = None) -> list[dict]:
    """Load evaluation data from JSONL file."""
    records: list[dict] = []
    if not path.exists():
        print(f"Warning: Evaluation data not found at {path}")
        return records

    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
                if max_samples and len(records) >= max_samples:
                    break
    return records


def evaluate_classification(
    records: list[dict], use_adapter: bool = True
) -> dict[str, float]:
    """Evaluate classification accuracy of the clause classifier.

    Args:
        records: List of evaluation records with 'answer' (clause text) and
                 'question' (which should hint at the clause type).
        use_adapter: Whether to use the QLoRA adapter or stub classifier.

    Returns:
        Dict with 'accuracy', 'total', 'correct' counts.
    """
    graph = build_graph()
    correct = 0
    total = 0
    skipped = 0

    for i, record in enumerate(records):
        clause_text = record.get("answer", "").strip()
        if not clause_text:
            skipped += 1
            continue

        # Infer expected clause type from the question (CUAD format)
        question = record.get("question", "").lower()
        expected_type = _infer_clause_type_from_question(question)

        if not expected_type:
            skipped += 1
            continue

        # Run classification through the graph
        state: ClauseState = {
            "contract_text": clause_text,
            "current_clause": clause_text,
            "use_qlora_adapter": use_adapter,
        }

        try:
            result = graph.invoke(state)
            predicted = result.get("predicted_clause_type", "").strip()

            # Count empty/None predictions as incorrect (not skipped)
            if not predicted:
                predicted = "UNPARSEABLE"

            expected = expected_type.strip()

            # Normalize and compare (partial match to handle variations)
            # Case-insensitive comparison
            predicted_lower = predicted.lower()
            expected_lower = expected.lower()

            if expected_lower in predicted_lower or predicted_lower in expected_lower:
                correct += 1

            # Always count this sample (even if prediction is bad)
            total += 1

            # Debug logging for first few samples
            if i < 5:
                print(f"  Sample {i+1}: Expected='{expected}', Predicted='{predicted}', Match={expected_lower in predicted_lower or predicted_lower in expected_lower}")

        except Exception as e:
            # Count exceptions as incorrect (not skipped)
            print(f"  Error on sample {i+1}: {e}")
            total += 1
            continue

    accuracy = correct / total if total > 0 else 0.0
    print(f"  Evaluated: {total} samples, Skipped: {skipped} (no clause type in question)")
    return {"accuracy": accuracy, "correct": correct, "total": total, "skipped": skipped}


def _infer_clause_type_from_question(question: str) -> Optional[str]:
    """Infer clause type from CUAD question text.

    Maps CUAD question patterns to the 41 official clause types.
    """
    q = question.lower()

    # Direct keyword matches for all 41 CUAD clause types
    if "document name" in q:
        return "Document Name"
    if "parties" in q and "party" in q:
        return "Parties"
    if "agreement date" in q:
        return "Agreement Date"
    if "effective date" in q:
        return "Effective Date"
    if "expiration date" in q:
        return "Expiration Date"
    if "renewal term" in q:
        return "Renewal Term"
    if "notice period" in q and ("terminat" in q or "renew" in q):
        return "Notice Period To Terminate Renewal"
    if "governing law" in q or "jurisdiction" in q:
        return "Governing Law"
    if "most favored nation" in q or "mfn" in q:
        return "Most Favored Nation"
    if "non-compete" in q or "non compete" in q or "noncompete" in q:
        return "Non-Compete"
    if "exclusivity" in q or "exclusive" in q:
        return "Exclusivity"
    if "no-solicit" in q and "customer" in q:
        return "No-Solicit Of Customers"
    if "no-solicit" in q and "employee" in q:
        return "No-Solicit Of Employees"
    if "non-disparagement" in q or "disparage" in q:
        return "Non-Disparagement"
    if "termination for convenience" in q:
        return "Termination For Convenience"
    if "terminat" in q:  # General termination
        return "Termination For Convenience"
    if "rofr" in q or "rofo" in q or "rofn" in q or "right of first" in q:
        return "Rofr/Rofo/Rofn"
    if "change of control" in q or "change-of-control" in q:
        return "Change Of Control"
    if "anti-assignment" in q or "assignment" in q:
        return "Anti-Assignment"
    if "revenue" in q or "profit sharing" in q:
        return "Revenue/Profit Sharing"
    if "price restriction" in q:
        return "Price Restrictions"
    if "minimum commitment" in q:
        return "Minimum Commitment"
    if "volume restriction" in q:
        return "Volume Restriction"
    if "ip ownership" in q or ("intellectual property" in q and "ownership" in q):
        return "IP Ownership Assignment"
    if "joint ip" in q:
        return "Joint IP Ownership"
    if "license grant" in q:
        return "License Grant"
    if "non-transferable" in q and "license" in q:
        return "Non-Transferable License"
    if "affiliate" in q and "licensor" in q:
        return "Affiliate License-Licensor"
    if "affiliate" in q and "licensee" in q:
        return "Affiliate License-Licensee"
    if ("unlimited" in q or "all-you-can-eat" in q) and "license" in q:
        return "Unlimited/All-You-Can-Eat-License"
    if ("irrevocable" in q or "perpetual" in q) and "license" in q:
        return "Irrevocable Or Perpetual License"
    if "source code escrow" in q or "escrow" in q:
        return "Source Code Escrow"
    if "post-termination" in q and "service" in q:
        return "Post-Termination Services"
    if "audit" in q:
        return "Audit Rights"
    if "uncapped liability" in q or ("liability" in q and "uncap" in q):
        return "Uncapped Liability"
    if "cap on liability" in q or ("liability" in q and "cap" in q):
        return "Cap On Liability"
    if "liquidated damage" in q:
        return "Liquidated Damages"
    if "warranty duration" in q:
        return "Warranty Duration"
    if "warrant" in q:  # General warranty
        return "Warranty Duration"
    if "insurance" in q:
        return "Insurance"
    if "covenant not to sue" in q:
        return "Covenant Not To Sue"
    if "third party beneficiary" in q or "third-party beneficiary" in q:
        return "Third Party Beneficiary"
    if "indemnif" in q:
        return "Indemnification"

    # Fallback for common patterns
    if "liabilit" in q:
        return "Cap On Liability"
    if "payment" in q or "fee" in q:
        return "Payment"  # Not official CUAD but common
    if "confidential" in q:
        return "Confidentiality"  # Not official CUAD but common

    return None


def evaluate_retrieval(records: list[dict], top_k: int = 5) -> dict[str, float]:
    """Evaluate retrieval hit rate: does the supporting span appear in top-k results?

    Args:
        records: List of evaluation records with 'supporting_span' and 'answer'.
        top_k: Number of top RAG results to check.

    Returns:
        Dict with 'hit_rate', 'hits', 'total' counts.
    """
    graph = build_graph()
    hits = 0
    total = 0

    for record in records:
        clause_text = record.get("answer", "").strip()
        supporting_span = record.get("supporting_span", "").strip()

        if not clause_text or not supporting_span:
            continue

        # Run retrieval through the graph
        state: ClauseState = {
            "contract_text": clause_text,
            "current_clause": clause_text,
            "use_qlora_adapter": False,  # Retrieval is independent of classifier
        }

        try:
            result = graph.invoke(state)
            rag_results = result.get("retrieved_rules", {}).get("rag_results", [])

            # Check if supporting span appears in any top-k result
            found = False
            for rag_result in rag_results[:top_k]:
                if supporting_span.lower() in rag_result.get("text", "").lower():
                    found = True
                    break

            if found:
                hits += 1
            total += 1

        except Exception as e:
            print(f"Error evaluating retrieval: {e}")
            continue

    hit_rate = hits / total if total > 0 else 0.0
    return {"hit_rate": hit_rate, "hits": hits, "total": total}


def evaluate_refusal(fake_clauses: list[str]) -> dict[str, float]:
    """Evaluate refusal rate: does citation validation refuse on fake/invalid clauses?

    Args:
        fake_clauses: List of synthetic clauses that should trigger refusal.

    Returns:
        Dict with 'refusal_rate', 'refused', 'total' counts.
    """
    graph = build_graph()
    refused = 0
    total = 0

    for clause in fake_clauses:
        state: ClauseState = {
            "contract_text": clause,
            "current_clause": clause,
            "use_qlora_adapter": False,
        }

        try:
            result = graph.invoke(state)

            # Check if the graph refused (refusal_reason present, citation_valid=False)
            if result.get("refusal_reason") or not result.get("citation_valid"):
                refused += 1
            total += 1

        except Exception as e:
            print(f"Error evaluating refusal: {e}")
            continue

    refusal_rate = refused / total if total > 0 else 0.0
    return {"refusal_rate": refusal_rate, "refused": refused, "total": total}


def main(max_samples: Optional[int] = None, use_adapter: bool = True) -> None:
    """Run full evaluation pipeline."""
    print("=" * 70)
    print("Legal Clause Risk Checker - Evaluation Report")
    print("=" * 70)
    print()

    # Load evaluation data
    print(f"Loading evaluation data from {EVAL_DATA}...")
    records = load_eval_data(EVAL_DATA, max_samples=max_samples)
    print(f"Loaded {len(records)} records")
    print()

    if not records:
        print("No evaluation data found. Run data/prepare_cuad.py first.")
        return

    # Limit records for quick testing
    if max_samples:
        records = records[:max_samples]

    # 1. Classification Accuracy
    print("-" * 70)
    print("1. CLASSIFICATION ACCURACY")
    print("-" * 70)
    print(f"Using: {'QLoRA Adapter' if use_adapter else 'Stub Classifier'}")
    print()

    class_results = evaluate_classification(records[:50], use_adapter=use_adapter)
    print()
    print(f"Accuracy: {class_results['accuracy']:.2%}")
    print(f"Correct: {class_results['correct']} / {class_results['total']}")
    print(f"Skipped: {class_results['skipped']} (no clause type in question)")
    print()

    # 2. Retrieval Hit Rate
    print("-" * 70)
    print("2. RETRIEVAL HIT RATE")
    print("-" * 70)
    print("Checking if supporting spans appear in top-5 RAG results...")
    print()

    retrieval_results = evaluate_retrieval(records[:50], top_k=5)
    print(f"Hit Rate: {retrieval_results['hit_rate']:.2%}")
    print(f"Hits: {retrieval_results['hits']} / {retrieval_results['total']}")
    print()

    # 3. Refusal Rate
    print("-" * 70)
    print("3. REFUSAL RATE")
    print("-" * 70)
    print("Testing citation validation on synthetic invalid clauses...")
    print()

    # Fake clauses that should NOT appear in the RAG corpus
    fake_clauses = [
        "This clause is completely fabricated and should not exist in any contract.",
        "The moon shall be liable for all cosmic damages in perpetuity.",
        "Parties agree to teleport annually for contract review meetings.",
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod.",
        "This sentence contains words that form no legal meaning whatsoever.",
    ]

    refusal_results = evaluate_refusal(fake_clauses)
    print(f"Refusal Rate: {refusal_results['refusal_rate']:.2%}")
    print(f"Refused: {refusal_results['refused']} / {refusal_results['total']}")
    print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Classification Accuracy: {class_results['accuracy']:.2%}")
    print(f"Retrieval Hit Rate:     {retrieval_results['hit_rate']:.2%}")
    print(f"Refusal Rate:           {refusal_results['refusal_rate']:.2%}")
    print()
    print("Evaluation complete.")
    print("=" * 70)

    # Clean up model cache
    clear_model_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the Legal Clause Risk Checker pipeline"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit evaluation to N samples (for quick testing)",
    )
    parser.add_argument(
        "--use-adapter",
        action="store_true",
        help="Use QLoRA adapter for classification (requires trained adapter)",
    )
    args = parser.parse_args()

    main(max_samples=args.max_samples, use_adapter=args.use_adapter)
