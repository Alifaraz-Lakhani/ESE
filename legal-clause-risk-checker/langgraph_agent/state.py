"""Typed state schema for the LangGraph evidence-checking state machine.

All nodes read from and write to this shared state dict. The schema tracks a
single clause as it moves through parse → classify → retrieve rules → generate
finding → validate citation → answer/refuse.
"""

from __future__ import annotations

from typing import Optional, TypedDict


class ClauseState(TypedDict, total=False):
    """Mutable state carried through the evidence-checking graph.

    Fields:
        contract_text: Full contract text supplied by the user (required).
        clauses: List of clauses parsed from the contract.
        current_clause: The clause currently being analyzed.
        predicted_clause_type: Clause type predicted by the classifier.
        retrieved_rules: Deterministic rule flags and/or RAG context.
        generated_finding: Structured finding from the generate node.
        citation_valid: Whether the quoted span appears verbatim in retrieved text.
        final_answer: Final answer when citation is valid.
        refusal_reason: Reason for refusal when citation is invalid.
        use_qlora_adapter: Whether to use the fine-tuned QLoRA adapter (default True).
    """

    contract_text: str
    clauses: list[str]
    current_clause: str
    predicted_clause_type: Optional[str]
    retrieved_rules: list[dict]
    generated_finding: Optional[dict]
    citation_valid: bool
    final_answer: Optional[str]
    refusal_reason: Optional[str]
    use_qlora_adapter: bool
