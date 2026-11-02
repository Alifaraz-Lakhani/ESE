"""LangGraph state machine definition for clause evidence checking.

The graph wires together six nodes:

    parse_contract  →  classify_clause  →  retrieve_rules  →  generate_finding
                                                  →  validate_citation  →  refuse_or_answer

A conditional edge after ``validate_citation`` routes to ``refuse_or_answer``,
which either sets ``final_answer`` (citation valid) or ``refusal_reason``
(citation invalid).
"""

from __future__ import annotations

from typing import Optional

from langgraph.graph import END, StateGraph

from .nodes import (
    classify_clause,
    generate_finding,
    parse_contract,
    refuse_or_answer,
    retrieve_rules,
    validate_citation,
)
from .state import ClauseState


def build_graph(checkpointer: Optional[object] = None) -> StateGraph:
    """Build and compile the clause evidence-checking graph.

    Args:
        checkpointer: Optional LangGraph checkpointer (e.g. SqliteSaver,
            MemorySaver) for persistence / human-in-the-loop workflows.

    Returns:
        A compiled LangGraph ``StateGraph`` ready for ``.invoke(...)``.
    """
    builder = StateGraph(ClauseState)

    # Nodes
    builder.add_node("parse_contract", parse_contract)
    builder.add_node("classify_clause", classify_clause)
    builder.add_node("retrieve_rules", retrieve_rules)
    builder.add_node("generate_finding", generate_finding)
    builder.add_node("validate_citation", validate_citation)
    builder.add_node("refuse_or_answer", refuse_or_answer)

    # Edges
    builder.set_entry_point("parse_contract")
    builder.add_edge("parse_contract", "classify_clause")
    builder.add_edge("classify_clause", "retrieve_rules")
    builder.add_edge("retrieve_rules", "generate_finding")
    builder.add_edge("generate_finding", "validate_citation")

    # Conditional edge after validate_citation: both routes go to
    # refuse_or_answer, which sets final_answer (valid) or refusal_reason.
    builder.add_conditional_edges(
        "validate_citation",
        _route_after_validation,
        {
            "answer": "refuse_or_answer",
            "refuse": "refuse_or_answer",
        },
    )
    builder.add_edge("refuse_or_answer", END)

    return builder.compile(checkpointer=checkpointer)


def _route_after_validation(state: ClauseState) -> str:
    """Route based on citation validity after validate_citation."""
    return "answer" if state.get("citation_valid") else "refuse"
