"""Streamlit entry point for the Legal Contract Clause Risk Checker.

Upload a PDF/TXT contract or paste contract text, then analyze each clause
through the LangGraph evidence-checking pipeline:

    parse_contract → classify_clause → retrieve_rules → generate_finding
    → validate_citation → refuse_or_answer

Displays a per-clause risk report (left panel: numbered clauses; right panel:
risk level, explanation, evidence) plus an agent-trace tab.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langgraph_agent.graph import build_graph  # noqa: E402
from langgraph_agent.state import ClauseState  # noqa: E402


st.set_page_config(
    page_title="Legal Contract Clause Risk Checker",
    page_icon="⚖️",
    layout="wide",
)

st.title("⚖️ Legal Contract Clause Risk Checker")
st.markdown(
    "Upload a PDF/TXT contract or paste contract text to analyze individual "
    "clauses for legal risk using an AI-powered evidence-checking pipeline."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF byte stream using PyMuPDF."""
    import fitz  # type: ignore  # PyMuPDF

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    return "\n".join(page.get_text() for page in doc)


def extract_text_from_txt(file_bytes: bytes) -> str:
    """Decode uploaded TXT bytes to a string."""
    return file_bytes.decode("utf-8", errors="replace")


def split_into_clauses(text: str) -> list[str]:
    """Split contract text into clauses on blank-line boundaries."""
    clauses = [c.strip() for c in text.split("\n\n") if c.strip()]
    return clauses or [text.strip()]


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Input")
    uploaded = st.file_uploader("Upload contract (PDF/TXT)", type=["pdf", "txt"])
    text_input = st.text_area("Or paste contract text:", height=200)

    st.divider()
    st.header("Settings")

    # QLoRA Adapter Toggle (Task 4)
    use_qlora = st.checkbox(
        "Use QLoRA Adapter",
        value=False,
        help="Enable fine-tuned QLoRA adapter for clause classification. "
             "Requires trained adapter in finetune/adapters/. "
             "Unchecked = stub classifier (keyword-based)."
    )

    if use_qlora:
        st.info("🔬 Fine-tuned model enabled. Classification will use the QLoRA adapter.")
    else:
        st.warning("📌 Using stub classifier (keyword-based). Check the box above to use fine-tuned model.")

    top_k = st.slider("RAG top-k", 1, 10, 5)

    st.divider()
    st.caption(
        "Educational demo; not a substitute for qualified legal review. "
        "Consult qualified counsel for contract review."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

contract_text = ""

if uploaded is not None:
    file_bytes = uploaded.read()
    if uploaded.type == "application/pdf":
        try:
            contract_text = extract_text_from_pdf(file_bytes)
        except Exception as exc:  # noqa: BLE001
            st.error(f"PDF extraction failed: {exc}")
    else:
        contract_text = extract_text_from_txt(file_bytes)
elif text_input.strip():
    contract_text = text_input

if contract_text.strip():
    clauses = split_into_clauses(contract_text)
    st.success(f"Loaded contract with {len(clauses)} clause(s).")

    graph = build_graph()

    # Analyze each clause through the LangGraph pipeline.
    results: list[dict] = []
    for i, clause in enumerate(clauses):
        initial_state: ClauseState = {
            "contract_text": contract_text,
            "clauses": clauses,
            "current_clause": clause,
            "use_qlora_adapter": use_qlora,  # Pass the toggle state to the pipeline
        }
        final_state = graph.invoke(initial_state)
        results.append(
            {
                "index": i + 1,
                "clause": clause,
                "result": final_state,
            }
        )

    # Tabs: Clause Review | Agent Trace
    tab_review, tab_trace = st.tabs(["Clause Review", "Agent Trace"])

    with tab_review:
        # Show model mode indicator at top
        if use_qlora:
            st.info("🔬 **Analysis Mode:** Fine-tuned QLoRA adapter")
        else:
            st.warning("📌 **Analysis Mode:** Keyword-based stub classifier")

        col_clauses, col_report = st.columns([1, 2])

        with col_clauses:
            st.subheader("Clauses")
            for r in results:
                preview = r["clause"][:400]
                if len(r["clause"]) > 400:
                    preview += "..."
                with st.expander(f"Clause {r['index']}", expanded=False):
                    st.write(r["clause"])

        with col_report:
            st.subheader("Risk Report")
            for r in results:
                finding = r["result"].get("generated_finding", {})
                final_answer = r["result"].get("final_answer")
                refusal_reason = r["result"].get("refusal_reason")

                st.markdown(f"### Clause {r['index']}")

                if refusal_reason:
                    st.error(f"🚫 Refused: {refusal_reason}")
                elif final_answer:
                    risk_level = (final_answer.get("risk_level") or "unknown").lower()
                    if risk_level == "high":
                        st.error(f"Risk Level: {risk_level.upper()}")
                    elif risk_level == "medium":
                        st.warning(f"Risk Level: {risk_level.upper()}")
                    else:
                        st.success(f"Risk Level: {risk_level.upper()}")

                    st.markdown("**Explanation:**")
                    st.write(final_answer.get("explanation", "No explanation."))

                    with st.expander("Evidence"):
                        st.write(
                            "**Quoted span:** "
                            + final_answer.get("quoted_span", "N/A")
                        )
                        st.write(
                            "**Clause type:** "
                            + str(final_answer.get("clause_type", "N/A"))
                        )
                else:
                    st.warning("No result available.")

                st.divider()

    with tab_trace:
        st.subheader("Agent Trace")
        st.markdown(
            "Execution path: **Parse Contract → Classify Clause → Retrieve Rules → "
            "Generate Finding → Validate Citation → Answer/Refuse**"
        )

        for r in results:
            state = r["result"]
            st.markdown(f"### Clause {r['index']}")

            steps = []
            if state.get("clauses"):
                steps.append(
                    ("✅ Parse Contract", f"{len(state['clauses'])} clause(s) found")
                )
            if state.get("predicted_clause_type"):
                steps.append(
                    ("✅ Classify Clause", f"Type: {state['predicted_clause_type']}")
                )
            if state.get("retrieved_rules"):
                flags = state["retrieved_rules"].get("flags", [])
                steps.append(
                    ("✅ Retrieve Rules", f"{len(flags)} risk flag(s)")
                )
            if state.get("generated_finding"):
                steps.append(
                    ("✅ Generate Finding", f"Risk: {state['generated_finding'].get('risk_level', 'N/A')}")
                )
            if "citation_valid" in state:
                steps.append(
                    ("✅ Validate Citation", f"Valid: {state['citation_valid']}")
                )
            if state.get("final_answer"):
                steps.append(("✅ Answer", "Final answer produced"))
            elif state.get("refusal_reason"):
                steps.append(("✅ Refuse", "Analysis refused"))

            for step_name, step_detail in steps:
                st.write(f"{step_name}: {step_detail}")

            st.divider()

else:
    st.info("Upload a PDF/TXT contract or paste contract text to begin.")

st.divider()
st.caption(
    "Educational demo; not a substitute for qualified legal review. "
    "Consult qualified counsel for contract review."
)
