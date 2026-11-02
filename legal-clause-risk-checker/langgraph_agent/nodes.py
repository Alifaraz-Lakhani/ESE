"""Retrieve, classify, generate, validate, and refuse nodes for the graph.

Each node is a pure function ``ClauseState -> ClauseState`` (it returns an
updated copy of the state dict). Nodes degrade gracefully when external
services (LLM, ChromaDB, fine-tuned adapter) are unavailable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from rag.retriever import retrieve_clauses
from rules.risk_rules import risk_flags, highest_risk_level
from .state import ClauseState

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ADAPTER_DIR = PROJECT_ROOT / "finetune" / "adapters"
MODEL_DIR = PROJECT_ROOT / "models" / "Qwen2.5-1.5B-Instruct"

# Global cache for the model to avoid reloading on every call
_ADAPTER_CACHE: Optional[tuple[Any, Any]] = None


def clear_model_cache() -> None:
    """Clear the global model cache. Useful for Streamlit reruns or memory management."""
    global _ADAPTER_CACHE
    _ADAPTER_CACHE = None


# ---------------------------------------------------------------------------
# Node 1: parse_contract
# ---------------------------------------------------------------------------

def parse_contract(state: ClauseState) -> ClauseState:
    """Split contract text into clauses and select the first as current.

    If ``current_clause`` is already set by the caller, it is respected and
    re-splitting is skipped — this lets the app iterate over clauses one at a
    time by pre-setting ``current_clause`` before invoking the graph.
    """
    contract_text = state.get("contract_text", "")

    # If current_clause already set by caller, respect it
    if state.get("current_clause"):
        clauses = state.get("clauses") or []
        if not clauses:
            clauses = [state["current_clause"]]
        return state

    clauses = [c.strip() for c in contract_text.split("\n\n") if c.strip()]
    if not clauses:
        clauses = [contract_text.strip()]
    current_clause = clauses[0] if clauses else ""
    return {**state, "clauses": clauses, "current_clause": current_clause}


# ---------------------------------------------------------------------------
# Node 2: classify_clause
# ---------------------------------------------------------------------------

def _stub_classify(clause_text: str) -> str:
    """Keyword-based fallback classifier when no fine-tuned adapter exists."""
    text = clause_text.lower()
    if "indemnif" in text:
        return "Indemnification"
    if "liabilit" in text:
        return "Limitation of Liability"
    if "terminat" in text:
        return "Termination"
    if "confidential" in text:
        return "Confidentiality"
    if "payment" in text or " pay " in text or text.startswith("pay"):
        return "Payment"
    if "governing law" in text:
        return "Governing Law"
    if "warrant" in text:
        return "Warranty"
    if "intellectual property" in text or " ip " in text:
        return "IP Ownership"
    return "Other"


def _load_adapter_classifier(use_adapter: bool = True):
    """Load fine-tuned Qwen model + LoRA adapter from ``finetune/adapters/``.

    CRITICAL: Uses global cache to avoid reloading on every call (RTX 4050 6GB VRAM constraint).
    Loads model in 4-bit quantization with low_cpu_mem_usage=True for 2GB system RAM.

    Args:
        use_adapter: If True, load the LoRA adapter; if False, use base model only.

    Returns:
        ``(model, tokenizer)`` or ``None`` if model cannot be loaded.
    """
    global _ADAPTER_CACHE

    # Return cached model if already loaded
    if _ADAPTER_CACHE is not None:
        return _ADAPTER_CACHE

    # Check if adapter exists (only if use_adapter=True)
    adapter_config_path = ADAPTER_DIR / "adapter_config.json"
    if use_adapter and not adapter_config_path.exists():
        return None

    try:
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # type: ignore
        from peft import PeftModel  # type: ignore

        # Use local model if available, otherwise download from HF
        if MODEL_DIR.exists():
            model_path = str(MODEL_DIR)
        else:
            model_path = "Qwen/Qwen2.5-1.5B-Instruct"

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 4-bit quantization config for RTX 4050 6GB VRAM
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        # Load base model with 4-bit quantization
        # CRITICAL: low_cpu_mem_usage=True for 2GB system RAM constraint
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        # Load LoRA adapter if requested and available
        if use_adapter and adapter_config_path.exists():
            model = PeftModel.from_pretrained(model, str(ADAPTER_DIR))

        model.eval()

        # Cache the loaded model globally
        _ADAPTER_CACHE = (model, tokenizer)
        return _ADAPTER_CACHE

    except Exception as e:  # noqa: BLE001 - degrade gracefully to stub
        print(f"Warning: Failed to load model: {e}")
        return None


def _classify_with_adapter(model: Any, tokenizer: Any, clause_text: str) -> str:
    """Use the fine-tuned adapter to classify a clause.

    Robust parsing extracts clause type from various output formats:
    - Direct label: "Indemnification"
    - With prefix: "Label: Indemnification"
    - JSON-like: {"clause_type": "Indemnification"}
    - Fuzzy match against CUAD labels
    """
    import torch  # type: ignore
    import re

    prompt = f"Clause: {clause_text}\nLabel: "
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=1024
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=64,  # Increased from 32 to handle longer outputs
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    generated = outputs[0][input_len:]
    raw_output = tokenizer.decode(generated, skip_special_tokens=True).strip()

    # Parse the output with robust extraction
    prediction = _extract_clause_type(raw_output)
    return prediction


def _extract_clause_type(raw_output: str) -> str:
    """Extract clause type from model output using multiple strategies.

    The model often generates extra text after the label, so we extract
    the FIRST valid clause type found.

    Tries in order:
    1. First line that matches a CUAD label
    2. After "Label:", "clause_type:", etc. (first match)
    3. Inside JSON-like structure
    4. Any CUAD label in first 100 chars
    5. Fallback to "Other"
    """
    import re

    # CUAD clause types (41 total)
    CUAD_LABELS = [
        "Document Name", "Parties", "Agreement Date", "Effective Date", "Expiration Date",
        "Renewal Term", "Notice Period To Terminate Renewal", "Governing Law",
        "Most Favored Nation", "Non-Compete", "Exclusivity", "No-Solicit Of Customers",
        "No-Solicit Of Employees", "Non-Disparagement", "Termination For Convenience",
        "Rofr/Rofo/Rofn", "Change Of Control", "Anti-Assignment", "Revenue/Profit Sharing",
        "Price Restrictions", "Minimum Commitment", "Volume Restriction",
        "IP Ownership Assignment", "Joint IP Ownership", "License Grant",
        "Non-Transferable License", "Affiliate License-Licensor", "Affiliate License-Licensee",
        "Unlimited/All-You-Can-Eat-License", "Irrevocable Or Perpetual License",
        "Source Code Escrow", "Post-Termination Services", "Audit Rights",
        "Uncapped Liability", "Cap On Liability", "Liquidated Damages",
        "Warranty Duration", "Insurance", "Covenant Not To Sue", "Third Party Beneficiary",
        "Indemnification",
    ]

    # Common abbreviations/variations that map to CUAD labels
    LABEL_ALIASES = {
        "termination": "Termination For Convenience",
        "liability": "Cap On Liability",
        "warranty": "Warranty Duration",
        "assignment": "Anti-Assignment",
        "ip ownership": "IP Ownership Assignment",
        "license": "License Grant",
        "renewal": "Renewal Term",
        "non compete": "Non-Compete",
        "change control": "Change Of Control",
        "audit": "Audit Rights",
        "insurance": "Insurance",
        "parties": "Parties",
    }

    # Normalize for comparison
    output_lower = raw_output.lower().strip()

    # Get first 200 chars for focused matching (model generates too much)
    output_first = raw_output[:200]
    output_first_lower = output_first.lower()

    # Strategy 1: Check FIRST LINE for exact label match
    first_line = raw_output.split('\n')[0].strip()
    for label in CUAD_LABELS:
        if label.lower() == first_line.lower():
            return label
        # Check if first line starts with the label
        if first_line.lower().startswith(label.lower()):
            return label

    # Strategy 2: Extract after common prefixes (in first 200 chars)
    patterns = [
        r'label:\s*([^\n,\.]+)',
        r'clause[_\s]?type:\s*([^\n,\.]+)',
        r'type:\s*([^\n,\.]+)',
        r'classification:\s*([^\n,\.]+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, output_first_lower, re.IGNORECASE)
        if match:
            extracted = match.group(1).strip()

            # First try exact match
            for label in CUAD_LABELS:
                if label.lower() == extracted.lower():
                    return label

            # Then try contains match (but check label is at start)
            for label in CUAD_LABELS:
                if extracted.lower().startswith(label.lower()):
                    return label

            # Finally try aliases
            for alias, full_label in LABEL_ALIASES.items():
                if alias in extracted.lower():
                    return full_label

    # Strategy 3: Check for CUAD labels at the START of output (first 100 chars)
    for label in CUAD_LABELS:
        # Case-insensitive match at the beginning
        if output_first_lower.startswith(label.lower()):
            return label
        # Or after some whitespace at the beginning
        if re.match(rf'^\s*{re.escape(label.lower())}', output_first_lower):
            return label

    # Strategy 4: JSON-like extraction
    json_match = re.search(r'["\']?clause_type["\']?\s*:\s*["\']([^"\']+)["\']', output_first, re.IGNORECASE)
    if json_match:
        extracted = json_match.group(1).strip()
        for label in CUAD_LABELS:
            if label.lower() == extracted.lower():
                return label

    # Strategy 5: Check aliases in first 100 chars
    for alias, full_label in LABEL_ALIASES.items():
        if alias in output_first_lower[:100]:
            return full_label

    # Strategy 6: ANY CUAD label found in first 100 chars (more lenient)
    for label in CUAD_LABELS:
        if label.lower() in output_first_lower[:100]:
            return label

    # Strategy 7: Partial word match for "Indemnity/Indemnification" edge cases
    if "indemni" in output_first_lower[:50]:
        return "Indemnification"
    if "terminat" in output_first_lower[:50]:
        return "Termination For Convenience"
    if "warrant" in output_first_lower[:50]:
        return "Warranty Duration"

    # Fallback
    return "Other"


def classify_clause(state: ClauseState) -> ClauseState:
    """Classify the current clause using the fine-tuned adapter or stub.

    Checks state for 'use_qlora_adapter' flag to decide whether to use adapter or base model.
    Falls back to keyword stub if model loading fails.
    """
    clause = state.get("current_clause", "")
    use_adapter = state.get("use_qlora_adapter", True)  # Default to using adapter

    loaded = _load_adapter_classifier(use_adapter=use_adapter)
    if loaded is not None:
        model, tokenizer = loaded
        predicted = _classify_with_adapter(model, tokenizer, clause)
    else:
        predicted = _stub_classify(clause)

    return {**state, "predicted_clause_type": predicted}


# ---------------------------------------------------------------------------
# Node 3: retrieve_rules
# ---------------------------------------------------------------------------

def retrieve_rules(state: ClauseState) -> ClauseState:
    """Retrieve deterministic rules and/or RAG context for the clause."""
    clause = state.get("current_clause", "")
    clause_type = state.get("predicted_clause_type")

    flags = risk_flags(clause, clause_type=clause_type)

    try:
        rag_results = retrieve_clauses(clause, clause_type=clause_type, top_k=5)
    except Exception:  # noqa: BLE001 - RAG is optional
        rag_results = []

    retrieved_rules = {
        "flags": flags,
        "rag_results": rag_results,
    }
    return {**state, "retrieved_rules": retrieved_rules}


# ---------------------------------------------------------------------------
# Node 4: generate_finding
# ---------------------------------------------------------------------------

def generate_finding(state: ClauseState) -> ClauseState:
    """Produce a structured finding for the current clause."""
    clause = state.get("current_clause", "")
    clause_type = state.get("predicted_clause_type") or "Other"
    rules = state.get("retrieved_rules", {})
    flags = rules.get("flags", [])

    risk_level = highest_risk_level(flags)

    flag_names = ", ".join(f.get("rule_name", "") for f in flags) or "none"
    explanation = (
        f"Clause classified as {clause_type}. Deterministic risk flags: "
        f"{flag_names}. Overall risk level: {risk_level}. "
        "Review flagged terms carefully before signing."
    )

    # Use a supporting span from the clause itself
    quoted_span = clause[:300] if len(clause) > 300 else clause

    finding = {
        "clause_type": clause_type,
        "risk_level": risk_level,
        "explanation": explanation,
        "quoted_span": quoted_span,
    }
    return {**state, "generated_finding": finding}


# ---------------------------------------------------------------------------
# Node 5: validate_citation
# ---------------------------------------------------------------------------

def validate_citation(state: ClauseState) -> ClauseState:
    """Deterministically check that quoted_span appears verbatim in retrieved text."""
    finding = state.get("generated_finding", {})
    rules = state.get("retrieved_rules", {})
    rag_results = rules.get("rag_results", [])

    quoted_span = finding.get("quoted_span", "")
    citation_valid = False

    if quoted_span:
        # Check against RAG results
        for result in rag_results:
            if quoted_span in result.get("text", ""):
                citation_valid = True
                break

        # Fallback: check against the current clause itself
        if not citation_valid:
            current = state.get("current_clause", "")
            if quoted_span in current:
                citation_valid = True

    return {**state, "citation_valid": citation_valid}


# ---------------------------------------------------------------------------
# Node 6: refuse_or_answer
# ---------------------------------------------------------------------------

def refuse_or_answer(state: ClauseState) -> ClauseState:
    """If citation is valid, produce final answer; otherwise refuse."""
    if state.get("citation_valid"):
        finding = state.get("generated_finding", {})
        final_answer = {
            "clause_type": finding.get("clause_type"),
            "risk_level": finding.get("risk_level"),
            "explanation": finding.get("explanation"),
            "quoted_span": finding.get("quoted_span"),
        }
        return {**state, "final_answer": final_answer}

    refusal_reason = (
        "Citation validation failed: the quoted span does not appear verbatim "
        "in any retrieved clause text. Refusing to answer."
    )
    return {**state, "refusal_reason": refusal_reason}
