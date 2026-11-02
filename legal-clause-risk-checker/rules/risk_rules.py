"""Deterministic risk flagging for legal clauses.

Inspects raw clause text (and optional clause type) and returns a list of
risk flags. Each flag is a dict with ``rule_name`` and ``risk_level``.
These rules are intentional, auditable, and require no LLM — they decide
the flag while the LLM explains the risk.
"""

from __future__ import annotations

import re
from typing import Optional

# (rule_name, compiled_regex, weight, risk_level).
# Weight contributes to the numeric risk score; risk_level is the severity band.
_RISK_PATTERNS: list[tuple[str, re.Pattern[str], float, str]] = [
    (
        "Uncapped Liability",
        re.compile(r"\bindemnif\w+\s+(?:and\s+)?hold\s+harmless\b", re.I),
        0.25,
        "high",
    ),
    (
        "Broad Indemnification",
        re.compile(r"\bindemnif\w+\s+from\s+any\s+and\s+all\s+claims\b", re.I),
        0.20,
        "high",
    ),
    (
        "Unlimited Liability",
        re.compile(r"\b(?:unlimited|uncapped)\s+liability\b", re.I),
        0.25,
        "high",
    ),
    (
        "Liability Disclaimer",
        re.compile(
            r"\b(?:disclaim|disclaimer)\s+(?:of\s+)?(?:all\s+)?(?:warranties?|liability)\b",
            re.I,
        ),
        0.20,
        "medium",
    ),
    (
        "Consequential Damages Waiver",
        re.compile(
            r"\b(?:consequential|indirect|incidental|special)\s+damages\b", re.I
        ),
        0.10,
        "medium",
    ),
    (
        "Auto-renewal / Evergreen",
        re.compile(r"\b(?:auto-?renew|evergreen|automatically\s+renew)\b", re.I),
        0.10,
        "low",
    ),
    (
        "Unilateral Termination Without Notice",
        re.compile(
            r"\bterminat\w+\s+(?:at\s+)?(?:any\s+time|will)|without\s+(?:cause|notice|prior\s+written\s+notice)|for\s+convenience|unilateral\s+terminat\w+\b",
            re.I,
        ),
        0.20,
        "high",
    ),
    (
        "IP Assignment (Broad)",
        re.compile(
            r"\b(?:assign|transfer)\s+.*\b(?:intellectual\s+property|IP)\b|\bIP\b.*\b(?:assign|transfer)\b",
            re.I,
        ),
        0.20,
        "high",
    ),
    (
        "Broad Confidentiality",
        re.compile(r"\bconfidential\w*\s+(?:information|material)\b", re.I),
        0.05,
        "low",
    ),
    (
        "Sole Discretion",
        re.compile(r"\bin\s+its\s+(?:sole|absolute)\s+discretion\b", re.I),
        0.10,
        "medium",
    ),
]


def risk_flags(clause_text: str, clause_type: Optional[str] = None) -> list[dict]:
    """Return a list of risk flag dicts with ``rule_name`` and ``risk_level``.

    Args:
        clause_text: Raw contract clause text.
        clause_type: Optional clause category (used for type-specific rules).

    Returns:
        List of dicts like ``{"rule_name": "Uncapped Liability", "risk_level": "high"}``.
    """
    flags: list[dict] = []

    for name, pattern, _weight, risk_level in _RISK_PATTERNS:
        if pattern.search(clause_text):
            flags.append({"rule_name": name, "risk_level": risk_level})

    # Type-specific heuristics
    if clause_type:
        ct = clause_type.lower()
        if "indemn" in ct and "cap" not in clause_text.lower():
            if not any(f["rule_name"] == "Uncapped Liability" for f in flags):
                flags.append({"rule_name": "Uncapped Liability", "risk_level": "high"})
        if "terminat" in ct and "notice" not in clause_text.lower():
            if not any(
                f["rule_name"] == "Unilateral Termination Without Notice" for f in flags
            ):
                flags.append(
                    {
                        "rule_name": "Unilateral Termination Without Notice",
                        "risk_level": "high",
                    }
                )

    return flags


def risk_score_from_flags(flags: list[dict]) -> float:
    """Convert a list of flag dicts into a deterministic risk score in [0, 1].

    Each flag contributes its weight; the score is capped at 1.0. This is
    intentionally simple and auditable — it is not a substitute for legal
    judgment.
    """
    weights = {
        "Uncapped Liability": 0.25,
        "Broad Indemnification": 0.20,
        "Unlimited Liability": 0.25,
        "Liability Disclaimer": 0.20,
        "Consequential Damages Waiver": 0.10,
        "Auto-renewal / Evergreen": 0.10,
        "Unilateral Termination Without Notice": 0.20,
        "IP Assignment (Broad)": 0.20,
        "Broad Confidentiality": 0.05,
        "Sole Discretion": 0.10,
    }
    score = sum(weights.get(f.get("rule_name", ""), 0.1) for f in flags)
    return round(min(score, 1.0), 2)


def highest_risk_level(flags: list[dict]) -> str:
    """Return the highest severity band among the flags.

    Order: high > medium > low. Returns "low" when there are no flags.
    """
    order = {"high": 3, "medium": 2, "low": 1}
    max_level = "low"
    max_order = 0
    for f in flags:
        level = f.get("risk_level", "low")
        if order.get(level, 0) > max_order:
            max_level = level
            max_order = order[level]
    return max_level
