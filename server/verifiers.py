"""
CascadeDebug Verifiers — Role-Specific Output Verification.

Programmatic verifiers for checking fix outputs. Used by reward_fix (r3).
All verification is deterministic — NO LLM-as-judge.
"""

from __future__ import annotations

import re


def verify_researcher_fix(
    fix_content: str,
    original_output: str,
    expected_keywords: list[str],
) -> float:
    """
    Verify a fix for a Researcher role output.
    Checks: factual accuracy keywords, explanation quality, specificity.

    Returns: score in [0.0, 1.0]
    """
    if not fix_content.strip():
        return 0.0

    score = 0.0
    fix_lower = fix_content.lower()

    # Keyword match (60% weight)
    if expected_keywords:
        matches = sum(1 for kw in expected_keywords if kw.lower() in fix_lower)
        score += 0.6 * (matches / len(expected_keywords))

    # Has explanation structure (20% weight)
    explanation_indicators = [
        "because", "therefore", "since", "this means",
        "which", "so that", "due to", "as a result",
        "complexity", "approach", "strategy", "analysis",
    ]
    has_explanation = any(ind in fix_lower for ind in explanation_indicators)
    score += 0.2 if has_explanation else 0.0

    # Reasonable length (20% weight)
    word_count = len(fix_content.split())
    if 10 <= word_count <= 200:
        score += 0.2
    elif word_count > 5:
        score += 0.1

    return min(score, 1.0)


def verify_coder_fix(
    fix_content: str,
    original_output: str,
    expected_keywords: list[str],
) -> float:
    """
    Verify a fix for a Coder role output.
    Checks: code-like structure, keyword matches, syntactic plausibility.

    Returns: score in [0.0, 1.0]
    """
    if not fix_content.strip():
        return 0.0

    score = 0.0
    fix_lower = fix_content.lower()

    # Keyword match (50% weight)
    if expected_keywords:
        matches = sum(1 for kw in expected_keywords if kw.lower() in fix_lower)
        score += 0.5 * (matches / len(expected_keywords))

    # Has code structure (30% weight)
    code_indicators = [
        "def ", "return ", "import ", "class ",
        "for ", "while ", "if ", "else:",
        "=", "(", ")", ":", "->",
        "SELECT", "FROM", "WHERE", "JOIN",
    ]
    code_count = sum(1 for ind in code_indicators if ind in fix_content)
    code_score = min(code_count / 3.0, 1.0)
    score += 0.3 * code_score

    # Reasonable code length (20% weight)
    line_count = len(fix_content.strip().split('\n'))
    if 2 <= line_count <= 50:
        score += 0.2
    elif line_count == 1 and len(fix_content) > 20:
        score += 0.1

    return min(score, 1.0)


def verify_analyst_fix(
    fix_content: str,
    original_output: str,
    expected_keywords: list[str],
) -> float:
    """
    Verify a fix for an Analyst role output.
    Checks: quantitative claims, keyword matches, structured reporting.

    Returns: score in [0.0, 1.0]
    """
    if not fix_content.strip():
        return 0.0

    score = 0.0
    fix_lower = fix_content.lower()

    # Keyword match (50% weight)
    if expected_keywords:
        matches = sum(1 for kw in expected_keywords if kw.lower() in fix_lower)
        score += 0.5 * (matches / len(expected_keywords))

    # Has quantitative content (30% weight)
    has_numbers = bool(re.search(r'\d+\.?\d*', fix_content))
    quant_terms = ["accuracy", "score", "result", "test", "validated", "verified", "performance", "p-value"]
    has_quant_term = any(t in fix_lower for t in quant_terms)
    if has_numbers and has_quant_term:
        score += 0.3
    elif has_numbers or has_quant_term:
        score += 0.15

    # Structured reporting (20% weight)
    report_indicators = [".", ":", ",", ";"]
    has_structure = sum(1 for ind in report_indicators if ind in fix_content) >= 2
    score += 0.2 if has_structure else 0.0

    return min(score, 1.0)


def verify_fix(
    role: str,
    fix_content: str,
    original_output: str,
    expected_keywords: list[str],
) -> float:
    """
    Route to the appropriate role-specific verifier.

    Args:
        role: "Researcher", "Coder", or "Analyst"
        fix_content: Agent's proposed fix text.
        original_output: The correct (pre-corruption) output.
        expected_keywords: Keywords that should appear in a correct fix.

    Returns: score in [0.0, 1.0]
    """
    if role == "Researcher":
        return verify_researcher_fix(fix_content, original_output, expected_keywords)
    elif role == "Coder":
        return verify_coder_fix(fix_content, original_output, expected_keywords)
    elif role == "Analyst":
        return verify_analyst_fix(fix_content, original_output, expected_keywords)
    else:
        # Fallback: generic keyword match
        if not fix_content.strip():
            return 0.0
        if expected_keywords:
            matches = sum(1 for kw in expected_keywords if kw.lower() in fix_content.lower())
            return matches / len(expected_keywords)
        return 0.0
