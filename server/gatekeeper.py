"""
CascadeDebug Gatekeeper — Deterministic Rule-Based.

The gatekeeper reviews the agent's proposed fix and either ACCEPTS or REJECTS it
based on deterministic rules (NOT an LLM). This is a Theme 1 multi-agent
negotiation component.

Rules:
  1. fix_content must not be empty
  2. fix_content must differ from the corrupted output
  3. fix_content must contain at least one expected fix keyword
  4. fault_step_id must be within valid range
  5. blame_role must be one of the valid roles

On rejection, the gatekeeper provides specific feedback about which constraint
was violated, so the agent can revise.
"""

from __future__ import annotations

VALID_ROLES = {"Researcher", "Coder", "Analyst"}


def evaluate_proposal(
    fault_step_id: int,
    blame_role: str,
    fix_content: str,
    pipeline_length: int,
    corrupted_output: str,
    expected_fix_keywords: list[str],
) -> dict:
    """
    Evaluate a fix proposal against deterministic rules.

    Args:
        fault_step_id: Agent's proposed faulty step (1-indexed).
        blame_role: Agent's proposed blame role.
        fix_content: Agent's proposed fix content.
        pipeline_length: Total number of steps in the pipeline.
        corrupted_output: The corrupted output at the injected step.
        expected_fix_keywords: Keywords that should appear in a correct fix.

    Returns:
        dict with:
          - accepted: bool
          - feedback: str or None (reason for rejection)
          - violated_rule: str or None (which rule was violated)
    """
    # Rule 1: fix_content must not be empty
    if not fix_content or not fix_content.strip():
        return {
            "accepted": False,
            "feedback": "REJECTED: Fix content is empty. Provide a substantive correction.",
            "violated_rule": "empty_fix",
        }

    # Rule 2: fault_step_id must be in valid range
    if fault_step_id < 1 or fault_step_id > pipeline_length:
        return {
            "accepted": False,
            "feedback": f"REJECTED: fault_step_id={fault_step_id} is out of range [1, {pipeline_length}]. Pick a valid step.",
            "violated_rule": "invalid_step_id",
        }

    # Rule 3: blame_role must be a valid role
    if blame_role not in VALID_ROLES:
        return {
            "accepted": False,
            "feedback": f"REJECTED: blame_role='{blame_role}' is not valid. Choose from: {sorted(VALID_ROLES)}.",
            "violated_rule": "invalid_role",
        }

    # Rule 4: fix_content must differ from corrupted output
    if fix_content.strip().lower() == corrupted_output.strip().lower():
        return {
            "accepted": False,
            "feedback": "REJECTED: Your fix is identical to the current (corrupted) output. Provide actual corrections.",
            "violated_rule": "identical_to_corrupt",
        }

    # Rule 5: fix_content must contain at least one expected fix keyword
    fix_lower = fix_content.lower()
    has_keyword = any(kw.lower() in fix_lower for kw in expected_fix_keywords)
    if not has_keyword and expected_fix_keywords:
        # Give a hint — mention the domain area, not the exact keyword
        hint_areas = {
            "O(": "complexity analysis",
            "range": "boundary handling",
            "sort": "sorting correctness",
            "return": "return value",
            "correct": "accuracy",
        }
        hint = "the domain-specific correction"
        for keyword_fragment, area in hint_areas.items():
            if any(keyword_fragment.lower() in kw.lower() for kw in expected_fix_keywords):
                hint = area
                break

        return {
            "accepted": False,
            "feedback": f"REJECTED: Fix does not address the core issue. Focus on {hint}.",
            "violated_rule": "missing_keywords",
        }

    # All rules passed
    return {
        "accepted": True,
        "feedback": None,
        "violated_rule": None,
    }
