"""
CascadeDebug Reward Functions — Phase 3.

Four independent reward signals for GRPO training:
  r1 = reward_localization   weight=0.35  (fault_step_id == injected_step)
  r2 = reward_blame          weight=0.20  (blame_role == injected_role)
  r3 = reward_fix            weight=0.35  (verifier score on fix content)
  r4 = reward_precision      weight=0.10  (no extra steps modified + turn efficiency)

  total = 0.35*r1 + 0.20*r2 + 0.35*r3 + 0.10*r4

All verifiers are programmatic — NO LLM-as-judge.
"""

from __future__ import annotations


# ──────────────────────────────────────────────────────────────────────────────
# Weights (locked — do not change without user approval)
# ──────────────────────────────────────────────────────────────────────────────

WEIGHTS = {
    "localization": 0.35,
    "blame": 0.20,
    "fix": 0.35,
    "precision": 0.10,
}


# ──────────────────────────────────────────────────────────────────────────────
# r1: Fault Localization Reward
# ──────────────────────────────────────────────────────────────────────────────

def reward_localization(
    predicted_step: int,
    true_step: int,
    curriculum_level: int = 1,
) -> float:
    """
    Reward for correctly identifying the faulty step.

    Args:
        predicted_step: Agent's predicted fault step (1-indexed).
        true_step: Ground truth injected step (1-indexed).
        curriculum_level: 1=easy (partial credit), 2+=strict.

    Returns:
        Float in [0.0, 1.0].
    """
    if predicted_step == true_step:
        return 1.0

    # Partial credit at Level 1 only: ±1 step → 0.3 reward
    if curriculum_level == 1 and abs(predicted_step - true_step) == 1:
        return 0.3

    return 0.0


# ──────────────────────────────────────────────────────────────────────────────
# r2: Blame Attribution Reward
# ──────────────────────────────────────────────────────────────────────────────

def reward_blame(
    predicted_role: str,
    true_role: str,
) -> float:
    """
    Reward for correctly attributing blame to the right role.

    Args:
        predicted_role: Agent's predicted role ("Researcher", "Coder", "Analyst").
        true_role: Ground truth role.

    Returns:
        Float in {0.0, 1.0}.
    """
    return 1.0 if predicted_role == true_role else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# r3: Fix Correctness Reward (Programmatic Verifier)
# ──────────────────────────────────────────────────────────────────────────────

def reward_fix(
    fix_content: str,
    original_output: str,
    expected_fix_keywords: list[str],
    corrupted_output: str,
) -> float:
    """
    Reward for quality of the proposed fix. Uses keyword matching + similarity.

    Scoring rubric (all programmatic, no LLM-as-judge):
      - keyword_score: fraction of expected_fix_keywords present in fix_content (0.5 weight)
      - not_corrupt_score: fix differs from corrupted output (0.2 weight)
      - length_score: fix has reasonable length vs original (0.15 weight)
      - original_sim_score: fix contains overlapping words with original (0.15 weight)

    Args:
        fix_content: Agent's proposed fix text.
        original_output: The correct (pre-corruption) output.
        expected_fix_keywords: Keywords that should appear in a correct fix.
        corrupted_output: The corrupted output the agent is trying to fix.

    Returns:
        Float in [0.0, 1.0].
    """
    if not fix_content or not fix_content.strip():
        return 0.0

    fix_lower = fix_content.lower()

    # 1. Keyword score (0.5 weight)
    if expected_fix_keywords:
        matches = sum(1 for kw in expected_fix_keywords if kw.lower() in fix_lower)
        keyword_score = matches / len(expected_fix_keywords)
    else:
        keyword_score = 0.5  # neutral if no keywords

    # 2. Not-corrupt score (0.2 weight): fix should differ from corrupted output
    if corrupted_output:
        corrupt_lower = corrupted_output.lower()
        not_corrupt_score = 0.0 if fix_lower == corrupt_lower else 1.0
    else:
        not_corrupt_score = 1.0

    # 3. Length score (0.15 weight): fix should have reasonable length
    if original_output:
        orig_len = len(original_output)
        fix_len = len(fix_content)
        if orig_len > 0:
            ratio = fix_len / orig_len
            # Ideal: 0.3 to 3.0 ratio
            if 0.3 <= ratio <= 3.0:
                length_score = 1.0
            elif ratio < 0.3:
                length_score = ratio / 0.3
            else:
                length_score = max(0.0, 1.0 - (ratio - 3.0) / 7.0)
        else:
            length_score = 0.5
    else:
        length_score = 0.5

    # 4. Original similarity score (0.15 weight): overlap with original output
    if original_output:
        orig_words = set(original_output.lower().split())
        fix_words = set(fix_lower.split())
        if orig_words:
            overlap = len(orig_words & fix_words) / len(orig_words)
            original_sim_score = min(overlap, 1.0)
        else:
            original_sim_score = 0.0
    else:
        original_sim_score = 0.0

    # Combine
    total = (
        0.50 * keyword_score
        + 0.20 * not_corrupt_score
        + 0.15 * length_score
        + 0.15 * original_sim_score
    )

    return round(min(max(total, 0.0), 1.0), 4)


# ──────────────────────────────────────────────────────────────────────────────
# r4: Surgical Precision Reward
# ──────────────────────────────────────────────────────────────────────────────

def reward_precision(
    turn: int,
    max_turns: int = 2,
    gatekeeper_accepted: bool = False,
) -> float:
    """
    Reward for surgical precision: fewer turns + gatekeeper acceptance.

    Scoring:
      - Turn 1 submit + accepted: 1.0
      - Turn 2 submit + accepted: 0.6
      - Not accepted: 0.2 (just for submitting)
      - Turn > max_turns: 0.0

    Args:
        turn: Current turn number when submitted.
        max_turns: Maximum allowed turns (default: 2).
        gatekeeper_accepted: Whether gatekeeper accepted the fix.

    Returns:
        Float in [0.0, 1.0].
    """
    if turn > max_turns:
        return 0.0

    if gatekeeper_accepted:
        # First-try bonus
        if turn == 1:
            return 1.0
        else:
            return 0.6
    else:
        return 0.2


# ──────────────────────────────────────────────────────────────────────────────
# Composite reward
# ──────────────────────────────────────────────────────────────────────────────

def compute_total_reward(
    predicted_step: int,
    true_step: int,
    predicted_role: str,
    true_role: str,
    fix_content: str,
    original_output: str,
    expected_fix_keywords: list[str],
    corrupted_output: str,
    turn: int,
    max_turns: int = 2,
    gatekeeper_accepted: bool = False,
    curriculum_level: int = 1,
) -> dict:
    """
    Compute all 4 reward components and the weighted total.

    Returns dict with keys: localization, blame, fix, precision, total, weights.
    """
    r1 = reward_localization(predicted_step, true_step, curriculum_level)
    r2 = reward_blame(predicted_role, true_role)
    r3 = reward_fix(fix_content, original_output, expected_fix_keywords, corrupted_output)
    r4 = reward_precision(turn, max_turns, gatekeeper_accepted)

    total = (
        WEIGHTS["localization"] * r1
        + WEIGHTS["blame"] * r2
        + WEIGHTS["fix"] * r3
        + WEIGHTS["precision"] * r4
    )

    return {
        "localization": round(r1, 4),
        "blame": round(r2, 4),
        "fix": round(r3, 4),
        "precision": round(r4, 4),
        "total": round(total, 4),
        "weights": WEIGHTS,
    }
