"""
CascadeDebug Graders — Phase 2 minimum compliance.

Each grader runs a deterministic episode and scores performance on a given
curriculum level. All scores in [0.0, 1.0].

These are stub implementations — Phase 3 fills in real verifier logic.

Called by the OpenEnv evaluation system via openenv.yaml graders section:
  graders:
    - task: localize_level1
      function: graders:level1_grader
    ...
"""

from __future__ import annotations

import random


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _run_scripted_episode(curriculum_level: int, seed: int = 42) -> dict:
    """
    Run a deterministic scripted episode and return raw scores.

    This is a stub. Phase 3 wires real environment + reward functions here.
    Returns dict of component scores:
      localization, blame, fix, precision, total
    """
    random.seed(seed)
    # Placeholder: scripted baseline returns ~random performance
    loc = random.uniform(0.1, 0.4)
    blame = random.uniform(0.1, 0.4)
    fix = random.uniform(0.1, 0.4)
    precision = random.uniform(0.5, 1.0)
    total = 0.35 * loc + 0.20 * blame + 0.35 * fix + 0.10 * precision
    return {
        "localization": loc,
        "blame": blame,
        "fix": fix,
        "precision": precision,
        "total": total,
        "curriculum_level": curriculum_level,
    }


# ---------------------------------------------------------------------------
# Grader functions (called by OpenEnv evaluation framework)
# ---------------------------------------------------------------------------

def level1_grader() -> float:
    """
    Grade agent on Level 1 task: 3-step pipeline, obvious errors.
    Returns float in [0.0, 1.0].
    Target: >0.4 for a trained model.
    """
    result = _run_scripted_episode(curriculum_level=1, seed=42)
    return float(result["total"])


def level2_grader() -> float:
    """
    Grade agent on Level 2 task: 4-step pipeline, subtle errors.
    Returns float in [0.0, 1.0].
    Target: >0.6 for a trained model.
    """
    result = _run_scripted_episode(curriculum_level=2, seed=43)
    return float(result["total"])


def level3_grader() -> float:
    """
    Grade agent on Level 3 task: 5-6 step pipeline, cross-step dependency errors.
    Returns float in [0.0, 1.0].
    Target: >0.7 for a trained model.
    """
    result = _run_scripted_episode(curriculum_level=3, seed=44)
    return float(result["total"])


def grade_all() -> dict:
    """Run all three graders and return combined results."""
    return {
        "localize_level1": level1_grader(),
        "localize_level2": level2_grader(),
        "localize_level3": level3_grader(),
    }
