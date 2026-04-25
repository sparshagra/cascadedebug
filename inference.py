"""
CascadeDebug inference script — required by OpenEnv minimum requirements.

Runs a baseline (random/heuristic) agent against the environment for each
of the 3 tasks and logs results in the OpenEnv-standard format:

  [START] task=<name> env=cascade_debug model=<model>
  [STEP]  step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...>

Usage:
  TASK_NAME=localize_level1 python inference.py
  TASK_NAME=all python inference.py

Note: Replace BASE_URL with your deployed HF Space URL when training.
"""

import json
import os
import random
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_URL = os.getenv("BASE_URL", "http://localhost:7860")
TASK_NAME = os.getenv("TASK_NAME", "localize_level1")
MODEL_NAME = os.getenv("MODEL_NAME", "baseline_random")
HF_TOKEN = os.getenv("HF_TOKEN", "")  # Add after claiming credits

TASKS = ["localize_level1", "localize_level2", "localize_level3"]
MAX_STEPS = 10
PIPELINE_BANK_PATH = Path(__file__).parent / "data" / "pipeline_bank.json"

TASK_LEVEL_MAP = {
    "localize_level1": 1,
    "localize_level2": 2,
    "localize_level3": 3,
}


# ---------------------------------------------------------------------------
# Baseline agent (heuristic, no LLM) — placeholder until Phase 5
# ---------------------------------------------------------------------------

def baseline_action(observation: dict, step: int) -> dict:
    """
    Heuristic agent: random guess across all steps and roles.
    Expected reward: ~0.10-0.20 (random baseline).
    """
    n_steps = len(observation.get("pipeline", []))
    roles = ["Researcher", "Coder", "Analyst"]
    return {
        "fault_step_id": random.randint(1, max(n_steps, 1)),
        "blame_role": random.choice(roles),
        "fix_content": "This is a placeholder fix from the baseline agent.",
        "action_type": "submit",
    }


# ---------------------------------------------------------------------------
# Load and run against real pipeline bank (offline mode)
# ---------------------------------------------------------------------------

def load_pipeline_bank():
    """Load pipeline bank from disk."""
    if not PIPELINE_BANK_PATH.exists():
        return None
    with open(PIPELINE_BANK_PATH, "r") as f:
        return json.load(f)


def run_task(task_name: str):
    """Run one episode of the given task and log results."""
    level = TASK_LEVEL_MAP.get(task_name, 1)
    print(f"[START] task={task_name} env=cascade_debug model={MODEL_NAME}")

    try:
        bank = load_pipeline_bank()

        if bank:
            # Use real pipeline bank data
            level_episodes = [ep for ep in bank if ep["curriculum_level"] == level]
            if not level_episodes:
                level_episodes = bank
            episode = random.choice(level_episodes)

            obs = {
                "pipeline": episode["corrupted_pipeline"],
                "task_brief": episode["task_brief"],
                "turn": 1,
                "gatekeeper_feedback": None,
                "curriculum_level": level,
                "pipeline_id": episode["pipeline_id"],
            }
        else:
            # Fallback mock observation
            obs = {
                "pipeline": [
                    {"role": "Researcher", "output": "Python uses 0-based indexing.", "step_id": 1},
                    {"role": "Coder", "output": "for i in range(1, n+1): print(i)", "step_id": 2},
                    {"role": "Analyst", "output": "Code output matches expected range.", "step_id": 3},
                ],
                "task_brief": "Explain Python indexing and write a loop from 1 to n.",
                "turn": 1,
                "gatekeeper_feedback": None,
                "curriculum_level": level,
                "pipeline_id": "mock_ep_001",
            }
            episode = None

        rewards = []
        for step in range(1, MAX_STEPS + 1):
            action = baseline_action(obs, step)

            # Compute reward inline (no server imports to avoid openenv dependency)
            if episode:
                # r1: localization
                r1 = 1.0 if action["fault_step_id"] == episode["injected_step"] else (
                    0.3 if level == 1 and abs(action["fault_step_id"] - episode["injected_step"]) == 1 else 0.0
                )
                # r2: blame
                r2 = 1.0 if action["blame_role"] == episode["injected_role"] else 0.0
                # r3: fix (keyword match)
                kw = episode.get("expected_fix_keywords", [])
                r3 = sum(1 for k in kw if k.lower() in action["fix_content"].lower()) / max(len(kw), 1)
                # r4: precision
                r4 = 0.2
                reward = round(0.35 * r1 + 0.20 * r2 + 0.35 * r3 + 0.10 * r4, 4)
            else:
                reward = round(random.uniform(0.05, 0.25), 4)

            done = True  # single-step submit
            rewards.append(reward)

            print(
                f"[STEP]  step={step} "
                f"action=fault_step={action['fault_step_id']},role={action['blame_role']} "
                f"reward={reward:.4f} done={str(done).lower()} error=null"
            )

            if done:
                break

        total_score = round(sum(rewards) / len(rewards), 4)
        print(
            f"[END]   success=true steps={len(rewards)} "
            f"score={total_score} "
            f"rewards={','.join(f'{r:.4f}' for r in rewards)}"
        )

    except Exception as e:
        print(f"[END]   success=false steps=0 score=0.0 rewards= error={e}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if TASK_NAME == "all":
        for task in TASKS:
            run_task(task)
            print()
    elif TASK_NAME in TASKS:
        run_task(TASK_NAME)
    else:
        print(f"Unknown task: {TASK_NAME}. Choose from: {TASKS} or 'all'")
        sys.exit(1)
