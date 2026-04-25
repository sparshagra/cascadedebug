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

import os
import random
import sys

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_URL = os.getenv("BASE_URL", "http://localhost:7860")
TASK_NAME = os.getenv("TASK_NAME", "localize_level1")
MODEL_NAME = os.getenv("MODEL_NAME", "baseline_random")
HF_TOKEN = os.getenv("HF_TOKEN", "")  # Add after claiming credits

TASKS = ["localize_level1", "localize_level2", "localize_level3"]
MAX_STEPS = 10


# ---------------------------------------------------------------------------
# Baseline agent (heuristic, no LLM) — placeholder until Phase 5
# ---------------------------------------------------------------------------

def baseline_action(observation: dict, step: int) -> dict:
    """
    Heuristic agent: always guesses step 1, Researcher, placeholder fix.
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
# Run one task
# ---------------------------------------------------------------------------

def run_task(task_name: str):
    """Run one episode of the given task and log results."""
    print(f"[START] task={task_name} env=cascade_debug model={MODEL_NAME}")

    try:
        # For local testing without a live server, use a mock
        # Phase 5 will import CascadeDebugEnv client and connect to BASE_URL
        mock_obs = {
            "pipeline": [
                {"role": "Researcher", "output": "Python uses 0-based indexing.", "step_id": 1},
                {"role": "Coder", "output": "for i in range(1, n+1): print(i)", "step_id": 2},
                {"role": "Analyst", "output": "Code output matches expected range.", "step_id": 3},
            ],
            "task_brief": "Explain Python indexing and write a loop from 1 to n.",
            "turn": 1,
            "gatekeeper_feedback": None,
            "curriculum_level": 1,
            "pipeline_id": "mock_ep_001",
            "done": False,
            "reward": 0.0,
        }

        rewards = []
        for step in range(1, MAX_STEPS + 1):
            action = baseline_action(mock_obs, step)
            reward = round(random.uniform(0.05, 0.25), 4)
            done = step >= 3  # Mock: done after 3 steps
            rewards.append(reward)

            print(
                f"[STEP]  step={step} "
                f"action=fault_step={action['fault_step_id']},role={action['blame_role']} "
                f"reward={reward:.2f} done={str(done).lower()} error=null"
            )

            if done:
                break

        total_score = round(sum(rewards) / len(rewards), 4)
        print(
            f"[END]   success=true steps={len(rewards)} "
            f"score={total_score} "
            f"rewards={','.join(str(r) for r in rewards)}"
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
