# Copyright (c) Meta Platforms, Inc. and affiliates.
# BSD-style license.

"""
CascadeDebug Environment — Server Implementation.

This module implements the core RL environment logic:
  - reset(): sample episode from pipeline bank, return observation (no ground truth)
  - step(): route action to gatekeeper → reward engine → return observation
  - state property: return full state including ground truth (for logging only)

Architecture:
  - Pipeline bank loaded lazily on first reset() (from data/pipeline_bank.json)
  - Episodes partitioned by curriculum level
  - Gatekeeper is deterministic rule-based (server/gatekeeper.py)
  - Reward functions are 4 independent signals (server/rewards.py)
  - Curriculum auto-advances based on rolling average reward
"""

import json
import random
from pathlib import Path
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import CascadeDebugAction, CascadeDebugObservation
except ImportError:
    from models import CascadeDebugAction, CascadeDebugObservation

# Import gatekeeper and rewards (sibling modules in server/)
try:
    from .gatekeeper import evaluate_proposal
    from .rewards import compute_total_reward
except ImportError:
    from server.gatekeeper import evaluate_proposal
    from server.rewards import compute_total_reward


PIPELINE_BANK_PATH = Path(__file__).parent.parent / "data" / "pipeline_bank.json"

# Rolling reward window for curriculum auto-advance
CURRICULUM_THRESHOLDS = {1: 0.4, 2: 0.6, 3: 0.7}
ROLLING_WINDOW = 100
MAX_TURNS = 2


class CascadeDebugEnvironment(Environment):
    """
    CascadeDebug RL Environment.

    Each episode:
    1. Samples a corrupted pipeline from the pre-computed bank
    2. Returns observation WITHOUT revealing which step was corrupted
    3. Agent localizes, blames, proposes fix, negotiates with gatekeeper
    4. Rewards: localization + blame + fix correctness + surgical precision

    Curriculum levels (auto-advance based on rolling avg reward):
      Level 1: 3-step, obvious errors, partial localization credit
      Level 2: 4-step, subtle errors, no partial credit
      Level 3: 5-6-step, cross-step dependency errors
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize environment. Pipeline bank loaded lazily on first reset."""
        self._pipeline_bank: list[dict] | None = None
        self._bank_by_level: dict[int, list[dict]] = {}
        self._curriculum_level: int = 1
        self._reward_history: list[float] = []

        # Per-episode state
        self._episode_id: str = ""
        self._episode_data: dict = {}
        self._injected_step: int = 0
        self._injected_role: str = ""
        self._error_type: str = ""
        self._current_pipeline: list[dict] = []
        self._task_brief: str = ""
        self._turn: int = 1
        self._submitted: bool = False
        self._gatekeeper_accepted: bool = False
        self._gatekeeper_feedback: str | None = None
        self._fix_history: list[dict] = []
        self._last_reward: dict = {}

        self._state = State(episode_id=str(uuid4()), step_count=0)

    # ------------------------------------------------------------------
    # Lazy pipeline bank loading
    # ------------------------------------------------------------------

    def _load_pipeline_bank(self):
        """Load pipeline bank from disk on first access."""
        if self._pipeline_bank is not None:
            return

        if not PIPELINE_BANK_PATH.exists():
            raise FileNotFoundError(
                f"Pipeline bank not found at {PIPELINE_BANK_PATH}. "
                "Run 'python data/generate_pipeline_bank.py' first (Phase 1)."
            )

        with open(PIPELINE_BANK_PATH, "r") as f:
            self._pipeline_bank = json.load(f)

        # Index by curriculum level
        self._bank_by_level = {}
        for ep in self._pipeline_bank:
            level = ep["curriculum_level"]
            if level not in self._bank_by_level:
                self._bank_by_level[level] = []
            self._bank_by_level[level].append(ep)

    # ------------------------------------------------------------------
    # Curriculum auto-advance
    # ------------------------------------------------------------------

    def _check_curriculum_advance(self):
        """Auto-advance curriculum level based on rolling average reward."""
        if self._curriculum_level >= 3:
            return  # Already at max level

        if len(self._reward_history) < ROLLING_WINDOW:
            return  # Not enough data

        recent = self._reward_history[-ROLLING_WINDOW:]
        avg = sum(recent) / len(recent)
        threshold = CURRICULUM_THRESHOLDS.get(self._curriculum_level, 1.0)

        if avg > threshold:
            self._curriculum_level += 1

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def reset(self) -> CascadeDebugObservation:
        """
        Reset: sample episode from pipeline bank, return observation.
        Ground truth (injected_step, injected_role) is hidden — only in state().
        """
        self._load_pipeline_bank()
        self._check_curriculum_advance()

        # Sample episode from current curriculum level
        level = self._curriculum_level
        if level in self._bank_by_level and self._bank_by_level[level]:
            episode = random.choice(self._bank_by_level[level])
        else:
            # Fallback: sample from all
            episode = random.choice(self._pipeline_bank)

        # Store episode data
        self._episode_data = episode
        self._episode_id = episode["pipeline_id"]
        self._injected_step = episode["injected_step"]
        self._injected_role = episode["injected_role"]
        self._error_type = episode["error_type"]
        self._task_brief = episode["task_brief"]

        # Agent sees the CORRUPTED pipeline (not clean)
        self._current_pipeline = episode["corrupted_pipeline"]

        # Reset episode state
        self._turn = 1
        self._submitted = False
        self._gatekeeper_accepted = False
        self._gatekeeper_feedback = None
        self._fix_history = []
        self._last_reward = {}

        self._state = State(episode_id=self._episode_id, step_count=0)

        return CascadeDebugObservation(
            pipeline=self._current_pipeline,
            task_brief=self._task_brief,
            turn=self._turn,
            gatekeeper_feedback=None,
            curriculum_level=self._curriculum_level,
            pipeline_id=self._episode_id,
            done=False,
            reward=0.0,
        )

    def step(self, action: CascadeDebugAction) -> CascadeDebugObservation:  # type: ignore[override]
        """
        Step: route action through gatekeeper (propose/revise) → reward engine (submit or max turns).

        Flow:
          1. If action_type == "propose" or "revise":
             - Run gatekeeper → if accepted, proceed to scoring
             - If rejected AND turn < MAX_TURNS, send feedback, increment turn
          2. If action_type == "submit" OR gatekeeper accepted OR turn >= MAX_TURNS:
             - Compute all 4 reward signals
             - Mark episode as done
        """
        self._state.step_count += 1

        # Store attempt in history
        self._fix_history.append({
            "turn": self._turn,
            "action_type": action.action_type,
            "fault_step_id": action.fault_step_id,
            "blame_role": action.blame_role,
            "fix_content": action.fix_content,
        })

        # ── Gatekeeper evaluation (propose/revise) ──────────────────────
        if action.action_type in ("propose", "revise"):
            gk_result = evaluate_proposal(
                fault_step_id=action.fault_step_id,
                blame_role=action.blame_role,
                fix_content=action.fix_content,
                pipeline_length=len(self._current_pipeline),
                corrupted_output=self._episode_data.get("corrupted_output", ""),
                expected_fix_keywords=self._episode_data.get("expected_fix_keywords", []),
            )

            self._gatekeeper_accepted = gk_result["accepted"]

            if not gk_result["accepted"] and self._turn < MAX_TURNS:
                # Rejected — give feedback, allow revision
                self._gatekeeper_feedback = gk_result["feedback"]
                self._turn += 1

                return CascadeDebugObservation(
                    pipeline=self._current_pipeline,
                    task_brief=self._task_brief,
                    turn=self._turn,
                    gatekeeper_feedback=self._gatekeeper_feedback,
                    curriculum_level=self._curriculum_level,
                    pipeline_id=self._episode_id,
                    done=False,
                    reward=0.0,
                )
            # If accepted or max turns reached, fall through to scoring

        # ── Reward computation (submit / accepted / max turns) ──────────
        self._submitted = True

        reward_result = compute_total_reward(
            predicted_step=action.fault_step_id,
            true_step=self._injected_step,
            predicted_role=action.blame_role,
            true_role=self._injected_role,
            fix_content=action.fix_content,
            original_output=self._episode_data.get("original_output", ""),
            expected_fix_keywords=self._episode_data.get("expected_fix_keywords", []),
            corrupted_output=self._episode_data.get("corrupted_output", ""),
            turn=self._turn,
            max_turns=MAX_TURNS,
            gatekeeper_accepted=self._gatekeeper_accepted,
            curriculum_level=self._curriculum_level,
        )

        self._last_reward = reward_result
        total_reward = reward_result["total"]

        # Track for curriculum advancement
        self._reward_history.append(total_reward)

        return CascadeDebugObservation(
            pipeline=self._current_pipeline,
            task_brief=self._task_brief,
            turn=self._turn,
            gatekeeper_feedback=self._gatekeeper_feedback,
            curriculum_level=self._curriculum_level,
            pipeline_id=self._episode_id,
            done=True,
            reward=total_reward,
        )

    @property
    def state(self) -> State:
        """Full internal state for logging. Never exposed to agent via observation."""
        return State(
            episode_id=self._episode_id,
            step_count=self._state.step_count,
            metadata={
                "injected_step": self._injected_step,
                "injected_role": self._injected_role,
                "error_type": self._error_type,
                "curriculum_level": self._curriculum_level,
                "turn": self._turn,
                "submitted": self._submitted,
                "gatekeeper_accepted": self._gatekeeper_accepted,
                "fix_history": self._fix_history,
                "last_reward": self._last_reward,
                "reward_history_length": len(self._reward_history),
            },
        )
