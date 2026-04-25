# Copyright (c) Meta Platforms, Inc. and affiliates.
# BSD-style license.

"""
CascadeDebug Environment — Server Implementation.

This module implements the core RL environment logic:
  - reset(): sample episode from pipeline bank, inject error, return obs
  - step(): route action to gatekeeper or reward engine
  - state property: return full state including ground truth (for logging only)

Phase 2 will fill in the full implementation.
Phase 1 will provide the pipeline bank (data/pipeline_bank.json).
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


PIPELINE_BANK_PATH = Path(__file__).parent.parent / "data" / "pipeline_bank.json"

# Rolling reward window for curriculum auto-advance
CURRICULUM_THRESHOLDS = {1: 0.4, 2: 0.6, 3: 0.7}
ROLLING_WINDOW = 100


class CascadeDebugEnvironment(Environment):
    """
    CascadeDebug RL Environment.

    Each episode:
    1. Samples a clean pipeline from the pre-computed bank
    2. Injects one silent error at a uniformly random step
    3. Returns observation WITHOUT revealing which step was corrupted
    4. Agent localizes, blames, proposes fix, negotiates with gatekeeper
    5. Rewards: localization + blame + fix correctness + surgical precision

    Curriculum levels (auto-advance based on rolling avg reward):
      Level 1: 3-step, obvious errors, partial localization credit
      Level 2: 4-step, subtle errors, no partial credit
      Level 3: 5-6-step, cross-step dependency errors
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize environment. Pipeline bank loaded lazily on first reset."""
        self._pipeline_bank: list[dict] | None = None
        self._curriculum_level: int = 1
        self._reward_history: list[float] = []

        # Per-episode hidden state
        self._episode_id: str = ""
        self._injected_step: int = 0
        self._injected_role: str = ""
        self._error_type: str = ""
        self._current_pipeline: list[dict] = []
        self._original_outputs: dict[int, str] = {}
        self._task_brief: str = ""
        self._turn: int = 1
        self._submitted: bool = False
        self._gatekeeper_accepted: bool = False
        self._fix_history: list[dict] = []

        self._state = State(episode_id=str(uuid4()), step_count=0)

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def reset(self) -> CascadeDebugObservation:
        """
        Reset: sample episode, inject error, return observation (no ground truth).
        TODO (Phase 2): full implementation.
        TODO (Phase 1): pipeline_bank.json must exist first.
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._episode_id = self._state.episode_id
        self._turn = 1
        self._submitted = False
        self._gatekeeper_accepted = False
        self._fix_history = []

        # Placeholder until pipeline bank exists (Phase 1)
        self._task_brief = "PLACEHOLDER — implement after Phase 1"
        self._current_pipeline = []
        self._injected_step = 0
        self._injected_role = ""
        self._error_type = ""

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
        Step: route to gatekeeper (propose/revise) or reward engine (submit).
        TODO (Phase 2): full implementation.
        """
        self._state.step_count += 1

        # Placeholder reward — Phase 2 wires real reward functions
        return CascadeDebugObservation(
            pipeline=self._current_pipeline,
            task_brief=self._task_brief,
            turn=self._turn,
            gatekeeper_feedback=None,
            curriculum_level=self._curriculum_level,
            pipeline_id=self._episode_id,
            done=True,
            reward=0.0,
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
            },
        )
