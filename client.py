# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Cascade Debug Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import CascadeDebugAction, CascadeDebugObservation


class CascadeDebugEnv(
    EnvClient[CascadeDebugAction, CascadeDebugObservation, State]
):
    """
    Client for the Cascade Debug Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> with CascadeDebugEnv(base_url="http://localhost:7860").sync() as client:
        ...     obs = client.reset()
        ...     print(obs.pipeline)
        ...     result = client.step(CascadeDebugAction(
        ...         fault_step_id=1,
        ...         blame_role="Researcher",
        ...         fix_content="Python uses 0-based indexing.",
        ...         action_type="submit"
        ...     ))
        ...     print(result.reward)
    """

    def _step_payload(self, action: CascadeDebugAction) -> Dict:
        """
        Convert CascadeDebugAction to JSON payload for step message.

        Args:
            action: CascadeDebugAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "fault_step_id": action.fault_step_id,
            "blame_role": action.blame_role,
            "fix_content": action.fix_content,
            "action_type": action.action_type,
        }

    def _parse_result(self, payload: Dict) -> StepResult[CascadeDebugObservation]:
        """
        Parse server response into StepResult[CascadeDebugObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with CascadeDebugObservation
        """
        obs_data = payload.get("observation", {})
        observation = CascadeDebugObservation(
            pipeline=obs_data.get("pipeline", []),
            task_brief=obs_data.get("task_brief", ""),
            turn=obs_data.get("turn", 1),
            gatekeeper_feedback=obs_data.get("gatekeeper_feedback"),
            curriculum_level=obs_data.get("curriculum_level", 1),
            pipeline_id=obs_data.get("pipeline_id", ""),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id, step_count, and metadata
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            metadata=payload.get("metadata", {}),
        )
