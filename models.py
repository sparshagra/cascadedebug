"""
Data models for the CascadeDebug Environment.

CascadeDebug is an RL environment that trains LLMs to localize faults in
multi-step professional pipelines (Researcher → Coder → Analyst) and
negotiate surgical fixes with a deterministic gatekeeper.
"""

from typing import Literal, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class CascadeDebugAction(Action):
    """
    Action submitted by the agent each turn.

    The agent reads the pipeline, identifies the faulty step, proposes a fix,
    and negotiates with the gatekeeper until accepted or submits final answer.
    """

    # Which pipeline step is faulty (1-indexed)
    fault_step_id: int = Field(
        ...,
        ge=1,
        le=6,
        description="Which step (1-indexed) the agent believes introduced the error.",
    )

    # Which role produced the faulty output
    blame_role: Literal["Researcher", "Coder", "Analyst"] = Field(
        ...,
        description="The role responsible for the faulty step output.",
    )

    # The corrected content for that step
    fix_content: str = Field(
        ...,
        min_length=1,
        description="The corrected output for the identified faulty step.",
    )

    # Action lifecycle: propose → (revise if rejected) → submit
    action_type: Literal["propose", "revise", "submit"] = Field(
        ...,
        description=(
            "propose: initial fix attempt. "
            "revise: revised fix after gatekeeper rejection. "
            "submit: final locked submission."
        ),
    )


class CascadeDebugObservation(Observation):
    """
    Observation returned to the agent each turn.

    Contains the pipeline step outputs and any gatekeeper feedback.
    Does NOT expose the injected_step (that is hidden in State).
    """

    pipeline: list[dict] = Field(
        default_factory=list,
        description=(
            "List of pipeline steps. Each entry: "
            "{'role': str, 'output': str, 'step_id': int}"
        ),
    )

    task_brief: str = Field(
        default="",
        description="The original task given to the pipeline.",
    )

    turn: int = Field(
        default=1,
        ge=1,
        description="Current negotiation turn (1 or 2).",
    )

    gatekeeper_feedback: Optional[str] = Field(
        default=None,
        description=(
            "Populated after a REJECT. Contains the violated constraint. "
            "None on first turn or after ACCEPT."
        ),
    )

    curriculum_level: int = Field(
        default=1,
        ge=1,
        le=3,
        description="Current curriculum level (1=easy, 2=medium, 3=hard).",
    )

    pipeline_id: str = Field(
        default="",
        description="Unique identifier for this episode's pipeline.",
    )


class CascadeDebugState(Field):
    """
    Full internal state — includes ground truth. Never sent to agent.
    Available via env.state for logging and debugging only.
    """

    pipeline_id: str = ""
    injected_step: int = 0          # ground truth — hidden from agent
    injected_role: str = ""         # ground truth — hidden from agent
    error_type: str = ""
    curriculum_level: int = 1
    turn: int = 1
    submitted: bool = False
    gatekeeper_accepted: bool = False
    fix_history: list = Field(default_factory=list)
