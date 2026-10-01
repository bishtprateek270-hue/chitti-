"""
Chitti Reinforcement Learning & Self-Correction Data Models.
Defines episodic trajectories, reward signals, self-reflexion lessons, and DPO pairs.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import enum
from typing import Any, Dict, List, Optional


class TaskOutcome(str, enum.Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    RECOVERED = "RECOVERED"
    USER_INTERRUPTED = "USER_INTERRUPTED"


@dataclass
class TrajectoryStep:
    """Represents a single state-action-reward transition step (s_t, a_t, r_t, s_{t+1})."""
    step_id: int
    action_type: str
    action_payload: Dict[str, Any]
    observation_before: str
    observation_after: str
    reward: float
    verified: bool
    error_message: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class ReflexionLesson:
    """Represents a distilled self-correction rule learned from a failure or mistake."""
    id: Optional[int] = None
    task_pattern: str = ""
    domain: str = "general"          # "browser", "os", "speech", "vision", "router", "code"
    failed_action: str = ""
    root_cause: str = ""
    correction_rule: str = ""        # Actionable guideline to avoid future repetition
    reward_delta: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class Episode:
    """Represents a complete task execution trajectory with cumulative reward and reflections."""
    episode_id: str
    task_description: str
    domain: str
    steps: List[TrajectoryStep] = field(default_factory=list)
    total_reward: float = 0.0
    outcome: TaskOutcome = TaskOutcome.SUCCESS
    lessons_learned: List[ReflexionLesson] = field(default_factory=list)
    user_feedback_score: Optional[float] = None
    start_time: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    end_time: Optional[str] = None


@dataclass
class DPOPreferencePair:
    """Represents a Direct Preference Optimization (DPO) pairwise training sample."""
    id: Optional[int] = None
    prompt: str = ""
    chosen_response: str = ""       # High-reward / corrected trajectory
    rejected_response: str = ""     # Low-reward / failed trajectory
    domain: str = "general"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
