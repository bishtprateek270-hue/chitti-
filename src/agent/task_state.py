"""
Chitti Agent Task State.
Tracks multi-step task execution, observations, step outcomes, and errors.
"""

import enum
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class TaskStatus(str, enum.Enum):
    IDLE = "IDLE"
    PLANNING = "PLANNING"
    EXECUTING = "EXECUTING"
    VERIFYING = "VERIFYING"
    WAITING_CONFIRMATION = "WAITING_CONFIRMATION"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class StepStatus(str, enum.Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


@dataclass
class AgentStep:
    step_id: int
    description: str
    action_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    status: StepStatus = StepStatus.PENDING
    result_message: Optional[str] = None
    observation: Optional[str] = None
    requires_confirmation: bool = False
    error: Optional[str] = None


@dataclass
class TaskState:
    task_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    task_description: str = ""
    status: TaskStatus = TaskStatus.IDLE
    current_step_index: int = 0
    steps: List[AgentStep] = field(default_factory=list)
    observations: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    pending_confirmation_step: Optional[AgentStep] = None

    @property
    def current_step(self) -> Optional[AgentStep]:
        if 0 <= self.current_step_index < len(self.steps):
            return self.steps[self.current_step_index]
        return None

    def add_observation(self, obs: str) -> None:
        self.observations.append(obs)

    def add_error(self, err: str) -> None:
        self.errors.append(err)

    def mark_completed(self) -> None:
        self.status = TaskStatus.COMPLETED
        self.end_time = time.time()

    def mark_failed(self, reason: str) -> None:
        self.status = TaskStatus.FAILED
        self.add_error(reason)
        self.end_time = time.time()
