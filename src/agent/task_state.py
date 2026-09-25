"""
Chitti Agent Task State & Lifecycle Manager (Phase 6).
Tracks multi-step task execution, step dependencies, observations, step outcomes,
recovery attempts, replanning, and cancellation.
"""

import enum
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


class TaskStatus(str, enum.Enum):
    TASK_CREATED = "TASK_CREATED"
    TASK_UNDERSTOOD = "TASK_UNDERSTOOD"
    PLANNING = "PLANNING"
    PLAN_READY = "PLAN_READY"
    EXECUTING = "EXECUTING"
    OBSERVING = "OBSERVING"
    VERIFYING = "VERIFYING"
    RECOVERING = "RECOVERING"
    REPLANNING = "REPLANNING"
    WAITING_FOR_CONFIRMATION = "WAITING_FOR_CONFIRMATION"
    WAITING_CONFIRMATION = "WAITING_CONFIRMATION"  # Alias for backward compatibility
    PAUSED = "PAUSED"
    CANCELLED = "CANCELLED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    IDLE = "IDLE"


class StepStatus(str, enum.Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    COMPLETED = "COMPLETED"  # Alias for SUCCESS
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    BLOCKED = "BLOCKED"


class ExecutionFlag(str, enum.Enum):
    TASK_UNDERSTOOD = "TASK_UNDERSTOOD"
    CODE_GENERATED = "CODE_GENERATED"
    CODE_VALIDATED = "CODE_VALIDATED"
    FILE_CREATED = "FILE_CREATED"
    FILE_OPENED = "FILE_OPENED"
    EDITOR_CONTENT_VERIFIED = "EDITOR_CONTENT_VERIFIED"
    FILE_SAVED = "FILE_SAVED"
    TOOLCHAIN_VERIFIED = "TOOLCHAIN_VERIFIED"
    CODE_EXECUTED = "CODE_EXECUTED"
    EXECUTION_VERIFIED = "EXECUTION_VERIFIED"


@dataclass
class AgentStep:
    step_id: int
    description: str
    action_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    status: StepStatus = StepStatus.PENDING
    depends_on: List[int] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 2
    result_message: Optional[str] = None
    observation: Optional[str] = None
    requires_confirmation: bool = False
    error: Optional[str] = None
    verification_strategy: Optional[str] = None
    expected_outcome: Optional[str] = None

    def mark_success(self, message: str, observation: Optional[str] = None) -> None:
        self.status = StepStatus.SUCCESS
        self.result_message = message
        self.observation = observation
        self.error = None

    def mark_failed(self, error: str) -> None:
        self.status = StepStatus.FAILED
        self.error = error

    def mark_blocked(self, reason: str) -> None:
        self.status = StepStatus.BLOCKED
        self.error = reason


@dataclass
class TaskContext:
    """Session-level short-term task context to link follow-up user requests."""
    current_goal: str = ""
    active_language: str = "python"
    workspace: str = "data/workspace"
    files_created: List[str] = field(default_factory=list)
    files_modified: List[str] = field(default_factory=list)
    commands_executed: List[str] = field(default_factory=list)
    last_error: Optional[str] = None
    last_output: Optional[str] = None
    active_application: Optional[str] = None
    active_project_path: Optional[str] = None
    custom_state: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskState:
    task_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    task_description: str = ""
    goal: str = ""
    status: TaskStatus = TaskStatus.TASK_CREATED
    current_step_index: int = 0
    steps: List[AgentStep] = field(default_factory=list)
    completed_steps: List[AgentStep] = field(default_factory=list)
    failed_steps: List[AgentStep] = field(default_factory=list)
    observations: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    decisions: List[str] = field(default_factory=list)
    verified_states: Dict[str, bool] = field(default_factory=dict)
    replan_count: int = 0
    max_replans: int = 2
    max_step_retries: int = 2
    max_recovery_attempts: int = 3
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    pending_confirmation_step: Optional[AgentStep] = None
    is_cancelled: bool = False
    is_paused: bool = False
    context: TaskContext = field(default_factory=TaskContext)

    def __post_init__(self):
        if not self.goal and self.task_description:
            self.goal = self.task_description

    def set_flag(self, flag: ExecutionFlag, value: bool = True) -> None:
        self.verified_states[flag.value] = value

    def get_flag(self, flag: ExecutionFlag) -> bool:
        return self.verified_states.get(flag.value, False)

    @property
    def current_step(self) -> Optional[AgentStep]:
        if 0 <= self.current_step_index < len(self.steps):
            return self.steps[self.current_step_index]
        return None

    def add_observation(self, obs: str) -> None:
        self.observations.append(obs)

    def add_error(self, err: str) -> None:
        self.errors.append(err)

    def add_decision(self, dec: str) -> None:
        self.decisions.append(dec)

    def cancel(self) -> None:
        self.is_cancelled = True
        self.status = TaskStatus.CANCELLED
        self.end_time = time.time()

    def pause(self) -> None:
        self.is_paused = True
        self.status = TaskStatus.PAUSED

    def resume(self) -> None:
        self.is_paused = False
        self.status = TaskStatus.EXECUTING

    def mark_completed(self) -> None:
        self.status = TaskStatus.COMPLETED
        self.end_time = time.time()

    def mark_failed(self, reason: str) -> None:
        self.status = TaskStatus.FAILED
        self.add_error(reason)
        self.end_time = time.time()

    def can_execute_step(self, step: AgentStep) -> Tuple[bool, Optional[str]]:
        """Checks if all prerequisite steps in step.depends_on have succeeded."""
        if not step.depends_on:
            return True, None
        completed_ids = {s.step_id for s in self.completed_steps if s.status in (StepStatus.SUCCESS, StepStatus.COMPLETED)}
        for dep_id in step.depends_on:
            if dep_id not in completed_ids:
                return False, f"Prerequisite step {dep_id} has not completed successfully."
        return True, None
