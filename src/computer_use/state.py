"""
Chitti Computer-Use Agent State & Action History.
Maintains short-lived execution state, verification status, and action history for computer tasks.
"""

import enum
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class VerificationStatus(str, enum.Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"


class RiskLevel(str, enum.Enum):
    LOW_RISK = "LOW_RISK"
    MEDIUM_RISK = "MEDIUM_RISK"
    HIGH_RISK = "HIGH_RISK"


@dataclass
class ActionHistoryEntry:
    """Represents an atomic action taken during a computer task."""
    step_id: int
    action: str
    target: str
    status: str
    message: str = ""
    observation: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_log_string(self) -> str:
        return f"[Step {self.step_id}] {self.action} on '{self.target}' -> {self.status}: {self.message}"


@dataclass
class ComputerTaskState:
    """
    Explicit Task State tracking the lifecycle and closed-loop progress of computer-use tasks.
    Maintains short-lived task history (not stored permanently in long-term memory).
    """
    task_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    user_request: str = ""
    current_goal: str = ""
    current_step_index: int = 0
    planned_steps: List[Dict[str, Any]] = field(default_factory=list)
    completed_steps: List[Dict[str, Any]] = field(default_factory=list)
    failed_steps: List[Dict[str, Any]] = field(default_factory=list)
    active_application: Optional[str] = None
    active_window: Optional[str] = None
    last_screen: Optional[str] = None
    last_action: Optional[str] = None
    verification_status: VerificationStatus = VerificationStatus.UNKNOWN
    retry_count: int = 0
    max_retries: int = 3
    action_history: List[str] = field(default_factory=list)
    history_entries: List[ActionHistoryEntry] = field(default_factory=list)
    observations: List[str] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    is_completed: bool = False
    is_failed: bool = False
    is_cancelled: bool = False
    is_blocked: bool = False
    blocked_reason: Optional[str] = None

    def record_action(
        self,
        step_id: int,
        action: str,
        target: str,
        status: str,
        message: str = "",
        observation: str = "",
    ) -> None:
        """Records an action in the short-lived task history."""
        self.last_action = action
        entry = ActionHistoryEntry(
            step_id=step_id,
            action=action,
            target=target,
            status=status,
            message=message,
            observation=observation,
        )
        self.history_entries.append(entry)
        summary = f"{action} -> {target} ({status})"
        self.action_history.append(summary)
        if observation:
            self.observations.append(observation)

    def can_retry(self) -> bool:
        """Determines if a failed action can be retried safely."""
        return self.retry_count < self.max_retries and not self.is_completed and not self.is_cancelled

    def increment_retry(self) -> int:
        self.retry_count += 1
        return self.retry_count

    def reset_retries(self) -> None:
        self.retry_count = 0

    def mark_completed(self, summary: str = "") -> None:
        self.is_completed = True
        self.verification_status = VerificationStatus.SUCCESS
        self.end_time = time.time()
        if summary:
            self.observations.append(summary)

    def mark_failed(self, error: str) -> None:
        self.is_failed = True
        self.verification_status = VerificationStatus.FAILED
        self.end_time = time.time()
        self.observations.append(f"Task Failed: {error}")

    def mark_blocked(self, reason: str) -> None:
        self.is_blocked = True
        self.blocked_reason = reason
        self.verification_status = VerificationStatus.FAILED
        self.end_time = time.time()
        self.observations.append(f"Task Blocked: {reason}")

    def get_execution_summary(self) -> str:
        """Returns a readable summary of actions taken."""
        lines = [f"Task: {self.user_request} (Goal: {self.current_goal})"]
        for h in self.action_history:
            lines.append(f" - {h}")
        status_str = "COMPLETED" if self.is_completed else ("BLOCKED" if self.is_blocked else ("FAILED" if self.is_failed else "IN_PROGRESS"))
        lines.append(f"Final Status: {status_str}")
        return "\n".join(lines)
