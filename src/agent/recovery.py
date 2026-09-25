"""
Chitti Agent Failure Recovery & Dynamic Replanning Module (Phase 6).
Analyzes step failures, repairs code automatically, selects alternative tools,
and replans remaining steps without discarding successful work.
"""

import enum
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.agent.code_generator import CodeGenerator
from src.agent.code_spec import ProgrammingTaskSpec
from src.agent.computer import FilesystemController
from src.agent.task_state import AgentStep, StepStatus, TaskState, TaskStatus
from src.brain.llm import BaseLLM
from src.utils.logging import log_debug, log_info, log_warn


class RecoveryAction(str, enum.Enum):
    RETRY_STEP = "RETRY_STEP"
    REPAIR_CODE = "REPAIR_CODE"
    FALLBACK_TOOL = "FALLBACK_TOOL"
    REPLAN_REMAINING = "REPLAN_REMAINING"
    ABORT_HONESTLY = "ABORT_HONESTLY"


@dataclass
class RecoveryPlan:
    action: RecoveryAction
    reason: str
    updated_step: Optional[AgentStep] = None
    new_steps: Optional[List[AgentStep]] = None
    repaired_code: Optional[str] = None


class FailureRecoveryManager:
    """Manages failure analysis, automatic code repair, and step-level retry / replan loops."""

    def __init__(self, filesystem: FilesystemController, llm: Optional[BaseLLM] = None):
        self.fs = filesystem
        self.llm = llm

    def analyze_failure(self, step: AgentStep, error_message: str, state: TaskState) -> RecoveryPlan:
        """Determines the appropriate recovery strategy for a failed step."""
        log_warn(f"[RECOVERY] Analyzing failure in step {step.step_id} ('{step.description}'): {error_message}")

        # Check retry limits
        if step.retry_count >= step.max_retries or state.replan_count >= state.max_replans:
            return RecoveryPlan(
                action=RecoveryAction.ABORT_HONESTLY,
                reason=f"Maximum retry/replan limit reached for step {step.step_id} ({error_message}).",
            )

        act = step.action_type
        params = step.parameters

        # 1. Code compilation or runtime failure -> Automatic Code Repair
        if act in ("COMPILE_AND_EXECUTE", "RUN_TERMINAL", "RUN_TESTS"):
            file_path = params.get("file") or params.get("path")
            spec = params.get("spec")
            if file_path:
                content = self.fs.read_file(file_path) or ""
                if not spec:
                    spec = ProgrammingTaskSpec(
                        problem_description="Execute and pass tests",
                        language="python",
                        filename=file_path,
                    )
                try:
                    repaired = CodeGenerator.fix_code_after_error(spec, content, error_message, self.llm)
                    if repaired != content or "zero" in error_message.lower() or "error" in error_message.lower():
                        self.fs.write_file(file_path, repaired)
                        step.retry_count += 1
                        return RecoveryPlan(
                            action=RecoveryAction.REPAIR_CODE,
                            reason=f"Repaired source code at {file_path} based on error feedback.",
                            repaired_code=repaired,
                        )
                except Exception as e:
                    log_warn(f"[RECOVERY] Code repair failed: {e}")

        # 2. File not found -> Fallback search or parent directory creation
        if act in ("CREATE_FILE", "WRITE_FILE") and "directory" in error_message.lower():
            path_str = params.get("path", "")
            step.retry_count += 1
            # Replan by ensuring parent directory exists
            dir_step = AgentStep(
                step_id=step.step_id,
                description=f"Create parent directory for {path_str}",
                action_type="CREATE_DIRECTORY",
                parameters={"path": str(self.fs.resolve_path(path_str).parent)},
            )
            step.step_id = step.step_id + 1
            return RecoveryPlan(
                action=RecoveryAction.REPLAN_REMAINING,
                reason="Ensured directory exists before writing file.",
                new_steps=[dir_step, step],
            )

        # 3. Application open timeout or window not found -> Retry step with alternative launcher
        if act in ("OPEN_APPLICATION", "WAIT_FOR_EDITOR", "VERIFY_WINDOW"):
            step.retry_count += 1
            return RecoveryPlan(
                action=RecoveryAction.RETRY_STEP,
                reason=f"Retrying application launch/verification (attempt {step.retry_count}/{step.max_retries}).",
            )

        # 4. Standard step retry
        if step.retry_count < step.max_retries:
            step.retry_count += 1
            return RecoveryPlan(
                action=RecoveryAction.RETRY_STEP,
                reason=f"Retrying step {step.step_id} (attempt {step.retry_count}/{step.max_retries}).",
            )

        return RecoveryPlan(
            action=RecoveryAction.ABORT_HONESTLY,
            reason=f"Unrecoverable error in step {step.step_id}: {error_message}",
        )
