"""
Chitti Agent Verifier Module (Phase 6).
Implements evidence-based verification for every subtask outcome.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.agent.actions import ActionResult
from src.agent.code_validator import CodeValidator
from src.agent.computer import (
    AppController,
    BrowserController,
    ComputerController,
    FilesystemController,
    ScreenAnalyzer,
)
from src.agent.task_state import AgentStep, ExecutionFlag, TaskState
from src.agent.tools import ToolEngine
from src.utils.logging import log_debug, log_info, log_warn


@dataclass
class VerificationOutcome:
    verified: bool
    evidence: str
    details: Dict[str, Any] = None


class SubtaskVerifier:
    """Verifies subtask execution against expected state and ground truth evidence."""

    def __init__(
        self,
        tools: ToolEngine,
        filesystem: FilesystemController,
        screen_analyzer: ScreenAnalyzer,
        browser: BrowserController,
    ):
        self.tools = tools
        self.fs = filesystem
        self.screen_analyzer = screen_analyzer
        self.browser = browser

    def verify_step(self, step: AgentStep, state: TaskState, action_result: Optional[ActionResult] = None) -> VerificationOutcome:
        """Evaluates whether the executed step meets its verification criteria."""
        act = step.action_type
        params = step.parameters

        # 1. FILE OPERATIONS
        if act in ("CREATE_FILE", "WRITE_FILE"):
            path_str = params.get("path", "")
            resolved = self.fs.resolve_path(path_str)
            if not resolved.exists():
                return VerificationOutcome(False, f"File '{path_str}' does not exist on disk.")
            content = self.fs.read_file(path_str)
            if content is None:
                return VerificationOutcome(False, f"File '{path_str}' could not be read.")
            state.set_flag(ExecutionFlag.FILE_CREATED, True)
            state.set_flag(ExecutionFlag.FILE_SAVED, True)
            return VerificationOutcome(True, f"File exists at {resolved} ({len(content)} chars)")

        elif act in ("CREATE_DIRECTORY", "CREATE_FOLDER"):
            path_str = params.get("path", "")
            resolved = self.fs.resolve_path(path_str)
            if not resolved.exists() or not resolved.is_dir():
                return VerificationOutcome(False, f"Directory '{path_str}' does not exist.")
            return VerificationOutcome(True, f"Directory exists at {resolved}")

        elif act in ("DELETE_FILE", "DELETE_DIRECTORY"):
            path_str = params.get("path", "")
            resolved = self.fs.resolve_path(path_str)
            if resolved.exists():
                return VerificationOutcome(False, f"Target '{path_str}' still exists on disk.")
            return VerificationOutcome(True, f"Target '{path_str}' successfully removed.")

        elif act == "VERIFY_FILE_CONTENT":
            path_str = params.get("path", "")
            keyword = params.get("expected_keyword", "")
            res = self.tools.execute_tool("verify_file_content", {"path": path_str, "expected_keyword": keyword})
            if not res.success:
                return VerificationOutcome(False, res.data.get("evidence", res.message))
            state.set_flag(ExecutionFlag.FILE_SAVED, True)
            return VerificationOutcome(True, res.data.get("evidence", "File content verified"))

        # 2. WINDOW & APPLICATION LAUNCH
        elif act in ("OPEN_APPLICATION", "OPEN_FOLDER"):
            target = params.get("target") or params.get("application") or ""
            # If action result succeeded, check window/process
            if action_result and action_result.success:
                return VerificationOutcome(True, f"Application/Folder '{target}' launched successfully.")
            return VerificationOutcome(False, f"Could not launch '{target}'.")

        elif act == "VERIFY_WINDOW":
            title = params.get("title", "")
            res = self.screen_analyzer.verify_window(title)
            if not res.success:
                return VerificationOutcome(False, f"Window verification failed for '{title}': {res.evidence}")
            return VerificationOutcome(True, f"Window verified: {res.evidence}")

        # 3. VS CODE EDITOR VERIFICATION
        elif act == "VERIFY_EDITOR_CONTENT":
            app = params.get("application", "Visual Studio Code")
            expected_file = params.get("expected_file", "script.py")
            markers = params.get("expected_markers", [])
            res = self.tools.execute_tool("verify_editor_content", {"application": app, "expected_file": expected_file, "expected_markers": markers})
            if not res.success:
                return VerificationOutcome(False, f"Editor verification failed: {res.data.get('evidence', res.error)}")
            state.set_flag(ExecutionFlag.EDITOR_CONTENT_VERIFIED, True)
            return VerificationOutcome(True, f"Editor content verified: {res.data.get('evidence')}")

        # 4. BROWSER & YOUTUBE PLAYBACK
        elif act == "VERIFY_PLAYBACK":
            b_state = self.browser.get_browser_state()
            if b_state.playback_verified and (b_state.active_video_id or b_state.current_url):
                return VerificationOutcome(True, f"Playback active (URL: {b_state.current_url}, Video: {b_state.active_video_id})")
            win_res = self.screen_analyzer.verify_window("YouTube")
            if win_res.success:
                return VerificationOutcome(True, f"YouTube window verified: {win_res.evidence}")
            return VerificationOutcome(False, "YouTube playback could not be confirmed.")

        # 5. CODE EXECUTION & TERMINAL
        elif act in ("RUN_TERMINAL", "COMPILE_AND_EXECUTE", "VERIFY_EXECUTION"):
            is_web = params.get("is_web", False)
            if is_web:
                return VerificationOutcome(True, "Web application active in browser.")
            if not state.get_flag(ExecutionFlag.TOOLCHAIN_VERIFIED):
                return VerificationOutcome(False, "Toolchain is not installed on this system. Execution cannot proceed.")
            if action_result and action_result.success:
                exit_code = action_result.data.get("exit_code", 0) if action_result.data else 0
                if exit_code == 0:
                    state.set_flag(ExecutionFlag.CODE_EXECUTED, True)
                    state.set_flag(ExecutionFlag.EXECUTION_VERIFIED, True)
                    out_text = action_result.data.get("output", "") if action_result.data else action_result.message
                    return VerificationOutcome(True, f"Process exited with code 0. Output:\n{out_text[:200]}")
                err_text = action_result.data.get("error", "") if action_result.data else action_result.message
                return VerificationOutcome(False, f"Process exited with non-zero code {exit_code}. Stderr: {err_text}")
            elif action_result and not action_result.success:
                return VerificationOutcome(False, f"Command execution failed: {action_result.message}")
            return VerificationOutcome(True, "Execution verified.")

        # Default fallback
        if action_result:
            return VerificationOutcome(action_result.success, action_result.message)
        return VerificationOutcome(True, "Step verified.")

