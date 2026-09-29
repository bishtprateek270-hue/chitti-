"""
Chitti Computer-Use Action Recovery Engine.
Implements intelligent error diagnosis, screen re-inspection, duplicate action prevention,
and safe recovery policies.
"""

from typing import Any, Dict, Optional, Tuple

from src.computer_use.actions import ComputerActionResult, ComputerActionStep, ComputerActionType
from src.computer_use.controller import ComputerController
from src.computer_use.screen import ScreenObserver
from src.computer_use.state import ComputerTaskState, VerificationStatus
from src.computer_use.verifier import ComputerActionVerifier
from src.utils.logging import log_info, log_warn


class ComputerActionRecovery:
    """Diagnoses action failures and executes safe recovery strategies."""

    def __init__(
        self,
        controller: ComputerController,
        screen_observer: ScreenObserver,
        verifier: ComputerActionVerifier,
    ):
        self.controller = controller
        self.screen_observer = screen_observer
        self.verifier = verifier

    def attempt_recovery(
        self,
        step: ComputerActionStep,
        state: ComputerTaskState,
        last_result: ComputerActionResult,
    ) -> Tuple[bool, str]:
        """
        Evaluates current screen state and attempts recovery without repeating irreversible actions blindly.
        Returns: (recovered: bool, message: str)
        """
        act_name = step.action_type.value if hasattr(step.action_type, "value") else str(step.action_type)
        log_warn(f"[RECOVERY] Initiating recovery for failed step {step.step_id} ({act_name}). Attempt {state.retry_count + 1}/{state.max_retries}")

        # Check retry limit
        if not state.can_retry():
            msg = f"Recovery limit reached ({state.max_retries} attempts). Aborting to prevent unsafe repetitions."
            log_warn(f"[RECOVERY] {msg}")
            state.mark_failed(msg)
            return False, msg

        state.increment_retry()

        # 1. Inspect current screen to understand the actual state
        obs = self.screen_observer.observe(capture_screenshot=True)

        # 2. Check for authentication or login barriers
        if obs.auth_required:
            msg = f"Action blocked: Authentication or QR code login required for {obs.auth_service or 'the application'}."
            log_warn(f"[RECOVERY] {msg}")
            state.mark_blocked(msg)
            return False, msg

        # 3. Duplicate Prevention for Communication Actions
        # If sending a message/email failed verification, check if the text already appears
        if act_name in ("SEND_MESSAGE", "SEND_EMAIL"):
            log_info(f"[RECOVERY] Checking whether {act_name} already took effect to avoid duplicate dispatch...")
            # If evidence of sent message is visible, do NOT resend
            if step.parameters.get("text") and step.parameters.get("text") in obs.summary_text:
                log_info("[RECOVERY] Message already detected on screen. Marking as completed without duplicate send.")
                return True, "Message was already dispatched successfully."

        # 4. Window focus recovery
        target_app = step.parameters.get("application") or step.parameters.get("service") or step.parameters.get("title")
        if target_app:
            if not self.screen_observer.is_window_visible(target_app):
                msg = f"Application '{target_app}' is not available or could not be launched."
                log_warn(f"[RECOVERY] {msg}")
                state.mark_failed(msg)
                return False, msg
            log_info(f"[RECOVERY] Restoring foreground focus for '{target_app}'...")
            self.controller.focus_window(target_app)
            self.controller.wait(0.2)

        # 5. Safe re-attempt of UI action
        log_info(f"[RECOVERY] Re-evaluating step {step.step_id} after environment recovery...")
        return True, "Environment recovered. Ready to re-verify."
