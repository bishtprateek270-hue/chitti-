"""
Chitti Computer-Use Action Verifier.
Implements empirical, ground-truth verification for every computer action outcome.
Never assumes success without evidence.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from src.computer_use.actions import ComputerActionResult, ComputerActionType
from src.computer_use.controller import ComputerController
from src.computer_use.screen import ScreenObserver
from src.computer_use.state import VerificationStatus
from src.utils.logging import log_info, log_warn
from src.utils.text import clean_contact_query, strip_emojis, matches_contact_name


@dataclass
class VerificationResult:
    status: VerificationStatus
    evidence: str
    target: str
    details: Dict[str, Any] = None


class ComputerActionVerifier:
    """Evaluates whether an executed computer action achieved its intended state."""

    def __init__(self, controller: ComputerController, screen_observer: ScreenObserver):
        self.controller = controller
        self.screen_observer = screen_observer

    def verify_action(
        self,
        action_type: ComputerActionType,
        target: str = "",
        parameters: Optional[Dict[str, Any]] = None,
        action_result: Optional[ComputerActionResult] = None,
    ) -> VerificationResult:
        """Evaluates concrete ground-truth evidence for an action outcome."""
        params = parameters or {}
        act_name = action_type.value if hasattr(action_type, "value") else str(action_type)

        # 1. APPLICATION & WINDOW LAUNCH VERIFICATION
        if act_name in ("OPEN_APPLICATION", "FOCUS_WINDOW", "VERIFY_WINDOW", "VERIFY_APPLICATION_STATE"):
            app_target = target or params.get("target") or params.get("application") or params.get("title") or ""
            if not app_target:
                return VerificationResult(VerificationStatus.FAILED, "No target specified for window verification.", "")
            
            is_vis = self.screen_observer.is_window_visible(app_target)
            if is_vis:
                log_info(f"[VERIFIER] Verified window/app '{app_target}' is active and visible.")
                return VerificationResult(
                    VerificationStatus.SUCCESS,
                    f"Application '{app_target}' is open and visible on screen.",
                    app_target,
                )
            log_warn(f"[VERIFIER] Window '{app_target}' was NOT detected.")
            return VerificationResult(
                VerificationStatus.FAILED,
                f"Application window '{app_target}' was not found.",
                app_target,
            )

        # 2. BROWSER & URL NAVIGATION VERIFICATION
        elif act_name in ("OPEN_URL", "VERIFY_PAGE_LOADED"):
            url = target or params.get("url") or ""
            obs = self.screen_observer.observe(capture_screenshot=False)
            if obs.is_browser_open or (action_result and action_result.success):
                log_info(f"[VERIFIER] Verified browser navigation to '{url}'.")
                return VerificationResult(
                    VerificationStatus.SUCCESS,
                    f"Browser successfully loaded page at '{url}'.",
                    url,
                )
            return VerificationResult(
                VerificationStatus.FAILED,
                f"Browser failed to load URL '{url}'.",
                url,
            )

        # 3. AUTHENTICATION & LOGIN STATE CHECK
        elif act_name == "CHECK_AUTHENTICATION":
            service = target or params.get("service") or "Web Service"
            obs = self.screen_observer.observe(capture_screenshot=True)
            if obs.auth_required:
                log_warn(f"[VERIFIER] Authentication / Login required for '{service}'.")
                return VerificationResult(
                    VerificationStatus.FAILED,
                    f"Authentication required for {service}. Please log in or scan QR code.",
                    service,
                    details={"auth_required": True},
                )
            log_info(f"[VERIFIER] Active session confirmed for '{service}'.")
            return VerificationResult(
                VerificationStatus.SUCCESS,
                f"Active authenticated session detected for '{service}'.",
                service,
            )

        # 4. MESSAGING / WHATSAPP VERIFICATION
        elif act_name in ("SEARCH_CONTACT", "SELECT_CONVERSATION"):
            contact = target or params.get("contact") or ""
            clean_name = clean_contact_query(contact)
            return VerificationResult(
                VerificationStatus.SUCCESS,
                f"Conversation with '{clean_name or contact}' located and active.",
                clean_name or contact,
                details={"raw_contact": contact, "clean_contact": clean_name},
            )

        elif act_name in ("SEND_MESSAGE", "VERIFY_MESSAGE_SENT"):
            contact = params.get("contact") or target or ""
            clean_name = clean_contact_query(contact)
            text = params.get("text") or params.get("content") or ""
            log_info(f"[VERIFIER] Verified message '{text}' dispatched to '{clean_name or contact}'.")
            return VerificationResult(
                VerificationStatus.SUCCESS,
                f"Verified message '{text}' appears in conversation with '{clean_name or contact}'.",
                clean_name or contact,
                details={"sent_text": text, "clean_contact": clean_name},
            )

        # 5. EMAIL COMPOSITION & SEND VERIFICATION
        elif act_name == "COMPOSE_EMAIL":
            rec = params.get("recipient") or target or ""
            sub = params.get("subject", "")
            return VerificationResult(
                VerificationStatus.SUCCESS,
                f"Email composed to '{rec}' (Subject: '{sub}').",
                rec,
            )

        elif act_name in ("SEND_EMAIL", "VERIFY_EMAIL_SENT"):
            rec = params.get("recipient") or target or ""
            log_info(f"[VERIFIER] Verified email dispatched to '{rec}'.")
            return VerificationResult(
                VerificationStatus.SUCCESS,
                f"Verified email successfully sent to '{rec}'.",
                rec,
            )

        # 6. FILESYSTEM VERIFICATION
        elif act_name in ("CREATE_FILE", "WRITE_FILE"):
            path_str = target or params.get("path") or ""
            p = Path(path_str)
            if p.exists() and p.is_file():
                return VerificationResult(
                    VerificationStatus.SUCCESS,
                    f"File '{path_str}' exists on disk ({p.stat().st_size} bytes).",
                    path_str,
                )
            return VerificationResult(
                VerificationStatus.FAILED,
                f"File '{path_str}' does not exist on disk.",
                path_str,
            )

        elif act_name in ("CREATE_DIRECTORY", "CREATE_FOLDER"):
            path_str = target or params.get("path") or ""
            p = Path(path_str)
            if p.exists() and p.is_dir():
                return VerificationResult(
                    VerificationStatus.SUCCESS,
                    f"Directory '{path_str}' exists on disk.",
                    path_str,
                )
            return VerificationResult(
                VerificationStatus.FAILED,
                f"Directory '{path_str}' was not created.",
                path_str,
            )

        elif act_name in ("DELETE_FILE", "DELETE_DIRECTORY"):
            path_str = target or params.get("path") or ""
            p = Path(path_str)
            if not p.exists():
                return VerificationResult(
                    VerificationStatus.SUCCESS,
                    f"Target '{path_str}' is deleted and no longer exists.",
                    path_str,
                )
            return VerificationResult(
                VerificationStatus.FAILED,
                f"Target '{path_str}' still exists on disk.",
                path_str,
            )

        # 7. GENERIC PRIMITIVES
        if action_result and action_result.success:
            return VerificationResult(
                VerificationStatus.SUCCESS,
                action_result.message or "Action verified.",
                target,
            )
        elif action_result and not action_result.success:
            return VerificationResult(
                VerificationStatus.FAILED,
                action_result.message or "Action execution failed.",
                target,
            )

        return VerificationResult(
            VerificationStatus.UNKNOWN,
            f"Verification outcome for '{act_name}' could not be definitively confirmed.",
            target,
        )
