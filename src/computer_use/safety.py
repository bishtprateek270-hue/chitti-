"""
Chitti Computer-Use Safety & Risk Classification.
Enforces security boundaries, prevents dangerous actions without confirmation,
and ensures safe device automation.
"""

import re
from typing import Any, Dict, Optional, Tuple

from src.computer_use.state import RiskLevel


class ComputerSafetyPolicy:
    """
    Evaluates actions against security policies and determines whether user confirmation is required.
    """

    # Low risk actions: Opening apps, web navigation, searching, typing, safe file reading/creation
    LOW_RISK_ACTIONS = {
        "OPEN_APPLICATION", "CLOSE_APPLICATION", "OPEN_URL", "SEARCH_WEB", "SEARCH_YOUTUBE",
        "PLAY_SONG", "INSPECT_SCREEN", "READ_SCREEN", "FIND_UI_ELEMENT", "CLICK",
        "DOUBLE_CLICK", "RIGHT_CLICK", "MOVE_MOUSE", "SCROLL", "DRAG", "WAIT",
        "WAIT_FOR_UI", "TYPE_TEXT", "PRESS_KEY", "HOTKEY", "COPY", "PASTE",
        "CLIPBOARD_READ", "CLIPBOARD_WRITE", "GET_ACTIVE_WINDOW", "FOCUS_WINDOW",
        "MINIMIZE_WINDOW", "MAXIMIZE_WINDOW", "VERIFY_UI_STATE", "VERIFY_TEXT",
        "VERIFY_ELEMENT", "VERIFY_APPLICATION_STATE", "CREATE_FILE", "WRITE_FILE",
        "CREATE_DIRECTORY", "CREATE_FOLDER", "TAKE_SCREENSHOT", "SET_VOLUME",
        "VERIFY_FILE_CONTENT", "VERIFY_EDITOR_CONTENT", "VERIFY_PAGE_LOADED",
    }

    # Medium risk actions: Modifying existing files, sending communication (emails / messages)
    MEDIUM_RISK_ACTIONS = {
        "MODIFY_FILE", "UPDATE_FILE", "SEND_EMAIL", "CONFIRM_SEND", "SEND_MESSAGE",
        "EXECUTE_COMMAND", "RUN_TERMINAL", "INSTALL_PACKAGE",
    }

    # High risk actions: Permanent deletion, system shutdown, formatting, security modifications
    HIGH_RISK_ACTIONS = {
        "DELETE_FILE", "DELETE_DIRECTORY", "DELETE_FOLDER", "FORMAT_DRIVE",
        "KILL_PROCESS", "SHUTDOWN", "RESTART", "MODIFY_SYSTEM_SETTINGS",
        "FINANCIAL_TRANSACTION",
    }

    @classmethod
    def evaluate_risk(cls, action_type: str, target: str = "", parameters: Optional[Dict[str, Any]] = None) -> RiskLevel:
        """Determines the risk classification for a given computer action."""
        act = action_type.upper().strip()
        params = parameters or {}

        # 1. High risk checks
        if act in cls.HIGH_RISK_ACTIONS:
            return RiskLevel.HIGH_RISK

        # Check for dangerous shell commands
        if act in ("RUN_TERMINAL", "EXECUTE_COMMAND"):
            cmd = (params.get("command") or target or "").lower()
            dangerous_patterns = [
                r"\brm\s+-rf\b", r"\bformat\b", r"\bdel\s+/[fqs]\b", r"\brd\s+/[sq]\b",
                r"\brmdir\s+/[sq]\b", r"\bdiskpart\b", r"\breg\s+delete\b", r"\bshutdown\b",
                r"\bdrop\s+database\b", r"\bdrop\s+table\b",
            ]
            if any(re.search(pat, cmd) for pat in dangerous_patterns):
                return RiskLevel.HIGH_RISK
            return RiskLevel.MEDIUM_RISK

        # 2. Medium risk checks
        if act in cls.MEDIUM_RISK_ACTIONS:
            return RiskLevel.MEDIUM_RISK

        if act == "SEND_EMAIL" or act == "CONFIRM_SEND":
            return RiskLevel.MEDIUM_RISK

        # 3. Default to Low Risk
        return RiskLevel.LOW_RISK

    @classmethod
    def requires_confirmation(cls, action_type: str, target: str = "", parameters: Optional[Dict[str, Any]] = None) -> Tuple[bool, Optional[str]]:
        """
        Determines if an action requires explicit user confirmation before execution.
        Returns: (requires_confirmation: bool, confirmation_prompt: Optional[str])
        """
        act = action_type.upper().strip()
        risk = cls.evaluate_risk(act, target=target, parameters=parameters)
        params = parameters or {}

        # High risk actions ALWAYS require confirmation
        if risk == RiskLevel.HIGH_RISK:
            tgt = target or params.get("path") or params.get("target") or "the specified files"
            prompt = f"This action will permanently delete or modify '{tgt}'. Do you want me to proceed? (Yes / No)"
            return True, prompt

        # Email dispatch confirmation
        if act in ("CONFIRM_SEND", "SEND_EMAIL"):
            rec = params.get("recipient") or target or "recipient"
            sub = params.get("subject", "")
            sub_text = f" with subject '{sub}'" if sub else ""
            prompt = f"I am ready to send an email to '{rec}'{sub_text}. Should I proceed? (Yes / No)"
            return True, prompt

        # Safe actions proceed automatically
        return False, None
