"""
Chitti Laptop Agent Manager Module.
Coordinates command parsing, safety validation, risk assessment, confirmation handling,
and deterministic tool execution for Windows laptop operations.
"""

import re
from typing import Optional, Tuple, Dict, Any

from src.agent.actions import ActionType, RiskLevel, StructuredAction, ActionResult
from src.agent.parser import ActionParser
from src.agent.validator import ActionValidator
from src.agent.executor import ActionExecutor
from src.config import get_config
from src.utils.logging import log_chitti, log_warning, log_error, log_debug


class LaptopAgentManager:
    """Coordinates safe laptop agent commands and tool execution."""

    def __init__(self, workspace_dir: Optional[str] = None, screenshots_dir: Optional[str] = None):
        cfg = get_config()
        self.workspace_dir = workspace_dir or cfg.agent.workspace_dir
        self.screenshots_dir = screenshots_dir or cfg.agent.screenshots_dir
        self.parser = ActionParser()
        self.validator = ActionValidator()
        self.executor = ActionExecutor()
        self.pending_destructive_action: Optional[StructuredAction] = None

    def handle_command(self, user_text: str, lang: str = "en") -> Optional[Tuple[bool, str, Optional[ActionResult]]]:
        """
        Parses and evaluates if the user's message is a laptop action command.
        Returns:
            - (True, confirmation_or_result_message, action_result) if handled as an action.
            - (False, error_or_rejection_message, None) if validation failed.
            - None if not an agent command (routes to memory/LLM).
        """
        raw = user_text.strip()
        lower = raw.lower()

        # 1. Handle Pending Destructive Action Confirmation
        if self.pending_destructive_action is not None:
            action = self.pending_destructive_action
            if re.search(r"\b(?:no|cancel|stop|abort|don'?t|nope|nahi|nahin|mat karo)\b", lower):
                self.pending_destructive_action = None
                log_chitti("[AGENT] Destructive action cancelled by user.")
                if lang == "hi":
                    return True, "कार्रवाई रद्द कर दी गई है। आपकी फ़ाइलें सुरक्षित हैं।", None
                elif lang in ("hinglish", "mixed"):
                    return True, "Action cancel kar diya gaya hai. Files safe hain.", None
                else:
                    return True, "Action cancelled. Your files remain untouched.", None

            elif re.search(r"\b(?:yes|proceed|confirm|sure|do it|yep|yeah|haan|sahi|ha|ha kar do)\b", lower):
                self.pending_destructive_action = None
                log_chitti(f"[AGENT] Destructive action confirmed for {action.action.value}")
                res = self.executor.execute(action, default_workspace=self.workspace_dir, screenshots_dir=self.screenshots_dir)
                resp_msg = self._format_response(res, lang=lang)
                return True, resp_msg, res

            else:
                prompt_msg = (
                    "यह कार्रवाई आपकी फ़ाइलों को हमेशा के लिए हटा सकती है। क्या आप जारी रखना चाहते हैं? (हाँ / नहीं)"
                    if lang == "hi" else
                    "Yeh action aapki files ko permanently modify ya delete kar sakta hai. Kya aap continue karna chahte hain? (Haan / Nahi)"
                    if lang in ("hinglish", "mixed") else
                    "This action may permanently modify or delete your files. Do you want me to continue? Please answer Yes or No."
                )
                return True, prompt_msg, None

        # 2. Parse command into structured action
        structured_action = self.parser.parse_command(raw)
        if not structured_action:
            return None

        log_chitti(f"[AGENT] Intent detected: {structured_action.action.value}")
        if structured_action.target:
            log_chitti(f"[AGENT] Target: {structured_action.target}")
        log_chitti(f"[AGENT] Risk: {structured_action.risk_level.value}")

        # 3. Validate structured action
        is_valid, reason, validated_action = self.validator.validate(structured_action)
        if not is_valid or not validated_action:
            log_chitti(f"[AGENT] Validation: FAILED | Reason: {reason}")
            log_chitti(f"[AGENT] Execution: BLOCKED")
            return False, reason, None

        log_chitti(f"[AGENT] Validation: PASSED")

        # 4. Check for Confirmation Requirement
        if validated_action.requires_confirmation or validated_action.risk_level == RiskLevel.HIGH:
            self.pending_destructive_action = validated_action
            log_chitti(f"[AGENT] Action requires user confirmation before proceeding.")
            confirm_msg = (
                f"यह कार्रवाई '{validated_action.target}' को हमेशा के लिए हटा सकती है। क्या आप जारी रखना चाहते हैं? (हाँ / नहीं)"
                if lang == "hi" else
                f"Yeh action '{validated_action.target}' ko delete kar sakta hai. Kya aap confirm karte hain? (Haan / Nahi)"
                if lang in ("hinglish", "mixed") else
                f"This action may permanently delete '{validated_action.target}'. Do you want me to continue? (Yes / No)"
            )
            return True, confirm_msg, None

        # 5. Execute Action
        result = self.executor.execute(validated_action, default_workspace=self.workspace_dir, screenshots_dir=self.screenshots_dir)
        response_msg = self._format_response(result, lang=lang)
        return True, response_msg, result

    def _format_response(self, result: ActionResult, lang: str = "en") -> str:
        """Formats natural multilingual response for action execution results."""
        if not result.success:
            if lang == "hi":
                return f"माफ़ कीजिए, कार्रवाई पूरी नहीं हो सकी: {result.message}"
            elif lang in ("hinglish", "mixed"):
                return f"Sorry, action execute nahi ho saka: {result.message}"
            else:
                return result.message

        if result.action == ActionType.OPEN_APPLICATION:
            if lang == "hi":
                return f"{result.target} खोल दिया गया है।"
            elif lang in ("hinglish", "mixed"):
                return f"{result.target} khol diya gaya hai."
            else:
                return f"{result.target} is open."

        elif result.action == ActionType.CLOSE_APPLICATION:
            if lang == "hi":
                return f"{result.target} बंद कर दिया गया है।"
            elif lang in ("hinglish", "mixed"):
                return f"{result.target} band kar diya gaya hai."
            else:
                return f"Closed {result.target}."

        elif result.action == ActionType.OPEN_FOLDER:
            if lang == "hi":
                return f"{result.target} फ़ोल्डर खोल दिया गया है।"
            elif lang in ("hinglish", "mixed"):
                return f"{result.target} folder open kar diya gaya hai."
            else:
                return f"Opened {result.target} folder."

        elif result.action == ActionType.OPEN_URL:
            if lang == "hi":
                return f"ब्राउज़र में {result.target} खोल दिया गया है।"
            elif lang in ("hinglish", "mixed"):
                return f"Browser mein {result.target} open kar diya hai."
            else:
                return f"Opened {result.target} in your browser."

        elif result.action == ActionType.TAKE_SCREENSHOT:
            filename = result.data.get("filename", "screenshot.png")
            if lang == "hi":
                return f"स्क्रीनशॉट ले लिया गया है और {filename} के रूप में सुरक्षित किया गया है।"
            elif lang in ("hinglish", "mixed"):
                return f"Screenshot le liya gaya hai aur {filename} mein save ho gaya hai."
            else:
                return f"Screenshot taken and saved as {filename}."

        elif result.action == ActionType.SET_VOLUME:
            op = result.data.get("operation", "")
            if lang == "hi":
                return "वॉल्यूम समायोजित कर दिया गया है।"
            elif lang in ("hinglish", "mixed"):
                return f"Volume {op} kar diya hai."
            else:
                return result.message

        return result.message
