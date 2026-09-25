"""
Chitti Laptop Agent Manager Module.
Coordinates command classification, multi-step planning, tool execution, safety validation,
risk assessment, general-purpose coding agent, and confirmation handling for Windows laptop operations.
"""

import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from src.agent.actions import ActionResult, ActionType, RiskLevel, StructuredAction
from src.agent.classifier import ClassificationResult, TaskClassifier, TaskIntent
from src.agent.code_generator import CodeGenerator
from src.agent.computer import (
    AppController,
    BrowserController,
    ComputerController,
    FilesystemController,
    ScreenAnalyzer,
    TerminalController,
)
from src.agent.executor import ActionExecutor
from src.agent.parser import ActionParser
from src.agent.planner import AgentPlanner, ComputerAgentLoop
from src.agent.projects import ProjectRegistry
from src.agent.task_state import ExecutionFlag, TaskState, TaskStatus
from src.agent.tools import ToolEngine
from src.agent.validator import ActionValidator
from src.brain.llm import BaseLLM
from src.config import get_config
from src.utils.logging import log_chitti, log_debug, log_error, log_info, log_warning


class LaptopAgentManager:
    """Coordinates safe laptop agent commands, computer control tools, and multi-step task execution."""

    def __init__(
        self,
        workspace_dir: Optional[str] = None,
        screenshots_dir: Optional[str] = None,
        project_registry_path: Optional[str] = None,
        llm: Optional[BaseLLM] = None,
    ):
        cfg = get_config()
        self.workspace_dir = workspace_dir or cfg.agent.workspace_dir
        self.screenshots_dir = screenshots_dir or cfg.agent.screenshots_dir
        self.project_registry_path = project_registry_path or cfg.agent.project_registry_path
        self.llm = llm

        # Core Computer Controllers
        self.computer = ComputerController(screenshots_dir=self.screenshots_dir)
        self.filesystem = FilesystemController(default_workspace=self.workspace_dir)
        self.terminal = TerminalController(default_cwd=self.workspace_dir)
        self.browser = BrowserController()
        self.apps = AppController()
        self.screen_analyzer = ScreenAnalyzer(self.computer)
        self.projects = ProjectRegistry(registry_file=self.project_registry_path)

        # Tool Engine
        self.tools = ToolEngine(
            computer=self.computer,
            filesystem=self.filesystem,
            terminal=self.terminal,
            browser=self.browser,
            apps=self.apps,
            screen_analyzer=self.screen_analyzer,
            projects=self.projects,
        )

        # Single-step & Multi-step Planners
        self.parser = ActionParser()
        self.validator = ActionValidator()
        self.executor = ActionExecutor()
        self.planner = AgentPlanner(project_registry=self.projects, filesystem=self.filesystem, llm=self.llm)
        self.loop = ComputerAgentLoop(
            tool_engine=self.tools,
            computer=self.computer,
            filesystem=self.filesystem,
            terminal=self.terminal,
            browser=self.browser,
            apps=self.apps,
            screen_analyzer=self.screen_analyzer,
            projects=self.projects,
            llm=self.llm,
        )

        # State tracking
        self.pending_destructive_action: Optional[StructuredAction] = None
        self.active_task_state: Optional[TaskState] = None

    def handle_command(self, user_text: str, lang: str = "en") -> Optional[Tuple[bool, str, Optional[ActionResult]]]:
        """
        Parses and evaluates if the user's message is a single-step or multi-step laptop action command.
        Returns:
            - (True, response_message, action_result) if handled as an action.
            - (False, error_message, None) if validation failed.
            - None if not an agent command (routes to conversation/memory/LLM).
        """
        raw = user_text.strip()
        lower = raw.lower()

        # 1. Handle Pending Confirmation State
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

        # 2. Intent Classification Check
        class_res = TaskClassifier.classify(raw)
        if class_res.intent in (TaskIntent.GENERAL_KNOWLEDGE, TaskIntent.PERSONAL_QUERY):
            # Pure knowledge or personal query - skip computer action
            return None

        # 3. Check for Multi-step Task Plan First
        task_plan = self.planner.plan_task(raw)
        if task_plan and task_plan.steps:
            log_chitti(f"[AGENT] Task detected: COMPUTER_TASK ({len(task_plan.steps)} steps)")
            for s in task_plan.steps:
                log_info(f"  - [{s.action_type}] {s.description}")

            self.active_task_state = task_plan
            success, msg = self.loop.execute_plan(task_plan)
            if task_plan.status == TaskStatus.WAITING_CONFIRMATION:
                self.pending_destructive_action = StructuredAction(
                    action=ActionType.DELETE_FOLDER if "DELETE_DIRECTORY" in (task_plan.current_step.action_type if task_plan.current_step else "") else ActionType.DELETE_FILE,
                    parameters=task_plan.current_step.parameters if task_plan.current_step else {},
                    risk_level=RiskLevel.HIGH,
                    requires_confirmation=True,
                    raw_input=raw,
                )
                return True, msg, None

            # Generate natural multilingual summary response
            resp_formatted = self._format_multistep_response(raw, task_plan, success, msg, lang=lang)
            return success, resp_formatted, ActionResult(action=ActionType.OPEN_APPLICATION, success=success, message=resp_formatted)

        # 4. Check for Single-step Structured Action
        structured_action = self.parser.parse_command(raw)
        if not structured_action:
            return None

        log_chitti(f"[AGENT] Intent detected: {structured_action.action.value}")
        if structured_action.target:
            log_chitti(f"[AGENT] Target: {structured_action.target}")
        log_chitti(f"[AGENT] Risk: {structured_action.risk_level.value}")

        # 5. Validate Single-step Action
        is_valid, reason, validated_action = self.validator.validate(structured_action)
        if not is_valid or not validated_action:
            log_chitti(f"[AGENT] Validation: FAILED | Reason: {reason}")
            log_chitti(f"[AGENT] Execution: BLOCKED")
            return False, reason, None

        log_chitti(f"[AGENT] Validation: PASSED")

        # 6. Check for Confirmation Requirement
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

        # 7. Execute Single-step Action
        result = self.executor.execute(validated_action, default_workspace=self.workspace_dir, screenshots_dir=self.screenshots_dir)
        response_msg = self._format_response(result, lang=lang)
        return True, response_msg, result

    def _format_multistep_response(self, raw_input: str, task_state: TaskState, success: bool, raw_msg: str, lang: str = "en") -> str:
        """Formats natural multilingual responses for multi-step agent plans dynamically."""
        lower = raw_input.lower()

        if not success:
            if lang == "hi":
                return f"माफ़ कीजिए, कार्य पूरा करने में समस्या आई: {raw_msg}"
            elif lang in ("hinglish", "mixed"):
                return f"Sorry, task complete karne me issue aaya: {raw_msg}"
            return f"I encountered an issue while executing the task: {raw_msg}"

        if "youtube" in lower or "song" in lower or "gaana" in lower:
            norm_artist = BrowserController.normalize_artist_query(raw_input)
            if not norm_artist or norm_artist == "Top Songs":
                norm_artist = "Sonu Nigam" if "sonu" in lower else ("Shreya Ghoshal" if "shreya" in lower else "requested")
            if lang == "hi":
                return f"YouTube खोल दिया गया है, {norm_artist} का गाना चुना गया है और प्लेबैक सत्यापित हो चुका है।"
            elif lang in ("hinglish", "mixed"):
                return f"YouTube open karke {norm_artist} ka song select kar diya hai aur playback verify ho gaya hai."
            return f"Done. I opened YouTube, selected a {norm_artist} song, and verified playback started."

        elif "vs code" in lower or "vscode" in lower or any(kw in lower for kw in ["code", "program", "file", "banao", "create", "likho", "implement"]):
            spec = CodeGenerator.parse_programming_task(raw_input)
            topic_title = spec.problem_description.title()
            lang_name = spec.language.upper()
            executed = task_state.get_flag(ExecutionFlag.CODE_EXECUTED) and task_state.get_flag(ExecutionFlag.EXECUTION_VERIFIED)

            if executed:
                if lang == "hi":
                    return f"VS Code खोल दिया गया है, {topic_title} का {lang_name} कोड ({spec.filename}) बनाकर सुरक्षित, सत्यापित और निष्पादित कर दिया गया है।"
                elif lang in ("hinglish", "mixed"):
                    return f"VS Code open kar diya hai, {spec.filename} create karke {topic_title} ka {lang_name} code save, verify aur execute kar diya hai."
                return f"VS Code is open and the {topic_title} {lang_name} code ({spec.filename}) has been created, verified, saved, and executed."
            else:
                if lang == "hi":
                    return f"VS Code खोल दिया गया है और {spec.filename} में {topic_title} का {lang_name} कोड लिखकर सुरक्षित और सत्यापित कर दिया गया है।"
                elif lang in ("hinglish", "mixed"):
                    return f"VS Code open hai aur {spec.filename} mein {topic_title} {lang_name} code likhkar save aur verify kar diya hai."
                return f"VS Code is open, and {spec.filename} has been created with the {topic_title} {lang_name} solution, verified in the editor, and saved."

        elif "notepad" in lower:
            if lang == "hi":
                return "Notepad खोल दिया गया है और टेक्स्ट लिख दिया गया है।"
            elif lang in ("hinglish", "mixed"):
                return "Notepad open kar diya hai aur text type ho gaya hai."
            return "Notepad opened and text was entered."

        elif "desktop" in lower and "folder" in lower:
            if lang == "hi":
                return "Desktop पर फ़ोल्डर बना दिया गया है।"
            elif lang in ("hinglish", "mixed"):
                return "Desktop par folder create kar diya gaya hai."
            return "Folder has been successfully created on your Desktop."

        return "Task completed successfully."

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
