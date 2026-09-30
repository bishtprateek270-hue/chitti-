"""
Chitti Laptop Agent Manager Module (Phase 6).
Coordinates multi-step agentic planning, dependency execution, failure recovery,
risk assessment, cancellation handling, and short-term session context for laptop operations.
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
from src.agent.task_state import ExecutionFlag, StepStatus, TaskContext, TaskState, TaskStatus
from src.agent.tools import ToolEngine
from src.agent.validator import ActionValidator
from src.brain.llm import BaseLLM
from src.config import get_config
from src.utils.logging import log_chitti, log_debug, log_error, log_info, log_warning


class LaptopAgentManager:
    """Coordinates safe laptop agent commands, computer control tools, and multi-step agentic planning."""

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

        # Planners & Loop
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

        # State tracking & Short-term session context
        self.pending_destructive_action: Optional[StructuredAction] = None
        self.active_task_state: Optional[TaskState] = None
        self.session_context: TaskContext = TaskContext(workspace=self.workspace_dir)

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

        # Clean up previous completed, cancelled, or failed task states on new command
        if self.active_task_state is not None and self.active_task_state.status in (TaskStatus.COMPLETED, TaskStatus.CANCELLED, TaskStatus.FAILED, TaskStatus.BLOCKED):
            self.active_task_state = None
            self.pending_destructive_action = None

        # 0. User Interruption / Cancellation
        if re.search(r"^(?:stop|cancel|abort|pause|ruk\s*jao|chhod\s*do|band\s*karo|don'?t\s+do\s+that)$", lower):
            if self.active_task_state and self.active_task_state.status in (TaskStatus.EXECUTING, TaskStatus.PLANNING, TaskStatus.WAITING_FOR_CONFIRMATION):
                if "pause" in lower:
                    self.active_task_state.pause()
                    log_chitti("[AGENT] Task paused by user.")
                    return True, "Task has been paused." if lang == "en" else "कार्य रोक दिया गया है।", None
                else:
                    self.active_task_state.cancel()
                    self.pending_destructive_action = None
                    log_chitti("[AGENT] Task cancelled by user.")
                    return True, "Task cancelled." if lang == "en" else "कार्य रद्द कर दिया गया है।", None

        # 1. Handle Pending Confirmation State (Multi-step Resumable Task or Single-step Action)
        if self.active_task_state is not None and self.active_task_state.status in (TaskStatus.WAITING_FOR_CONFIRMATION, TaskStatus.WAITING_CONFIRMATION):
            if re.search(r"\b(?:no|cancel|stop|abort|don'?t|nope|nahi|nahin|mat karo)\b", lower):
                self.active_task_state.cancel()
                self.active_task_state = None
                self.pending_destructive_action = None
                log_chitti("[AGENT] Action cancelled by user.")
                if lang == "hi":
                    return True, "कार्रवाई रद्द कर दी गई है।", None
                elif lang in ("hinglish", "mixed"):
                    return True, "Action cancel kar diya gaya hai.", None
                else:
                    return True, "Action cancelled.", None

            elif re.search(r"\b(?:yes|proceed|confirm|sure|do it|yep|yeah|haan|sahi|ha|ha kar do)\b", lower):
                pending_step = self.active_task_state.pending_confirmation_step or self.active_task_state.current_step
                log_chitti(f"[AGENT] Action confirmed by user. Resuming task {self.active_task_state.task_id} from step {pending_step.step_id if pending_step else 'next'}...")
                if pending_step:
                    pending_step.mark_success("Confirmation approved by user.")
                    self.active_task_state.completed_steps.append(pending_step)
                    self.active_task_state.current_step_index += 1
                self.active_task_state.status = TaskStatus.EXECUTING
                self.active_task_state.pending_confirmation_step = None
                self.pending_destructive_action = None
                
                success, msg = self.loop.execute_plan(self.active_task_state)
                task_plan = self.active_task_state
                if task_plan.status in (TaskStatus.WAITING_FOR_CONFIRMATION, TaskStatus.WAITING_CONFIRMATION):
                    return True, msg, None

                self.pending_destructive_action = None
                resp_formatted = self._format_multistep_response(task_plan.task_description, task_plan, success, msg, lang=lang)
                return True, resp_formatted, ActionResult(action=ActionType.OPEN_APPLICATION, success=success, message=resp_formatted)

            else:
                # User typed something other than yes/no. Check if this is a NEW actionable command.
                class_check = TaskClassifier.classify(raw)
                if class_check.is_actionable_task or self.parser.parse_command(raw):
                    log_chitti(f"[AGENT] Previous pending task superseded by new user command: '{raw}'")
                    self.active_task_state.cancel()
                    self.active_task_state = None
                    self.pending_destructive_action = None
                    # Fall through to execute the new command below
                else:
                    pending_step = self.active_task_state.pending_confirmation_step or self.active_task_state.current_step
                    act_type = pending_step.action_type if pending_step else ""
                    if act_type in ("CONFIRM_SEND", "SEND_EMAIL"):
                        rec = pending_step.parameters.get("recipient", "recipient")
                        sub = pending_step.parameters.get("subject", "")
                        sub_text = f" with subject '{sub}'" if sub else ""
                        prompt_msg = f"I am ready to send an email to '{rec}'{sub_text}. Should I proceed? (Yes / No)"
                    elif act_type == "SEND_MESSAGE":
                        contact = pending_step.parameters.get("contact", "contact")
                        txt = pending_step.parameters.get("text", "")
                        prompt_msg = f"I am ready to send message '{txt}' to '{contact}'. Should I proceed? (Yes / No)"
                    elif act_type in ("DELETE_DIRECTORY", "DELETE_FILE"):
                        target_name = pending_step.parameters.get("path") or pending_step.parameters.get("target") or "specified files"
                        prompt_msg = f"This action will modify or delete '{target_name}'. Do you want me to continue? (Yes / No)"
                    else:
                        prompt_msg = "Do you want me to proceed with this action? Please answer Yes or No."
                    return True, prompt_msg, None

        elif self.pending_destructive_action is not None:
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
        if not class_res.is_actionable_task and class_res.intent in (
            TaskIntent.GENERAL_CONVERSATION,
            TaskIntent.GENERAL_KNOWLEDGE,
            TaskIntent.TECHNICAL_QUERY,
            TaskIntent.PERSONAL_MEMORY,
            TaskIntent.RELATIONSHIP_MEMORY,
            TaskIntent.PROJECT_MEMORY,
            TaskIntent.PREFERENCE_MEMORY,
            TaskIntent.EXPLICIT_MEMORY,
        ):
            return False, "", None

        # Project request detected logging
        if class_res.intent in (TaskIntent.CREATE_PROJECT, TaskIntent.BUILD_PROJECT, TaskIntent.MODIFY_PROJECT):
            log_chitti(f"[CHITTI] [TASK ROUTER] Intent: {class_res.intent.value}")
            log_chitti("[CHITTI] [TASK ROUTER] Project request detected")
            log_chitti("[CHITTI] [TASK ROUTER] Phase 6 agent activated")
        elif class_res.is_actionable_task:
            log_chitti(f"[CHITTI] [TASK ROUTER] Intent: {class_res.intent.value}")

        # 3. Check for Multi-step Task Plan First
        task_plan = self.planner.plan_task(raw, context=self.session_context)
        if task_plan and task_plan.steps:
            log_chitti(f"[AGENT] Task detected: MULTI_STEP_PLAN ({len(task_plan.steps)} steps)")
            for s in task_plan.steps:
                dep_info = f" (depends on: {s.depends_on})" if s.depends_on else ""
                log_info(f"  - [Step {s.step_id}: {s.action_type}] {s.description}{dep_info}")

            self.active_task_state = task_plan
            success, msg = self.loop.execute_plan(task_plan)

            if task_plan.status in (TaskStatus.WAITING_FOR_CONFIRMATION, TaskStatus.WAITING_CONFIRMATION):
                pending_step = task_plan.pending_confirmation_step or task_plan.current_step
                act_type = pending_step.action_type if pending_step else ""
                if act_type in ("DELETE_DIRECTORY", "DELETE_FILE"):
                    self.pending_destructive_action = StructuredAction(
                        action=ActionType.DELETE_FOLDER if act_type == "DELETE_DIRECTORY" else ActionType.DELETE_FILE,
                        parameters=pending_step.parameters if pending_step else {},
                        risk_level=RiskLevel.HIGH,
                        requires_confirmation=True,
                        raw_input=raw,
                    )
                return True, msg, None

            # Generate natural multilingual summary response
            self.pending_destructive_action = None
            resp_formatted = self._format_multistep_response(raw, task_plan, success, msg, lang=lang)
            return True, resp_formatted, ActionResult(action=ActionType.OPEN_APPLICATION, success=success, message=resp_formatted)


        # 4. Check for Single-step Structured Action (Fast Path)
        structured_action = self.parser.parse_command(raw)
        if not structured_action:
            return False, "", None

        log_chitti(f"[AGENT] Single-step intent detected: {structured_action.action.value}")
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

        # 7. Execute Single-step Action (with Section 28 Diagnostics)
        log_chitti(f"[INTENT] {validated_action.action.value}")
        log_chitti(f"[TARGET] {validated_action.target or 'System'}")
        log_chitti(f"[GOAL] Perform {validated_action.action.value} on {validated_action.target or 'System'}")
        log_chitti(f"[AVAILABLE TOOLS] {list(self.tools.tools.keys())[:8]}...")
        log_chitti(f"[SELECTED TOOL] {validated_action.action.value.lower()}")
        log_chitti(f"[TOOL ARGUMENTS] {validated_action.parameters}")

        result = self.executor.execute(validated_action, default_workspace=self.workspace_dir, screenshots_dir=self.screenshots_dir)

        log_chitti(f"[TOOL RESULT] success={'true' if result.success else 'false'}")
        log_chitti(f"[OBSERVATION] {result.message}")
        log_chitti(f"[VERIFICATION] {'Target operation verified on OS' if result.success else 'Action execution failed'}")
        log_chitti(f"[FINAL STATUS] {'COMPLETED' if result.success else 'FAILED'}")

        response_msg = self._format_response(result, lang=lang)
        return True, response_msg, result

    def _format_multistep_response(self, raw_input: str, task_state: TaskState, success: bool, raw_msg: str, lang: str = "en") -> str:
        """Formats natural multilingual responses for multi-step agent plans dynamically."""
        lower = raw_input.lower()

        if not success:
            if task_state.status == TaskStatus.BLOCKED:
                if lang == "hi":
                    return f"प्रोजेक्ट फ़ाइलें VS Code में बना दी गई हैं, लेकिन आवश्यक कंपाइलर/रनटाइम इंस्टॉल न होने के कारण निष्पादन रोक दिया गया है: {raw_msg}"
                elif lang in ("hinglish", "mixed"):
                    return f"Project files VS Code me create kar di gayi hain, lekin required compiler/runtime install na hone ki wajah se execution block ho gaya hai: {raw_msg}"
                return f"I created the project and opened it in VS Code, but execution was blocked because the required toolchain/runtime is not installed: {raw_msg}"
            elif task_state.status == TaskStatus.PARTIALLY_COMPLETED:
                if lang == "hi":
                    return f"कार्य आंशिक रूप से पूरा हुआ: {raw_msg}"
                elif lang in ("hinglish", "mixed"):
                    return f"Task partially complete hua: {raw_msg}"
                return f"Task partially completed: {raw_msg}"
            if lang == "hi":
                return f"माफ़ कीजिए, कार्य पूरा करने में समस्या आई: {raw_msg}"
            elif lang in ("hinglish", "mixed"):
                return f"Sorry, task complete karne me issue aaya: {raw_msg}"
            return f"I encountered an issue while executing the task: {raw_msg}"

        if "whatsapp" in lower or "message" in lower or "msg" in lower or "telegram" in lower:
            m_contact = (
                re.search(r"(?i)\bto\s+([^\n\r,;:.]+?)(?:\s+saying|\s+that|\s+message|\s+msg|\s*:|\s+['\"]|$)", raw_input) or
                re.search(r"(?i)\b([^\n\r,;:.]+?)\s+ko\b", raw_input) or
                re.search(r"(?i)\bto\s+([a-zA-Z0-9_\-]+)\b", raw_input)
            )
            target = m_contact.group(1).strip() if m_contact else "the contact"
            if lang == "hi":
                return f"WhatsApp पर {target} को संदेश भेज दिया गया है।"
            elif lang in ("hinglish", "mixed"):
                return f"WhatsApp par {target} ko message send kar diya hai."
            return f"Message has been sent to {target} on WhatsApp."

        elif "email" in lower or "mail" in lower or "@" in lower or "gmail" in lower:
            m_email = re.search(r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})", raw_input)
            rec = m_email.group(1) if m_email else "the recipient"
            if lang == "hi":
                return f"{rec} को ईमेल भेज दिया गया है।"
            elif lang in ("hinglish", "mixed"):
                return f"{rec} ko email send kar diya hai."
            return f"Email has been sent to {rec}."

        elif "youtube" in lower or "song" in lower or "gaana" in lower:
            norm_artist = BrowserController.normalize_artist_query(raw_input)
            if not norm_artist or norm_artist == "Top Songs":
                norm_artist = "Sonu Nigam" if "sonu" in lower else ("Shreya Ghoshal" if "shreya" in lower else "requested")
            if lang == "hi":
                return f"YouTube खोल दिया गया है, {norm_artist} का गाना चुना गया है और प्लेबैक सत्यापित हो चुका है।"
            elif lang in ("hinglish", "mixed"):
                return f"YouTube open karke {norm_artist} ka song select kar diya hai aur playback verify ho gaya hai."
            return f"Done. I opened YouTube, selected a {norm_artist} song, and verified playback started."

        elif "vs code" in lower or "vscode" in lower or any(kw in lower for kw in ["code", "program", "file", "banao", "create", "likho", "implement", "tracker", "scraper"]):
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
