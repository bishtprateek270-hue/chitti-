"""
Chitti Closed-Loop Computer-Use Action Planner & Execution Engine.
Translates high-level natural language requests into structured multi-step plans,
executes atomic device primitives, observes resulting screens, and verifies completion.
"""

import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from src.computer_use.actions import (
    ComputerActionResult,
    ComputerActionStep,
    ComputerActionType,
)
from src.computer_use.controller import ComputerController
from src.computer_use.recovery import ComputerActionRecovery
from src.computer_use.safety import ComputerSafetyPolicy
from src.computer_use.screen import ScreenObserver
from src.computer_use.state import (
    ComputerTaskState,
    RiskLevel,
    VerificationStatus,
)
from src.computer_use.ui_detector import UIElementDetector
from src.computer_use.verifier import ComputerActionVerifier
from src.utils.logging import log_chitti, log_info, log_warn
from src.utils.text import clean_contact_query, strip_emojis


class ComputerUseEngine:
    """
    Central Closed-Loop Computer-Use Agent Engine.
    Executes full device control tasks across applications, browsers, and the Windows OS.
    """

    def __init__(self, screenshots_dir: str = "data/screenshots"):
        self.controller = ComputerController(screenshots_dir=screenshots_dir)
        self.screen_observer = ScreenObserver(controller=self.controller)
        self.ui_detector = UIElementDetector(controller=self.controller)
        self.verifier = ComputerActionVerifier(controller=self.controller, screen_observer=self.screen_observer)
        self.recovery = ComputerActionRecovery(
            controller=self.controller,
            screen_observer=self.screen_observer,
            verifier=self.verifier,
        )

    def plan_task(self, user_request: str) -> ComputerTaskState:
        """Translates user natural language into a structured ComputerTaskState plan."""
        raw = user_request.strip()
        lower = raw.lower()

        state = ComputerTaskState(user_request=raw)
        steps: List[ComputerActionStep] = []

        # 1. WHATSAPP WEB MESSAGING
        if "whatsapp" in lower and any(w in lower for w in ["send", "message", "msg", "bhejo"]):
            # Extract contact & content (supporting emojis and clean search query)
            m_contact = (
                re.search(r"(?i)\bto\s+([^\n\r,;:.]+?)(?:\s+saying|\s+that|\s+message|\s+msg|\s*:|\s+on\s+whatsapp|$)", raw) or
                re.search(r"(?i)\b([^\n\r,;:.]+?)\s+ko\b", raw) or
                re.search(r"(?i)\b(?:send|message|msg)\s+([^\n\r,;:'\"]+?)\s*(?:saying|that|message|msg|:|\s+['\"])", raw)
            )
            contact_raw = m_contact.group(1).strip() if m_contact else "Contact"
            contact = re.sub(r"(?i)\s+(?:on|via|using)\s+.*$", "", contact_raw).strip() or "Contact"
            search_query = clean_contact_query(contact)

            # Extract message body
            m_msg = (
                re.search(r"['\"](.*?)['\"]", raw) or
                re.search(r"(?i)\b(?:send|message|msg)\s+([^\n\r,;:.]+?)\s+to\b", raw) or
                re.search(r"(?i)(?:saying|that|message|msg)\s+(.+)$", raw)
            )
            content = m_msg.group(1).strip() if m_msg else "Hello"

            state.current_goal = f"Send WhatsApp message '{content}' to {contact}"

            steps = [
                ComputerActionStep(
                    step_id=1,
                    description="Open WhatsApp Web in browser",
                    action_type=ComputerActionType.OPEN_URL,
                    parameters={"url": "https://web.whatsapp.com", "service": "WhatsApp Web"},
                ),
                ComputerActionStep(
                    step_id=2,
                    description="Verify page load and active session",
                    action_type=ComputerActionType.CHECK_AUTHENTICATION,
                    parameters={"service": "WhatsApp Web"},
                    depends_on=[1],
                ),
                ComputerActionStep(
                    step_id=3,
                    description=f"Search for contact '{contact}' (query: '{search_query}')",
                    action_type=ComputerActionType.SEARCH_CONTACT,
                    parameters={"contact": contact, "search_contact": search_query, "service": "WhatsApp Web"},
                    depends_on=[2],
                ),
                ComputerActionStep(
                    step_id=4,
                    description=f"Select conversation with '{contact}'",
                    action_type=ComputerActionType.SELECT_CONVERSATION,
                    parameters={"contact": contact, "search_contact": search_query, "service": "WhatsApp Web"},
                    depends_on=[3],
                ),
                ComputerActionStep(
                    step_id=5,
                    description=f"Type message '{content}'",
                    action_type=ComputerActionType.TYPE_TEXT,
                    parameters={"text": content},
                    depends_on=[4],
                ),
                ComputerActionStep(
                    step_id=6,
                    description=f"Send message to '{contact}'",
                    action_type=ComputerActionType.SEND_MESSAGE,
                    parameters={"contact": contact, "text": content, "service": "WhatsApp Web"},
                    depends_on=[5],
                ),
                ComputerActionStep(
                    step_id=7,
                    description=f"Verify message '{content}' appears in conversation",
                    action_type=ComputerActionType.VERIFY_MESSAGE_SENT,
                    parameters={"contact": contact, "text": content},
                    depends_on=[6],
                ),
            ]

        # 2. GMAIL / EMAIL SENDING
        elif any(w in lower for w in ["email", "mail", "gmail", "outlook"]) and any(w in lower for w in ["send", "mail", "email", "shoot", "dispatch"]):
            m_email = re.search(r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})", raw)
            recipient = m_email.group(1) if m_email else "recipient@example.com"

            m_sub = re.search(r"(?i)\bsubject\s+([^,]+?)(?:\s+and\s+body|\s+body|\s+saying|$)", raw)
            subject = m_sub.group(1).strip() if m_sub else "Message from Chitti"

            m_body = re.search(r"(?i)(?:body|saying)\s+(.+)$", raw) or re.search(r"['\"](.*?)['\"]", raw)
            body = m_body.group(1).strip() if m_body else "Hello"

            state.current_goal = f"Send email to {recipient} with subject '{subject}'"

            steps = [
                ComputerActionStep(
                    step_id=1,
                    description="Open Gmail in browser",
                    action_type=ComputerActionType.OPEN_URL,
                    parameters={"url": "https://mail.google.com", "service": "Gmail / Webmail"},
                ),
                ComputerActionStep(
                    step_id=2,
                    description="Verify page load and account session",
                    action_type=ComputerActionType.CHECK_AUTHENTICATION,
                    parameters={"service": "Gmail / Webmail"},
                    depends_on=[1],
                ),
                ComputerActionStep(
                    step_id=3,
                    description=f"Compose email to '{recipient}'",
                    action_type=ComputerActionType.COMPOSE_EMAIL,
                    parameters={"recipient": recipient, "subject": subject, "content": body},
                    depends_on=[2],
                ),
                ComputerActionStep(
                    step_id=4,
                    description=f"Confirm sending email to '{recipient}'",
                    action_type=ComputerActionType.CONFIRM_SEND,
                    parameters={"recipient": recipient, "subject": subject},
                    risk_level=RiskLevel.MEDIUM_RISK,
                    requires_confirmation=True,
                    depends_on=[3],
                ),
                ComputerActionStep(
                    step_id=5,
                    description=f"Dispatch email to '{recipient}'",
                    action_type=ComputerActionType.SEND_EMAIL,
                    parameters={"recipient": recipient, "subject": subject, "content": body},
                    depends_on=[4],
                ),
                ComputerActionStep(
                    step_id=6,
                    description=f"Verify email dispatched to '{recipient}'",
                    action_type=ComputerActionType.VERIFY_EMAIL_SENT,
                    parameters={"recipient": recipient},
                    depends_on=[5],
                ),
            ]

        # 3. NOTEPAD / TEXT EDITOR
        elif "notepad" in lower and any(w in lower for w in ["write", "type", "enter"]):
            m_text = re.search(r"['\"](.*?)['\"]", raw) or re.search(r"(?i)(?:write|type)\s+(.+)$", raw)
            text_to_write = m_text.group(1).strip() if m_text else "Hello Chitti"

            state.current_goal = f"Open Notepad and type '{text_to_write}'"

            steps = [
                ComputerActionStep(
                    step_id=1,
                    description="Launch Notepad application",
                    action_type=ComputerActionType.OPEN_APPLICATION,
                    parameters={"application": "notepad"},
                ),
                ComputerActionStep(
                    step_id=2,
                    description="Verify Notepad window is active",
                    action_type=ComputerActionType.FOCUS_WINDOW,
                    parameters={"title": "Notepad"},
                    depends_on=[1],
                ),
                ComputerActionStep(
                    step_id=3,
                    description=f"Type text '{text_to_write}'",
                    action_type=ComputerActionType.TYPE_TEXT,
                    parameters={"text": text_to_write},
                    depends_on=[2],
                ),
                ComputerActionStep(
                    step_id=4,
                    description="Verify text entered in Notepad",
                    action_type=ComputerActionType.VERIFY_UI_STATE,
                    parameters={"expected_text": text_to_write},
                    depends_on=[3],
                ),
            ]

        # 4. GENERAL APPLICATION LAUNCH
        elif any(w in lower for w in ["open", "launch", "start", "kholo", "chalao"]) and not any(w in lower for w in ["http", ".com", "youtube", "whatsapp", "gmail"]):
            app_match = re.search(r"(?i)\b(?:open|launch|start|kholo|chalao)\s+(?:the\s+)?([a-zA-Z0-9_\-\s]+?)(?:\s+app|\s+application)?$", raw)
            app_name = app_match.group(1).strip() if app_match else "application"

            state.current_goal = f"Open application '{app_name}'"

            steps = [
                ComputerActionStep(
                    step_id=1,
                    description=f"Launch application '{app_name}'",
                    action_type=ComputerActionType.OPEN_APPLICATION,
                    parameters={"application": app_name},
                ),
                ComputerActionStep(
                    step_id=2,
                    description=f"Verify '{app_name}' window is open and active",
                    action_type=ComputerActionType.VERIFY_APPLICATION_STATE,
                    parameters={"application": app_name},
                    depends_on=[1],
                ),
            ]

        # 5. FILESYSTEM CREATION (Direct API + Verification)
        elif any(w in lower for w in ["create", "make", "banao"]) and any(w in lower for w in ["folder", "directory"]):
            m_folder = re.search(r"(?i)\b(?:folder|directory)\s+(?:named|called)?\s*([a-zA-Z0-9_\-]+)", raw) or re.search(r"(?i)\b([a-zA-Z0-9_\-]+)\s+folder", raw)
            folder_name = m_folder.group(1).strip() if m_folder else "NewFolder"
            folder_path = os.path.expanduser(f"~/Desktop/{folder_name}") if "desktop" in lower else f"data/workspace/{folder_name}"

            state.current_goal = f"Create folder '{folder_name}'"

            steps = [
                ComputerActionStep(
                    step_id=1,
                    description=f"Create directory '{folder_name}' on filesystem",
                    action_type=ComputerActionType.CREATE_DIRECTORY,
                    parameters={"path": folder_path},
                ),
                ComputerActionStep(
                    step_id=2,
                    description=f"Verify directory '{folder_name}' exists on disk",
                    action_type=ComputerActionType.CREATE_DIRECTORY,
                    parameters={"path": folder_path},
                    depends_on=[1],
                ),
            ]

        # 6. BROWSER SEARCH
        elif any(w in lower for w in ["search", "google", "find"]) and any(w in lower for w in ["chrome", "edge", "browser", "web", "online"]):
            m_q = re.search(r"(?i)\b(?:for|search)\s+(.+)$", raw)
            query = m_q.group(1).strip() if m_q else "Python documentation"

            state.current_goal = f"Search web for '{query}'"

            steps = [
                ComputerActionStep(
                    step_id=1,
                    description=f"Open search for '{query}' in browser",
                    action_type=ComputerActionType.SEARCH_WEB,
                    parameters={"query": query, "url": f"https://www.google.com/search?q={query}"},
                ),
                ComputerActionStep(
                    step_id=2,
                    description="Verify search results page loaded",
                    action_type=ComputerActionType.VERIFY_PAGE_LOADED,
                    parameters={"url": "google.com"},
                    depends_on=[1],
                ),
            ]

        # Default Generic Plan
        else:
            state.current_goal = raw
            steps = [
                ComputerActionStep(
                    step_id=1,
                    description=f"Inspect system state for '{raw}'",
                    action_type=ComputerActionType.INSPECT_SCREEN,
                    parameters={},
                )
            ]

        state.planned_steps = [{"step_id": s.step_id, "action": s.action_type.value, "desc": s.description} for s in steps]
        state._internal_steps = steps
        return state

    def execute_plan(self, state: ComputerTaskState) -> Tuple[bool, str]:
        """
        Executes the closed-loop computer-use plan:
        Action Execution -> Screen Observation -> Verification -> Recovery.
        """
        log_chitti(f"[COMPUTER AGENT] Starting execution for: '{state.user_request}' ({len(state._internal_steps)} steps)")
        
        for step in state._internal_steps:
            act_name = step.action_type.value if hasattr(step.action_type, "value") else str(step.action_type)
            log_info(f"[COMPUTER AGENT] [STEP {step.step_id}/{len(state._internal_steps)}] {step.description}")

            step_success = False
            while not step_success:
                # 1. Execute Atomic Action
                exec_res = self._execute_step_action(step)
                
                # 2. Screen Observation
                obs = self.screen_observer.observe(capture_screenshot=True)

                # 3. Ground-Truth Verification
                ver_res = self.verifier.verify_action(
                    action_type=step.action_type,
                    target=step.parameters.get("target") or step.parameters.get("application") or step.parameters.get("url") or "",
                    parameters=step.parameters,
                    action_result=exec_res,
                )

                # 4. Handle Verification Outcome
                if ver_res.status == VerificationStatus.SUCCESS:
                    state.record_action(
                        step_id=step.step_id,
                        action=act_name,
                        target=str(step.parameters),
                        status="SUCCESS",
                        message=ver_res.evidence,
                        observation=obs.summary_text,
                    )
                    state.completed_steps.append({"step_id": step.step_id, "action": act_name, "status": "SUCCESS"})
                    log_info(f"[COMPUTER AGENT] Step {step.step_id} SUCCESS: {ver_res.evidence}")
                    step_success = True
                else:
                    # 5. Recovery Loop
                    recovered, rec_msg = self.recovery.attempt_recovery(step, state, exec_res)
                    if not recovered:
                        state.record_action(
                            step_id=step.step_id,
                            action=act_name,
                            target=str(step.parameters),
                            status="FAILED",
                            message=rec_msg,
                            observation=obs.summary_text,
                        )
                        state.mark_failed(rec_msg)
                        log_warn(f"[COMPUTER AGENT] Step {step.step_id} FAILED: {rec_msg}")
                        return False, rec_msg

        state.mark_completed("All planned computer actions executed and verified successfully.")
        log_info("[COMPUTER AGENT] FINAL VERIFICATION: PASSED")
        return True, "Task completed successfully."

    def _execute_step_action(self, step: ComputerActionStep) -> ComputerActionResult:
        """Executes the low-level device primitive for an atomic step."""
        act = step.action_type
        params = step.parameters
        act_name = act.value if hasattr(act, "value") else str(act)

        try:
            if act_name == "OPEN_URL":
                url = params.get("url", "https://google.com")
                import webbrowser
                webbrowser.open(url)
                self.controller.wait(0.3)
                return ComputerActionResult(action_type=act, success=True, message=f"Opened {url}")

            elif act_name == "SEARCH_WEB":
                query = params.get("query", "")
                url = params.get("url", f"https://www.google.com/search?q={query}")
                import webbrowser
                webbrowser.open(url)
                self.controller.wait(0.3)
                return ComputerActionResult(action_type=act, success=True, message=f"Searched web for '{query}'")

            elif act_name == "OPEN_APPLICATION":
                app = params.get("application", "")
                import subprocess
                try:
                    subprocess.Popen(app, shell=True)
                    self.controller.wait(0.5)
                    return ComputerActionResult(action_type=act, success=True, message=f"Launched {app}")
                except Exception as e:
                    return ComputerActionResult(action_type=act, success=False, error=str(e))

            elif act_name == "FOCUS_WINDOW":
                title = params.get("title", "")
                ok = self.controller.focus_window(title)
                return ComputerActionResult(action_type=act, success=ok, message=f"Focused {title}")

            elif act_name == "TYPE_TEXT":
                txt = params.get("text", "")
                ok = self.controller.type_text(txt)
                return ComputerActionResult(action_type=act, success=ok, message=f"Typed text: {txt}")

            elif act_name == "PRESS_KEY":
                k = params.get("key", "enter")
                ok = self.controller.press_key(k)
                return ComputerActionResult(action_type=act, success=ok, message=f"Pressed {k}")

            elif act_name == "HOTKEY":
                keys = params.get("keys", ["ctrl", "enter"])
                ok = self.controller.hotkey(*keys)
                return ComputerActionResult(action_type=act, success=ok, message=f"Executed hotkey: {keys}")

            elif act_name == "SEARCH_CONTACT":
                contact = params.get("contact", "")
                search_query = params.get("search_contact") or clean_contact_query(contact)
                service = params.get("service", "WhatsApp Web")
                self.controller.focus_window(service)
                self.controller.press_key("esc")
                self.controller.wait(0.2)
                self.controller.hotkey("ctrl", "alt", "/")
                self.controller.wait(0.2)
                
                win = self.controller.find_window(service) if hasattr(self.controller, "find_window") else None
                if win and win.width > 0:
                    search_x = win.left + min(240, int(win.width * 0.18))
                    search_y = win.top + min(180, int(win.height * 0.18))
                else:
                    screen_w, screen_h = self.controller.get_screen_size()
                    search_x = min(240, int(screen_w * 0.15))
                    search_y = min(180, int(screen_h * 0.18))

                self.controller.click(search_x, search_y)
                self.controller.wait(0.2)
                self.controller.hotkey("ctrl", "a")
                self.controller.wait(0.1)
                self.controller.press_key("backspace")
                self.controller.wait(0.1)
                self.controller.type_text(search_query)
                self.controller.wait(0.4)
                self.controller.press_key("enter")
                return ComputerActionResult(action_type=act, success=True, message=f"Searched contact '{search_query}' (for {contact})")

            elif act_name == "SELECT_CONVERSATION":
                contact = params.get("contact", "")
                service = params.get("service", "WhatsApp Web")
                self.controller.focus_window(service)
                self.controller.wait(0.2)
                
                win = self.controller.find_window(service) if hasattr(self.controller, "find_window") else None
                if win and win.width > 0:
                    chat_item_x = win.left + min(240, int(win.width * 0.18))
                    chat_item_y = win.top + min(270, int(win.height * 0.28))
                else:
                    screen_w, screen_h = self.controller.get_screen_size()
                    chat_item_x = min(240, int(screen_w * 0.15))
                    chat_item_y = min(270, int(screen_h * 0.28))

                self.controller.click(chat_item_x, chat_item_y)
                self.controller.wait(0.3)
                self.controller.press_key("enter")
                self.controller.wait(0.3)
                
                if win and win.width > 0:
                    msg_x = win.left + int(win.width * 0.55)
                    msg_y = win.top + (win.height - 50)
                    self.controller.click(msg_x, msg_y)
                    self.controller.wait(0.1)

                return ComputerActionResult(action_type=act, success=True, message=f"Selected chat {contact}")

            elif act_name == "SEND_MESSAGE":
                self.controller.press_key("enter")
                return ComputerActionResult(action_type=act, success=True, message="Sent message")

            elif act_name == "COMPOSE_EMAIL":
                rec = params.get("recipient", "")
                sub = params.get("subject", "")
                body = params.get("content", "")
                self.controller.focus_window("Gmail")
                self.controller.press_key("c")
                self.controller.wait(0.1)
                self.controller.type_text(rec)
                self.controller.press_key("enter")
                self.controller.press_key("tab")
                self.controller.type_text(sub)
                self.controller.press_key("tab")
                self.controller.type_text(body)
                return ComputerActionResult(action_type=act, success=True, message=f"Composed email to {rec}")

            elif act_name == "SEND_EMAIL":
                self.controller.focus_window("Gmail")
                self.controller.hotkey("ctrl", "enter")
                return ComputerActionResult(action_type=act, success=True, message="Dispatched email")

            elif act_name in ("CREATE_DIRECTORY", "CREATE_FOLDER"):
                p = params.get("path", "")
                os.makedirs(p, exist_ok=True)
                return ComputerActionResult(action_type=act, success=True, message=f"Created folder {p}")

            elif act_name in ("CREATE_FILE", "WRITE_FILE"):
                p = params.get("path", "")
                content = params.get("content", "")
                with open(p, "w", encoding="utf-8") as f:
                    f.write(content)
                return ComputerActionResult(action_type=act, success=True, message=f"Wrote file {p}")

            return ComputerActionResult(action_type=act, success=True, message="Step executed.")

        except Exception as e:
            return ComputerActionResult(action_type=act, success=False, error=str(e))
