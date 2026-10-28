"""
Chitti Generic Natural Intent & Context Understanding Engine.
Infers user intent, entities, targets, context dependencies, and appropriate execution paths
from natural English, Hindi, and Hinglish speech without rigid regex or command phrases.
"""

import os
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.utils.logging import log_chitti, log_debug, log_info
from src.utils.metrics import ExecutionPath
from src.utils.text import normalize_typos, polish_message_text, clean_contact_query


class IntentType(str, Enum):
    # PATH A: Simple Deterministic Commands
    SIMPLE_COMMAND = "SIMPLE_COMMAND"          # Stop, mute, volume, screenshot, calc
    STANDBY_SLEEP = "STANDBY_SLEEP"            # Sleep, quiet, standby
    WAKE = "WAKE"                              # Wake up

    # PATH B: Normal Conversation & Assistance
    CONVERSATION = "CONVERSATION"              # Small talk, opinion, question, general statement
    TECHNICAL_QUERY = "TECHNICAL_QUERY"        # How-to, explain concept, theoretical answer
    CLARIFICATION_NEEDED = "CLARIFICATION"     # Ambiguous request needing one question

    # PATH C: Computer & OS Tasks
    OPEN_PROJECT = "OPEN_PROJECT"              # Locate and open an existing project
    OPEN_APPLICATION = "OPEN_APPLICATION"      # Open app or website
    CLOSE_APPLICATION = "CLOSE_APPLICATION"    # Close app
    SEND_MESSAGE = "SEND_MESSAGE"              # Message via WhatsApp/Slack/Email
    FILE_OPERATION = "FILE_OPERATION"          # Create/find folder or file

    # PATH D: Coding & Diagnostics Tasks
    DEBUG_CODE = "DEBUG_CODE"                  # Diagnose why code isn't working
    CREATE_PROJECT = "CREATE_PROJECT"          # Build website, app, script
    RUN_PROJECT = "RUN_PROJECT"                # Execute current project/tests

    # PATH E: Complex Multi-step Tasks
    COMPLEX_TASK = "COMPLEX_TASK"              # Multi-step goal requiring planner loop


@dataclass
class SemanticIntent:
    """Structured internal representation of the user's inferred intent."""
    intent: IntentType
    goal: str
    target: Optional[str] = None
    recipient: Optional[str] = None
    channel: Optional[str] = None
    content: Optional[str] = None
    context_required: bool = False
    missing_information: List[str] = field(default_factory=list)
    clarification_prompt: Optional[str] = None
    risk: str = "low"  # low, medium, high
    execution_path: ExecutionPath = ExecutionPath.PATH_B_CONVERSATION
    parameters: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.90

    def log(self):
        """Emits structured observability log."""
        log_chitti(f"[INTENT ENGINE] Inferred Intent: {self.intent.value} (Path: {self.execution_path.value})")
        log_chitti(f"[INTENT ENGINE] Goal: {self.goal}")
        if self.target:
            log_chitti(f"[INTENT ENGINE] Target: {self.target}")
        if self.recipient:
            log_chitti(f"[INTENT ENGINE] Recipient: {self.recipient}")
        if self.context_required:
            log_chitti("[INTENT ENGINE] Context resolution: REQUIRED")
        if self.missing_information:
            log_chitti(f"[INTENT ENGINE] Missing info: {self.missing_information}")


class NaturalIntentEngine:
    """
    Understands ordinary natural English, Hindi, and Hinglish speech.
    Extracts goals, entities, context dependencies, and selects optimal execution path.
    """

    @classmethod
    def analyze(cls, text: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> SemanticIntent:
        """
        Analyzes natural user input and resolves semantic intent with context awareness.
        """
        raw = text.strip()
        normalized = normalize_typos(raw)
        lower = normalized.lower()

        # Clean invocation prefix & trailing filler
        clean = re.sub(r"^(?:chitti,?\s*|hey chitti,?\s*|bhai,?\s*|yaar,?\s*|please\s+|can\s+you\s+)", "", normalized, flags=re.IGNORECASE).strip()
        clean = re.sub(r"(?i)\s+(?:for\s+me|in\s+(?:any\s+|my\s+)?browser|browser\s+me(?:in)?|on\s+(?:my\s+)?computer)$", "", clean).strip()
        clean = clean.rstrip(".!?,")
        clean_lower = clean.lower()

        # -------------------------------------------------------------
        # 1. EMERGENCY STOP / SLEEP / WAKE (PATH A - FASTEST)
        # -------------------------------------------------------------
        from src.core.phrase_matcher import get_phrase_matcher, ControlCommand
        matcher = get_phrase_matcher()
        ctrl = matcher.classify_command(raw)
        if ctrl.command == ControlCommand.STOP:
            return SemanticIntent(
                intent=IntentType.SIMPLE_COMMAND,
                goal="Stop current speech and task execution immediately",
                risk="low",
                execution_path=ExecutionPath.PATH_A_SIMPLE,
                parameters={"action": "stop"},
                confidence=1.0,
            )
        if ctrl.command == ControlCommand.SLEEP:
            return SemanticIntent(
                intent=IntentType.STANDBY_SLEEP,
                goal="Enter quiet standby sleep mode",
                risk="low",
                execution_path=ExecutionPath.PATH_A_SIMPLE,
                parameters={"action": "sleep"},
                confidence=1.0,
            )
        if ctrl.command == ControlCommand.WAKE:
            # If there is no actionable task following the wake word, treat as standalone WAKE
            has_actionable_task = bool(ctrl.extracted_tail) or bool(re.search(r"(?i)\b(?:open|kholo|play|baja|search|repo|repository|project|code|write|create|find|run|type|send|batao|tell|dikhao|chalao)\b", clean_lower))
            if not has_actionable_task:
                return SemanticIntent(
                    intent=IntentType.WAKE,
                    goal="Wake assistant from sleep",
                    risk="low",
                    execution_path=ExecutionPath.PATH_A_SIMPLE,
                    parameters={"action": "wake"},
                    confidence=1.0,
                )

        # Fast simple commands (volume, mute, calculator, screenshot)
        if re.search(r"(?i)\b(?:volume\s+(?:kam|badhao|mute|up|down)|increase\s+volume|decrease\s+volume|mute\s+volume)\b", clean_lower) or re.search(r"(?:आवाज़\s*(?:कम|बढ़ाओ|म्यूट))", raw):
            return SemanticIntent(
                intent=IntentType.SIMPLE_COMMAND,
                goal="Adjust system volume",
                target="Volume",
                risk="low",
                execution_path=ExecutionPath.PATH_A_SIMPLE,
                parameters={"action": "volume"},
                confidence=0.98,
            )
        if re.search(r"(?i)\b(?:take|capture)\s+(?:a\s+)?screenshot|screenshot\s+(?:le\s+lo|kheecho)\b", clean_lower):
            return SemanticIntent(
                intent=IntentType.SIMPLE_COMMAND,
                goal="Capture screen snapshot",
                target="Screenshot",
                risk="low",
                execution_path=ExecutionPath.PATH_A_SIMPLE,
                parameters={"action": "screenshot"},
                confidence=0.98,
            )

        # Standalone application tokens (e.g. "Calculator", "Notepad", "Chrome")
        APP_ALIASES = {"calculator", "notepad", "chrome", "google chrome", "vscode", "vs code", "terminal", "edge", "spotify", "browser"}
        if clean_lower in APP_ALIASES:
            return SemanticIntent(
                intent=IntentType.OPEN_APPLICATION,
                goal=f"Open {clean.title()}",
                target=clean.title(),
                risk="low",
                execution_path=ExecutionPath.PATH_A_SIMPLE,
                parameters={"application": clean.title()},
                confidence=0.95,
            )

        # -------------------------------------------------------------
        # 1.2. CONTEXTUAL DIRECTIVES, PIVOTS, CORRECTIONS & FOLLOW-UPS
        # -------------------------------------------------------------
        # Stop talking / pause directive
        if re.search(r"(?i)\b(?:stop\s+talking(?:\s+for\s+a\s+second)?|pause(?:\s+speech)?|ruko|thoda\s+ruko|chup\s+ho\s+jao)\b", clean_lower):
            return SemanticIntent(
                intent=IntentType.SIMPLE_COMMAND,
                goal="Pause or halt speech output immediately",
                risk="low",
                execution_path=ExecutionPath.PATH_A_SIMPLE,
                parameters={"action": "pause"},
                confidence=0.98,
            )

        # Pivot / Forget previous request
        if re.search(r"(?i)\b(?:wait,?\s*forget\s+that|forget\s+that|chhodo\s+use|leave\s+that|i\s+want\s+to\s+do\s+something\s+else)\b", clean_lower):
            return SemanticIntent(
                intent=IntentType.CONVERSATION,
                goal="Cancel previous context and pivot to new task",
                risk="low",
                execution_path=ExecutionPath.PATH_B_CONVERSATION,
                parameters={"action": "pivot", "cancel_previous": True},
                confidence=0.95,
            )

        # Correction / Alternative file ("No, not that one. Open the other file.")
        if re.search(r"(?i)\b(?:no,?\s*not\s+that\s+one|ye\s+nahi|not\s+this\s+one)\b.*(?:open\s+(?:the\s+)?other\s+file|dusri\s+file\s+kholo)?", clean_lower) or re.search(r"(?i)\bopen\s+(?:the\s+)?other\s+file\b", clean_lower):
            return SemanticIntent(
                intent=IntentType.FILE_OPERATION,
                goal="Open alternative file as corrected by user",
                target="other file",
                context_required=True,
                risk="low",
                execution_path=ExecutionPath.PATH_C_COMPUTER,
                parameters={"action": "open_alternative"},
                confidence=0.95,
            )

        # Context-dependent file opening ("Can you open that file for me?")
        if re.search(r"(?i)\b(?:open\s+(?:that|the)\s+file|us\s+file\s+ko\s+kholo|file\s+kholo)\b", clean_lower) and not re.search(r"(?i)\b(?:project|code|editor)\b", clean_lower):
            return SemanticIntent(
                intent=IntentType.FILE_OPERATION,
                goal="Open requested file from recent context",
                target="context file",
                context_required=True,
                risk="low",
                execution_path=ExecutionPath.PATH_C_COMPUTER,
                parameters={"action": "open_file"},
                confidence=0.92,
            )

        # Verification of previous action ("Can you check whether that actually worked?")
        if re.search(r"(?i)\b(?:check\s+(?:whether\s+that|if\s+(?:it|that))\s+(?:actually\s+)?worked|did\s+it\s+work|kaam\s+kiya\s+kya|check\s+karo\s+hua\s+ya\s+nahi)\b", clean_lower):
            return SemanticIntent(
                intent=IntentType.COMPLEX_TASK,
                goal="Verify whether the previous operation succeeded",
                context_required=True,
                risk="low",
                execution_path=ExecutionPath.PATH_C_COMPUTER,
                parameters={"action": "verify_task"},
                confidence=0.94,
            )

        # Follow-up style refinements ("Wait, explain it more simply", "Explain it again, but make it easier", "Actually, make the answer shorter")
        is_refinement = bool(
            re.search(r"(?i)\b(?:explain\s+(?:it|that)\s+(?:again,?\s*but\s+)?(?:more\s+|much\s+)?(?:simply|easier|simple)|make\s+(?:the\s+answer|it)\s+(?:shorter|simpler|concise|easier)|aasan\s+bhasha\s+me(?:in)?\s+samjhao|chhota\s+karo)\b", clean_lower)
            or re.search(r"(?i)\b(?:explain\s+it\s+again|make\s+it\s+easier)\b", clean_lower)
        )
        if is_refinement:
            recent_topic = cls._extract_recent_topic(conversation_history)
            target_topic = recent_topic or "previous topic"
            refine_type = "shorter" if ("short" in clean_lower or "chhota" in clean_lower) else "simpler"
            return SemanticIntent(
                intent=IntentType.TECHNICAL_QUERY,
                goal=f"Explain {target_topic} in a {refine_type} manner",
                target=target_topic,
                context_required=True,
                risk="low",
                execution_path=ExecutionPath.PATH_B_CONVERSATION,
                parameters={"topic": target_topic, "refinement": refine_type, "original_query": clean},
                confidence=0.95,
            )

        # -------------------------------------------------------------
        # 1.5. COMPLEX MULTI-STEP TASKS (PATH E)
        # -------------------------------------------------------------
        is_multi_step = bool(
            re.search(r"(?i)\b(?:find|locate|search)\s+.*,\s*(?:rename|move|copy)\s+.*,\s*(?:and|aur)?\s*(?:attach|send|upload)\b", clean_lower)
            or re.search(r"(?i)\b(?:find|locate)\s+.*,\s*(?:rename|move|copy)\s+.*(?:and|aur)\s+(?:attach|send|upload)\b", clean_lower)
        )
        if is_multi_step:
            return SemanticIntent(
                intent=IntentType.COMPLEX_TASK,
                goal=clean,
                risk="medium",
                execution_path=ExecutionPath.PATH_E_COMPLEX,
                parameters={"task": clean},
                confidence=0.95,
            )

        # -------------------------------------------------------------
        # 1E. DIRECT PROJECT & REPOSITORY OPENING ("open project repo", "open 3D wallpaper project", "open repo")
        # -------------------------------------------------------------
        is_creation_query = bool(re.search(r"(?i)\b(?:create|created|creating|build|built|building|make|made|making|develop|developed|developing|implement|implemented|implementing|generate|generated|generating|write|wrote|writing)\b", clean_lower))
        m_proj = None if is_creation_query else re.search(r"(?i)\b(?:open|kholo|show|bring\s+up|load)\s+(?:the\s+|my\s+|our\s+)?([a-zA-Z0-9_\-\s]+?)(?:\s+(?:in\s+(?:browser|chrome|brave|edge|vs\s*code|vscode|editor)))?$", clean_lower)
        if m_proj:
            candidate_target = m_proj.group(1).strip()
            if candidate_target.lower() in ("it", "this", "that", "them", "file", "the file", "app", "window"):
                candidate_target = ""
            if candidate_target:
                from src.agent.projects import ProjectRegistry
                reg = ProjectRegistry()
                p_path = reg.get_project_path(candidate_target)
                if p_path:
                in_browser = bool(re.search(r"(?i)\b(?:browser|chrome|brave|edge|website|web)\b", clean_lower))
                if in_browser or (candidate_target in ("repo", "repository", "github repo", "my github repo", "project repo", "project repository") and not re.search(r"(?i)\b(?:code|editor|folder|local|directory|desktop)\b", clean_lower)):
                    from src.agent.registry import DEFAULT_URL_MAP
                    repo_url = DEFAULT_URL_MAP.get(candidate_target, DEFAULT_URL_MAP.get("my repo", "https://github.com/bishtprateek270-hue?tab=repositories"))
                    return SemanticIntent(
                        intent=IntentType.OPEN_APPLICATION,
                        goal=f"Open {candidate_target} in browser",
                        target=candidate_target,
                        risk="low",
                        execution_path=ExecutionPath.PATH_C_COMPUTER,
                        parameters={"url": repo_url, "target": candidate_target, "action": "open_url"},
                        confidence=0.95,
                    )
                else:
                    return SemanticIntent(
                        intent=IntentType.OPEN_PROJECT,
                        goal=f"Open {candidate_target} project in VS Code / Editor",
                        target=candidate_target,
                        risk="low",
                        execution_path=ExecutionPath.PATH_C_COMPUTER,
                        parameters={
                            "project_name": candidate_target,
                            "project_path": p_path,
                            "action": "open_project",
                        },
                        confidence=0.95,
                    )

        # -------------------------------------------------------------
        # 2. CONTEXTUAL PROJECT OPENING ("Yaar, kal jo Python project banaya tha, usko kholna")
        # -------------------------------------------------------------
        is_web_or_repo_query = bool(re.search(r"(?i)\b(?:repo|repository|github|git|deployment|ldeployment|live|browser|website|link|url|vercel)\b", clean_lower))
        is_open_past_project = not is_web_or_repo_query and (
            bool(re.search(r"(?i)\b(?:kal|pichla|purana|previous|yesterday|earlier|last)\b.*(?:project|code|work)\b.*(?:kholna|kholo|open|chalao|dikhao)", clean_lower))
            or bool(re.search(r"(?i)\b(?:project|code)\b.*(?:banaya\s+tha|bana\s+tha|created|worked\s+on)\b.*(?:kholna|kholo|open|bring\s+up)", clean_lower))
            or bool(re.search(r"(?i)\b(?:open|bring\s+up|load)\s+(?:that|the|my)?\s*(?:recent|previous|yesterday'?s?|last|same)\s+(?:project|code)\b", clean_lower))
            or bool(re.search(r"(?i)\b(?:open|bring\s+up|load)\s+(?:that|my)\s+(?:project|code)\s*(?:again|also|once\s+more|phir\s+se)?\b", clean_lower))
            or bool(re.search(r"(?i)\b(?:open|bring\s+up|load)\s+(?:the\s+)?(?:project|code)\s+(?:again|also|once\s+more|phir\s+se)\b", clean_lower))
            or bool(re.search(r"(?i)\b(?:let'?s?\s+work\s+on\s+that|work\s+on\s+the|continue\s+with)\s+([a-zA-Z0-9_\-]+)?\s*(?:project|code)\b", clean_lower))
            or bool(re.search(r"(?i)\b(?:let'?s?\s+)?continue\s+(?:the\s+|that\s+|working\s+on\s+(?:the\s+|that\s+)?)?project\b", clean_lower))
            or bool(re.search(r"(?:मेरा\s+)?(?:प्रोजेक्ट|प्रोग्राम|कोड)\s*(?:खोलो|चलाओ|दिखाओ)", raw))
        )
        if is_open_past_project:
            # Extract target technology / hint if available (e.g., Python)
            m_lang = re.search(r"(?i)\b(python|javascript|typescript|react|html|css|cpp|c\+\+|java|rust|go|django|flask|fastapi)\b", clean_lower)
            lang_hint = m_lang.group(1).lower() if m_lang else None

            # Resolve project using registry / context
            resolved_proj, proj_path = cls._resolve_recent_project(lang_hint=lang_hint)
            target_desc = resolved_proj or (f"{lang_hint} project" if lang_hint else "recent project")

            return SemanticIntent(
                intent=IntentType.OPEN_PROJECT,
                goal=f"Locate and open {target_desc} in VS Code / editor",
                target=target_desc,
                context_required=True,
                risk="low",
                execution_path=ExecutionPath.PATH_C_COMPUTER,
                parameters={
                    "project_name": resolved_proj,
                    "project_path": proj_path,
                    "language_hint": lang_hint,
                    "action": "open_project",
                },
                confidence=0.95,
            )

        # -------------------------------------------------------------
        # 3. CODE DIAGNOSIS & REASONING ("This code isn't working. Can you figure out why?")
        # -------------------------------------------------------------
        is_code_debug = (
            bool(re.search(r"(?i)\b(?:code|program|script|function|it)\s+(?:isn'?t|is\s+not|not)\s+working\b", clean_lower))
            or bool(re.search(r"(?i)\b(?:why\s+is\s+(?:this|my|the)\s+code\s+(?:failing|crashing|broken|throwing|not\s+working))\b", clean_lower))
            or bool(re.search(r"(?i)\b(?:figure\s+out\s+why|find\s+(?:the\s+)?(?:error|bug|issue)|diagnose|fix\s+(?:this|my)\s+code)\b", clean_lower))
            or bool(re.search(r"(?i)\b(?:check|inspect|see)\s+.*(?:issues?|problems?|errors?|bugs?)\s+.*(?:code|program|script)\b", clean_lower))
            or bool(re.search(r"(?i)\b(?:ye|yeh|mera)\s+code\s+(?:chal\s+nahi\s+raha|kaam\s+nahi\s+kar\s+raha|fail\s+ho\s+raha|error\s+de\s+raha)\b", clean_lower))
            or bool(re.search(r"(?i)\b(?:dekhna\s+kya\s+(?:gadbad|error|issue|problem)\s+hai|code\s+me(?:in)?\s+kya\s+error\s+hai)\b", clean_lower))
        )
        if is_code_debug:
            return SemanticIntent(
                intent=IntentType.DEBUG_CODE,
                goal="Inspect code context, diagnose errors, and propose fixes",
                target="Current active code / project",
                context_required=True,
                risk="low",
                execution_path=ExecutionPath.PATH_D_CODING,
                parameters={"query": clean},
                confidence=0.94,
            )

        # -------------------------------------------------------------
        # 4. NATURAL MESSAGING ("I need to tell Rahul that I'll be late.")
        # -------------------------------------------------------------
        # Check for message command without recipient first
        m_send_no_rec = re.search(r"(?i)\b(?:i\s+need\s+to\s+)?(?:send\s+(?:a\s+)?message|message\s+bhejo)\s+(?:saying|that|to\s+say)\s+(.+)$", clean)
        # Look for natural tell / message patterns in English and Hindi with recipient
        m_tell_en = re.search(r"(?i)\b(?:i\s+need\s+to\s+)?tell\s+([A-Za-z0-9_\s]+?)\s+(?:that|to)\s+(.+)$", clean)
        m_tell_hi = re.search(r"(?i)\b([A-Za-z0-9_\s]+?)\s+ko\s+(?:bolna|keh\s+dena|bata\s+dena)\s+(?:ki\s+)?(.+)$", clean)
        m_msg_direct = re.search(r"(?i)\b(?:send\s+(?:a\s+)?message\s+to|message|msg)\s+([A-Za-z0-9_\s]+?)\s+(?:saying|that|:)\s*(.+)$", clean)
        m_whatsapp_hi = re.search(r"(?i)\b([A-Za-z0-9_\s]+?)\s+ko\s+(?:whatsapp|watsapp|message)\s+(?:pe|par)?\s*(?:ye\s+)?bhejo\s*[:\s]*(.+)$", clean)

        if m_send_no_rec and not (m_tell_en or m_tell_hi or m_msg_direct or m_whatsapp_hi):
            body_no_rec = polish_message_text(m_send_no_rec.group(1).strip().strip("\"'")).rstrip(".!?,")
            return SemanticIntent(
                intent=IntentType.SEND_MESSAGE,
                goal=f"Send message '{body_no_rec}'",
                content=body_no_rec,
                channel="WhatsApp",
                missing_information=["recipient"],
                clarification_prompt="Who would you like me to send this message to?",
                context_required=True,
                risk="low",
                execution_path=ExecutionPath.PATH_C_COMPUTER,
                parameters={"text": body_no_rec},
                confidence=0.90,
            )

        m_msg_match = m_tell_en or m_tell_hi or m_msg_direct or m_whatsapp_hi
        if m_msg_match:
            contact = m_msg_match.group(1).strip()
            msg_content = m_msg_match.group(2).strip().strip("\"'")
            contact = re.sub(r"(?i)^(?:to|ko|that)\s+", "", contact).strip()

            # Polish content and strip trailing period
            msg_content = polish_message_text(msg_content).rstrip(".!?,")
            msg_content = re.sub(r"(?i)\s+(?:on|via|using|pe|par)\s+(?:whatsapp|telegram|slack|discord|teams|web|browser|app).*$", "", msg_content).strip() or "Hi"
            channel = "WhatsApp"  # Default desktop messaging channel
            if "email" in clean_lower or "@" in contact:
                channel = "Email"
            elif "slack" in clean_lower:
                channel = "Slack"
            elif "telegram" in clean_lower:
                channel = "Telegram"

            # Check if critical recipient is missing
            missing = []
            clarification = None
            if not contact or contact.lower() in ("someone", "him", "her", "them"):
                missing.append("recipient")
                clarification = "Who would you like me to send this message to?"

            return SemanticIntent(
                intent=IntentType.SEND_MESSAGE,
                goal=f"Send message to {contact} via {channel}",
                target=contact,
                recipient=contact,
                channel=channel,
                content=msg_content,
                context_required=True if missing else False,
                missing_information=missing,
                clarification_prompt=clarification,
                risk="low",
                execution_path=ExecutionPath.PATH_C_COMPUTER,
                parameters={
                    "recipient": contact,
                    "channel": channel,
                    "text": msg_content,
                    "contact": contact,
                },
                confidence=0.92,
            )

        # -------------------------------------------------------------
        # 5. NATURAL PROJECT CREATION ("Can you make a simple website for my project?")
        # -------------------------------------------------------------
        is_create_project = (
            bool(re.search(r"(?i)\b(?:make|made|making|create|created|creating|build|built|building|develop|developed|developing|generate|generated|generating|implement|implemented|implementing|write|wrote|writing)\s+(?:a\s+|an\s+)?(?:simple\s+|modern\s+|interactive\s+|quick\s+|small\s+|fully\s+functional\s+)?([a-zA-Z0-9_\-\s]+?\s+)?(?:website|site|web|portal|store|shop|ecommerce|commerce|app|application|project|dashboard|portfolio|tracker|tool|system|game|calculator|program|script)\b", clean_lower))
            or bool(re.search(r"(?i)\b(?:mere\s+liye\s+)?(?:ek\s+)?([a-zA-Z0-9_\-\s]+?)\s+(?:website|site|web|store|shop|ecommerce|app|project|dashboard|calculator)\s*(?:banao|bana\s+do|bana\s+dijiye|create\s+karo)\b", clean_lower))
        )
        # Guard: Check that it is not an inquiry / conversational question ("How do I create...", "Why create...")
        is_pure_question = bool(re.search(r"(?i)^(?:how\s+(?:do\s+i|can\s+i|to)|why\s+would|what\s+is\s+a)\b", clean_lower))

        if is_create_project and not is_pure_question:
            m_target = re.search(r"(?i)\b(?:make|made|making|create|created|creating|build|built|building|develop|developed|developing|implement|implemented|implementing|write|wrote|writing|banao|bana\s+do)\s+(?:a\s+|an\s+|the\s+)?([a-zA-Z0-9_\-\s]+)", clean)
            target_str = m_target.group(1).strip() if m_target else "website"
            target_str = re.sub(r"(?i)\s+(?:for\s+my\s+project|for\s+me|using\s+.*)$", "", target_str).strip()

            return SemanticIntent(
                intent=IntentType.CREATE_PROJECT,
                goal=f"Create and build software project: {clean}",
                target=target_str or "project",
                context_required=False,
                risk="low",
                execution_path=ExecutionPath.PATH_D_CODING,
                parameters={"task": clean, "target": target_str},
                confidence=0.96,
            )

        # -------------------------------------------------------------
        # 6. PURE STATEMENTS / CASUAL CHAT ("I'm tired of this project", "Hello", "How are you")
        # Do NOT trigger computer automation for mere statements of feeling!
        # -------------------------------------------------------------
        is_feeling_or_casual = (
            bool(re.search(r"(?i)\b(?:i'?m\s+(?:tired|bored|exhausted|happy|frustrated|done\s+for\s+today))\b", clean_lower))
            or bool(re.search(r"(?i)\b(?:aaj\s+man\s+nahi\s+hai|thak\s+gaya|bore\s+ho\s+raha)\b", clean_lower))
            or clean_lower in ("hello", "hi", "hey", "namaste", "sup", "yo", "kaise ho", "kya haal hai", "thanks", "thank you")
        )
        if is_feeling_or_casual:
            return SemanticIntent(
                intent=IntentType.CONVERSATION,
                goal=f"Conversational response to: {clean}",
                target=None,
                risk="low",
                execution_path=ExecutionPath.PATH_B_CONVERSATION,
                parameters={"statement": clean},
                confidence=0.95,
            )

        # -------------------------------------------------------------
        # 7. EXPLICIT APPLICATION LAUNCH & WEB TASKS ("open notepad", "chrome kholo", "calc")
        # -------------------------------------------------------------
        m_app_en = re.search(r"(?i)^(?:open|launch|start|run|show)\s+(?:the\s+|app\s+|my\s+)?([a-zA-Z0-9_\-\.\:\/\s]+)$", clean)
        m_app_hi = re.search(r"(?i)^([a-zA-Z0-9_\-\.\:\/\s]+?)(?:\s+ko|\s+app)?\s+(?:kholo|chalao|open\s+karo|chala\s+do)$", clean)
        m_app = m_app_en or m_app_hi
        if m_app:
            app_raw = m_app.group(1).strip()
            # If not a complex sentence with and/aur
            if not re.search(r"(?i)\b(?:and|aur|project|website|saying|that|with)\b", app_raw):
                return SemanticIntent(
                    intent=IntentType.OPEN_APPLICATION,
                    goal=f"Open {app_raw}",
                    target=app_raw,
                    risk="low",
                    execution_path=ExecutionPath.PATH_A_SIMPLE,
                    parameters={"application": app_raw},
                    confidence=0.95,
                )

        # -------------------------------------------------------------
        # 8. GENERAL TECHNICAL KNOWLEDGE ("What is a neural network?", "Explain this simply")
        # -------------------------------------------------------------
        if re.search(r"(?i)^(?:what\s+is|what\s+are|explain|how\s+does|difference\s+between|kya\s+hota\s+hai|samjhao)\b", clean_lower):
            return SemanticIntent(
                intent=IntentType.TECHNICAL_QUERY,
                goal=f"Explain technical concept: {clean}",
                risk="low",
                execution_path=ExecutionPath.PATH_B_CONVERSATION,
                parameters={"query": clean},
                confidence=0.92,
            )

        # -------------------------------------------------------------
        # DEFAULT: Conversational assistance via PATH B
        # -------------------------------------------------------------
        return SemanticIntent(
            intent=IntentType.CONVERSATION,
            goal=clean,
            risk="low",
            execution_path=ExecutionPath.PATH_B_CONVERSATION,
            parameters={"query": clean},
            confidence=0.85,
        )

    @classmethod
    def _resolve_recent_project(cls, lang_hint: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
        """
        Intelligently resolves the most recent project matching the given language or context.
        Checks ProjectRegistry, Desktop, and user workspace.
        """
        try:
            from src.agent.projects import ProjectRegistry
            reg = ProjectRegistry()
            projects = reg.list_projects()

            # 1. Filter by language hint if specified (e.g. Python -> look for .py or python in name/files)
            if lang_hint:
                for name, path in reversed(list(projects.items())):
                    if lang_hint in name.lower():
                        return name, path
                    p = Path(path)
                    if p.exists() and p.is_dir():
                        if list(p.glob("*.py")) if lang_hint == "python" else False:
                            return name, path

            # 2. Return the most recently registered or modified project (excluding "chitti" itself)
            candidates = [(name, path) for name, path in projects.items() if name.lower() != "chitti"]
            if candidates:
                return candidates[-1]

            # 3. Check Desktop or Projects directory for recent Python folders
            desktop = Path.home() / "OneDrive" / "Desktop"
            if not desktop.exists():
                desktop = Path.home() / "Desktop"

            if desktop.exists():
                folders = [f for f in desktop.iterdir() if f.is_dir() and not f.name.startswith(".")]
                if lang_hint == "python":
                    py_folders = [f for f in folders if "python" in f.name.lower() or list(f.glob("*.py"))]
                    if py_folders:
                        best = sorted(py_folders, key=lambda x: x.stat().st_mtime, reverse=True)[0]
                        reg.register_project(best.name, str(best.resolve()))
                        return best.name, str(best.resolve())

                if folders:
                    best = sorted(folders, key=lambda x: x.stat().st_mtime, reverse=True)[0]
                    reg.register_project(best.name, str(best.resolve()))
                    return best.name, str(best.resolve())

        except Exception as e:
            log_debug(f"[INTENT ENGINE] Project resolution notice: {e}")

        return None, None

    @classmethod
    def _extract_recent_topic(cls, conversation_history: Optional[List[Any]]) -> Optional[str]:
        """Extracts the subject or technical topic of the most recent user conversation turn."""
        if not conversation_history:
            return None
        messages = conversation_history
        if hasattr(conversation_history, "_messages"):
            messages = conversation_history._messages
        elif hasattr(conversation_history, "messages"):
            messages = conversation_history.messages

        for msg in reversed(messages):
            content = ""
            if isinstance(msg, dict):
                if msg.get("role") == "user":
                    content = msg.get("content", "")
            elif hasattr(msg, "role") and getattr(msg, "role") == "user":
                content = getattr(msg, "content", "")
            elif isinstance(msg, str):
                content = msg

            if content:
                topic = re.sub(
                    r"(?i)^(?:can\s+you\s+|please\s+|explain\s+|what\s+is\s+|what\s+are\s+|tell\s+me\s+about\s+|how\s+does\s+)",
                    "",
                    content,
                ).strip().rstrip(".!?")
                if topic and len(topic) > 2 and topic.lower() not in ("hi", "hello", "thanks", "ok", "yes", "no"):
                    return topic
        return None
