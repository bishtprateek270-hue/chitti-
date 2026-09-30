"""
Chitti Action Intent & User Goal Extraction Engine.
Separates User Intent from Target dynamically without hardcoded application/website lists.
Analyzes natural language requests into structured generic action models.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from src.agent.registry import AppDiscovery, FolderDiscovery, ResourceDiscovery, DEFAULT_URL_MAP
from src.utils.logging import log_chitti, log_debug, log_info
from src.utils.text import clean_contact_query, normalize_typos, polish_message_text, strip_emojis


class ActionIntentType(str, Enum):
    # Communication Actions
    SEND_MESSAGE = "SEND_MESSAGE"
    SEND_EMAIL = "SEND_EMAIL"
    WRITE_EMAIL = "WRITE_EMAIL"

    # Browser & Media Actions
    OPEN_URL = "OPEN_URL"
    SEARCH_WEB = "SEARCH_WEB"
    PLAY_MEDIA = "PLAY_MEDIA"

    # Computer & Application Actions
    OPEN_APPLICATION = "OPEN_APPLICATION"
    CLOSE_APPLICATION = "CLOSE_APPLICATION"
    TYPE_TEXT = "TYPE_TEXT"
    CONTROL_APPLICATION = "CONTROL_APPLICATION"

    # Filesystem Actions
    CREATE_FILE = "CREATE_FILE"
    CREATE_FOLDER = "CREATE_FOLDER"
    DELETE_FILE = "DELETE_FILE"
    DELETE_FOLDER = "DELETE_FOLDER"
    OPEN_FOLDER = "OPEN_FOLDER"
    OPEN_FILE = "OPEN_FILE"

    # Project & Coding Actions (Phase 6)
    CREATE_PROJECT = "CREATE_PROJECT"
    MODIFY_PROJECT = "MODIFY_PROJECT"
    RUN_PROJECT = "RUN_PROJECT"
    DEBUG_PROJECT = "DEBUG_PROJECT"
    TEST_PROJECT = "TEST_PROJECT"

    # System Control Actions
    TAKE_SCREENSHOT = "TAKE_SCREENSHOT"
    ANALYZE_SCREEN = "ANALYZE_SCREEN"
    DIAGNOSE_SCREEN_ERROR = "DIAGNOSE_SCREEN_ERROR"
    FIND_UI_ELEMENT = "FIND_UI_ELEMENT"
    SET_VOLUME = "SET_VOLUME"
    GET_SYSTEM_INFO = "GET_SYSTEM_INFO"

    # Fallback / Informational
    CONVERSATIONAL = "CONVERSATIONAL"
    UNKNOWN = "UNKNOWN"


@dataclass
class ActionIntent:
    """Generic Action Model representing a computer-use task independently of specific apps."""
    intent: ActionIntentType
    goal: str = ""
    target: Optional[str] = None
    destination: Optional[str] = None
    recipient: Optional[str] = None
    content: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    execution_required: bool = True
    verification_required: bool = True
    channel: Optional[str] = None
    application: Optional[str] = None
    subject: Optional[str] = None
    attachments: List[str] = field(default_factory=list)
    preconditions: List[str] = field(default_factory=list)
    requires_browser: bool = False
    requires_authentication: bool = False
    requires_confirmation: bool = False

    def log_intent(self):
        """Emits structured observable logs for this action intent."""
        log_chitti("[CHITTI] [GOAL]")
        log_chitti(f"[CHITTI] Intent: {self.intent.value}")
        if self.application:
            log_chitti(f"[CHITTI] Application: {self.application}")
        if self.recipient:
            log_chitti(f"[CHITTI] Recipient: {self.recipient}")
        elif self.target:
            log_chitti(f"[CHITTI] Target: {self.target}")
        if self.content:
            log_chitti(f"[CHITTI] Content: {self.content}")
        log_chitti(f"[CHITTI] Final Goal: {self.goal}")

        # Standard analyzer tags for subsystem compatibility
        log_chitti(f"[CHITTI] [ACTION ANALYZER] Intent: {self.intent.value}")
        if self.application:
            log_chitti(f"[CHITTI] [ACTION ANALYZER] Application: {self.application}")
        if self.target:
            log_chitti(f"[CHITTI] [ACTION ANALYZER] Target: {self.target}")
        if self.recipient:
            log_chitti(f"[CHITTI] [ACTION ANALYZER] Recipient: {self.recipient}")
        if self.content:
            log_chitti(f"[CHITTI] [ACTION ANALYZER] Content: {self.content}")
        log_chitti(f"[CHITTI] [TASK PLANNER] Goal: {self.goal}")


class ActionIntentAnalyzer:
    """
    Extracts structured user goals and entities dynamically from arbitrary requests.
    Decoupled from application-specific hardcoding.
    """

    @classmethod
    def _normalize_text(cls, text: str) -> str:
        """Corrects typos in common action verbs and application names."""
        return normalize_typos(text)

    @classmethod
    def extract_intent(cls, user_text: str) -> ActionIntent:
        """
        Analyzes the user's request and returns a structured ActionIntent.
        """
        raw = user_text.strip()
        normalized = cls._normalize_text(raw)
        lower = normalized.lower()

        # Clean conversational prefixes and trailing fillers
        clean = re.sub(r"^(?:chitti,?\s*|hey chitti,?\s*|bhai,?\s*|please\s+|can\s+you\s+)", "", normalized, flags=re.IGNORECASE).strip()
        clean = re.sub(r"(?i)\s+(?:for\s+me|in\s+(?:any\s+|my\s+)?browser|browser\s+me(?:in)?|on\s+(?:my\s+)?computer)$", "", clean).strip()
        clean_lower = clean.lower()

        # -------------------------------------------------------------
        # 1. EMAIL DRAFTING vs EMAIL SENDING
        # -------------------------------------------------------------
        email_addr_match = re.search(r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})", clean)

        # 1A. Explicit Email Drafting ("write an email", "draft an email") WITHOUT actual send command
        is_draft_only = bool(re.search(r"(?i)\b(?:write|draft|generate|compose|suggest)\s+(?:an?\s+)?email\b", clean)) and not email_addr_match and not re.search(r"(?i)\b(?:send|shoot|dispatch|mail\s+this|email\s+this)\b", clean)
        if is_draft_only:
            m_target = re.search(r"(?i)\b(?:to|for)\s+([a-zA-Z0-9_\-\s]+?)(?:\s+asking|\s+about|\s+regarding|\s+with|\s*$)", clean)
            target_name = m_target.group(1).strip() if m_target else "Recipient"
            return ActionIntent(
                intent=ActionIntentType.WRITE_EMAIL,
                goal=f"Draft an email to {target_name}",
                channel="Email",
                target=target_name,
                destination=target_name,
                execution_required=True,
                verification_required=False,
            )

        # 1B. Actionable Email Sending ("send email to...", "email <msg> to <recipient>...", "mail this to...")
        if email_addr_match or bool(re.search(r"(?i)\b(?:send\s+(?:an?\s+)?email|email\s+(?:this|the|a)?|mail\s+(?:this|the|a)?)\b", clean)):
            recipient = email_addr_match.group(1) if email_addr_match else ""
            if not recipient:
                m_rec = re.search(r"(?i)\b(?:to|ko)\s+([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}|[a-zA-Z0-9_\-]+)", clean)
                recipient = m_rec.group(1).strip() if m_rec else "Recipient"

            # Extract subject if specified
            subject = "Message from Chitti"
            m_sub = re.search(r"(?i)\b(?:subject|regarding|about)\s+[\"']?([^\"',]+)[\"']?", clean)
            if m_sub:
                subject = m_sub.group(1).strip().title()
                clean_no_sub = re.sub(r"(?i)\b(?:with\s+)?(?:subject|regarding|about)\s+[\"']?[^\"',]+[\"']?", "", clean).strip()
            else:
                clean_no_sub = clean

            # Extract message content / body
            content = ""
            m_saying = re.search(r"(?i)\b(?:with\s+message|saying|body|with\s+body|that)\s+([\"']?.+?[\"']?)(?:\s+from\s+my\s+side|\s*$)", clean_no_sub)
            if m_saying:
                content = m_saying.group(1).strip().strip("\"'")
            else:
                # Pattern 1: send <msg> [email] to <recipient>
                m_msg_before_to = re.search(r"(?i)\b(?:email|send|mail)\s+(?:an?\s+)?(?:message\s+|msg\s+|email\s+)?(?!(?:an?\s+)?(?:email|mail|message)\s+to)(.+?)\s+(?:an?\s+)?(?:message\s+|msg\s+|email\s+)?to\s+", clean_no_sub)
                # Pattern 2: send email to <recipient> [:] <msg>
                m_msg_after_to = re.search(r"(?i)\b(?:to|ko)\s+[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:\s*[:,\-]?\s*)(.+)$", clean_no_sub)
                
                if m_msg_before_to:
                    content = m_msg_before_to.group(1).strip().strip("\"'")
                elif m_msg_after_to:
                    content = m_msg_after_to.group(1).strip().strip("\"'")

            # Thorough cleanup of extracted content
            if content:
                content = re.sub(r"(?i)\s+(?:an?\s+)?(?:email|mail|message|msg|text)$", "", content).strip()
                content = re.sub(r"(?i)^(?:an?\s+)?(?:email|mail|message|msg|text)\s+(?:saying|that|with\s+body|body)?\s*", "", content).strip()
                content = re.sub(r"(?i)^(?:saying|that|body|with\s+message|with\s+body|:)\s*", "", content).strip()
                content = re.sub(r"(?i)\s+from\s+my\s+side$", "", content).strip()
                content = polish_message_text(content)

            if not content or content.lower() in ("an email", "email", "a message", "message", "this", "mail"):
                content = "Hello, I am reaching out to you."

            return ActionIntent(
                intent=ActionIntentType.SEND_EMAIL,
                goal=f"Send email to {recipient}",
                channel="Email",
                application="Gmail / Webmail",
                recipient=recipient,
                target=recipient,
                destination=recipient,
                content=content,
                subject=subject,
                parameters={"recipient": recipient, "subject": subject, "content": content},
                execution_required=True,
                requires_browser=True,
                requires_authentication=True,
                requires_confirmation=True,
                verification_required=True,
            )

        # -------------------------------------------------------------
        # 2. MESSAGING & CHAT APPS
        # -------------------------------------------------------------
        is_messaging = bool(
            re.search(r"(?i)\b(?:whatsapp|watsapp|telegram|slack|discord|teams|signal)\b", clean) or
            re.search(r"(?i)\b(?:message|msg|send\s+message)\s+.*to\s+.*", clean) or
            re.search(r"(?i)\b.*ko\s+.*(?:message|bhejo|msg)\b", clean)
        )
        if is_messaging:
            # Determine channel & application
            if "telegram" in lower:
                channel, app, base_url = "Telegram", "Telegram Web", "https://web.telegram.org"
            elif "slack" in lower:
                channel, app, base_url = "Slack", "Slack", "https://app.slack.com"
            elif "discord" in lower:
                channel, app, base_url = "Discord", "Discord", "https://discord.com/app"
            elif "teams" in lower:
                channel, app, base_url = "Teams", "Microsoft Teams", "https://teams.microsoft.com"
            else:
                channel, app, base_url = "WhatsApp", "WhatsApp Web", "https://web.whatsapp.com"

            # Check if this is sending a real message or merely opening the app
            m_saying = re.search(r"(?i)\b(?:open\s+.*(?:and|aur)\s+)?(?:send(?:\s+a)?\s+(?:message|msg|text)|message|msg)\s+to\s+(?P<contact>[^\n\r,;:.]+?)\s+(?:saying|that|:)\s*(?P<msg>[\"']?[^\"']+?[\"']?)$", clean)
            m_msg_quotes = re.search(r"(?i)\b(?:open\s+.*(?:and|aur)\s+)?(?:send(?:\s+a)?\s+(?:message|msg|text)|message|msg|send)\s+(?P<msg>[\"'][^\"']+[\"'])\s+to\s+(?P<contact>[^\n\r,;:.]+)", clean)
            m_contact_ko = re.search(r"(?i)(?:(?:whatsapp|watsapp)\s+(?:par|pe)?\s*)?(?P<contact>[^\n\r,;:.]+?)\s+ko\s+(?P<msg>[\"']?[^\"']+?[\"']?)\s*(?:message\s+karo|bhejo|send\s+karo|message\s+kar|msg\s+bhejo)", clean)
            m_send_contact_msg = re.search(r"(?i)\b(?:send|message|msg)\s+(?:to\s+)?(?P<contact>[^\n\r,;:'\"]+?)\s*(?:saying|that|message|msg|:)\s*(?P<msg>[\"']?[^\"']+?[\"']?)$", clean) or \
                                 re.search(r"(?i)\b(?:send|message|msg)\s+(?:to\s+)?(?P<contact>[^\n\r,;:'\"]+?)\s+(?P<msg>['\"][^'\"]+?['\"])", clean)
            m_send_to_contact = re.search(r"(?i)\b(?:open\s+.*(?:and|aur)\s+)?(?:send|message|msg)\s+to\s+(?P<contact>[A-Za-z0-9_\s]+?)\s+(?P<msg>[\"'][^\"']+[\"']|[^\s\"']+(?:\s+[^\s\"']+)?)$", clean)
            m_msg_to = re.search(r"(?i)\b(?:open\s+.*(?:and|aur)\s+)?(?:send(?:\s+a)?\s+(?:message|msg|text)|send|message|msg)\s+(?P<msg>[\"'][^\"']+[\"']|[^\s\"']+(?:\s+[^\s\"']+)?)\s+to\s+(?P<contact>[^\n\r,;:.]+)", clean)

            m_msg_matched = m_saying or m_msg_quotes or m_contact_ko or m_send_contact_msg or m_msg_to or m_send_to_contact

            if m_msg_matched:
                msg_text = (m_msg_matched.group("msg") or "").strip().strip("\"'")
                contact_name = (m_msg_matched.group("contact") or "").strip().strip("\"'")

                # Clean message text and contact name
                msg_text = re.sub(r"(?i)\s+(?:an?\s+)?(?:email|mail|message|msg|text)$", "", msg_text).strip()
                msg_text = re.sub(r"(?i)^(?:saying|that|:)\s*", "", msg_text).strip()
                msg_text = re.sub(r"(?i)\s+(?:message|msg|text)$", "", msg_text).strip()
                msg_text = re.sub(r"(?i)\s+(?:on|via|using|pe|par)\s+(?:whatsapp|telegram|slack|discord|teams|web|browser|app).*$", "", msg_text).strip() or "Hi"
                msg_text = polish_message_text(msg_text)
                
                contact_name = re.sub(r"(?i)\s+(?:on|via|using|pe|par)\s+(?:whatsapp|telegram|slack|discord|teams|web|browser|app).*$", "", contact_name).strip()
                contact_name = re.sub(r"(?i)\s+(?:on|via|using|pe|par)\s+.*$", "", contact_name).strip()
                contact_name = re.sub(r"(?i)\s+(?:message|msg|text)$", "", contact_name).strip()
                contact_name = re.sub(r"(?i)^(?:to\s+|for\s+|ko\s+)", "", contact_name).strip() or "Contact"
                search_query = clean_contact_query(contact_name)

                return ActionIntent(
                    intent=ActionIntentType.SEND_MESSAGE,
                    goal=f"Send \"{msg_text}\" to {contact_name} using {app}",
                    channel=channel,
                    application=app,
                    target=contact_name,
                    recipient=contact_name,
                    destination=contact_name,
                    content=msg_text,
                    parameters={"url": base_url, "contact": contact_name, "search_contact": search_query, "text": msg_text},
                    execution_required=True,
                    requires_browser=True,
                    requires_authentication=True,
                    requires_confirmation=False,
                    verification_required=True,
                )
            else:
                return ActionIntent(
                    intent=ActionIntentType.OPEN_URL,
                    goal=f"Open {app} in browser",
                    channel=channel,
                    application=app,
                    target=app,
                    destination=base_url,
                    parameters={"url": base_url},
                    execution_required=True,
                    requires_browser=True,
                    requires_authentication=True,
                    verification_required=True,
                )

        # -------------------------------------------------------------
        # 3. YOUTUBE & MEDIA PLAYBACK
        # -------------------------------------------------------------
        is_media = bool(
            re.search(r"(?i)\b(?:play\s+.*(?:song|music|track)|play\s+.*on\s+youtube|youtube\s+(?:pe|par).*chalao|gaana\s+chalao|search\s+youtube\s+for)\b", clean) or
            ("youtube" in lower and any(w in lower for w in ["play", "search", "song", "gaana", "music"]))
        )
        if is_media:
            m_yt = re.search(r"(?i)\b(?:play\s+(?:a\s+)?(.*)\s+(?:song|music|track)|go\s+to\s+youtube\s+and\s+(?:play|search(?:\s+for)?)\s+(.*)|(?:search\s+for\s+|play\s+)?(.*)\s+on\s+youtube|youtube\s+(?:pe\s+|par\s+)(.*)\s+(?:chalao|play\s+karo|search\s+karo)|search\s+youtube\s+for\s+(.*)|(.*)\s+(?:ka\s+gaana|song)\s+(?:chalao|play\s+karo))\b", clean)
            if m_yt:
                query = (m_yt.group(1) or m_yt.group(2) or m_yt.group(3) or m_yt.group(4) or m_yt.group(5) or m_yt.group(6) or "").strip()
            else:
                query = re.sub(r"(?i)\b(?:youtube|play|song|music|gaana|search|for|chalao|karo)\b", "", clean).strip()

            query = query or "Trending Songs"
            return ActionIntent(
                intent=ActionIntentType.PLAY_MEDIA,
                goal=f"Play '{query}' on YouTube",
                channel="Browser",
                application="YouTube",
                target=query,
                destination="YouTube",
                content=query,
                parameters={"query": query},
                execution_required=True,
                requires_browser=True,
                verification_required=True,
            )

        # -------------------------------------------------------------
        # 4. VISION GROUNDING & SCREEN REASONING (Phase 1)
        # -------------------------------------------------------------
        # 4A. Error Diagnosis on Screen
        is_diagnose_error = bool(
            re.search(r"(?i)\b(?:look\s+at\s+(?:my\s+)?screen\s+and\s+(?:tell|find|fix|check|diagnose|explain)|explain\s+(?:the\s+|this\s+)?error|why\s+(?:is\s+my\s+code\s+failing|did\s+it\s+fail|is\s+there\s+an\s+error)|diagnose\s+(?:the\s+)?(?:screen|error)|what\s+is\s+wrong\s+with\s+(?:my\s+code|this))\b", clean) or
            re.search(r"(?i)\b(?:screen\s+pe\s+(?:kya\s+error\s+hai|error\s+dekho|error\s+batao|kya\s+gadbad\s+hai)|meri\s+screen\s+dekho\s+aur\s+error\s+batao|error\s+solve\s+karo\s+screen\s+dekh\s+ke)\b", clean) or
            re.search(r"(?i)\b(?:read\s+this\s+error|read\s+the\s+error\s+on\s+screen)\b", clean)
        )
        if is_diagnose_error:
            return ActionIntent(
                intent=ActionIntentType.DIAGNOSE_SCREEN_ERROR,
                goal="Inspect active window and diagnose visible errors",
                channel="Vision Grounding",
                application="Screen Vision",
                target="Active Error",
                parameters={"query": clean},
                execution_required=True,
                verification_required=True,
            )

        # 4B. Find Interactive UI Elements (Buttons, Inputs)
        m_find_ui = re.search(r"(?i)\b(?:find|locate)\s+(?:the\s+)?([A-Za-z0-9_\-\s]+?)\s+(?:button|input|icon|control|link)\s+on\s+screen\b", clean) or \
                    re.search(r"(?i)\bscreen\s+pe\s+([A-Za-z0-9_\-\s]+?)\s+(?:button|dhoondo|kahan\s+hai)\b", clean)
        if m_find_ui:
            elem_name = m_find_ui.group(1).strip()
            return ActionIntent(
                intent=ActionIntentType.FIND_UI_ELEMENT,
                goal=f"Locate '{elem_name}' control on screen",
                channel="Vision Grounding",
                application="Screen Vision",
                target=elem_name,
                parameters={"element": elem_name},
                execution_required=True,
                verification_required=True,
            )

        # 4C. General Screen Analysis & Summarization ("look at my screen", "screen pe kya hai")
        is_screen_analysis = bool(
            re.search(r"(?i)\b(?:look\s+at\s+(?:my\s+)?screen|what\s+is\s+on\s+my\s+screen|summarize\s+(?:what\s+is\s+on\s+)?(?:my\s+)?screen|explain\s+what\s+is\s+on\s+(?:my\s+)?screen|inspect\s+(?:the\s+)?active\s+window|read\s+my\s+screen)\b", clean) or
            re.search(r"(?i)\b(?:meri\s+screen\s+dekho|screen\s+(?:pe\s+)?kya\s+hai(?:\s+batao)?|screen\s+(?:ko\s+)?summarize\s+karo|screen\s+(?:ko\s+)?explain\s+karo|screen\s+dekho)\b", clean)
        )
        if is_screen_analysis and not any(w in clean_lower for w in ["screenshot le lo", "screenshot kheecho", "capture screen"]):
            return ActionIntent(
                intent=ActionIntentType.ANALYZE_SCREEN,
                goal="Capture and visually analyze current display content",
                channel="Vision Grounding",
                application="Screen Vision",
                target="Screen",
                parameters={"query": clean, "focus_active_window": True},
                execution_required=True,
                verification_required=True,
            )

        # -------------------------------------------------------------
        # 5. BROWSER WEB SEARCH
        # -------------------------------------------------------------
        m_search = re.search(r"(?i)\b(?:open\s+(?:chrome|browser|edge)\s+(?:and|aur)\s+search(?:\s+for)?\s+(.*))\b", clean) or \
                   re.search(r"(?i)\b(?:search\s+(?:for\s+)?(.*)\s+on\s+(?:google|chrome|browser|web|bing))\b", clean) or \
                   re.search(r"(?i)\b(?:google|search)\s+(?:for\s+)?(.*)\b", clean)
        if m_search and not any(w in lower for w in ["create", "build", "make", "banao", "likho"]):
            search_query = m_search.group(1).strip()
            if search_query:
                return ActionIntent(
                    intent=ActionIntentType.SEARCH_WEB,
                    goal=f"Search web for '{search_query}'",
                    channel="Browser",
                    application="Browser",
                    target=search_query,
                    content=search_query,
                    parameters={"query": search_query},
                    execution_required=True,
                    requires_browser=True,
                    verification_required=True,
                )

        # -------------------------------------------------------------
        # 6. PROJECT CREATION / CODING ACTIONS (Phase 6)
        # -------------------------------------------------------------
        is_coding = bool(
            re.search(r"(?i)\b(?:open\s+(?:vs\s*code|vscode|ide)\s*(?:and|aur|,)?\s*)?(?:create|build|make|develop|implement|generate|design|write)\s+.*(?:app|application|project|website|calculator|tracker|system|dashboard|timer|todo|game|api|script|program|code)", clean) or
            re.search(r"(?i)\b(?:python|cpp|c\+\+|java|react|html|css|flask|fastapi)\b.*(?:banao|create|build|write|implement|run)", clean) or
            re.search(r"(?i)\b(?:banao|bana\s+do|create\s+karo|develop\s+karo)\b", clean) or
            re.search(r"(?i)\b(?:calculator|tracker|todo|dashboard|timer|system|game|app|project|website|api)\s+(?:with\s+.*ui|and\s+run|aur\s+run|banao|likho)\b", clean)
        )
        if is_coding:
            return ActionIntent(
                intent=ActionIntentType.CREATE_PROJECT,
                goal=clean,
                channel="Project Agent",
                application="Visual Studio Code",
                target="Project",
                execution_required=True,
                verification_required=True,
            )

        # -------------------------------------------------------------
        # 7. SYSTEM CONTROLS (Screenshot, Volume, System Info)
        # -------------------------------------------------------------
        if bool(re.search(r"(?i)\b(?:take|capture)\s+(?:a\s+)?screenshot|screenshot\s+(?:le\s+lo|kheecho|lo)\b", clean)):
            return ActionIntent(
                intent=ActionIntentType.TAKE_SCREENSHOT,
                goal="Capture display screenshot",
                channel="System",
                application="Screen Capture",
                target="Screen",
                execution_required=True,
                verification_required=True,
            )

        if bool(re.search(r"(?i)\b(?:increase|decrease|raise|lower|turn\s+up|turn\s+down|mute|unmute)\s+volume|volume\s+(?:kam|badhao|mute)\b", clean)):
            return ActionIntent(
                intent=ActionIntentType.SET_VOLUME,
                goal="Adjust system audio volume",
                channel="System",
                target="Volume",
                execution_required=True,
                verification_required=False,
            )

        if bool(re.search(r"(?i)\b(?:how\s+much\s+(?:ram|storage|disk)|what\s+is\s+my\s+(?:ram|gpu|cpu|os|specs)|system\s+info)\b", clean)):
            return ActionIntent(
                intent=ActionIntentType.GET_SYSTEM_INFO,
                goal="Retrieve system specifications",
                channel="System",
                target="Specs",
                execution_required=True,
                verification_required=False,
            )

        # -------------------------------------------------------------
        # 7. FILESYSTEM OPERATIONS (Create, Delete, Open Folders/Files)
        # -------------------------------------------------------------
        m_desk_folder = re.search(r"(?i)\bcreate\s+(?:a\s+)?folder\s+(?:called|named)?\s*([A-Za-z0-9_\-]+)\s+(?:on|in)\s+(?:my\s+)?desktop\b", clean)
        if m_desk_folder:
            f_name = m_desk_folder.group(1).strip()
            return ActionIntent(
                intent=ActionIntentType.CREATE_FOLDER,
                goal=f"Create folder '{f_name}' on Desktop",
                channel="Filesystem",
                target=f_name,
                destination="Desktop",
                parameters={"name": f_name, "location": "Desktop"},
                execution_required=True,
                verification_required=True,
            )

        m_create_file = re.search(r"(?i)\b(?:create|make)\s+(?:a\s+)?(?:text\s+)?file\s+([A-Za-z0-9_\-\.]+)(?:\s+with\s+(?:content|text)\s+(.*))?", clean)
        if m_create_file:
            fname = m_create_file.group(1).strip()
            fcont = m_create_file.group(2).strip() if m_create_file.group(2) else ""
            return ActionIntent(
                intent=ActionIntentType.CREATE_FILE,
                goal=f"Create file '{fname}'",
                channel="Filesystem",
                target=fname,
                content=fcont,
                parameters={"name": fname, "content": fcont},
                execution_required=True,
                verification_required=True,
            )

        m_del_folder = re.search(r"(?i)\b(?:delete|remove)\s+(?:the\s+)?folder\s+([A-Za-z0-9_\-\.\s]+)", clean)
        if m_del_folder:
            f_target = m_del_folder.group(1).strip()
            return ActionIntent(
                intent=ActionIntentType.DELETE_FOLDER,
                goal=f"Delete folder '{f_target}'",
                channel="Filesystem",
                target=f_target,
                parameters={"target": f_target},
                execution_required=True,
                requires_confirmation=True,
                verification_required=True,
            )

        m_del_file = re.search(r"(?i)\b(?:delete|remove)\s+(?:the\s+)?file\s+([A-Za-z0-9_\-\.\s]+)", clean)
        if m_del_file:
            f_target = m_del_file.group(1).strip()
            return ActionIntent(
                intent=ActionIntentType.DELETE_FILE,
                goal=f"Delete file '{f_target}'",
                channel="Filesystem",
                target=f_target,
                parameters={"target": f_target},
                execution_required=True,
                requires_confirmation=True,
                verification_required=True,
            )

        # -------------------------------------------------------------
        # 8. DYNAMIC RESOURCE & APPLICATION OPENING ("open X", "launch X", "start X", "X kholo", "X chalao")
        # -------------------------------------------------------------
        # English verb-first pattern: "open X", "launch X", "start X", "run X"
        m_open_en = re.search(r"(?i)^(?:open|launch|start|run|visit|go\s+to|show)\s+(?:the\s+|app\s+|my\s+|a\s+|an\s+)?([A-Za-z0-9_\-\.\:\/\s]+)$", clean)
        # Hindi/Hinglish target-first pattern: "X kholo", "X chalao", "X open karo"
        m_open_hi = re.search(r"(?i)^([A-Za-z0-9_\-\.\:\/\s]+?)(?:\s+ko|\s+app)?\s+(?:kholo|chalao|open\s+karo|chala\s+do|dikhao|खोलो|चलाओ|दिखाओ)\s*[.!?]?$", clean)

        m_open = m_open_en or m_open_hi
        if m_open:
            raw_target = m_open.group(1).strip().rstrip(".!? \t\n")

            # Reject compound instructions that belong to multi-step planning or project generation
            if re.search(r"(?i)\b(?:and|aur|with|to|saying|build|create|make|develop|write|run|search|type|implement|banao|likho)\b", raw_target):
                raw_target = ""

        if m_open and raw_target:
            # Check if this target is explicitly a folder / directory
            is_explicit_folder = (
                "folder" in clean_lower
                or "directory" in clean_lower
                or raw_target.lower().strip() in ["downloads", "documents", "desktop", "pictures", "videos", "music", "home", "explorer", "file explorer"]
                or raw_target.startswith(("/", "\\", "C:", "./", "../"))
            )
            if is_explicit_folder:
                f_name = re.sub(r"(?i)\b(?:folder|directory|a|an|the|my|this|some)\b", "", raw_target).strip()
                if not f_name or f_name.lower() in ("folder", "directory", "dir", "a", "the", "my"):
                    f_name = "Desktop"
                return ActionIntent(
                    intent=ActionIntentType.OPEN_FOLDER,
                    goal=f"Open folder '{f_name}'",
                    channel="Filesystem",
                    target=f_name,
                    destination=f_name,
                    parameters={"target": f_name},
                    execution_required=True,
                    verification_required=True,
                )

            # Check if this target is a URL / Website / Web Service
            if ResourceDiscovery.is_url(raw_target) or "browser" in clean_lower or "website" in clean_lower or "site" in clean_lower or "web" in clean_lower:
                target_url = ResourceDiscovery.to_url(raw_target)
                site_name = raw_target.title()
                return ActionIntent(
                    intent=ActionIntentType.OPEN_URL,
                    goal=f"Open {site_name} in browser",
                    channel="Browser",
                    application="Browser",
                    target=site_name,
                    destination=target_url,
                    parameters={"url": target_url, "site_name": site_name},
                    execution_required=True,
                    requires_browser=True,
                    verification_required=True,
                )

            # Default to dynamic application launch
            app_title = raw_target.title()
            if raw_target.lower() in ("vs code", "vscode", "visual studio code"):
                app_title = "Visual Studio Code"
            elif raw_target.lower() in ("chrome", "google chrome"):
                app_title = "Google Chrome"
            elif raw_target.lower() in ("edge", "microsoft edge"):
                app_title = "Microsoft Edge"

            return ActionIntent(
                intent=ActionIntentType.OPEN_APPLICATION,
                goal=f"Open {app_title}",
                channel="Desktop",
                application=app_title,
                target=raw_target,
                destination=raw_target,
                parameters={"application": raw_target, "target": raw_target},
                execution_required=True,
                verification_required=True,
            )

        # -------------------------------------------------------------
        # 9. CLOSE APPLICATION COMMANDS
        # -------------------------------------------------------------
        m_close_app = re.search(r"(?i)^(?:close|exit|terminate|kill|shut\s+down)\s+(?:the\s+|app\s+)?([A-Za-z0-9_\-\s]+)", clean) or \
                      re.search(r"(?i)^([A-Za-z0-9_\-\s]+?)\s+(?:band\s+karo|band\s+kar\s+do|close\s+karo|बंद\s+करो)(?:\s+|$|[.,!?])", clean)
        if m_close_app:
            app_target = m_close_app.group(1).strip().rstrip(".!? \t\n")
            if app_target.lower() not in {"this", "window", "folder", "chitti"}:
                return ActionIntent(
                    intent=ActionIntentType.CLOSE_APPLICATION,
                    goal=f"Close {app_target}",
                    channel="Desktop",
                    application=app_target,
                    target=app_target,
                    parameters={"target": app_target},
                    execution_required=True,
                    verification_required=True,
                )

        # -------------------------------------------------------------
        # 10. FALLBACK (Conversational)
        # -------------------------------------------------------------
        return ActionIntent(
            intent=ActionIntentType.CONVERSATIONAL,
            goal=clean,
            channel="Chat",
            target="Conversation",
            execution_required=False,
            verification_required=False,
        )
