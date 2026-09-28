"""
Chitti Action Intent & User Goal Extraction Engine (Phase 5 & 6).
Analyzes natural language requests to extract structured user goals, channels,
targets, recipients, message content, preconditions, and safety requirements.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from src.utils.logging import log_chitti, log_debug, log_info


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

    # Filesystem Actions
    CREATE_FILE = "CREATE_FILE"
    CREATE_FOLDER = "CREATE_FOLDER"
    DELETE_FILE = "DELETE_FILE"
    DELETE_FOLDER = "DELETE_FOLDER"
    OPEN_FOLDER = "OPEN_FOLDER"

    # Project & Coding Actions (Phase 6)
    CREATE_PROJECT = "CREATE_PROJECT"
    MODIFY_PROJECT = "MODIFY_PROJECT"
    RUN_PROJECT = "RUN_PROJECT"
    DEBUG_PROJECT = "DEBUG_PROJECT"
    TEST_PROJECT = "TEST_PROJECT"

    # System Control Actions
    TAKE_SCREENSHOT = "TAKE_SCREENSHOT"
    SET_VOLUME = "SET_VOLUME"
    GET_SYSTEM_INFO = "GET_SYSTEM_INFO"

    # Fallback
    CONVERSATIONAL = "CONVERSATIONAL"
    UNKNOWN = "UNKNOWN"


@dataclass
class ActionIntent:
    intent: ActionIntentType
    goal: str
    channel: Optional[str] = None
    application: Optional[str] = None
    target: Optional[str] = None
    recipient: Optional[str] = None
    content: Optional[str] = None
    subject: Optional[str] = None
    attachments: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    preconditions: List[str] = field(default_factory=list)
    requires_browser: bool = False
    requires_authentication: bool = False
    requires_confirmation: bool = False
    verification_required: bool = True

    def log_intent(self):
        """Emits structured observable logs for this action intent."""
        log_chitti(f"[CHITTI] [GOAL]")
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

        # Also emit standard analyzer tags
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
    Extracts structured user goals and entities from arbitrary user requests.
    Decoupled from application-specific hardcoding.
    """

    @classmethod
    def _normalize_text(cls, text: str) -> str:
        """Corrects typos in common action verbs and application names."""
        normalized = text
        normalized = re.sub(r"(?i)\b(?:opem|opne|oppen|oepn)\b", "open", normalized)
        normalized = re.sub(r"(?i)\b(?:lauch|luanch|lanuch)\b", "launch", normalized)
        normalized = re.sub(r"(?i)\b(?:messag|mesage|mesg|msg)\b", "message", normalized)
        normalized = re.sub(r"(?i)\b(?:serach|sreach)\b", "search", normalized)
        normalized = re.sub(r"(?i)\b(?:watsapp|whatapp|whatspp|whatsap|watsap|wtsp)\b", "whatsapp", normalized)
        normalized = re.sub(r"(?i)\b(?:vscdoe|vscde)\b", "vscode", normalized)
        normalized = re.sub(r"(?i)\b(?:youtub|yotube|utube)\b", "youtube", normalized)
        normalized = re.sub(r"(?i)\b(?:chrone|chorme|crm)\b", "chrome", normalized)
        normalized = re.sub(r"(?i)\b(?:notepd|notepadd)\b", "notepad", normalized)
        return normalized

    @classmethod
    def extract_intent(cls, user_text: str) -> ActionIntent:
        """
        Analyzes the user's request and returns a structured ActionIntent.
        """
        raw = user_text.strip()
        normalized = cls._normalize_text(raw)
        lower = normalized.lower()

        # Clean conversational prefixes
        clean = re.sub(r"^(?:chitti,?\s*|hey chitti,?\s*|bhai,?\s*|please\s+)", "", normalized, flags=re.IGNORECASE).strip()
        clean_lower = clean.lower()

        # -------------------------------------------------------------
        # 1. EMAIL DRAFTING vs EMAIL SENDING
        # -------------------------------------------------------------
        email_addr_match = re.search(r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})", clean)

        # 1A. Explicit Email Drafting ("write an email", "draft an email", "compose an email for me") WITHOUT actual send command
        is_draft_only = bool(re.search(r"(?i)\b(?:write|draft|generate|compose|suggest)\s+(?:an?\s+)?email\b", clean)) and not email_addr_match and not re.search(r"(?i)\b(?:send|shoot|dispatch|mail\s+this|email\s+this)\b", clean)
        if is_draft_only:
            m_target = re.search(r"(?i)\b(?:to|for)\s+([a-zA-Z0-9_\-\s]+?)(?:\s+asking|\s+about|\s+regarding|\s+with|\s*$)", clean)
            target_name = m_target.group(1).strip() if m_target else "Recipient"
            return ActionIntent(
                intent=ActionIntentType.WRITE_EMAIL,
                goal=f"Draft an email to {target_name}",
                channel="Email",
                target=target_name,
                requires_browser=False,
                requires_authentication=False,
                requires_confirmation=False,
                verification_required=False,
            )

        # 1B. Real Actionable Email Sending ("send email to...", "email <msg> to <recipient>...", "mail this to...")
        if email_addr_match or bool(re.search(r"(?i)\b(?:send\s+(?:an?\s+)?email|email\s+(?:this|the|a)?|mail\s+(?:this|the|a)?)\b", clean)):
            recipient = email_addr_match.group(1) if email_addr_match else ""
            if not recipient:
                m_rec = re.search(r"(?i)\b(?:to|ko)\s+([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}|[a-zA-Z0-9_\-]+)", clean)
                recipient = m_rec.group(1).strip() if m_rec else "Recipient"

            # Extract message content / body
            content = ""
            m_saying = re.search(r"(?i)\b(?:with\s+message|saying|body|with\s+body)\s+([\"']?.+?[\"']?)(?:\s+from\s+my\s+side|\s*$)", clean)
            if m_saying:
                content = m_saying.group(1).strip().strip("\"'")
            else:
                m_msg_body = re.search(r"(?i)\b(?:email|send|mail)\s+(?:message\s+|msg\s+)?([\"'][^\"']+[\"'])(?:\s+(?:message\s+|msg\s+)?to\b|\s*$)", clean) or \
                             re.search(r"(?i)\b(?:email|send|mail)\s+(?:message\s+|msg\s+)?(?!(?:an?\s+)?(?:email|mail|message)\s+to)(.+?)\s+(?:message\s+|msg\s+)?to\s+", clean)
                if m_msg_body:
                    content = m_msg_body.group(1).strip().strip("\"'")
                    content = re.sub(r"(?i)\s+(?:message|msg)$", "", content).strip()
            
            if not content or content.lower() in ("an email", "email", "a message", "message", "this"):
                content = "Hello, I am reaching out to you."

            # Clean trailing instructions like 'from my side'
            content = re.sub(r"(?i)\s+from\s+my\s+side$", "", content).strip()

            subject = "Message from Chitti"
            m_sub = re.search(r"(?i)\b(?:subject|regarding|about)\s+[\"']?([^\"',]+)[\"']?", clean)
            if m_sub:
                subject = m_sub.group(1).strip().title()

            return ActionIntent(
                intent=ActionIntentType.SEND_EMAIL,
                goal=f"Send email to {recipient}",
                channel="Email",
                application="Gmail / Webmail",
                recipient=recipient,
                target=recipient,
                content=content,
                subject=subject,
                requires_browser=True,
                requires_authentication=True,
                requires_confirmation=True,
                verification_required=True,
            )

        # -------------------------------------------------------------
        # 2. MESSAGING & CHAT APPS (WhatsApp, Telegram, Slack, Discord)
        # -------------------------------------------------------------
        is_messaging = bool(
            re.search(r"(?i)\b(?:whatsapp|watsapp|telegram|slack|discord)\b", clean) or
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
            else:
                channel, app, base_url = "WhatsApp", "WhatsApp Web", "https://web.whatsapp.com"

            # Check if this is merely opening the service or sending a real message
            m_msg_full = re.search(r"(?i)\b(?:message|msg|send(?:\s+a)?\s+message|send)\s+([\"']?[^\"']+?[\"']?)\s+to\s+([a-zA-Z0-9_\-\s]+)\b", clean) or \
                         re.search(r"(?i)\b(?:open\s+.*(?:and|aur)\s+)?(?:message|msg|send(?:\s+a)?\s+message|send)\s+([\"']?[^\"']+?[\"']?)\s+to\s+([a-zA-Z0-9_\-\s]+)\b", clean) or \
                         re.search(r"(?i)\b([a-zA-Z0-9_\-]+)\s+ko\s+([\"']?[^\"']+?[\"']?)\s*(?:message\s+karo|bhejo|send\s+karo|message\s+kar|msg\s+bhejo)\b", clean) or \
                         re.search(r"(?i)\b(?:whatsapp|watsapp)\s+(?:par|pe)?\s*([a-zA-Z0-9_\-]+)\s+ko\s+([\"']?[^\"']+?[\"']?)\s*(?:message\s+karo|bhejo|send\s+karo|message\s+kar)\b", clean)

            if m_msg_full:
                g1 = (m_msg_full.group(1) or "").strip().strip("\"'")
                g2 = (m_msg_full.group(2) or "").strip().strip("\"'")

                # Distinguish contact vs message content
                if any(w in g1.lower() for w in ["hi", "hello", "hey", "namaste", "good", "how", "what", "meet", "doc", "link", "thanks", "ok"]):
                    msg_text, contact_name = g1, g2
                else:
                    contact_name, msg_text = g1, g2

                # Clean message text and contact name
                msg_text = re.sub(r"(?i)\s+(?:message|msg|text)$", "", msg_text).strip() or "Hi"
                contact_name = re.sub(r"(?i)\s+(?:on|via|using)\s+.*$", "", contact_name).strip()
                contact_name = re.sub(r"(?i)\s+(?:message|msg|text)$", "", contact_name).strip() or "Contact"

                return ActionIntent(
                    intent=ActionIntentType.SEND_MESSAGE,
                    goal=f"Send \"{msg_text}\" to {contact_name} using {app}",
                    channel=channel,
                    application=app,
                    target=contact_name,
                    content=msg_text,
                    parameters={"url": base_url, "contact": contact_name, "text": msg_text},
                    requires_browser=True,
                    requires_authentication=True,
                    requires_confirmation=False,
                    verification_required=True,
                )
            else:
                # Merely opening the messaging service
                return ActionIntent(
                    intent=ActionIntentType.OPEN_URL,
                    goal=f"Open {app} in browser",
                    channel=channel,
                    application=app,
                    target=app,
                    parameters={"url": base_url},
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
                content=query,
                parameters={"query": query},
                requires_browser=True,
                verification_required=True,
            )

        # -------------------------------------------------------------
        # 4. BROWSER WEB SEARCH & URL NAVIGATION
        # -------------------------------------------------------------
        m_search = re.search(r"(?i)\b(?:open\s+(?:chrome|browser|edge)\s+(?:and|aur)\s+search(?:\s+for)?\s+(.*))\b", clean) or \
                   re.search(r"(?i)\b(?:search\s+(?:for\s+)?(.*)\s+on\s+(?:google|chrome|browser|web|bing))\b", clean) or \
                   re.search(r"(?i)\b(?:google|search)\s+(?:for\s+)?(.*)\b", clean)
        if m_search and not any(w in lower for w in ["youtube", "create", "build", "make"]):
            search_query = m_search.group(1).strip()
            return ActionIntent(
                intent=ActionIntentType.SEARCH_WEB,
                goal=f"Search web for '{search_query}'",
                channel="Browser",
                application="Chrome",
                target=search_query,
                content=search_query,
                parameters={"query": search_query},
                requires_browser=True,
                verification_required=True,
            )

        # Direct Web URL or famous site navigation (e.g. open youtube, open reddit, open github)
        m_site = re.search(r"(?i)\b(?:open|launch|visit|navigate|go\s+to)\s+(https?://\S+|www\.\S+|[a-zA-Z0-9_\-\.]+\.[a-zA-Z]{2,}(?:/\S*)?|youtube|reddit|github|twitter|x\.com|wikipedia|amazon|netflix|spotify|chatgpt)\b", clean)
        if m_site:
            site_raw = m_site.group(1).lower().strip()
            url_map = {
                "youtube": "https://youtube.com",
                "reddit": "https://reddit.com",
                "github": "https://github.com",
                "twitter": "https://twitter.com",
                "x.com": "https://x.com",
                "wikipedia": "https://wikipedia.org",
                "amazon": "https://amazon.com",
                "netflix": "https://netflix.com",
                "spotify": "https://open.spotify.com",
                "chatgpt": "https://chatgpt.com",
            }
            target_url = url_map.get(site_raw, site_raw if site_raw.startswith("http") else f"https://{site_raw}")
            site_name = site_raw.title()
            return ActionIntent(
                intent=ActionIntentType.OPEN_URL,
                goal=f"Open {site_name} in browser",
                channel="Browser",
                application="Browser",
                target=site_name,
                parameters={"url": target_url, "site_name": site_name},
                requires_browser=True,
                verification_required=True,
            )

        # -------------------------------------------------------------
        # 5. DESKTOP APPLICATION LAUNCH & CONTROLS
        # -------------------------------------------------------------
        m_notepad_type = re.search(r"(?i)\bopen\s+notepad\s+(?:and|aur)\s+type\s+(.*)", clean) or \
                         re.search(r"(?i)\bnotepad\s+(?:kholo|open\s+karo)\s+aur\s+(?:type\s+karo\s+|likho\s+)(.*)", clean)
        if m_notepad_type:
            text_val = m_notepad_type.group(1).strip()
            return ActionIntent(
                intent=ActionIntentType.TYPE_TEXT,
                goal=f"Open Notepad and type '{text_val}'",
                channel="Desktop",
                application="Notepad",
                target="Notepad",
                content=text_val,
                parameters={"target": "Notepad", "text": text_val},
                verification_required=True,
            )

        m_app = re.search(r"(?i)\b(?:open|launch|kholo|chalao)\s+(vs\s*code|vscode|notepad|chrome|browser|edge|calculator|terminal|powershell|cmd|explorer)\b", clean)
        if m_app:
            app_raw = m_app.group(1).lower().strip()
            app_map = {
                "vs code": "Visual Studio Code",
                "vscode": "Visual Studio Code",
                "notepad": "Notepad",
                "chrome": "Google Chrome",
                "browser": "Browser",
                "edge": "Microsoft Edge",
                "calculator": "Calculator",
                "terminal": "Terminal",
                "powershell": "PowerShell",
                "cmd": "Command Prompt",
                "explorer": "File Explorer",
            }
            app_name = app_map.get(app_raw, app_raw.title())
            return ActionIntent(
                intent=ActionIntentType.OPEN_APPLICATION,
                goal=f"Open {app_name}",
                channel="Desktop",
                application=app_name,
                target=app_name,
                parameters={"application": app_name},
                verification_required=True,
            )

        # -------------------------------------------------------------
        # 6. SYSTEM CONTROLS (Screenshot, Volume)
        # -------------------------------------------------------------
        if bool(re.search(r"(?i)\b(?:take|capture)\s+(?:a\s+)?screenshot|screenshot\s+(?:le\s+lo|kheecho)\b", clean)):
            return ActionIntent(
                intent=ActionIntentType.TAKE_SCREENSHOT,
                goal="Capture display screenshot",
                channel="System",
                application="Screen Capture",
                target="Screen",
                verification_required=True,
            )

        # -------------------------------------------------------------
        # 7. FILESYSTEM OPERATIONS
        # -------------------------------------------------------------
        m_desk_folder = re.search(r"(?i)\bcreate\s+(?:a\s+)?folder\s+(?:called|named)?\s*([A-Za-z0-9_\-]+)\s+(?:on|in)\s+(?:my\s+)?desktop\b", clean)
        if m_desk_folder:
            f_name = m_desk_folder.group(1).strip()
            return ActionIntent(
                intent=ActionIntentType.CREATE_FOLDER,
                goal=f"Create folder '{f_name}' on Desktop",
                channel="Filesystem",
                target=f_name,
                parameters={"name": f_name, "location": "Desktop"},
                verification_required=True,
            )

        # -------------------------------------------------------------
        # 8. PROJECT CREATION / CODING ACTIONS (Phase 6)
        # -------------------------------------------------------------
        is_coding = bool(
            re.search(r"(?i)\b(?:create|build|make|develop|implement|generate|design|write)\s+.*(?:app|application|project|website|calculator|tracker|system|dashboard|timer|todo|game|api|script|program)", clean) or
            re.search(r"(?i)\b(?:python|cpp|c\+\+|java|react|html|css|flask|fastapi)\b.*(?:banao|create|build|write|implement|run)", clean) or
            re.search(r"(?i)\b(?:banao|bana\s+do|create\s+karo|develop\s+karo)\b", clean)
        )
        if is_coding:
            return ActionIntent(
                intent=ActionIntentType.CREATE_PROJECT,
                goal=clean,
                channel="Project Agent",
                application="Visual Studio Code",
                target="Project",
                verification_required=True,
            )

        # -------------------------------------------------------------
        # 9. FALLBACK
        # -------------------------------------------------------------
        return ActionIntent(
            intent=ActionIntentType.CONVERSATIONAL,
            goal=clean,
            channel="Chat",
            target="Conversation",
            verification_required=False,
        )
