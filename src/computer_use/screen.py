"""
Chitti Screen Observation & Scene Understanding.
Inspects screen state, active windows, loaded web applications, and authentication/login requirements.
"""

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil

from src.computer_use.controller import ComputerController, WindowInfo
from src.utils.logging import log_info, log_warn


@dataclass
class ScreenObservation:
    """Empirical observation of the screen state following an action."""
    screenshot_path: str
    screen_width: int
    screen_height: int
    active_window: Optional[WindowInfo]
    visible_windows: List[WindowInfo]
    is_browser_open: bool = False
    active_service: Optional[str] = None
    auth_required: bool = False
    auth_service: Optional[str] = None
    summary_text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class ScreenObserver:
    """Inspects screen and application state for closed-loop execution."""

    APP_PROCESS_ALIASES = {
        "chrome": ["chrome.exe", "chrome", "google chrome"],
        "google chrome": ["chrome.exe", "chrome", "google chrome"],
        "edge": ["msedge.exe", "msedge", "microsoft edge"],
        "brave": ["brave.exe", "brave"],
        "notepad": ["notepad.exe", "notepad"],
        "calculator": ["calculator.exe", "calculatorapp.exe", "calc.exe"],
        "terminal": ["powershell.exe", "cmd.exe", "windowsterminal.exe", "wt.exe"],
        "vs code": ["code.exe", "code", "visual studio code", "vscode"],
        "vscode": ["code.exe", "code", "visual studio code", "vscode"],
        "whatsapp web": ["whatsapp", "whatsapp web", "chrome.exe", "msedge.exe", "brave.exe"],
        "whatsapp": ["whatsapp", "whatsapp web", "chrome.exe", "msedge.exe", "brave.exe"],
        "gmail": ["gmail", "inbox", "mail.google.com", "chrome.exe", "msedge.exe", "brave.exe"],
        "gmail / webmail": ["gmail", "inbox", "mail.google.com", "chrome.exe", "msedge.exe", "brave.exe"],
        "outlook": ["outlook.exe", "outlook", "chrome.exe", "msedge.exe", "brave.exe"],
        "youtube": ["youtube", "chrome.exe", "msedge.exe", "brave.exe"],
    }

    AUTH_KEYWORDS = [
        "scan qr code", "link a device", "sign in", "login", "log in", "enter password",
        "choose an account", "verify your identity", "authentication required",
    ]

    def __init__(self, controller: ComputerController):
        self.controller = controller

    def observe(self, capture_screenshot: bool = True) -> ScreenObservation:
        """Captures screen and evaluates active application, window hierarchy, and auth state."""
        ss_path = self.controller.screenshot() if capture_screenshot else ""
        width, height = self.controller.get_screen_size()
        active = self.controller.get_active_window()
        visible = self.controller.list_windows()

        active_title = active.title.lower() if active and active.title else ""
        all_titles = [w.title for w in visible if w.title]

        # Check for active web / communication services
        active_service = None
        if any("whatsapp" in t.lower() for t in all_titles) or "whatsapp" in active_title:
            active_service = "WhatsApp Web"
        elif any("gmail" in t.lower() or "inbox" in t.lower() for t in all_titles) or "gmail" in active_title:
            active_service = "Gmail / Webmail"
        elif any("outlook" in t.lower() for t in all_titles) or "outlook" in active_title:
            active_service = "Outlook"
        elif any("youtube" in t.lower() for t in all_titles) or "youtube" in active_title:
            active_service = "YouTube"

        # Check for authentication / QR / login prompts
        auth_req = False
        auth_svc = None
        for kw in self.AUTH_KEYWORDS:
            if kw in active_title or any(kw in t.lower() for t in all_titles):
                auth_req = True
                auth_svc = active_service
                break

        summary = f"Screen {width}x{height}. Active: '{active.title if active else 'None'}'. Visible: {', '.join(all_titles[:4])}"
        if auth_req:
            summary += f" [AUTH/LOGIN REQUIRED for {auth_svc or 'service'}]"

        log_info(f"[SCREEN OBSERVER] {summary}")

        return ScreenObservation(
            screenshot_path=ss_path,
            screen_width=width,
            screen_height=height,
            active_window=active,
            visible_windows=visible,
            is_browser_open=bool(active_service or any("chrome" in t.lower() or "brave" in t.lower() or "edge" in t.lower() for t in all_titles)),
            active_service=active_service,
            auth_required=auth_req,
            auth_service=auth_svc,
            summary_text=summary,
        )

    def is_window_visible(self, query: str) -> bool:
        """Checks if a window matching query or alias is visible."""
        q = query.lower().strip()
        tokens = [t for t in re.split(r"[\s\/\-_]+", q) if len(t) > 2 and t not in ("web", "app", "window", "application")]
        visible = self.controller.list_windows()

        for w in visible:
            w_title = w.title.lower()
            if q in w_title or (tokens and any(t in w_title for t in tokens)):
                return True
            aliases = self.APP_PROCESS_ALIASES.get(q, [])
            if any(a in w_title for a in aliases):
                return True

        # Fallback process check
        expected_aliases = self.APP_PROCESS_ALIASES.get(q, [q])
        for proc in psutil.process_iter(['name']):
            try:
                pname = (proc.info['name'] or "").lower()
                if q in pname or any(a in pname for a in expected_aliases) or (tokens and any(t in pname for t in tokens)):
                    return True
            except Exception:
                continue

        return False
