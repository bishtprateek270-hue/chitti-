"""
Chitti Screen Analyzer & UI Understanding.
Analyzes screenshots, running windows, and UI state to provide honest, verifiable evidence.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import psutil

from src.agent.computer.controller import ComputerController, WindowInfo
from src.utils.logging import log_debug, log_info, log_warn


@dataclass
class VerificationResult:
    success: bool
    evidence: str
    target: str


@dataclass
class ScreenAnalysisResult:
    screenshot_path: str
    screen_width: int
    screen_height: int
    active_window: Optional[WindowInfo]
    visible_windows: List[WindowInfo]
    summary: str


class ScreenAnalyzer:
    """Inspects screen and UI state for verification and multi-step reasoning."""

    def __init__(self, controller: ComputerController):
        self.controller = controller

    def capture_and_analyze(self) -> ScreenAnalysisResult:
        """Takes a screenshot and inspects the active and visible UI windows."""
        path = self.controller.take_screenshot()
        width, height = self.controller.get_screen_size()
        active = self.controller.get_active_window()
        visible = self.controller.list_windows()

        active_title = active.title if active else "None"
        win_titles = [w.title for w in visible[:5]]

        summary = f"Screen {width}x{height}. Active: '{active_title}'. Visible windows: {', '.join(win_titles)}"
        log_info(f"[SCREEN ANALYZER] {summary}")

        return ScreenAnalysisResult(
            screenshot_path=path,
            screen_width=width,
            screen_height=height,
            active_window=active,
            visible_windows=visible,
            summary=summary,
        )

    APP_PROCESS_ALIASES = {
        "visual studio code": ["code.exe", "code", "visual studio code", "vscode"],
        "vs code": ["code.exe", "code", "visual studio code", "vscode"],
        "vscode": ["code.exe", "code", "visual studio code", "vscode"],
        "chrome": ["chrome.exe", "chrome", "google chrome"],
        "google chrome": ["chrome.exe", "chrome", "google chrome"],
        "edge": ["msedge.exe", "msedge", "microsoft edge"],
        "msedge": ["msedge.exe", "msedge", "microsoft edge"],
        "notepad": ["notepad.exe", "notepad"],
        "calculator": ["calculator.exe", "calculatorapp.exe", "calc.exe"],
        "terminal": ["powershell.exe", "cmd.exe", "windowsterminal.exe", "wt.exe"],
        "youtube": ["chrome.exe", "msedge.exe", "firefox.exe", "brave.exe", "opera.exe", "chrome", "msedge", "firefox", "brave", "opera"],
    }

    def verify_window(self, title_query: str) -> VerificationResult:
        """
        Explicitly verifies whether a window or process matching title_query exists.
        Returns a VerificationResult with success boolean and evidence.
        """
        query = title_query.lower().strip()
        visible = self.controller.list_windows()

        # 1. Check visible window titles
        for w in visible:
            w_title = w.title.lower()
            if query in w_title:
                log_info(f"[VERIFY] Window '{w.title}' detected matching query '{title_query}'")
                return VerificationResult(
                    success=True,
                    evidence=f"Window '{w.title}' is visible on screen.",
                    target=title_query,
                )
            # Check aliases against window title
            aliases = self.APP_PROCESS_ALIASES.get(query, [])
            if any(alias in w_title for alias in aliases):
                log_info(f"[VERIFY] Window '{w.title}' detected matching alias for '{title_query}'")
                return VerificationResult(
                    success=True,
                    evidence=f"Window '{w.title}' is visible on screen.",
                    target=title_query,
                )

        # 2. Check running processes as secondary verification
        expected_aliases = self.APP_PROCESS_ALIASES.get(query, [query])
        for proc in psutil.process_iter(['name']):
            try:
                pname = (proc.info['name'] or "").lower()
                if query in pname or any(alias in pname for alias in expected_aliases):
                    log_info(f"[VERIFY] Running process '{pname}' detected matching '{title_query}'")
                    return VerificationResult(
                        success=True,
                        evidence=f"Application process '{pname}' is actively running.",
                        target=title_query,
                    )
            except Exception:
                continue

        log_warn(f"[VERIFY] Verification FAILED: No window or process matching '{title_query}' found.")
        return VerificationResult(
            success=False,
            evidence=f"Window or process matching '{title_query}' was not detected.",
            target=title_query,
        )

    def verify_window_present(self, title_query: str) -> bool:
        """Boolean check for window presence."""
        res = self.verify_window(title_query)
        return res.success

    def verify_active_window(self, title_query: str) -> bool:
        """Verifies if the foreground window matches title_query."""
        query = title_query.lower().strip()
        active = self.controller.get_active_window()
        if active and query in active.title.lower():
            return True
        return False
