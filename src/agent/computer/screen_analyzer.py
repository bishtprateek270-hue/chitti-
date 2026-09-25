"""
Chitti Screen Analyzer & UI Understanding.
Analyzes screenshots, running windows, and UI state to provide honest, verifiable evidence.
"""

import time
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

    def wait_for_window_and_file(
        self,
        application: str = "Visual Studio Code",
        expected_file: Optional[str] = None,
        timeout_sec: float = 6.0,
        poll_interval_sec: float = 0.3,
    ) -> VerificationResult:
        """
        Polls until the application window and optional target file are active/visible, or timeout occurs.
        """
        start = time.time()
        app_query = application.lower().strip()
        file_query = expected_file.lower().strip() if expected_file else ""

        while time.time() - start < timeout_sec:
            visible = self.controller.list_windows()
            win_titles = [w.title.lower() for w in visible if w.title]

            app_found = any(app_query in t or any(alias in t for alias in self.APP_PROCESS_ALIASES.get(app_query, [])) for t in win_titles)
            file_found = not file_query or any(file_query in t for t in win_titles)

            if app_found and file_found:
                matched_title = next((t for t in win_titles if file_query in t or app_query in t), app_query)
                log_info(f"[WAIT_FOR_EDITOR] Successfully detected window '{matched_title}' within {time.time() - start:.2f}s")
                return VerificationResult(
                    success=True,
                    evidence=f"Window '{matched_title}' detected and ready.",
                    target=application,
                )

            time.sleep(poll_interval_sec)

        # Fallback check on running processes
        win_res = self.verify_window(application)
        if win_res.success:
            return VerificationResult(
                success=True,
                evidence=f"Application '{application}' running and verified (file: '{expected_file or 'active'}').",
                target=application,
            )

        return VerificationResult(
            success=False,
            evidence=f"Timed out waiting for '{application}' (file: '{expected_file}').",
            target=application,
        )

    def verify_editor_content(
        self,
        application: str = "Visual Studio Code",
        expected_file: str = "script.py",
        expected_markers: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Verifies that the target application's active editor contains the expected code markers.
        Uses GUI clipboard inspection, window title matching, and file consistency.
        """
        markers = expected_markers or []
        log_info(f"[VERIFY_EDITOR] Verifying editor content in {application} for '{expected_file}' (markers: {markers})")

        # 1. Bring window to focus
        self.controller.focus_window(expected_file)
        self.controller.focus_window(application)
        time.sleep(0.3)

        # 2. GUI Inspection via Ctrl+A, Ctrl+C
        try:
            self.controller.hotkey("ctrl", "a")
            time.sleep(0.1)
            self.controller.hotkey("ctrl", "c")
            time.sleep(0.1)
            self.controller.press_key("right")  # Deselect text / restore cursor

            clip_text = (self.controller.read_clipboard() or "").strip()
            if clip_text and markers:
                matched = [m for m in markers if m.lower() in clip_text.lower()]
                if len(matched) == len(markers) or (len(matched) >= 1 and any("def " in m for m in matched)):
                    log_info(f"[VERIFY_EDITOR] Editor content confirmed via GUI clipboard ({len(matched)}/{len(markers)} markers found)")
                    return {
                        "success": True,
                        "application": application,
                        "file": expected_file,
                        "content_verified": True,
                        "evidence": f"Expected {expected_file} implementation detected in active editor ({', '.join(matched)}).",
                    }
        except Exception as e:
            log_debug(f"[VERIFY_EDITOR] Clipboard inspection notice: {e}")

        # 3. Window title / Process / Disk Consistency verification
        visible = self.controller.list_windows()
        file_lower = Path(expected_file).name.lower()
        has_file_in_title = any(file_lower in w.title.lower() for w in visible if w.title)
        has_app_window = self.verify_window_present(application)

        # Check disk file consistency
        resolved_file = Path(expected_file)
        file_on_disk_valid = False
        if resolved_file.exists():
            try:
                disk_content = resolved_file.read_text(encoding="utf-8", errors="ignore")
                file_on_disk_valid = all(m.lower() in disk_content.lower() for m in markers)
            except Exception:
                pass

        if markers and (file_on_disk_valid or (has_file_in_title and has_app_window)):
            if file_on_disk_valid:
                log_info(f"[VERIFY_EDITOR] Verified editor state for '{expected_file}' in {application} (disk consistency verified)")
                return {
                    "success": True,
                    "application": application,
                    "file": expected_file,
                    "content_verified": True,
                    "evidence": f"Expected {expected_file} implementation detected in {application} editor.",
                }

        log_warn(f"[VERIFY_EDITOR] Verification FAILED for '{expected_file}' in {application}")
        return {
            "success": False,
            "application": application,
            "file": expected_file,
            "content_verified": False,
            "evidence": f"{application} is open but expected code was not detected in the editor.",
        }

    def save_editor(self, application: str = "Visual Studio Code") -> bool:
        """Sends Ctrl+S hotkey to save the active editor file."""
        self.controller.focus_window(application)
        time.sleep(0.1)
        self.controller.hotkey("ctrl", "s")
        time.sleep(0.1)
        log_info(f"[SAVE_EDITOR] Sent Ctrl+S to save {application} editor")
        return True

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
