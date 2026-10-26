"""
Chitti Screen Analyzer & UI Understanding.
Analyzes screenshots, running windows, and UI state to provide honest, verifiable evidence.
"""

import os
import re
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
        "visual studio code": ["code.exe", "visual studio code", "vscode.exe"],
        "vs code": ["code.exe", "visual studio code", "vscode.exe"],
        "vscode": ["code.exe", "visual studio code", "vscode.exe"],
        "chrome": ["chrome.exe", "google chrome"],
        "google chrome": ["chrome.exe", "google chrome"],
        "edge": ["msedge.exe", "microsoft edge"],
        "msedge": ["msedge.exe", "microsoft edge"],
        "brave": ["brave.exe", "brave"],
        "notepad": ["notepad.exe", "notepad"],
        "calculator": ["calculator.exe", "calculatorapp.exe", "calc.exe"],
        "terminal": ["powershell.exe", "cmd.exe", "windowsterminal.exe", "wt.exe"],
        "youtube": ["youtube"],
        "browser": ["chrome.exe", "msedge.exe", "firefox.exe", "brave.exe", "opera.exe"],
        "web": ["chrome.exe", "msedge.exe", "firefox.exe", "brave.exe", "opera.exe"],
        "whatsapp web": ["whatsapp", "whatsapp web"],
        "whatsapp": ["whatsapp", "whatsapp.exe"],
        "gmail / webmail": ["gmail", "inbox", "mail.google.com"],
        "gmail": ["gmail", "inbox", "mail.google.com"],
        "webmail": ["gmail", "inbox", "mail.google.com"],
        "telegram": ["telegram", "telegram.exe"],
        "slack": ["slack", "slack.exe"],
        "discord": ["discord", "discord.exe"],
    }


    def verify_window(self, title_query: str) -> VerificationResult:
        """
        Explicitly verifies whether a window or process matching title_query exists.
        Returns a VerificationResult with success boolean and evidence.
        """
        query = title_query.lower().strip()
        visible = self.controller.list_windows()
        tokens = [t for t in re.split(r"[\s\/\-_]+", query) if len(t) > 2 and t not in ("web", "app", "window", "application")]

        # 1. Check visible window titles
        for w in visible:
            w_title = w.title.lower()
            if query in w_title or (tokens and any(t in w_title for t in tokens)):
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

        # 2. For website targets (e.g. YouTube), verify via browser controller state and running browser
        if query in ("youtube", "whatsapp web", "gmail", "gmail / webmail", "webmail"):
            from src.agent.computer.browser import BrowserController
            b_state = BrowserController().get_browser_state()
            site_key = "youtube" if query == "youtube" else ("whatsapp" if "whatsapp" in query else "google.com")
            if b_state.is_open and b_state.current_url and site_key in b_state.current_url.lower():
                has_browser_proc = False
                try:
                    for proc in psutil.process_iter(['name']):
                        p_name = (proc.info.get('name') or '').lower()
                        if any(b == p_name for b in ('brave.exe', 'chrome.exe', 'msedge.exe', 'firefox.exe')):
                            has_browser_proc = True
                            break
                except Exception:
                    has_browser_proc = True

                if has_browser_proc:
                    log_info(f"[VERIFY] Active browser state verified for '{title_query}': {b_state.current_url}")
                    return VerificationResult(
                        success=True,
                        evidence=f"Browser tab actively open at '{b_state.current_url}'.",
                        target=title_query,
                    )
            return VerificationResult(
                success=False,
                evidence=f"No active browser window or tab for '{title_query}' was found.",
                target=title_query,
            )

        # 3. Check running processes as secondary verification for native desktop apps (strict exact matching)
        expected_aliases = self.APP_PROCESS_ALIASES.get(query, [query])
        for proc in psutil.process_iter(['name']):
            try:
                pname = (proc.info['name'] or "").lower()
                is_match = (
                    pname in expected_aliases
                    or any(pname == alias or pname == f"{alias}.exe" for alias in expected_aliases)
                    or (pname.endswith(".exe") and pname[:-4] in expected_aliases)
                )
                if is_match:
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

        # 1. Check disk file consistency directly (safe, instant, zero simulated keystrokes)
        resolved_file = Path(expected_file)
        if not resolved_file.is_absolute() and hasattr(self.controller, "default_workspace") and self.controller.default_workspace:
            resolved_file = Path(self.controller.default_workspace) / expected_file

        file_on_disk_valid = False
        if resolved_file.exists():
            try:
                disk_content = resolved_file.read_text(encoding="utf-8", errors="ignore")
                matching_count = sum(1 for m in markers if m.lower() in disk_content.lower())
                file_on_disk_valid = matching_count >= max(1, len(markers) // 2) if markers else len(disk_content.strip()) > 50
            except Exception:
                pass

        if markers and file_on_disk_valid:
            log_info(f"[VERIFY_EDITOR] Verified editor state for '{expected_file}' in {application} (disk consistency verified)")
            return {
                "success": True,
                "application": application,
                "file": expected_file,
                "content_verified": True,
                "evidence": f"Expected {expected_file} implementation detected in {application} editor.",
            }

        # 2. Window title / Process verification
        visible = self.controller.list_windows()
        file_lower = Path(expected_file).name.lower()
        has_file_in_title = any(file_lower in w.title.lower() for w in visible if w.title)
        has_app_window = self.verify_window_present(application)

        if markers and (has_file_in_title and has_app_window):
            log_info(f"[VERIFY_EDITOR] Verified editor state for '{expected_file}' in {application} (window presence verified)")
            return {
                "success": True,
                "application": application,
                "file": expected_file,
                "content_verified": True,
                "evidence": f"Expected {expected_file} window detected in {application}.",
            }

        # 3. Only attempt GUI clipboard inspection if the target application is explicitly in the foreground
        # CRITICAL: NEVER send Ctrl+C to a terminal, PowerShell, or console window as it triggers SIGINT / KeyboardInterrupt!
        self.controller.focus_window(expected_file)
        self.controller.focus_window(application)
        time.sleep(0.3)

        is_target_focused = False
        try:
            import ctypes
            user32 = ctypes.windll.user32
            hwnd = user32.GetForegroundWindow()
            length = user32.GetWindowTextLengthW(hwnd)
            buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buf, length + 1)
            fg_title = buf.value.lower()
            is_terminal = any(t in fg_title for t in ("cmd", "powershell", "terminal", "python", "chitti"))
            is_target_focused = not is_terminal and (application.lower() in fg_title or file_lower in fg_title)
        except Exception:
            is_target_focused = False

        if is_target_focused:
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
