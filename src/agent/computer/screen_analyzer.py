"""
Chitti Screen Analyzer & UI Understanding.
Analyzes screenshots and visible UI to verify actions and guide agent decisions.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.agent.computer.controller import ComputerController, WindowInfo
from src.utils.logging import log_debug, log_info, log_warn


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

    def verify_window_present(self, title_query: str) -> bool:
        """Verifies if a window matching title_query is visible on screen."""
        query = title_query.lower().strip()
        visible = self.controller.list_windows()
        for w in visible:
            if query in w.title.lower():
                return True
        return False

    def verify_active_window(self, title_query: str) -> bool:
        """Verifies if the foreground window matches title_query."""
        query = title_query.lower().strip()
        active = self.controller.get_active_window()
        if active and query in active.title.lower():
            return True
        return False
