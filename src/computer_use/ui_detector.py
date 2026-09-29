"""
Chitti Semantic UI Element Detector.
Locates UI elements semantically using Windows UI trees, semantic target models, and relative geometry
rather than fragile fixed pixel coordinates.
"""

import ctypes
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.computer_use.controller import ComputerController, WindowInfo
from src.utils.logging import log_debug, log_info, log_warn
from src.utils.text import clean_contact_query, matches_contact_name, strip_emojis

try:
    import win32gui
    import win32con
    HAS_WIN32 = True
except Exception:
    win32gui = None
    win32con = None
    HAS_WIN32 = False


@dataclass
class UIElement:
    """Represents a detected UI element."""
    name: str
    control_type: str
    left: int
    top: int
    right: int
    bottom: int
    center_x: int
    center_y: int
    handle: int = 0
    is_enabled: bool = True
    is_visible: bool = True
    suggested_shortcut: Optional[str] = None


class UIElementDetector:
    """Detects and resolves UI targets semantically across Windows applications and web browsers."""

    # Semantic action mappings for keyboard shortcuts
    SEMANTIC_SHORTCUTS = {
        "search": ("hotkey", ["ctrl", "alt", "/"]),
        "search_web": ("hotkey", ["ctrl", "k"]),
        "search_browser": ("hotkey", ["ctrl", "e"]),
        "compose": ("press_key", ["c"]),
        "send": ("hotkey", ["ctrl", "enter"]),
        "send_message": ("press_key", ["enter"]),
        "save": ("hotkey", ["ctrl", "s"]),
        "select_all": ("hotkey", ["ctrl", "a"]),
        "copy": ("hotkey", ["ctrl", "c"]),
        "paste": ("hotkey", ["ctrl", "v"]),
        "new_tab": ("hotkey", ["ctrl", "t"]),
        "close_tab": ("hotkey", ["ctrl", "w"]),
    }

    def __init__(self, controller: ComputerController):
        self.controller = controller

    def locate_element(self, description: str, window_title: Optional[str] = None) -> Optional[UIElement]:
        """
        Locates a UI element by semantic description within the active or specified window.
        Uses native Windows UI control enumeration, relative window geometry, and semantic shortcuts.
        """
        desc_lower = description.lower().strip()
        log_info(f"[UI DETECTOR] Locating semantic element: '{description}' (Window: '{window_title or 'active'}')")

        # 1. Target window resolution
        active_win = self.controller.get_active_window()
        if window_title:
            self.controller.focus_window(window_title)
            active_win = self.controller.get_active_window()

        win_rect = (active_win.left, active_win.top, active_win.left + active_win.width, active_win.top + active_win.height) if active_win else (0, 0, 1920, 1080)
        wl, wt, wr, wb = win_rect
        ww = wr - wl
        wh = wb - wt

        # 2. Native Win32 Child Controls Enumeration
        if HAS_WIN32 and active_win and active_win.handle:
            found_child: Optional[UIElement] = None
            clean_desc = clean_contact_query(desc_lower)
            def enum_child_proc(hwnd, lparam):
                nonlocal found_child
                try:
                    if win32gui.IsWindowVisible(hwnd):
                        txt = win32gui.GetWindowText(hwnd).lower()
                        cls_name = win32gui.GetClassName(hwnd).lower()
                        rect = win32gui.GetWindowRect(hwnd)
                        cl, ct, cr, cb = rect
                        if (cr - cl) > 0 and (cb - ct) > 0:
                            if (desc_lower in txt or (clean_desc and clean_desc in txt) or
                                matches_contact_name(clean_desc or desc_lower, txt) or
                                desc_lower in cls_name):
                                cx = (cl + cr) // 2
                                cy = (ct + cb) // 2
                                found_child = UIElement(
                                    name=txt or desc_lower,
                                    control_type=cls_name,
                                    left=cl, top=ct, right=cr, bottom=cb,
                                    center_x=cx, center_y=cy,
                                    handle=hwnd,
                                )
                                return False
                except Exception:
                    pass
                return True

            try:
                win32gui.EnumChildWindows(active_win.handle, enum_child_proc, None)
            except Exception:
                pass

            if found_child:
                log_info(f"[UI DETECTOR] Found native Win32 control '{found_child.name}' at ({found_child.center_x}, {found_child.center_y})")
                return found_child

        # 3. Relative Semantic Layout Computation (Resolution-Independent)
        # For browser & web applications (WhatsApp Web, Gmail, etc.)
        if "search" in desc_lower:
            # Search bar is typically in the top-left or top-center (approx 20-30% from top)
            cx = wl + int(ww * 0.25)
            cy = wt + int(wh * 0.15)
            return UIElement(
                name="Search Box",
                control_type="Edit/Search",
                left=wl + int(ww * 0.1), top=wt + int(wh * 0.1),
                right=wl + int(ww * 0.4), bottom=wt + int(wh * 0.2),
                center_x=cx, center_y=cy,
                suggested_shortcut="ctrl+alt+/",
            )

        elif "compose" in desc_lower or "new message" in desc_lower:
            # Compose button is typically top-left in webmail (approx 10-15% X, 20% Y)
            cx = wl + int(ww * 0.10)
            cy = wt + int(wh * 0.20)
            return UIElement(
                name="Compose Button",
                control_type="Button",
                left=wl + int(ww * 0.05), top=wt + int(wh * 0.15),
                right=wl + int(ww * 0.15), bottom=wt + int(wh * 0.25),
                center_x=cx, center_y=cy,
                suggested_shortcut="c",
            )

        elif "send" in desc_lower:
            # Send button in chat/email is typically bottom-right or bottom-left of compose
            cx = wl + int(ww * 0.90)
            cy = wt + int(wh * 0.92)
            return UIElement(
                name="Send Button",
                control_type="Button",
                left=wl + int(ww * 0.85), top=wt + int(wh * 0.88),
                right=wl + int(ww * 0.95), bottom=wt + int(wh * 0.96),
                center_x=cx, center_y=cy,
                suggested_shortcut="ctrl+enter",
            )

        elif "message" in desc_lower or "input" in desc_lower or "body" in desc_lower:
            # Message input is typically bottom center
            cx = wl + int(ww * 0.55)
            cy = wt + int(wh * 0.92)
            return UIElement(
                name="Message Input",
                control_type="Edit",
                left=wl + int(ww * 0.35), top=wt + int(wh * 0.88),
                right=wl + int(ww * 0.85), bottom=wt + int(wh * 0.96),
                center_x=cx, center_y=cy,
            )

        # Fallback to center of window
        cx = wl + ww // 2
        cy = wt + wh // 2
        log_info(f"[UI DETECTOR] Resolved generic element target '{description}' at window center ({cx}, {cy})")
        return UIElement(
            name=description,
            control_type="GenericControl",
            left=wl, top=wt, right=wr, bottom=wb,
            center_x=cx, center_y=cy,
        )
