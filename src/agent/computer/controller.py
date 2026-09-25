"""
Chitti Computer Controller.
Provides high-level programmatic control over mouse, keyboard, clipboard, screen, and window management on Windows.
"""

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.utils.logging import log_debug, log_info, log_warn

# Try imports for GUI automation with graceful ctypes / win32 fallbacks
try:
    import pyautogui
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.05
    HAS_PYAUTOGUI = True
except Exception as e:
    pyautogui = None
    HAS_PYAUTOGUI = False

try:
    import pyperclip
    HAS_PYPERCLIP = True
except Exception:
    pyperclip = None
    HAS_PYPERCLIP = False

try:
    import pygetwindow as gw
    HAS_PYGETWINDOW = True
except Exception:
    gw = None
    HAS_PYGETWINDOW = False

try:
    from PIL import Image, ImageGrab
    HAS_PIL = True
except Exception:
    ImageGrab = None
    HAS_PIL = False

try:
    import win32gui
    import win32con
    import win32process
    import win32api
    HAS_WIN32 = True
except Exception:
    win32gui = None
    win32con = None
    HAS_WIN32 = False

import ctypes


@dataclass
class WindowInfo:
    """Represents a visible Windows application window."""
    title: str
    handle: int
    left: int
    top: int
    width: int
    height: int
    is_active: bool
    is_minimized: bool


class ComputerController:
    """Central interface for low-level & high-level computer operations."""

    def __init__(self, screenshots_dir: str = "data/screenshots"):
        self.screenshots_dir = Path(screenshots_dir)
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # MOUSE CONTROLS
    # -------------------------------------------------------------

    def get_screen_size(self) -> Tuple[int, int]:
        """Returns the screen resolution (width, height)."""
        if HAS_PYAUTOGUI:
            return pyautogui.size()
        user32 = ctypes.windll.user32
        return (user32.GetSystemMetrics(0), user32.GetSystemMetrics(1))

    def get_mouse_position(self) -> Tuple[int, int]:
        """Returns the current mouse cursor coordinates (x, y)."""
        if HAS_PYAUTOGUI:
            return pyautogui.position()
        pt = ctypes.wintypes.POINT()
        ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
        return (pt.x, pt.y)

    def move_mouse(self, x: int, y: int, duration: float = 0.2) -> bool:
        """Moves the mouse cursor to (x, y)."""
        try:
            if HAS_PYAUTOGUI:
                pyautogui.moveTo(x, y, duration=duration)
                return True
            ctypes.windll.user32.SetCursorPos(int(x), int(y))
            return True
        except Exception as e:
            log_warn(f"Failed to move mouse to ({x}, {y}): {e}")
            return False

    def click(self, x: Optional[int] = None, y: Optional[int] = None, button: str = "left", clicks: int = 1) -> bool:
        """Clicks at (x, y) or at the current cursor position."""
        try:
            if HAS_PYAUTOGUI:
                if x is not None and y is not None:
                    pyautogui.click(x=x, y=y, clicks=clicks, button=button)
                else:
                    pyautogui.click(clicks=clicks, button=button)
                return True

            # Native ctypes fallback
            if x is not None and y is not None:
                self.move_mouse(x, y)
            time.sleep(0.05)
            user32 = ctypes.windll.user32
            if button.lower() == "left":
                for _ in range(clicks):
                    user32.mouse_event(0x0002, 0, 0, 0, 0) # LEFTDOWN
                    user32.mouse_event(0x0004, 0, 0, 0, 0) # LEFTUP
                    time.sleep(0.05)
            elif button.lower() == "right":
                user32.mouse_event(0x0008, 0, 0, 0, 0) # RIGHTDOWN
                user32.mouse_event(0x0010, 0, 0, 0, 0) # RIGHTUP
            return True
        except Exception as e:
            log_warn(f"Failed to click mouse: {e}")
            return False

    def double_click(self, x: Optional[int] = None, y: Optional[int] = None) -> bool:
        """Performs a double click."""
        return self.click(x=x, y=y, button="left", clicks=2)

    def right_click(self, x: Optional[int] = None, y: Optional[int] = None) -> bool:
        """Performs a right click."""
        return self.click(x=x, y=y, button="right", clicks=1)

    def scroll(self, amount: int) -> bool:
        """Scrolls vertically (positive for up, negative for down)."""
        try:
            if HAS_PYAUTOGUI:
                pyautogui.scroll(amount)
                return True
            ctypes.windll.user32.mouse_event(0x0800, 0, 0, amount * 120, 0) # MOUSEEVENTF_WHEEL
            return True
        except Exception as e:
            log_warn(f"Failed to scroll: {e}")
            return False

    def drag(self, x: int, y: int, duration: float = 0.5) -> bool:
        """Drags mouse to (x, y)."""
        try:
            if HAS_PYAUTOGUI:
                pyautogui.dragTo(x, y, duration=duration, button="left")
                return True
            return False
        except Exception as e:
            log_warn(f"Failed to drag mouse: {e}")
            return False

    # -------------------------------------------------------------
    # KEYBOARD CONTROLS
    # -------------------------------------------------------------

    def type_text(self, text: str, interval: float = 0.02) -> bool:
        """Types the given string into the active window."""
        try:
            if HAS_PYAUTOGUI:
                pyautogui.write(text, interval=interval)
                return True
            # Fallback to clipboard paste for Unicode / special characters
            self.write_clipboard(text)
            self.paste()
            return True
        except Exception as e:
            log_warn(f"Failed to type text: {e}")
            return False

    def press_key(self, key: str) -> bool:
        """Presses a single key (e.g. 'enter', 'esc', 'tab', 'backspace')."""
        try:
            clean_key = key.lower().strip()
            if HAS_PYAUTOGUI:
                pyautogui.press(clean_key)
                return True
            return False
        except Exception as e:
            log_warn(f"Failed to press key {key}: {e}")
            return False

    def hotkey(self, *keys: str) -> bool:
        """Executes a key combination (e.g. hotkey('ctrl', 'c'), hotkey('alt', 'tab'))."""
        try:
            clean_keys = [k.lower().strip() for k in keys]
            if HAS_PYAUTOGUI:
                pyautogui.hotkey(*clean_keys)
                return True
            return False
        except Exception as e:
            log_warn(f"Failed to execute hotkey {keys}: {e}")
            return False

    # -------------------------------------------------------------
    # CLIPBOARD CONTROLS
    # -------------------------------------------------------------

    def copy(self) -> str:
        """Triggers Ctrl+C and returns the clipboard contents."""
        self.hotkey("ctrl", "c")
        time.sleep(0.1)
        return self.read_clipboard()

    def paste(self) -> bool:
        """Triggers Ctrl+V into the active input."""
        return self.hotkey("ctrl", "v")

    def read_clipboard(self) -> str:
        """Reads text from the system clipboard."""
        try:
            if HAS_PYPERCLIP:
                return pyperclip.paste() or ""
            return ""
        except Exception as e:
            log_warn(f"Failed to read clipboard: {e}")
            return ""

    def write_clipboard(self, text: str) -> bool:
        """Writes text to the system clipboard."""
        try:
            if HAS_PYPERCLIP:
                pyperclip.copy(text)
                return True
            return False
        except Exception as e:
            log_warn(f"Failed to write clipboard: {e}")
            return False

    # -------------------------------------------------------------
    # SCREEN CONTROLS
    # -------------------------------------------------------------

    def take_screenshot(self, filename: Optional[str] = None) -> str:
        """Captures the screen and saves it to data/screenshots."""
        if not filename:
            filename = f"screenshot_{int(time.time() * 1000)}.png"
        save_path = self.screenshots_dir / filename

        if HAS_PIL and ImageGrab:
            img = ImageGrab.grab()
            img.save(str(save_path))
            log_info(f"Captured screenshot: {save_path}")
            return str(save_path.resolve())

        raise RuntimeError("Screen capture requires Pillow (PIL.ImageGrab).")

    # -------------------------------------------------------------
    # WINDOW MANAGEMENT
    # -------------------------------------------------------------

    def list_windows(self) -> List[WindowInfo]:
        """Lists all open, visible top-level windows with their coordinates."""
        windows: List[WindowInfo] = []
        if HAS_PYGETWINDOW and gw:
            try:
                all_wins = gw.getAllWindows()
                for w in all_wins:
                    if w.title and w.visible and w.width > 0 and w.height > 0:
                        windows.append(WindowInfo(
                            title=w.title,
                            handle=w._hWnd if hasattr(w, "_hWnd") else 0,
                            left=w.left,
                            top=w.top,
                            width=w.width,
                            height=w.height,
                            is_active=w.isActive if hasattr(w, "isActive") else False,
                            is_minimized=w.isMinimized if hasattr(w, "isMinimized") else False,
                        ))
                return windows
            except Exception as e:
                log_debug(f"PyGetWindow list failed: {e}")

        # Fallback to win32gui
        if HAS_WIN32:
            def enum_cb(hwnd, results):
                try:
                    if win32gui.IsWindowVisible(hwnd):
                        title = win32gui.GetWindowText(hwnd)
                        if title:
                            rect = win32gui.GetWindowRect(hwnd)
                            left, top, right, bottom = rect
                            w = right - left
                            h = bottom - top
                            if w > 0 and h > 0:
                                results.append(WindowInfo(
                                    title=title,
                                    handle=hwnd,
                                    left=left,
                                    top=top,
                                    width=w,
                                    height=h,
                                    is_active=(hwnd == win32gui.GetForegroundWindow()),
                                    is_minimized=win32gui.IsIconic(hwnd) != 0,
                                ))
                except Exception:
                    pass
                return True

            res: List[WindowInfo] = []
            try:
                win32gui.EnumWindows(enum_cb, res)
            except Exception as e:
                log_debug(f"win32gui EnumWindows caught: {e}")
            return res

        return windows

    def get_active_window(self) -> Optional[WindowInfo]:
        """Returns information about the currently focused foreground window."""
        windows = self.list_windows()
        for w in windows:
            if w.is_active:
                return w
        if HAS_WIN32:
            hwnd = win32gui.GetForegroundWindow()
            if hwnd:
                title = win32gui.GetWindowText(hwnd)
                rect = win32gui.GetWindowRect(hwnd)
                return WindowInfo(
                    title=title,
                    handle=hwnd,
                    left=rect[0],
                    top=rect[1],
                    width=rect[2] - rect[0],
                    height=rect[3] - rect[1],
                    is_active=True,
                    is_minimized=False,
                )
        return None

    def focus_window(self, title_query: str) -> bool:
        """Brings the window matching title_query into the foreground."""
        query = title_query.lower().strip()
        if HAS_PYGETWINDOW and gw:
            try:
                candidates = [w for w in gw.getAllWindows() if query in w.title.lower()]
                if candidates:
                    target = candidates[0]
                    if target.isMinimized:
                        target.restore()
                    target.activate()
                    time.sleep(0.1)
                    return True
            except Exception as e:
                log_debug(f"PyGetWindow focus error: {e}")

        if HAS_WIN32:
            for w in self.list_windows():
                if query in w.title.lower():
                    try:
                        win32gui.ShowWindow(w.handle, win32con.SW_RESTORE)
                        win32gui.SetForegroundWindow(w.handle)
                        time.sleep(0.1)
                        return True
                    except Exception as e:
                        log_warn(f"Failed to focus window via win32: {e}")
        return False

    def minimize_window(self, title_query: str) -> bool:
        """Minimizes the window matching title_query."""
        query = title_query.lower().strip()
        if HAS_PYGETWINDOW and gw:
            try:
                for w in gw.getAllWindows():
                    if query in w.title.lower():
                        w.minimize()
                        return True
            except Exception:
                pass
        return False

    def maximize_window(self, title_query: str) -> bool:
        """Maximizes the window matching title_query."""
        query = title_query.lower().strip()
        if HAS_PYGETWINDOW and gw:
            try:
                for w in gw.getAllWindows():
                    if query in w.title.lower():
                        w.maximize()
                        return True
            except Exception:
                pass
        return False

    def close_window(self, title_query: str) -> bool:
        """Closes the window matching title_query."""
        query = title_query.lower().strip()
        if HAS_PYGETWINDOW and gw:
            try:
                for w in gw.getAllWindows():
                    if query in w.title.lower():
                        w.close()
                        return True
            except Exception:
                pass
        return False
