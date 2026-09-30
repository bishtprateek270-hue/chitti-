"""
Chitti Computer-Use Device Automation Controller.
Provides low-level & high-level device control: mouse, keyboard, screen capture, clipboard, and window management.
"""

import ctypes
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.utils.logging import log_debug, log_info, log_warn

# Try importing standard GUI libraries with robust ctypes/win32 fallbacks
import importlib

try:
    pyautogui = importlib.import_module("pyautogui")
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.05
    HAS_PYAUTOGUI = True
except Exception:
    pyautogui = None
    HAS_PYAUTOGUI = False

try:
    pyperclip = importlib.import_module("pyperclip")
    HAS_PYPERCLIP = True
except Exception:
    pyperclip = None
    HAS_PYPERCLIP = False

try:
    gw = importlib.import_module("pygetwindow")
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
    win32process = None
    win32api = None
    HAS_WIN32 = False


@dataclass
class WindowInfo:
    """Represents an open Windows application window."""
    title: str
    handle: int
    left: int
    top: int
    width: int
    height: int
    is_active: bool
    is_minimized: bool


class ComputerController:
    """Central device automation controller for Windows."""

    def __init__(self, screenshots_dir: str = "data/screenshots"):
        self.screenshots_dir = Path(screenshots_dir)
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)
        self._clipboard_text: str = ""

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

    def move_mouse(self, x: int, y: int, duration: float = 0.1) -> bool:
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

            if x is not None and y is not None:
                self.move_mouse(x, y)
            time.sleep(0.02)
            user32 = ctypes.windll.user32
            if button.lower() == "left":
                for _ in range(clicks):
                    user32.mouse_event(0x0002, 0, 0, 0, 0)  # MOUSEEVENTF_LEFTDOWN
                    user32.mouse_event(0x0004, 0, 0, 0, 0)  # MOUSEEVENTF_LEFTUP
                    time.sleep(0.03)
            elif button.lower() == "right":
                user32.mouse_event(0x0008, 0, 0, 0, 0)  # MOUSEEVENTF_RIGHTDOWN
                user32.mouse_event(0x0010, 0, 0, 0, 0)  # MOUSEEVENTF_RIGHTUP
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
            ctypes.windll.user32.mouse_event(0x0800, 0, 0, int(amount * 120), 0)
            return True
        except Exception as e:
            log_warn(f"Failed to scroll: {e}")
            return False

    def drag(self, x: int, y: int, duration: float = 0.3) -> bool:
        """Drags mouse to (x, y)."""
        try:
            if HAS_PYAUTOGUI:
                pyautogui.dragTo(x, y, duration=duration, button="left")
                return True
            # Native fallback: left down, move, left up
            user32 = ctypes.windll.user32
            user32.mouse_event(0x0002, 0, 0, 0, 0)
            time.sleep(0.05)
            self.move_mouse(x, y)
            time.sleep(duration)
            user32.mouse_event(0x0004, 0, 0, 0, 0)
            return True
        except Exception as e:
            log_warn(f"Failed to drag mouse: {e}")
            return False

    def wait(self, seconds: float = 1.0) -> bool:
        """Pauses execution for a specified duration."""
        time.sleep(max(0.05, float(seconds)))
        return True

    # -------------------------------------------------------------
    # KEYBOARD CONTROLS
    # -------------------------------------------------------------

    def type_text(self, text: str, interval: float = 0.01) -> bool:
        """Types text into the active input."""
        try:
            if HAS_PYAUTOGUI:
                pyautogui.write(text, interval=interval)
                return True
            self.write_clipboard(text)
            self.paste()
            return True
        except Exception as e:
            log_warn(f"Failed to type text: {e}")
            return False

    def press_key(self, key: str) -> bool:
        """Presses a single key (e.g. 'enter', 'tab', 'c', 'esc')."""
        try:
            clean_key = key.lower().strip()
            if HAS_PYAUTOGUI:
                try:
                    pyautogui.press(clean_key)
                    return True
                except Exception:
                    pass

            KEY_MAP = {
                "enter": 0x0D, "return": 0x0D, "esc": 0x1B, "escape": 0x1B,
                "tab": 0x09, "backspace": 0x08, "space": 0x20, "up": 0x26,
                "down": 0x28, "left": 0x25, "right": 0x27,
            }
            vk = KEY_MAP.get(clean_key, ord(clean_key.upper()) if len(clean_key) == 1 else 0)
            if vk and hasattr(ctypes, "windll"):
                user32 = ctypes.windll.user32
                user32.keybd_event(vk, 0, 0, 0)
                time.sleep(0.02)
                user32.keybd_event(vk, 0, 2, 0)
                return True
            return True
        except Exception as e:
            log_warn(f"Failed to press key '{key}': {e}")
            return False

    def hotkey(self, *keys: str) -> bool:
        """Executes a key shortcut combination (e.g. hotkey('ctrl', 'enter'))."""
        clean_keys = [k.lower().strip() for k in keys]
        if HAS_PYAUTOGUI:
            try:
                pyautogui.hotkey(*clean_keys)
                return True
            except Exception:
                pass

        VK_MAP = {
            "ctrl": 0x11, "control": 0x11, "alt": 0x12, "shift": 0x10,
            "enter": 0x0D, "tab": 0x09, "esc": 0x1B,
            "s": 0x53, "c": 0x43, "v": 0x56, "a": 0x41, "z": 0x5A, "k": 0x4B,
        }
        try:
            user32 = ctypes.windll.user32
            for k in clean_keys:
                vk = VK_MAP.get(k, ord(k.upper()) if len(k) == 1 else 0)
                if vk:
                    user32.keybd_event(vk, 0, 0, 0)
            time.sleep(0.03)
            for k in reversed(clean_keys):
                vk = VK_MAP.get(k, ord(k.upper()) if len(k) == 1 else 0)
                if vk:
                    user32.keybd_event(vk, 0, 2, 0)
            return True
        except Exception as e:
            log_warn(f"Failed to execute hotkey {keys}: {e}")
            return False

    # -------------------------------------------------------------
    # CLIPBOARD CONTROLS
    # -------------------------------------------------------------

    def copy(self) -> str:
        """Triggers Ctrl+C and returns the clipboard contents."""
        self.hotkey("ctrl", "c")
        time.sleep(0.05)
        return self.read_clipboard()

    def paste(self) -> bool:
        """Triggers Ctrl+V into the active input."""
        return self.hotkey("ctrl", "v")

    def read_clipboard(self) -> str:
        """Reads text from system clipboard."""
        try:
            if HAS_PYPERCLIP:
                txt = pyperclip.paste()
                if txt:
                    self._clipboard_text = txt
                    return txt
            return self._clipboard_text
        except Exception:
            return self._clipboard_text

    def write_clipboard(self, text: str) -> bool:
        """Writes text to system clipboard."""
        self._clipboard_text = text
        try:
            if HAS_PYPERCLIP:
                pyperclip.copy(text)
            return True
        except Exception:
            return True

    # -------------------------------------------------------------
    # SCREEN CONTROLS
    # -------------------------------------------------------------

    def screenshot(self, filename: Optional[str] = None) -> str:
        """Captures the current screen and returns the saved file path."""
        if not filename:
            filename = f"screenshot_{int(time.time() * 1000)}.png"
        save_path = self.screenshots_dir / filename

        if HAS_PIL and ImageGrab:
            try:
                img = ImageGrab.grab()
                img.save(str(save_path))
                return str(save_path.resolve())
            except Exception as e:
                log_debug(f"ImageGrab notice: {e}")
                img = Image.new("RGB", (1920, 1080), color=(30, 30, 30))
                img.save(str(save_path))
                return str(save_path.resolve())

        if HAS_PIL:
            img = Image.new("RGB", (1920, 1080), color=(30, 30, 30))
            img.save(str(save_path))
            return str(save_path.resolve())

        return str(save_path.resolve())

    # -------------------------------------------------------------
    # WINDOW MANAGEMENT
    # -------------------------------------------------------------

    def list_windows(self) -> List[WindowInfo]:
        """Lists all visible top-level windows."""
        windows: List[WindowInfo] = []
        if HAS_PYGETWINDOW and gw:
            try:
                for w in gw.getAllWindows():
                    if w.title and w.visible and w.width > 0 and w.height > 0:
                        windows.append(WindowInfo(
                            title=w.title,
                            handle=w._hWnd if hasattr(w, "_hWnd") else 0,
                            left=w.left, top=w.top, width=w.width, height=w.height,
                            is_active=w.isActive if hasattr(w, "isActive") else False,
                            is_minimized=w.isMinimized if hasattr(w, "isMinimized") else False,
                        ))
                return windows
            except Exception:
                pass

        if HAS_WIN32:
            def enum_cb(hwnd, results):
                try:
                    if win32gui.IsWindowVisible(hwnd):
                        title = win32gui.GetWindowText(hwnd)
                        if title:
                            rect = win32gui.GetWindowRect(hwnd)
                            left, top, right, bottom = rect
                            w, h = right - left, bottom - top
                            if w > 0 and h > 0:
                                results.append(WindowInfo(
                                    title=title, handle=hwnd,
                                    left=left, top=top, width=w, height=h,
                                    is_active=(hwnd == win32gui.GetForegroundWindow()),
                                    is_minimized=win32gui.IsIconic(hwnd) != 0,
                                ))
                except Exception:
                    pass
                return True

            res: List[WindowInfo] = []
            try:
                win32gui.EnumWindows(enum_cb, res)
            except Exception:
                pass
            return res

        return windows

    def get_active_window(self) -> Optional[WindowInfo]:
        """Returns the currently active foreground window."""
        windows = self.list_windows()
        for w in windows:
            if w.is_active:
                return w
        if HAS_WIN32:
            try:
                hwnd = win32gui.GetForegroundWindow()
                if hwnd:
                    title = win32gui.GetWindowText(hwnd)
                    rect = win32gui.GetWindowRect(hwnd)
                    return WindowInfo(
                        title=title, handle=hwnd,
                        left=rect[0], top=rect[1], width=rect[2] - rect[0], height=rect[3] - rect[1],
                        is_active=True, is_minimized=False,
                    )
            except Exception:
                pass
        return None

    def find_window(self, title_query: str) -> Optional[WindowInfo]:
        """Finds a window by title query or keywords."""
        query = title_query.lower().strip()
        tokens = [t for t in re.split(r"[\s\/\-_]+", query) if len(t) > 2 and t not in ("web", "app", "window", "application")]
        wins = self.list_windows()
        for w in wins:
            if query in w.title.lower():
                return w
        if tokens:
            for w in wins:
                if any(t in w.title.lower() for t in tokens):
                    return w
        return None

    def focus_window(self, title_query: str) -> bool:
        """Brings the window matching title_query into the foreground with rock-solid focus."""
        target_win = self.find_window(title_query)
        if not target_win:
            if HAS_PYGETWINDOW and gw:
                try:
                    for w in gw.getAllWindows():
                        if any(t in w.title.lower() for t in re.split(r"[\s\/\-_]+", title_query.lower()) if len(t) > 2):
                            if w.isMinimized:
                                w.restore()
                            w.activate()
                            time.sleep(0.1)
                            return True
                except Exception:
                    pass
            return False

        hwnd = target_win.handle
        if hwnd and hasattr(ctypes, "windll"):
            try:
                user32 = ctypes.windll.user32
                if target_win.is_minimized or user32.IsIconic(hwnd):
                    user32.ShowWindow(hwnd, 9)  # SW_RESTORE
                else:
                    user32.ShowWindow(hwnd, 5)  # SW_SHOW

                fg_window = user32.GetForegroundWindow()
                if fg_window and fg_window != hwnd:
                    fg_thread = user32.GetWindowThreadProcessId(fg_window, None)
                    current_thread = user32.GetWindowThreadProcessId(hwnd, None)
                    user32.AttachThreadInput(fg_thread, current_thread, True)
                    user32.BringWindowToTop(hwnd)
                    user32.SetForegroundWindow(hwnd)
                    user32.AttachThreadInput(fg_thread, current_thread, False)
                else:
                    user32.BringWindowToTop(hwnd)
                    user32.SetForegroundWindow(hwnd)

                time.sleep(0.15)
                return True
            except Exception as e:
                log_debug(f"Win32 focus notice: {e}")

        if HAS_PYGETWINDOW and gw:
            try:
                for w in gw.getAllWindows():
                    if (hwnd and getattr(w, "_hWnd", 0) == hwnd) or (target_win.title and target_win.title in w.title):
                        if w.isMinimized:
                            w.restore()
                        w.activate()
                        time.sleep(0.1)
                        return True
            except Exception as e:
                log_debug(f"PyGetWindow fallback notice: {e}")

        return False

    def close_window(self, title_query: str) -> bool:
        """Closes the window matching title_query."""
        query = title_query.lower().strip()
        if HAS_WIN32:
            wins = self.list_windows()
            for w in wins:
                if query in w.title.lower():
                    try:
                        win32gui.PostMessage(w.handle, win32con.WM_CLOSE, 0, 0)
                        return True
                    except Exception:
                        pass
        return False
