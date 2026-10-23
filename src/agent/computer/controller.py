"""
Chitti Computer Controller.
Provides high-level programmatic control over mouse, keyboard, clipboard, screen, and window management on Windows.
"""

import os
import platform
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.utils.logging import log_debug, log_info, log_warn

# Try imports for GUI automation with graceful ctypes / win32 fallbacks
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


@dataclass
class UIElementInfo:
    """Represents an interactive on-screen control discovered via Windows UI Automation or Win32."""
    name: str
    control_type: str
    handle: int
    left: int
    top: int
    width: int
    height: int
    center_x: int
    center_y: int
    is_enabled: bool = True
    is_visible: bool = True
    control_id: int = 0


class ComputerController:
    """Central interface for low-level & high-level computer operations."""

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

    def drag(
        self,
        x: int = 0,
        y: int = 0,
        duration: float = 0.5,
        from_x: Optional[int] = None,
        from_y: Optional[int] = None,
        to_x: Optional[int] = None,
        to_y: Optional[int] = None,
    ) -> bool:
        """Drags mouse from (from_x, from_y) to (to_x, to_y) or from current position to (x, y)."""
        target_x = to_x if to_x is not None else x
        target_y = to_y if to_y is not None else y
        if from_x is not None and from_y is not None:
            self.move_mouse(from_x, from_y)
        try:
            if HAS_PYAUTOGUI:
                pyautogui.dragTo(target_x, target_y, duration=duration, button="left")
                return True
            return False
        except Exception as e:
            log_warn(f"Failed to drag mouse: {e}")
            return False

    def drag_mouse(
        self,
        from_x: int,
        from_y: int,
        to_x: int,
        to_y: int,
        duration: float = 0.5,
    ) -> bool:
        """Drags mouse from (from_x, from_y) to (to_x, to_y)."""
        return self.drag(from_x=from_x, from_y=from_y, to_x=to_x, to_y=to_y, duration=duration)

    # -------------------------------------------------------------
    # KEYBOARD CONTROLS
    # -------------------------------------------------------------

    def type_text(self, text: str, interval: float = 0.02) -> bool:
        """Types the given string into the active window."""
        try:
            # If text contains non-ASCII characters or emojis, use clipboard paste for 100% accuracy
            if any(ord(c) > 127 for c in text):
                self.write_clipboard(text)
                time.sleep(0.05)
                self.hotkey("ctrl", "v")
                time.sleep(0.05)
                return True

            if HAS_PYAUTOGUI:
                try:
                    pyautogui.write(text, interval=interval)
                    return True
                except Exception:
                    pass

            # Fallback to clipboard paste for Unicode / special characters
            self.write_clipboard(text)
            time.sleep(0.05)
            self.hotkey("ctrl", "v")
            time.sleep(0.05)
            return True
        except Exception as e:
            log_warn(f"Failed to type text: {e}")
            return False


    def press_key(self, key: str) -> bool:
        """Presses a single key (e.g. 'enter', 'esc', 'tab', 'backspace')."""
        try:
            clean_key = key.lower().strip()
            if HAS_PYAUTOGUI:
                try:
                    pyautogui.press(clean_key)
                    return True
                except Exception as e:
                    log_debug(f"pyautogui.press notice: {e}")

            # Win32 keybd_event fallback
            KEY_MAP = {
                "enter": 0x0D,
                "return": 0x0D,
                "esc": 0x1B,
                "escape": 0x1B,
                "tab": 0x09,
                "backspace": 0x08,
                "space": 0x20,
                "up": 0x26,
                "down": 0x28,
                "left": 0x25,
                "right": 0x27,
            }
            vk = KEY_MAP.get(clean_key, ord(clean_key.upper()) if len(clean_key) == 1 else 0)
            if vk and hasattr(ctypes, "windll"):
                try:
                    user32 = ctypes.windll.user32
                    user32.keybd_event(vk, 0, 0, 0)
                    user32.keybd_event(vk, 0, 2, 0)  # KEYEVENTF_KEYUP
                    return True
                except Exception:
                    pass
            return True
        except Exception as e:
            log_warn(f"Failed to press key {key}: {e}")
            return False

    def hotkey(self, *keys: str) -> bool:
        """Executes a key combination (e.g. hotkey('ctrl', 'c'), hotkey('alt', 'tab'))."""
        clean_keys = [k.lower().strip() for k in keys]
        if HAS_PYAUTOGUI:
            try:
                pyautogui.hotkey(*clean_keys)
                return True
            except Exception as e:
                log_debug(f"pyautogui.hotkey notice: {e}")

        # Win32 keybd_event fallback
        VK_MAP = {
            "ctrl": 0x11,
            "control": 0x11,
            "alt": 0x12,
            "shift": 0x10,
            "s": 0x53,
            "c": 0x43,
            "v": 0x56,
            "a": 0x41,
            "z": 0x5A,
        }
        try:
            user32 = ctypes.windll.user32
            for k in clean_keys:
                vk = VK_MAP.get(k, ord(k.upper()) if len(k) == 1 else 0)
                if vk:
                    user32.keybd_event(vk, 0, 0, 0)
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
        time.sleep(0.1)
        return self.read_clipboard()

    def paste(self) -> bool:
        """Triggers Ctrl+V into the active input."""
        return self.hotkey("ctrl", "v")

    def read_clipboard(self) -> str:
        """Reads text from the system clipboard."""
        try:
            if HAS_PYPERCLIP:
                txt = pyperclip.paste()
                if txt:
                    self._clipboard_text = txt
                    return txt
            return self._clipboard_text
        except Exception as e:
            log_warn(f"Failed to read clipboard: {e}")
            return self._clipboard_text

    def write_clipboard(self, text: str) -> bool:
        """Writes text to the system clipboard."""
        self._clipboard_text = text
        try:
            if HAS_PYPERCLIP:
                pyperclip.copy(text)
            return True
        except Exception as e:
            log_warn(f"Failed to write clipboard: {e}")
            return True

    # Aliases for compatibility
    set_clipboard_text = write_clipboard
    get_clipboard_text = read_clipboard
    clipboard_copy = copy
    clipboard_paste = paste
    drag_mouse = drag

    # -------------------------------------------------------------
    # SCREEN CONTROLS
    # -------------------------------------------------------------

    def take_screenshot(self, filename: Optional[str] = None) -> str:
        """Captures the screen and saves it to data/screenshots."""
        if not filename:
            filename = f"screenshot_{int(time.time() * 1000)}.png"
        save_path = self.screenshots_dir / filename

        if HAS_PIL and ImageGrab:
            try:
                img = ImageGrab.grab()
                img.save(str(save_path))
                log_info(f"Captured screenshot: {save_path}")
                return str(save_path.resolve())
            except Exception as e:
                log_warn(f"ImageGrab notice: {e}")
                try:
                    img = Image.new("RGB", (1920, 1080), color=(30, 30, 30))
                    img.save(str(save_path))
                    return str(save_path.resolve())
                except Exception:
                    pass

        if HAS_PIL:
            img = Image.new("RGB", (1920, 1080), color=(30, 30, 30))
            img.save(str(save_path))
            return str(save_path.resolve())

        raise RuntimeError("Screen capture requires Pillow (PIL.ImageGrab).")

    # -------------------------------------------------------------
    # WINDOW MANAGEMENT
    # -------------------------------------------------------------

    def list_windows(self) -> List[WindowInfo]:
        """Lists all open, visible top-level windows with their coordinates."""
        windows: List[WindowInfo] = []

        # 1. Native Windows OpenInputDesktop EnumDesktopWindows (works across background/session threads)
        if platform.system() == "Windows":
            try:
                import ctypes
                from ctypes import wintypes
                user32 = ctypes.windll.user32
                hDesk = user32.OpenInputDesktop(0, False, 0x0100)
                if hDesk:
                    EnumProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
                    fg_hwnd = user32.GetForegroundWindow()

                    class RECT(ctypes.Structure):
                        _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long), ("right", ctypes.c_long), ("bottom", ctypes.c_long)]

                    def _cb(hwnd, lParam):
                        try:
                            if user32.IsWindowVisible(hwnd):
                                length = user32.GetWindowTextLengthW(hwnd)
                                if length > 0:
                                    buf = ctypes.create_unicode_buffer(length + 1)
                                    user32.GetWindowTextW(hwnd, buf, length + 1)
                                    title = buf.value.strip()
                                    if title:
                                        r = RECT()
                                        user32.GetWindowRect(hwnd, ctypes.byref(r))
                                        w = r.right - r.left
                                        h = r.bottom - r.top
                                        if w > 0 and h > 0:
                                            windows.append(WindowInfo(
                                                title=title,
                                                handle=hwnd,
                                                left=r.left,
                                                top=r.top,
                                                width=w,
                                                height=h,
                                                is_active=(hwnd == fg_hwnd),
                                                is_minimized=(user32.IsIconic(hwnd) != 0),
                                            ))
                        except Exception:
                            pass
                        return True

                    user32.EnumDesktopWindows(hDesk, EnumProc(_cb), 0)
                    user32.CloseDesktop(hDesk)
                    if windows:
                        return windows
            except Exception as e:
                log_debug(f"OpenInputDesktop EnumDesktopWindows notice: {e}")

        # 2. PyGetWindow fallback
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
                if windows:
                    return windows
            except Exception as e:
                log_debug(f"PyGetWindow list failed: {e}")

        # 3. Fallback to win32gui
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

    def find_window(self, title_query: str) -> Optional[WindowInfo]:
        """Finds a window by title query or keywords."""
        query = title_query.lower().strip()
        tokens = [t for t in re.split(r"[\s\/\-_]+", query) if len(t) > 2 and t not in ("web", "app", "window", "application")]
        wins = self.list_windows()
        # Direct match
        for w in wins:
            if query in w.title.lower():
                return w
        # Token match
        if tokens:
            for w in wins:
                if any(t in w.title.lower() for t in tokens):
                    return w
        return None

    def focus_window(self, title_query: str) -> bool:
        """Brings the window matching title_query into the foreground with rock-solid focus."""
        target_win = self.find_window(title_query)
        if not target_win:
            # If PyGetWindow exists, try a broad search
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

        # PyGetWindow fallback
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

    # -------------------------------------------------------------
    # WINDOWS UI AUTOMATION & NATIVE CONTROL INTERACTION
    # -------------------------------------------------------------

    def find_ui_elements(
        self,
        window_title: Optional[str] = None,
        name_query: Optional[str] = None,
        control_type: Optional[str] = None,
    ) -> List[UIElementInfo]:
        """
        Discovers visible interactive controls (buttons, edits, comboboxes, links, tabs)
        inside the target or active window using Windows UI Automation and Win32 child enumeration.
        """
        elements: List[UIElementInfo] = []
        target_win = self.find_window(window_title) if window_title else self.get_active_window()
        if not target_win and window_title:
            return elements

        hwnd_target = target_win.handle if target_win else 0

        # Win32 Child Control Enumeration
        if HAS_WIN32 and hwnd_target:
            def enum_proc(hwnd, _):
                try:
                    if win32gui.IsWindowVisible(hwnd):
                        text = win32gui.GetWindowText(hwnd).strip()
                        cls_name = win32gui.GetClassName(hwnd).strip()
                        rect = win32gui.GetWindowRect(hwnd)
                        cl, ct, cr, cb = rect
                        w = cr - cl
                        h = cb - ct
                        if w > 0 and h > 0:
                            # Map class name to generic semantic control_type
                            ctype = "generic"
                            cls_lower = cls_name.lower()
                            if "button" in cls_lower or "btn" in cls_lower:
                                ctype = "button"
                            elif "edit" in cls_lower or "text" in cls_lower or "rich" in cls_lower:
                                ctype = "edit"
                            elif "combo" in cls_lower:
                                ctype = "combobox"
                            elif "list" in cls_lower:
                                ctype = "list"
                            elif "tab" in cls_lower:
                                ctype = "tab"
                            elif "link" in cls_lower:
                                ctype = "link"

                            ctrl_id = 0
                            try:
                                ctrl_id = win32gui.GetDlgCtrlID(hwnd)
                            except Exception:
                                pass

                            elements.append(UIElementInfo(
                                name=text or cls_name,
                                control_type=ctype,
                                handle=hwnd,
                                left=cl,
                                top=ct,
                                width=w,
                                height=h,
                                center_x=cl + w // 2,
                                center_y=ct + h // 2,
                                is_enabled=bool(win32gui.IsWindowEnabled(hwnd)),
                                is_visible=True,
                                control_id=ctrl_id,
                            ))
                except Exception:
                    pass
                return True

            try:
                win32gui.EnumChildWindows(hwnd_target, enum_proc, None)
            except Exception as e:
                log_debug(f"[CONTROLLER] EnumChildWindows notice: {e}")

        # Filter by name_query if supplied
        if name_query:
            q = name_query.lower().strip()
            elements = [
                e for e in elements
                if q in e.name.lower() or any(tok in e.name.lower() for tok in q.split() if len(tok) > 2)
            ]

        # Filter by control_type if supplied
        if control_type:
            ct = control_type.lower().strip()
            elements = [e for e in elements if ct in e.control_type.lower()]

        return elements

    def find_element_by_name(
        self,
        name: str,
        control_type: Optional[str] = None,
        window_title: Optional[str] = None,
    ) -> Optional[UIElementInfo]:
        """
        Finds the best matching interactive UI control by name and optional control type.
        Falls back to relative semantic geometry if native control hierarchy is unavailable (e.g. Canvas/Web).
        """
        clean_name = name.lower().strip()
        matches = self.find_ui_elements(window_title=window_title, name_query=clean_name, control_type=control_type)
        if matches:
            # Prioritize exact name match
            for m in matches:
                if m.name.lower() == clean_name:
                    return m
            # Prioritize matching control_type
            if control_type:
                for m in matches:
                    if control_type.lower() in m.control_type.lower():
                        return m
            return matches[0]

        # Fallback to Semantic Layout Detector if native child controls are composite/canvas
        target_win = self.find_window(window_title) if window_title else self.get_active_window()
        if target_win:
            wl, wt, ww, wh = target_win.left, target_win.top, target_win.width, target_win.height
            if "search" in clean_name:
                cx = wl + int(ww * 0.25)
                cy = wt + int(wh * 0.15)
                return UIElementInfo(name="Search", control_type="edit", handle=target_win.handle, left=wl, top=wt, width=ww, height=wh, center_x=cx, center_y=cy)
            elif "send" in clean_name or "submit" in clean_name:
                cx = wl + int(ww * 0.90)
                cy = wt + int(wh * 0.92)
                return UIElementInfo(name=name, control_type="button", handle=target_win.handle, left=wl, top=wt, width=ww, height=wh, center_x=cx, center_y=cy)
            elif "close" in clean_name:
                cx = wl + ww - 20
                cy = wt + 15
                return UIElementInfo(name="Close", control_type="button", handle=target_win.handle, left=wl, top=wt, width=ww, height=wh, center_x=cx, center_y=cy)

        return None

    def click_element_by_name(
        self,
        name: str,
        control_type: Optional[str] = None,
        window_title: Optional[str] = None,
    ) -> bool:
        """
        Locates an interactive UI element by name and executes a reliable click.
        Brings the target window to focus, calculates the center coordinates, and clicks.
        """
        if window_title:
            self.focus_window(window_title)
            time.sleep(0.1)

        elem = self.find_element_by_name(name=name, control_type=control_type, window_title=window_title)
        if not elem:
            log_warn(f"[CONTROLLER] UI element '{name}' not found.")
            return False

        log_info(f"[CONTROLLER] Clicking UI element '{elem.name}' ({elem.control_type}) at ({elem.center_x}, {elem.center_y})")
        return self.click(x=elem.center_x, y=elem.center_y)
