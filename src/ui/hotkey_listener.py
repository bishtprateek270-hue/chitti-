"""
Chitti Safe Global Background Hotkey Listener (Phase 4).
Uses native Win32 RegisterHotKey to capture global shortcuts (Alt+Space, Ctrl+Alt+C)
WITHOUT any low-level keyboard hooks or interference with normal typing.
"""

import ctypes
from ctypes import wintypes
import os
import platform
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from src.utils.logging import log_chitti, log_debug, log_info, log_warn, log_error


# Windows Modifier Key Constants
MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
MOD_WIN = 0x0008
MOD_NOREPEAT = 0x4000

# Virtual-Key Codes
VK_SPACE = 0x20
VK_C = 0x43
VK_MENU = 0x12     # Alt
VK_CONTROL = 0x11  # Ctrl
VK_SHIFT = 0x10    # Shift
VK_ESCAPE = 0x1B

WM_HOTKEY = 0x0312
WM_QUIT = 0x0012


class GlobalHotkeyManager:
    """
    Manages global OS hotkey bindings safely using native Windows RegisterHotKey.
    Does NOT hook into or block standard keyboard events.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._callbacks: Dict[int, List[Callable[[], None]]] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._hotkey_counter = 1
        self._registered_keys: Dict[int, Dict[str, Any]] = {}
        self._thread_id: Optional[int] = None
        self._last_trigger_time = 0.0

    @property
    def is_running(self) -> bool:
        return self._running

    def register_hotkey(
        self,
        modifiers: int = (MOD_ALT | MOD_NOREPEAT),
        vk_code: int = VK_SPACE,
        callback: Optional[Callable[[], None]] = None,
    ) -> int:
        """
        Registers a global hotkey combination.
        Default: Alt + Space (ID #1).
        Ctrl + Alt + C (ID #2).
        """
        with self._lock:
            hk_id = self._hotkey_counter
            self._hotkey_counter += 1

            if callback:
                if hk_id not in self._callbacks:
                    self._callbacks[hk_id] = []
                self._callbacks[hk_id].append(callback)

            self._registered_keys[hk_id] = {
                "modifiers": modifiers,
                "vk_code": vk_code,
            }
            log_debug(f"[HOTKEY] Registered hotkey #{hk_id} (mod={modifiers}, vk={vk_code})")
            return hk_id

    def unregister_hotkey(self, hk_id: int) -> bool:
        """Removes a registered hotkey."""
        with self._lock:
            if hk_id in self._callbacks:
                del self._callbacks[hk_id]
            if hk_id in self._registered_keys:
                del self._registered_keys[hk_id]
                return True
            return False

    def simulate_hotkey_press(self, hk_id: int = 1):
        """Dispatches a hotkey trigger event with debouncing."""
        now = time.time()
        if now - self._last_trigger_time < 0.4:
            return
        self._last_trigger_time = now

        with self._lock:
            callbacks = list(self._callbacks.get(hk_id, self._callbacks.get(1, [])))

        log_chitti(f"[HOTKEY] ⚡ Hotkey #{hk_id} activated. Triggering {len(callbacks)} action(s).")
        for cb in callbacks:
            try:
                cb()
            except Exception as e:
                log_warn(f"[HOTKEY] Callback exception for hotkey #{hk_id}: {e}")

    def start(self):
        """Starts the background Win32 message loop thread."""
        if self._running:
            return

        if platform.system() != "Windows":
            log_warn("[HOTKEY] Native Windows hotkeys are only supported on Windows OS.")
            self._running = True
            return

        self._running = True
        self._thread = threading.Thread(target=self._msg_loop, daemon=True, name="ChittiHotkeyThread")
        self._thread.start()
        log_info("[HOTKEY] Safe Global Hotkey Listener started (Alt+Space, Ctrl+Space, Alt+C).")

    def stop(self):
        """Stops the message loop and unregisters all hotkeys."""
        self._running = False
        if platform.system() == "Windows" and self._thread_id:
            try:
                user32 = ctypes.windll.user32
                user32.PostThreadMessageW(self._thread_id, WM_QUIT, 0, 0)
            except Exception:
                pass

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        log_info("[HOTKEY] Safe Global Hotkey Listener stopped.")

    def _msg_loop(self):
        """Standard Win32 message loop using RegisterHotKey with robust GetAsyncKeyState edge polling."""
        if platform.system() == "Windows":
            try:
                user32 = ctypes.windll.user32
                hDesk = user32.OpenDesktopW("Default", 0, False, 0x01FF)
                if hDesk:
                    user32.SetThreadDesktop(hDesk)
            except Exception:
                pass

        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        self._thread_id = kernel32.GetCurrentThreadId()

        # Explicit Win32 function signatures to prevent 64-bit calling convention issues
        try:
            user32.RegisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int, wintypes.UINT, wintypes.UINT]
            user32.RegisterHotKey.restype = wintypes.BOOL
            user32.UnregisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int]
            user32.UnregisterHotKey.restype = wintypes.BOOL
            user32.PeekMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT, wintypes.UINT]
            user32.PeekMessageW.restype = wintypes.BOOL
            user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
            user32.GetAsyncKeyState.restype = wintypes.SHORT
        except Exception:
            pass

        # Register hotkeys on this thread
        with self._lock:
            for hk_id, spec in self._registered_keys.items():
                try:
                    res = user32.RegisterHotKey(None, hk_id, spec["modifiers"], spec["vk_code"])
                    if not res:
                        log_debug(f"[HOTKEY] RegisterHotKey returned {res} for id {hk_id} (fallback active).")
                except Exception as e:
                    log_debug(f"[HOTKEY] RegisterHotKey exception: {e}")

        msg = wintypes.MSG()
        prev_alt_space = False
        prev_ctrl_space = False
        prev_ctrl_alt_c = False
        prev_alt_c = False

        try:
            while self._running:
                # 1. Process Win32 HotKey Messages if posted
                while user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, 1):  # PM_REMOVE = 1
                    if msg.message == WM_HOTKEY:
                        hk_id = msg.wParam
                        self.simulate_hotkey_press(hk_id)
                    elif msg.message == WM_QUIT:
                        return
                    user32.TranslateMessage(ctypes.byref(msg))
                    user32.DispatchMessageW(ctypes.byref(msg))

                # 2. Safe, read-only state query with rising-edge detection (handles both main and left/right keys)
                alt = bool(user32.GetAsyncKeyState(VK_MENU) & 0x8000) or bool(user32.GetAsyncKeyState(0xA4) & 0x8000) or bool(user32.GetAsyncKeyState(0xA5) & 0x8000)
                ctrl = bool(user32.GetAsyncKeyState(VK_CONTROL) & 0x8000) or bool(user32.GetAsyncKeyState(0xA2) & 0x8000) or bool(user32.GetAsyncKeyState(0xA3) & 0x8000)
                space = bool(user32.GetAsyncKeyState(VK_SPACE) & 0x8000)
                c_key = bool(user32.GetAsyncKeyState(VK_C) & 0x8000)

                is_alt_space = alt and space and not ctrl
                is_ctrl_space = ctrl and space and not alt
                is_ctrl_alt_c = ctrl and alt and c_key
                is_alt_c = alt and c_key and not ctrl

                if (is_alt_space and not prev_alt_space) or (is_ctrl_space and not prev_ctrl_space):
                    self.simulate_hotkey_press(1)
                elif (is_ctrl_alt_c and not prev_ctrl_alt_c) or (is_alt_c and not prev_alt_c):
                    self.simulate_hotkey_press(1)

                prev_alt_space = is_alt_space
                prev_ctrl_space = is_ctrl_space
                prev_ctrl_alt_c = is_ctrl_alt_c
                prev_alt_c = is_alt_c

                time.sleep(0.02)
        finally:
            with self._lock:
                for hk_id in self._registered_keys:
                    try:
                        user32.UnregisterHotKey(None, hk_id)
                    except Exception:
                        pass
