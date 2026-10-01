"""
Chitti Global Background Hotkey Listener (Phase 4).
Captures global system hotkeys (e.g. Alt+Space, Ctrl+Alt+C) across Windows
to instantly summon or dismiss the floating dynamic island overlay.
"""

import ctypes
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
VK_ESCAPE = 0x1B


class GlobalHotkeyManager:
    """
    Manages global OS hotkey bindings using the native Windows Win32 API.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._callbacks: Dict[int, List[Callable[[], None]]] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._hotkey_counter = 1
        self._registered_keys: Dict[int, Dict[str, Any]] = {}

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
        Default: Alt + Space.
        Returns the hotkey ID.
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
        """Simulates a hotkey trigger event for automated testing or CLI triggers."""
        with self._lock:
            callbacks = list(self._callbacks.get(hk_id, []))

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
        log_info("[HOTKEY] Global Hotkey Listener started (Alt+Space active).")

    def stop(self):
        """Stops the message loop and unregisters hotkeys."""
        self._running = False
        if platform.system() == "Windows":
            try:
                # Post WM_QUIT to thread
                user32 = ctypes.windll.user32
                user32.PostQuitMessage(0)
            except Exception:
                pass

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        log_info("[HOTKEY] Global Hotkey Listener stopped.")

    def _msg_loop(self):
        """Win32 Message Loop listening for WM_HOTKEY."""
        user32 = ctypes.windll.user32

        # Register all pending hotkeys on this thread
        with self._lock:
            for hk_id, spec in self._registered_keys.items():
                res = user32.RegisterHotKey(None, hk_id, spec["modifiers"], spec["vk_code"])
                if not res:
                    log_warn(f"[HOTKEY] Win32 RegisterHotKey returned 0 for #{hk_id} (code may already be bound).")

        WM_HOTKEY = 0x0312
        msg = ctypes.wintypes.MSG()

        try:
            while self._running:
                # Peek or Get message with timeout
                if user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, 1):  # PM_REMOVE = 1
                    if msg.message == WM_HOTKEY:
                        hk_id = msg.wParam
                        self.simulate_hotkey_press(hk_id)
                    user32.TranslateMessage(ctypes.byref(msg))
                    user32.DispatchMessageW(ctypes.byref(msg))
                else:
                    time.sleep(0.05)
        finally:
            # Unregister on exit
            with self._lock:
                for hk_id in self._registered_keys:
                    user32.UnregisterHotKey(None, hk_id)
