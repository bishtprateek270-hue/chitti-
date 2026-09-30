"""
Chitti Screen Reader & Visual Frame Capture.
Captures full-screen screenshots, active window boundaries, cropped ROI regions,
and prepares visual base64 payloads for multimodal model inference.
"""

import base64
import ctypes
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.utils.logging import log_debug, log_info, log_warn

# GUI Automation and Image Grab Imports with Fallbacks
try:
    from PIL import Image, ImageGrab
    HAS_PIL = True
except Exception:
    ImageGrab = None
    HAS_PIL = False

try:
    import win32gui
    import win32process
    import psutil
    HAS_WIN32 = True
except Exception:
    win32gui = None
    win32process = None
    psutil = None
    HAS_WIN32 = False

try:
    import pygetwindow as gw
    HAS_PYGETWINDOW = True
except Exception:
    gw = None
    HAS_PYGETWINDOW = False


@dataclass
class WindowRect:
    """Represents the bounding geometry and metadata of an on-screen window."""
    title: str
    handle: int
    left: int
    top: int
    width: int
    height: int
    process_name: str = ""
    is_active: bool = True

    @property
    def box(self) -> Tuple[int, int, int, int]:
        """Returns (left, top, right, bottom) bounding box tuple."""
        return (self.left, self.top, self.left + self.width, self.top + self.height)


class ScreenReader:
    """Provides high-fidelity visual capture of screens and windows."""

    def __init__(self, output_dir: str = "data/screenshots/vision"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_screen_size(self) -> Tuple[int, int]:
        """Returns primary display resolution (width, height)."""
        try:
            user32 = ctypes.windll.user32
            return (user32.GetSystemMetrics(0), user32.GetSystemMetrics(1))
        except Exception:
            return (1920, 1080)

    def capture_full_screen(self, output_path: Optional[str] = None) -> str:
        """
        Captures the entire desktop display across all primary/virtual monitors.
        Returns the absolute filepath to the saved PNG image.
        """
        if not output_path:
            filename = f"fullscreen_{int(time.time() * 1000)}.png"
            target = self.output_dir / filename
        else:
            target = Path(output_path)
            target.parent.mkdir(parents=True, exist_ok=True)

        if HAS_PIL and ImageGrab:
            try:
                img = ImageGrab.grab(all_screens=True)
                img.save(str(target))
                log_info(f"[VISION] Captured full-screen screenshot: {target}")
                return str(target.resolve())
            except Exception as e:
                log_warn(f"[VISION] ImageGrab grab failed: {e}. Falling back to default PIL grab.")
                try:
                    img = ImageGrab.grab()
                    img.save(str(target))
                    return str(target.resolve())
                except Exception as inner_e:
                    log_warn(f"[VISION] Secondary grab failed: {inner_e}")

        # Synthetic fallback for headless test environments
        if HAS_PIL:
            w, h = self.get_screen_size()
            img = Image.new("RGB", (w, h), color=(35, 39, 42))
            img.save(str(target))
            return str(target.resolve())

        raise RuntimeError("ScreenReader requires Pillow (PIL) for image operations.")

    def get_active_window(self) -> Optional[WindowRect]:
        """Inspects and returns the active foreground window geometry and title."""
        if HAS_WIN32 and win32gui:
            try:
                hwnd = win32gui.GetForegroundWindow()
                if hwnd and win32gui.IsWindowVisible(hwnd):
                    title = win32gui.GetWindowText(hwnd)
                    rect = win32gui.GetWindowRect(hwnd)
                    left, top, right, bottom = rect
                    w = max(0, right - left)
                    h = max(0, bottom - top)

                    pname = ""
                    if psutil and win32process:
                        try:
                            _, pid = win32process.GetWindowThreadProcessId(hwnd)
                            pname = psutil.Process(pid).name()
                        except Exception:
                            pname = ""

                    return WindowRect(
                        title=title or "Active Window",
                        handle=hwnd,
                        left=left,
                        top=top,
                        width=w,
                        height=h,
                        process_name=pname,
                        is_active=True,
                    )
            except Exception as e:
                log_debug(f"[VISION] win32gui GetForegroundWindow error: {e}")

        if HAS_PYGETWINDOW and gw:
            try:
                w = gw.getActiveWindow()
                if w and w.title:
                    return WindowRect(
                        title=w.title,
                        handle=w._hWnd if hasattr(w, "_hWnd") else 0,
                        left=w.left,
                        top=w.top,
                        width=w.width,
                        height=w.height,
                        is_active=True,
                    )
            except Exception as e:
                log_debug(f"[VISION] PyGetWindow active window error: {e}")

        # Fallback dummy window rect
        w_size, h_size = self.get_screen_size()
        return WindowRect(
            title="Desktop / Foreground Window",
            handle=0,
            left=0,
            top=0,
            width=w_size,
            height=h_size,
            is_active=True,
        )

    def capture_active_window(self, output_path: Optional[str] = None) -> Tuple[str, Optional[WindowRect]]:
        """
        Captures only the bounding box of the active foreground window.
        Returns the saved screenshot path and the WindowRect metadata.
        """
        active_win = self.get_active_window()
        if not output_path:
            filename = f"active_win_{int(time.time() * 1000)}.png"
            target = self.output_dir / filename
        else:
            target = Path(output_path)
            target.parent.mkdir(parents=True, exist_ok=True)

        if not active_win or active_win.width <= 0 or active_win.height <= 0:
            full_path = self.capture_full_screen(output_path=str(target))
            return full_path, active_win

        if HAS_PIL and ImageGrab:
            try:
                bbox = (active_win.left, active_win.top, active_win.left + active_win.width, active_win.top + active_win.height)
                img = ImageGrab.grab(bbox=bbox)
                img.save(str(target))
                log_info(f"[VISION] Captured active window ('{active_win.title}'): {target}")
                return str(target.resolve()), active_win
            except Exception as e:
                log_warn(f"[VISION] Active window bbox grab failed: {e}. Falling back to full screen.")

        full_path = self.capture_full_screen(output_path=str(target))
        return full_path, active_win

    def crop_region(self, image_path: str, rect: Tuple[int, int, int, int], output_path: Optional[str] = None) -> str:
        """
        Crops a specific bounding box (left, top, right, bottom) from an existing screenshot.
        """
        if not HAS_PIL:
            raise RuntimeError("Pillow is required for image cropping.")

        img = Image.open(image_path)
        cropped = img.crop(rect)
        if not output_path:
            filename = f"crop_{int(time.time() * 1000)}.png"
            target = self.output_dir / filename
        else:
            target = Path(output_path)
            target.parent.mkdir(parents=True, exist_ok=True)

        cropped.save(str(target))
        return str(target.resolve())

    @staticmethod
    def encode_base64(image_path: str) -> str:
        """Encodes an image file as a base64 string for multimodal LLM vision APIs."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
