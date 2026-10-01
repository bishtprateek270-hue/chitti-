"""
Chitti Desktop UI Package (Phase 4).
Exposes the floating dynamic island HUD overlay, visualizer models, and global hotkey listener.
"""

from src.ui.visualizer import HUDMode, HUDState, HUDStepInfo, HUD_THEME
from src.ui.hotkey_listener import GlobalHotkeyManager
from src.ui.hud_overlay import FloatingHUD

__all__ = [
    "HUDMode",
    "HUDState",
    "HUDStepInfo",
    "HUD_THEME",
    "GlobalHotkeyManager",
    "FloatingHUD",
]
