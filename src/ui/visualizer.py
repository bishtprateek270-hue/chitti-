"""
Chitti HUD State & Visualizer Models (Phase 4).
Provides state definitions, progress calculation, color palettes,
and waveform animations for the floating desktop dynamic island overlay.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class HUDMode(str, Enum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"
    EXECUTING = "EXECUTING"
    SUCCESS = "SUCCESS"
    ERROR = "ERROR"


# Curated Dark Glassmorphism Color Palette
HUD_THEME = {
    "bg_dark": "#0d1117",
    "bg_card": "#161b22",
    "bg_input": "#21262d",
    "border": "#30363d",
    "text_primary": "#f0f6fc",
    "text_secondary": "#8b949e",
    "accent_blue": "#58a6ff",
    "accent_cyan": "#39c5cf",
    "accent_green": "#3fb950",
    "accent_orange": "#d29922",
    "accent_red": "#f85149",
    "accent_purple": "#bc8cff",
}


@dataclass
class HUDStepInfo:
    """Represents a single step in a multi-step plan execution."""
    step_id: int
    description: str
    status: str = "PENDING"  # PENDING, RUNNING, COMPLETED, FAILED


@dataclass
class HUDState:
    """Represents the complete runtime visual state of the floating HUD."""
    mode: HUDMode = HUDMode.IDLE
    status_text: str = "Chitti is ready"
    active_prompt: str = ""
    current_step: int = 0
    total_steps: int = 0
    steps: List[HUDStepInfo] = field(default_factory=list)
    audio_level: float = 0.0  # 0.0 to 1.0 for voice pulse
    last_response: str = ""
    is_visible: bool = False
    updated_at: float = field(default_factory=time.time)

    @property
    def progress_ratio(self) -> float:
        if self.total_steps <= 0:
            return 0.0
        return min(1.0, max(0.0, self.current_step / self.total_steps))

    @property
    def mode_color(self) -> str:
        colors = {
            HUDMode.IDLE: HUD_THEME["accent_blue"],
            HUDMode.LISTENING: HUD_THEME["accent_cyan"],
            HUDMode.THINKING: HUD_THEME["accent_purple"],
            HUDMode.SPEAKING: HUD_THEME["accent_green"],
            HUDMode.EXECUTING: HUD_THEME["accent_orange"],
            HUDMode.SUCCESS: HUD_THEME["accent_green"],
            HUDMode.ERROR: HUD_THEME["accent_red"],
        }
        return colors.get(self.mode, HUD_THEME["accent_blue"])

    @property
    def mode_icon(self) -> str:
        icons = {
            HUDMode.IDLE: "⚪",
            HUDMode.LISTENING: "🎙️",
            HUDMode.THINKING: "🧠",
            HUDMode.SPEAKING: "🔊",
            HUDMode.EXECUTING: "⚡",
            HUDMode.SUCCESS: "✅",
            HUDMode.ERROR: "⚠️",
        }
        return icons.get(self.mode, "🤖")
