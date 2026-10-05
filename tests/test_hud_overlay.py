"""
Test Suite for Phase 4: Floating Desktop HUD & Global Hotkey Overlay.
Tests:
- HUDState and Visualizer models
- GlobalHotkeyManager registration and simulated hotkey dispatch
- FloatingHUD thread-safe queue updates, mode switching, and step progress
- Dark glassmorphism theme consistency
"""

import time
from unittest.mock import MagicMock
import pytest

from src.ui.visualizer import HUDMode, HUDState, HUDStepInfo, HUD_THEME
from src.ui.hotkey_listener import GlobalHotkeyManager, VK_SPACE, MOD_ALT
from src.ui.hud_overlay import FloatingHUD


class TestHUDOverlayPhase4:
    """Test suite for Phase 4 HUD and Hotkey components."""

    def test_hud_state_and_visualizer_models(self):
        """Verify HUDState progress calculation, color mapping, and icon badges."""
        state = HUDState()
        assert state.mode == HUDMode.IDLE
        assert state.mode_icon == "🤖"
        assert state.progress_ratio == 0.0

        # Update mode
        state.mode = HUDMode.EXECUTING
        assert state.mode_color == HUD_THEME["accent_orange"]
        assert state.mode_icon == "⚡"

        # Update step progress
        state.current_step = 2
        state.total_steps = 4
        assert state.progress_ratio == 0.5

    def test_global_hotkey_registration_and_simulation(self):
        """Verify hotkey registration and simulated trigger callback."""
        manager = GlobalHotkeyManager()
        called = []

        hk_id = manager.register_hotkey(
            modifiers=MOD_ALT,
            vk_code=VK_SPACE,
            callback=lambda: called.append("hotkey_pressed"),
        )
        assert hk_id == 1

        # Simulate hotkey press
        manager.simulate_hotkey_press(hk_id=1)
        assert len(called) == 1
        assert called[0] == "hotkey_pressed"

        # Unregister
        unreg = manager.unregister_hotkey(hk_id=1)
        assert unreg is True

    def test_floating_hud_state_queue_and_apis(self):
        """Verify FloatingHUD non-blocking queue operations."""
        submitted = []
        hud = FloatingHUD(on_command_submit=lambda txt: submitted.append(txt))

        # Test state setters
        hud.set_mode(HUDMode.THINKING, "Thinking about solution...")
        hud.set_progress(1, 3, "Opening VS Code")
        hud.set_response("Here is the answer")

        # Verify items queued
        assert not hud._queue.empty()
        item = hud._queue.get()
        assert item[0] == "set_mode"
        assert item[1] == HUDMode.THINKING

    def test_floating_hud_command_submission(self):
        """Verify submitting command from HUD triggers callback."""
        submitted = []
        hud = FloatingHUD(on_command_submit=lambda txt: submitted.append(txt))

        hud._submit_text("look at my screen")
        time.sleep(0.1)

        assert len(submitted) == 1
        assert submitted[0] == "look at my screen"

    def test_hud_theme_palette(self):
        """Verify dark glassmorphism theme colors."""
        assert HUD_THEME["bg_dark"] == "#0d1117"
        assert HUD_THEME["accent_blue"] == "#58a6ff"
        assert HUD_THEME["accent_cyan"] == "#39c5cf"
        assert HUD_THEME["accent_green"] == "#3fb950"
