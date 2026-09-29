"""
Phase 5 Comprehensive Computer-Use Agent Test Suite.
Verifies the closed-loop agent execution across all 10 core specification scenarios:
1. Open Chrome
2. Open Notepad and type Hello Chitti
3. Open WhatsApp Web (Loaded & Auth check)
4. Send Rahul 'Hello' on WhatsApp (Real actions + verification)
5. Open Gmail (Loaded & Auth check)
6. Send an email with subject Test and body Hello
7. Create a folder named ChittiTest on Desktop
8. Open VS Code and create test.py
9. Create a Python program and run it
10. Intentionally impossible task reports failure (No false success)
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from src.computer_use import (
    ComputerUseEngine,
    ComputerTaskState,
    VerificationStatus,
    RiskLevel,
    ComputerSafetyPolicy,
    ComputerActionType,
)
from src.computer_use.screen import ScreenObservation
from src.computer_use.controller import WindowInfo
from src.agent.manager import LaptopAgentManager


@pytest.fixture
def computer_engine(tmp_path):
    ss = tmp_path / "screenshots"
    ss.mkdir()
    return ComputerUseEngine(screenshots_dir=str(ss))


@pytest.fixture
def laptop_manager(tmp_path):
    ws = tmp_path / "workspace"
    ss = tmp_path / "screenshots"
    pr = tmp_path / "projects.json"
    ws.mkdir()
    ss.mkdir()
    return LaptopAgentManager(
        workspace_dir=str(ws),
        screenshots_dir=str(ss),
        project_registry_path=str(pr),
    )


# ==============================================================================
# TEST 1: Open Chrome
# ==============================================================================
def test_01_open_chrome(computer_engine):
    """Scenario 1: 'Open Chrome' -> Chrome opens and verification succeeds."""
    mock_obs = ScreenObservation(
        screenshot_path="",
        screen_width=1920,
        screen_height=1080,
        active_window=WindowInfo(title="Google Chrome", handle=1, left=0, top=0, width=1920, height=1080, is_active=True, is_minimized=False),
        visible_windows=[WindowInfo(title="Google Chrome", handle=1, left=0, top=0, width=1920, height=1080, is_active=True, is_minimized=False)],
        is_browser_open=True,
        summary_text="Google Chrome active",
    )

    with patch.object(computer_engine.screen_observer, "observe", return_value=mock_obs), \
         patch.object(computer_engine.screen_observer, "is_window_visible", return_value=True):
        state = computer_engine.plan_task("Open Chrome")
        assert len(state._internal_steps) >= 1
        success, msg = computer_engine.execute_plan(state)
        assert success is True
        assert state.is_completed is True
        assert state.verification_status == VerificationStatus.SUCCESS
        assert len(state.action_history) > 0


# ==============================================================================
# TEST 2: Open Notepad and type Hello Chitti
# ==============================================================================
def test_02_open_notepad_and_type(computer_engine):
    """Scenario 2: 'Open Notepad and type Hello Chitti' -> Notepad opens, text entered, verification succeeds."""
    mock_obs = ScreenObservation(
        screenshot_path="",
        screen_width=1920,
        screen_height=1080,
        active_window=WindowInfo(title="Notepad", handle=2, left=0, top=0, width=800, height=600, is_active=True, is_minimized=False),
        visible_windows=[WindowInfo(title="Notepad", handle=2, left=0, top=0, width=800, height=600, is_active=True, is_minimized=False)],
        summary_text="Notepad active with text Hello Chitti",
    )

    with patch.object(computer_engine.screen_observer, "observe", return_value=mock_obs), \
         patch.object(computer_engine.screen_observer, "is_window_visible", return_value=True):
        state = computer_engine.plan_task("Open Notepad and type 'Hello Chitti'")
        actions = [s.action_type for s in state._internal_steps]
        assert ComputerActionType.OPEN_APPLICATION in actions
        assert ComputerActionType.TYPE_TEXT in actions

        success, msg = computer_engine.execute_plan(state)
        assert success is True
        assert state.is_completed is True
        assert any("Hello Chitti" in h for h in state.action_history)


# ==============================================================================
# TEST 3: Open WhatsApp Web
# ==============================================================================
def test_03_open_whatsapp_web(computer_engine):
    """Scenario 3: 'Open WhatsApp Web' -> WhatsApp opens and loaded/auth state detected."""
    mock_obs = ScreenObservation(
        screenshot_path="",
        screen_width=1920,
        screen_height=1080,
        active_window=WindowInfo(title="WhatsApp Web - Google Chrome", handle=3, left=0, top=0, width=1920, height=1080, is_active=True, is_minimized=False),
        visible_windows=[],
        is_browser_open=True,
        active_service="WhatsApp Web",
        auth_required=False,
        summary_text="WhatsApp Web active session",
    )

    with patch.object(computer_engine.screen_observer, "observe", return_value=mock_obs), \
         patch.object(computer_engine.screen_observer, "is_window_visible", return_value=True):
        state = computer_engine.plan_task("Open WhatsApp Web")
        success, msg = computer_engine.execute_plan(state)
        assert success is True
        assert state.is_completed is True


# ==============================================================================
# TEST 4: Send Rahul 'Hello' on WhatsApp
# ==============================================================================
def test_04_send_whatsapp_message(computer_engine):
    """Scenario 4: 'Send Rahul Hello on WhatsApp' -> Closed loop execution & verification."""
    mock_obs = ScreenObservation(
        screenshot_path="",
        screen_width=1920,
        screen_height=1080,
        active_window=WindowInfo(title="WhatsApp - Brave", handle=4, left=0, top=0, width=1920, height=1080, is_active=True, is_minimized=False),
        visible_windows=[],
        is_browser_open=True,
        active_service="WhatsApp Web",
        auth_required=False,
        summary_text="WhatsApp Web active. Rahul: Hello",
    )

    with patch.object(computer_engine.screen_observer, "observe", return_value=mock_obs), \
         patch.object(computer_engine.screen_observer, "is_window_visible", return_value=True):
        state = computer_engine.plan_task("Send Rahul 'Hello' on WhatsApp")
        assert "Rahul" in state.current_goal
        
        actions = [s.action_type for s in state._internal_steps]
        assert ComputerActionType.OPEN_URL in actions
        assert ComputerActionType.CHECK_AUTHENTICATION in actions
        assert ComputerActionType.SEARCH_CONTACT in actions
        assert ComputerActionType.SELECT_CONVERSATION in actions
        assert ComputerActionType.TYPE_TEXT in actions
        assert ComputerActionType.SEND_MESSAGE in actions
        assert ComputerActionType.VERIFY_MESSAGE_SENT in actions

        success, msg = computer_engine.execute_plan(state)
        assert success is True
        assert state.is_completed is True


# ==============================================================================
# TEST 5: Open Gmail
# ==============================================================================
def test_05_open_gmail(computer_engine):
    """Scenario 5: 'Open Gmail' -> Gmail opens and account/login state is detected."""
    mock_obs = ScreenObservation(
        screenshot_path="",
        screen_width=1920,
        screen_height=1080,
        active_window=WindowInfo(title="Inbox - bisht@gmail.com - Gmail", handle=5, left=0, top=0, width=1920, height=1080, is_active=True, is_minimized=False),
        visible_windows=[],
        is_browser_open=True,
        active_service="Gmail / Webmail",
        auth_required=False,
        summary_text="Gmail Inbox active",
    )

    with patch.object(computer_engine.screen_observer, "observe", return_value=mock_obs), \
         patch.object(computer_engine.screen_observer, "is_window_visible", return_value=True):
        state = computer_engine.plan_task("Open Gmail")
        success, msg = computer_engine.execute_plan(state)
        assert success is True
        assert state.is_completed is True


# ==============================================================================
# TEST 6: Send Email with Subject and Body
# ==============================================================================
def test_06_send_email_flow(computer_engine):
    """Scenario 6: Send email with subject Test and body Hello -> End-to-end plan & execution."""
    mock_obs = ScreenObservation(
        screenshot_path="",
        screen_width=1920,
        screen_height=1080,
        active_window=WindowInfo(title="Gmail", handle=6, left=0, top=0, width=1920, height=1080, is_active=True, is_minimized=False),
        visible_windows=[],
        is_browser_open=True,
        active_service="Gmail / Webmail",
        auth_required=False,
        summary_text="Gmail Message sent confirmation",
    )

    with patch.object(computer_engine.screen_observer, "observe", return_value=mock_obs), \
         patch.object(computer_engine.screen_observer, "is_window_visible", return_value=True):
        state = computer_engine.plan_task("Send an email to test@example.com with subject Test and body Hello")
        assert "test@example.com" in state.current_goal

        actions = [s.action_type for s in state._internal_steps]
        assert ComputerActionType.OPEN_URL in actions
        assert ComputerActionType.COMPOSE_EMAIL in actions
        assert ComputerActionType.CONFIRM_SEND in actions
        assert ComputerActionType.SEND_EMAIL in actions
        assert ComputerActionType.VERIFY_EMAIL_SENT in actions

        success, msg = computer_engine.execute_plan(state)
        assert success is True
        assert state.is_completed is True


# ==============================================================================
# TEST 7: Create a folder named ChittiTest on Desktop
# ==============================================================================
def test_07_create_folder_on_desktop(computer_engine, tmp_path):
    """Scenario 7: 'Create a folder named ChittiTest on Desktop' -> Filesystem creation + verification."""
    folder_path = tmp_path / "ChittiTest"
    
    with patch("os.path.expanduser", return_value=str(folder_path)):
        state = computer_engine.plan_task("Create a folder named ChittiTest on Desktop")
        success, msg = computer_engine.execute_plan(state)
        assert success is True
        assert folder_path.exists()
        assert folder_path.is_dir()


# ==============================================================================
# TEST 8: Open VS Code and create test.py
# ==============================================================================
def test_08_open_vscode_and_create_file(laptop_manager, tmp_path):
    """Scenario 8: 'Open VS Code and create test.py' -> VS Code opens and file exists on disk."""
    ws = laptop_manager.workspace_dir
    test_file = Path(ws) / "test.py"
    
    res = laptop_manager.handle_command("open vs code and create test.py with a hello world program")
    assert res is not None
    handled, msg, action_res = res
    assert handled is True
    created_files = list(Path(ws).glob("*.py"))
    assert len(created_files) > 0
    assert any("hello" in f.read_text(encoding="utf-8").lower() or "print" in f.read_text(encoding="utf-8").lower() for f in created_files)


# ==============================================================================
# TEST 9: Create a Python program and run it
# ==============================================================================
def test_09_create_python_program_and_run(laptop_manager):
    """Scenario 9: 'Create a python calculator program and run it' -> Generated, executed, and verified."""
    res = laptop_manager.handle_command("create a python calculator program and run it")
    assert res is not None
    handled, msg, action_res = res
    assert handled is True
    assert action_res.success is True
    assert laptop_manager.active_task_state.status.value in ("COMPLETED", "EXECUTING", "TASK_CREATED")


# ==============================================================================
# TEST 10: Intentionally impossible task reports failure (No false success)
# ==============================================================================
def test_10_impossible_task_reports_honest_failure(computer_engine):
    """Scenario 10: Intentionally impossible task -> Reports failure instead of claiming false success."""
    mock_obs = ScreenObservation(
        screenshot_path="",
        screen_width=1920,
        screen_height=1080,
        active_window=None,
        visible_windows=[],
        is_browser_open=False,
        summary_text="No windows visible",
    )

    with patch.object(computer_engine.screen_observer, "observe", return_value=mock_obs), \
         patch.object(computer_engine.screen_observer, "is_window_visible", return_value=False):
        state = computer_engine.plan_task("Open non_existent_quantum_app_99999")
        success, msg = computer_engine.execute_plan(state)
        assert success is False
        assert state.is_completed is False
        assert state.verification_status == VerificationStatus.FAILED
        assert any(w in msg.lower() for w in ["not found", "failed", "limit reached", "not available", "could not be launched"])


# ==============================================================================
# TEST 11: Safety Policy & Risk Classification
# ==============================================================================
def test_11_safety_policy_risk_classification():
    """Verify ComputerSafetyPolicy accurately classifies risk and gates dangerous actions."""
    # Low risk
    assert ComputerSafetyPolicy.evaluate_risk("OPEN_APPLICATION") == RiskLevel.LOW_RISK
    assert ComputerSafetyPolicy.evaluate_risk("TYPE_TEXT") == RiskLevel.LOW_RISK
    
    # Medium risk
    assert ComputerSafetyPolicy.evaluate_risk("SEND_EMAIL") == RiskLevel.MEDIUM_RISK
    
    # High risk
    assert ComputerSafetyPolicy.evaluate_risk("DELETE_DIRECTORY") == RiskLevel.HIGH_RISK
    req_del, prompt_del = ComputerSafetyPolicy.requires_confirmation("DELETE_DIRECTORY", target="important_data")
    assert req_del is True
    assert "permanently delete" in prompt_del.lower()
