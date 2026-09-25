"""
Comprehensive Unit & Integration Tests for Chitti Phase 5: Controlled Laptop Agent Foundation.
Tests action registry, parser, validator, executor, multilingual commands, safety gates, and destructive confirmations.
"""

import os
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock

from src.agent.actions import ActionType, RiskLevel, StructuredAction, ActionRegistry
from src.agent.parser import ActionParser
from src.agent.validator import ActionValidator
from src.agent.executor import ActionExecutor
from src.agent.manager import LaptopAgentManager


@pytest.fixture
def agent_manager(tmp_path):
    workspace = tmp_path / "workspace"
    screenshots = tmp_path / "screenshots"
    workspace.mkdir()
    screenshots.mkdir()
    return LaptopAgentManager(workspace_dir=str(workspace), screenshots_dir=str(screenshots))


# 1. SPECIFICATION TEST: Application Opening (English, Hindi, Hinglish)
def test_parse_open_application_english():
    action = ActionParser.parse_command("Open Chrome")
    assert action is not None
    assert action.action == ActionType.OPEN_APPLICATION
    assert action.parameters["target"].lower() == "chrome"


def test_parse_open_application_hindi():
    action = ActionParser.parse_command("Chrome kholo")
    assert action is not None
    assert action.action == ActionType.OPEN_APPLICATION
    assert action.parameters["target"].lower() == "chrome"

    action_devanagari = ActionParser.parse_command("Chrome खोलो")
    assert action_devanagari is not None
    assert action_devanagari.action == ActionType.OPEN_APPLICATION


def test_parse_open_application_hinglish():
    action = ActionParser.parse_command("Chrome ko open karo")
    assert action is not None
    assert action.action == ActionType.OPEN_APPLICATION
    assert action.parameters["target"].lower() == "chrome"

    action_chalao = ActionParser.parse_command("VS Code chala do")
    assert action_chalao is not None
    assert action_chalao.action == ActionType.OPEN_APPLICATION
    assert "code" in action_chalao.parameters["target"].lower() or "vs code" in action_chalao.parameters["target"].lower()


# 2. SPECIFICATION TEST: Folder Opening
def test_parse_open_folder():
    action = ActionParser.parse_command("Open Downloads")
    assert action is not None
    assert action.action == ActionType.OPEN_FOLDER
    assert "downloads" in action.parameters["target"].lower()

    action_hi = ActionParser.parse_command("Downloads folder kholo")
    assert action_hi is not None
    assert action_hi.action == ActionType.OPEN_FOLDER


# 3. SPECIFICATION TEST: Screenshot
def test_parse_take_screenshot():
    action = ActionParser.parse_command("Take a screenshot")
    assert action is not None
    assert action.action == ActionType.TAKE_SCREENSHOT

    action_hi = ActionParser.parse_command("Screenshot le lo")
    assert action_hi is not None
    assert action_hi.action == ActionType.TAKE_SCREENSHOT

    action_capture = ActionParser.parse_command("Screen capture karo")
    assert action_capture is not None
    assert action_capture.action == ActionType.TAKE_SCREENSHOT


# 4. SPECIFICATION TEST: Non-Agent Queries Return None
def test_general_knowledge_no_laptop_action():
    action1 = ActionParser.parse_command("What is deep learning?")
    assert action1 is None

    action2 = ActionParser.parse_command("Explain CNN.")
    assert action2 is None

    action3 = ActionParser.parse_command("Solve this LeetCode problem.")
    assert action3 is None


def test_personal_queries_no_laptop_action():
    action1 = ActionParser.parse_command("What is my name?")
    assert action1 is None

    action2 = ActionParser.parse_command("Who created you?")
    assert action2 is None

    action3 = ActionParser.parse_command("Tell me about my college.")
    assert action3 is None


# 5. SPECIFICATION TEST: Destructive Action Confirmation
def test_destructive_action_requires_confirmation(agent_manager):
    # Command to delete a folder
    handled, msg, result = agent_manager.handle_command("Delete folder TestFolder", lang="en")
    assert handled is True
    assert "continue" in msg.lower() or "delete" in msg.lower()
    assert agent_manager.pending_destructive_action is not None
    assert agent_manager.pending_destructive_action.action == ActionType.DELETE_FOLDER

    # User cancels
    handled_cancel, msg_cancel, _ = agent_manager.handle_command("No, cancel that.", lang="en")
    assert handled_cancel is True
    assert "cancel" in msg_cancel.lower() or "safe" in msg_cancel.lower()
    assert agent_manager.pending_destructive_action is None


# 6. SPECIFICATION TEST: Invalid Application Graceful Failure
def test_invalid_application_graceful_failure(agent_manager):
    handled, msg, result = agent_manager.handle_command("Open NonExistentFakeAppXYZ123", lang="en")
    assert handled is False
    assert "couldn't find" in msg.lower() or "not found" in msg.lower()
    assert result is None


# 7. SPECIFICATION TEST: System Information Queries
def test_system_info_parsing_and_execution(agent_manager):
    action = ActionParser.parse_command("How much RAM do I have?")
    assert action is not None
    assert action.action == ActionType.GET_SYSTEM_INFO

    is_valid, reason, validated = ActionValidator.validate(action)
    assert is_valid is True

    result = ActionExecutor.execute(validated)
    assert result.success is True
    assert "GB" in result.message or "RAM" in result.message


# 8. SPECIFICATION TEST: Volume Control Operations
def test_volume_control_operations(agent_manager):
    action_inc = ActionParser.parse_command("Volume badhao")
    assert action_inc is not None
    assert action_inc.action == ActionType.SET_VOLUME
    assert action_inc.parameters["operation"] == "increase"

    action_mute = ActionParser.parse_command("Mute audio")
    assert action_mute is not None
    assert action_mute.action == ActionType.SET_VOLUME
    assert action_mute.parameters["operation"] == "mute"


# 9. SPECIFICATION TEST: URL Opening
def test_open_url_commands(agent_manager):
    action_gh = ActionParser.parse_command("Open GitHub")
    assert action_gh is not None
    assert action_gh.action == ActionType.OPEN_URL
    assert "github.com" in action_gh.parameters["url"]

    action_custom = ActionParser.parse_command("Open https://www.python.org")
    assert action_custom is not None
    assert action_custom.action == ActionType.OPEN_URL
    assert "python.org" in action_custom.parameters["url"]


# 10. SPECIFICATION TEST: File & Folder Creation
def test_create_folder_and_text_file(agent_manager, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)

    # 1. Create Folder
    action_folder = ActionParser.parse_command("Create folder ProjectPhoenix")
    assert action_folder is not None
    assert action_folder.action == ActionType.CREATE_FOLDER

    is_val, _, val_folder = ActionValidator.validate(action_folder)
    assert is_val is True

    res_folder = ActionExecutor.execute(val_folder, default_workspace=str(workspace))
    assert res_folder.success is True

    # 2. Create Text File
    action_file = ActionParser.parse_command("Create a text file notes.txt with content Hello Chitti Agent")
    assert action_file is not None
    assert action_file.action == ActionType.CREATE_TEXT_FILE

    is_val_file, _, val_file = ActionValidator.validate(action_file)
    assert is_val_file is True

    res_file = ActionExecutor.execute(val_file, default_workspace=str(workspace))
    assert res_file.success is True
