"""
Comprehensive Unit & Integration Tests for Chitti Phase 5: Full Laptop Computer-Use Agent.
Tests computer controller, mouse/keyboard/clipboard, window management, filesystem, terminal risk analysis,
browser automation, project registry, multi-step planner, execution loop, failure recovery, and safety gates.
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
from src.agent.projects import ProjectRegistry
from src.agent.task_state import TaskState, TaskStatus, StepStatus, AgentStep
from src.agent.planner import AgentPlanner, ComputerAgentLoop
from src.agent.computer import (
    ComputerController,
    FilesystemController,
    TerminalController,
    TerminalRiskLevel,
    BrowserController,
    AppController,
    ScreenAnalyzer,
)


@pytest.fixture
def agent_manager(tmp_path):
    workspace = tmp_path / "workspace"
    screenshots = tmp_path / "screenshots"
    registry_file = tmp_path / "projects.json"
    workspace.mkdir()
    screenshots.mkdir()
    return LaptopAgentManager(
        workspace_dir=str(workspace),
        screenshots_dir=str(screenshots),
        project_registry_path=str(registry_file),
    )


# -------------------------------------------------------------
# 1. APPLICATION & INTENT PARSING (English, Hindi, Hinglish)
# -------------------------------------------------------------

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


def test_parse_open_folder():
    action = ActionParser.parse_command("Open Downloads")
    assert action is not None
    assert action.action == ActionType.OPEN_FOLDER
    assert "downloads" in action.parameters["target"].lower()

    action_hi = ActionParser.parse_command("Downloads folder kholo")
    assert action_hi is not None
    assert action_hi.action == ActionType.OPEN_FOLDER


def test_parse_take_screenshot():
    action = ActionParser.parse_command("Take a screenshot")
    assert action is not None
    assert action.action == ActionType.TAKE_SCREENSHOT

    action_hi = ActionParser.parse_command("Screenshot le lo")
    assert action_hi is not None
    assert action_hi.action == ActionType.TAKE_SCREENSHOT


# -------------------------------------------------------------
# 2. ISOLATION: GENERAL KNOWLEDGE & PERSONAL QUERIES
# -------------------------------------------------------------

def test_general_knowledge_no_laptop_action():
    assert ActionParser.parse_command("What is deep learning?") is None
    assert ActionParser.parse_command("Explain CNN.") is None
    assert ActionParser.parse_command("Solve this LeetCode problem.") is None


def test_personal_queries_no_laptop_action():
    assert ActionParser.parse_command("What is my name?") is None
    assert ActionParser.parse_command("Who created you?") is None
    assert ActionParser.parse_command("Tell me about my college.") is None


# -------------------------------------------------------------
# 3. COMPUTER CONTROLLER: MOUSE, KEYBOARD, CLIPBOARD, WINDOWS
# -------------------------------------------------------------

def test_computer_controller_screen_and_mouse(tmp_path):
    controller = ComputerController(screenshots_dir=str(tmp_path))
    size = controller.get_screen_size()
    assert len(size) == 2
    assert size[0] > 0 and size[1] > 0

    pos = controller.get_mouse_position()
    assert len(pos) == 2


def test_computer_controller_clipboard_and_keyboard(tmp_path):
    controller = ComputerController(screenshots_dir=str(tmp_path))
    # Test clipboard write & read
    test_text = "Chitti Autonomous Agent Test"
    ok = controller.write_clipboard(test_text)
    if ok:
        assert controller.read_clipboard() == test_text


def test_computer_controller_window_listing(tmp_path):
    controller = ComputerController(screenshots_dir=str(tmp_path))
    windows = controller.list_windows()
    assert isinstance(windows, list)


# -------------------------------------------------------------
# 4. FILESYSTEM CONTROLLER
# -------------------------------------------------------------

def test_filesystem_crud_and_search(tmp_path):
    fs = FilesystemController(default_workspace=str(tmp_path))

    # 1. Create file & read
    file_path = fs.create_file("test_doc.txt", "Hello Chitti Filesystem")
    assert Path(file_path).exists()
    content = fs.read_file("test_doc.txt")
    assert "Hello Chitti Filesystem" in content

    # 2. Append file
    fs.append_file("test_doc.txt", "\nSecond line added.")
    content_updated = fs.read_file("test_doc.txt")
    assert "Second line added." in content_updated

    # 3. Create folder
    dir_path = fs.create_directory("subfolder")
    assert Path(dir_path).is_dir()

    # 4. Copy file
    copied_path = fs.copy_file("test_doc.txt", "subfolder/copied_doc.txt")
    assert Path(copied_path).exists()

    # 5. Move file
    moved_path = fs.move_file("subfolder/copied_doc.txt", "subfolder/moved_doc.txt")
    assert Path(moved_path).exists()
    assert not Path("subfolder/copied_doc.txt").exists()

    # 6. Rename file
    renamed_path = fs.rename_file(moved_path, "renamed_doc.txt")
    assert Path(renamed_path).exists()

    # 7. Search files
    matches = fs.search_files("*.txt")
    assert len(matches) >= 2

    # 8. Get file info
    info = fs.get_file_info("test_doc.txt")
    assert info.name == "test_doc.txt"
    assert not info.is_dir
    assert info.size_bytes > 0

    # 9. Delete file
    fs.delete_file("test_doc.txt")
    assert not Path(file_path).exists()

    # 10. Delete directory
    fs.delete_directory("subfolder", recursive=True)
    assert not Path(dir_path).exists()


# -------------------------------------------------------------
# 5. TERMINAL CONTROLLER & RISK CLASSIFICATION
# -------------------------------------------------------------

def test_terminal_risk_classification():
    term = TerminalController()

    # LOW risk commands
    assert term.classify_risk("python --version") == TerminalRiskLevel.LOW
    assert term.classify_risk("git status") == TerminalRiskLevel.LOW
    assert term.classify_risk("pytest tests/") == TerminalRiskLevel.LOW
    assert term.classify_risk("dir") == TerminalRiskLevel.LOW
    assert not term.is_destructive("python --version")

    # MEDIUM risk commands
    assert term.classify_risk("pip install requests") == TerminalRiskLevel.MEDIUM
    assert term.classify_risk("npm install express") == TerminalRiskLevel.MEDIUM
    assert term.classify_risk("python main.py") == TerminalRiskLevel.MEDIUM

    # HIGH risk commands (destructive)
    assert term.classify_risk("del file.txt") == TerminalRiskLevel.HIGH
    assert term.classify_risk("rmdir folder") == TerminalRiskLevel.HIGH
    assert term.classify_risk("git reset --hard") == TerminalRiskLevel.HIGH
    assert term.is_destructive("del file.txt")

    # CRITICAL risk commands
    assert term.classify_risk("format C:") == TerminalRiskLevel.CRITICAL
    assert term.classify_risk("diskpart") == TerminalRiskLevel.CRITICAL
    assert term.classify_risk("rmdir /s /q C:\\") == TerminalRiskLevel.CRITICAL
    assert term.is_destructive("format C:")


def test_terminal_command_execution(tmp_path):
    term = TerminalController(default_cwd=str(tmp_path))
    result = term.execute_command("echo HelloChittiTerminal", timeout_sec=10)
    assert result.is_success
    assert "HelloChittiTerminal" in result.output
    assert result.elapsed_time_sec >= 0


# -------------------------------------------------------------
# 6. BROWSER & APP CONTROLLERS
# -------------------------------------------------------------

def test_browser_controller():
    browser = BrowserController()
    with patch("webbrowser.open", return_value=True) as mock_open:
        ok = browser.open_url("https://www.python.org")
        assert ok is True
        mock_open.assert_called_once()

    with patch("webbrowser.open", return_value=True) as mock_search:
        ok_search = browser.search_web("Python documentation")
        assert ok_search is True
        mock_search.assert_called_once()


def test_app_controller_lifecycle():
    apps = AppController()
    # Test running applications list
    running = apps.list_running_applications()
    assert isinstance(running, list)
    assert len(running) > 0


# -------------------------------------------------------------
# 7. PROJECT & ENVIRONMENT REGISTRY
# -------------------------------------------------------------

def test_project_registry(tmp_path):
    reg_file = tmp_path / "test_projects.json"
    registry = ProjectRegistry(registry_file=str(reg_file))

    # Register project
    proj_dir = tmp_path / "DocForensics"
    proj_dir.mkdir()
    registry.register_project("DocForensics", str(proj_dir))

    # Resolve project
    resolved = registry.get_project_path("DocForensics")
    assert resolved == str(proj_dir.resolve())

    # Fuzzy match
    fuzzy = registry.get_project_path("docforensics")
    assert fuzzy == str(proj_dir.resolve())

    # List & remove
    projs = registry.list_projects()
    assert "docforensics" in projs
    registry.remove_project("DocForensics")
    assert "docforensics" not in registry.list_projects()


# -------------------------------------------------------------
# 8. MULTI-STEP PLANNER & AGENT LOOP
# -------------------------------------------------------------

def test_plan_open_chrome_and_search(tmp_path):
    registry = ProjectRegistry(registry_file=str(tmp_path / "projects.json"))
    planner = AgentPlanner(project_registry=registry)

    state = planner.plan_task("Open Chrome and search for the latest Python documentation")
    assert state is not None
    assert len(state.steps) == 2
    assert state.steps[0].action_type == "SEARCH_WEB"
    assert "Python documentation" in state.steps[0].parameters["query"]


def test_plan_open_vscode_project(tmp_path):
    registry = ProjectRegistry(registry_file=str(tmp_path / "projects.json"))
    proj_dir = tmp_path / "Chitti"
    proj_dir.mkdir()
    registry.register_project("Chitti", str(proj_dir))

    planner = AgentPlanner(project_registry=registry)
    state = planner.plan_task("Open VS Code and open my Chitti project")
    assert state is not None
    assert len(state.steps) == 3
    assert state.steps[0].action_type == "RESOLVE_PROJECT"
    assert state.steps[1].action_type == "OPEN_APPLICATION"


def test_plan_notepad_and_type(tmp_path):
    registry = ProjectRegistry(registry_file=str(tmp_path / "projects.json"))
    planner = AgentPlanner(project_registry=registry)

    state = planner.plan_task("Open Notepad and type Hello Chitti")
    assert state is not None
    assert len(state.steps) == 3
    assert state.steps[0].action_type == "OPEN_APPLICATION"
    assert state.steps[1].action_type == "TYPE_TEXT"
    assert state.steps[1].parameters["text"] == "Hello Chitti"


def test_plan_and_execute_fibonacci_generation(agent_manager, tmp_path):
    # End-to-end task execution: "Create a Python file in my project and write a program that calculates Fibonacci numbers"
    handled, msg, result = agent_manager.handle_command(
        "Create a Python file in my project and write a program that calculates Fibonacci numbers",
        lang="en"
    )
    assert handled is True
    assert "completed successfully" in msg.lower()

    # Verify fibonacci.py exists in workspace
    fib_file = Path(agent_manager.workspace_dir) / "fibonacci.py"
    assert fib_file.exists()
    content = fib_file.read_text(encoding="utf-8")
    assert "def fibonacci" in content


def test_plan_and_execute_folder_and_file_creation(agent_manager, tmp_path):
    # End-to-end task execution: "Open the folder ProjectAlpha and create test.txt"
    handled, msg, result = agent_manager.handle_command(
        "Open the folder ProjectAlpha and create test.txt",
        lang="en"
    )
    assert handled is True
    created_file = Path(agent_manager.workspace_dir) / "ProjectAlpha" / "test.txt"
    assert created_file.exists()


def test_plan_destructive_action_confirmation_cycle(agent_manager):
    # Command: "Delete ChittiTest"
    handled, msg, result = agent_manager.handle_command("Delete folder ChittiTest", lang="en")
    assert handled is True
    assert "continue" in msg.lower() or "delete" in msg.lower()
    assert agent_manager.pending_destructive_action is not None

    # User confirms
    handled_conf, msg_conf, res_conf = agent_manager.handle_command("Yes, proceed.", lang="en")
    assert handled_conf is True
    assert agent_manager.pending_destructive_action is None
