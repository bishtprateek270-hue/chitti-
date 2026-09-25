"""
Comprehensive Phase 6 Test Suite: General-Purpose Project Building & Agentic Planning.
Tests dynamic understanding, technology selection, project scaffolding, multi-file generation,
requirement extraction, anti-hardcoding adherence, error recovery, and honest verification across all 10 test cases.
"""

import os
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.agent.code_generator import CodeGenerator
from src.agent.code_spec import ProgrammingTaskSpec
from src.agent.code_validator import CodeValidator
from src.agent.computer import (
    AppController,
    BrowserController,
    ComputerController,
    FilesystemController,
    ScreenAnalyzer,
    TerminalController,
)
from src.agent.manager import LaptopAgentManager
from src.agent.planner import AgentPlanner, ComputerAgentLoop
from src.agent.projects import ProjectRegistry
from src.agent.task_state import AgentStep, ExecutionFlag, StepStatus, TaskState, TaskStatus
from src.agent.toolchain import ToolchainManager
from src.agent.tools import ToolEngine


@pytest.fixture
def workspace_tmp(tmp_path):
    ws = tmp_path / "test_workspace"
    ws.mkdir(parents=True, exist_ok=True)
    yield ws
    shutil.rmtree(ws, ignore_errors=True)


@pytest.fixture
def agent_manager(workspace_tmp):
    mgr = LaptopAgentManager(
        workspace_dir=str(workspace_tmp),
        screenshots_dir=str(workspace_tmp / "screenshots"),
        project_registry_path=str(workspace_tmp / "projects.json"),
    )
    return mgr


# =========================================================================
# SECTION 24: ALL 10 PROJECT TEST CASES
# =========================================================================

def test_case_1_calculator_with_good_ui_and_run():
    """Test 1: 'Create a calculator with a good UI and run it.'"""
    prompt = "Create a calculator with a good UI and run it."
    spec = CodeGenerator.parse_programming_task(prompt)
    
    assert spec.ui_required is True
    assert spec.execution_requested is True
    assert spec.language == "html"
    assert spec.project_type == "web_app"
    assert spec.filename.endswith(".html")
    assert not spec.filename.startswith("go_to")
    
    # Generate code and validate
    spec = CodeGenerator.generate_solution(spec)
    val = CodeValidator.validate_code(spec.files[0].content, language=spec.language)
    assert val.valid is True
    assert "calculateResult" in spec.files[0].content or "function" in spec.files[0].content
    assert "TODO" not in spec.files[0].content


def test_case_2_todo_app_with_add_delete_complete_browser():
    """Test 2: 'Create a todo app with add, delete and complete functionality and open it in the browser.'"""
    prompt = "Create a todo app with add, delete and complete functionality and open it in the browser."
    spec = CodeGenerator.parse_programming_task(prompt)
    
    assert spec.ui_required is True
    assert spec.execution_requested is True
    assert any("add" in r.lower() for r in spec.requirements)
    assert any("delete" in r.lower() or "remove" in r.lower() for r in spec.requirements)
    assert any("complete" in r.lower() or "status" in r.lower() for r in spec.requirements)
    
    spec = CodeGenerator.generate_solution(spec)
    content = spec.files[0].content
    assert "addItem" in content or "add" in content.lower()
    assert "deleteItem" in content or "delete" in content.lower()


def test_case_3_expense_tracker_with_clean_ui():
    """Test 3: 'Build a simple expense tracker with a clean UI.'"""
    prompt = "Build a simple expense tracker with a clean UI."
    spec = CodeGenerator.parse_programming_task(prompt)
    
    assert spec.ui_required is True
    assert spec.language == "html"
    assert "expense_tracker" in spec.filename or "expense" in spec.filename
    
    spec = CodeGenerator.generate_solution(spec)
    content = spec.files[0].content
    assert "Expense" in content or "amount" in content.lower()
    assert "TODO" not in content


def test_case_4_python_student_management_system():
    """Test 4: 'Create a Python student management system.'"""
    prompt = "Create a Python student management system."
    spec = CodeGenerator.parse_programming_task(prompt)
    
    assert spec.language == "python"
    assert spec.project_type == "management_system"
    assert spec.filename.endswith(".py")
    assert "student" in spec.filename
    
    spec = CodeGenerator.generate_solution(spec)
    content = spec.files[0].content
    val = CodeValidator.validate_code(content, language="python")
    assert val.valid is True
    assert "class Student" in content or "class " in content
    assert "def add_" in content or "def add" in content


def test_case_5_cpp_library_management_program():
    """Test 5: 'Build a C++ library management program.'"""
    prompt = "Build a C++ library management program."
    spec = CodeGenerator.parse_programming_task(prompt)
    
    assert spec.language == "cpp"
    assert spec.project_type == "management_system"
    assert spec.filename.endswith(".cpp")
    
    spec = CodeGenerator.generate_solution(spec)
    content = spec.files[0].content
    assert "#include <iostream>" in content
    assert "class " in content
    assert "int main" in content


def test_case_6_weather_dashboard():
    """Test 6: 'Create a simple weather dashboard.'"""
    prompt = "Create a simple weather dashboard."
    spec = CodeGenerator.parse_programming_task(prompt)
    
    assert spec.ui_required is True
    assert spec.language == "html"
    assert "weather" in spec.filename or "dashboard" in spec.filename
    
    spec = CodeGenerator.generate_solution(spec)
    val = CodeValidator.validate_code(spec.files[0].content, language="html")
    assert val.valid is True


def test_case_7_portfolio_website():
    """Test 7: 'Build a portfolio website.'"""
    prompt = "Build a portfolio website."
    spec = CodeGenerator.parse_programming_task(prompt)
    
    assert spec.ui_required is True
    assert spec.project_type == "web_app"
    assert spec.filename.endswith(".html")
    
    spec = CodeGenerator.generate_solution(spec)
    content = spec.files[0].content
    assert "Portfolio" in content or "<!DOCTYPE html>" in content


def test_case_8_flask_api_for_managing_notes():
    """Test 8: 'Create a Flask API for managing notes.'"""
    prompt = "Create a Flask API for managing notes."
    spec = CodeGenerator.parse_programming_task(prompt)
    
    assert spec.language == "python"
    assert spec.framework == "flask"
    assert spec.project_type == "rest_api"
    assert "flask" in spec.dependencies
    
    spec = CodeGenerator.generate_solution(spec)
    content = spec.files[0].content
    val = CodeValidator.validate_code(content, language="python")
    assert val.valid is True
    assert "@app.route" in content
    assert "jsonify" in content


def test_case_9_java_student_management_application():
    """Test 9: 'Create a Java student management application.'"""
    prompt = "Create a Java student management application."
    spec = CodeGenerator.parse_programming_task(prompt)
    
    assert spec.language == "java"
    assert spec.project_type == "management_system"
    assert spec.filename.endswith(".java")
    assert spec.filename[0].isupper()  # PascalCase naming for Java
    
    spec = CodeGenerator.generate_solution(spec)
    content = spec.files[0].content
    assert "public class" in content
    assert "public static void main" in content


def test_case_10_unseen_custom_project():
    """Test 10: Completely unseen novel project (e.g. 'Create an automated workout and fitness nutrition logger with a good UI and run it.')."""
    prompt = "Create an automated workout and fitness nutrition logger with a good UI and run it."
    spec = CodeGenerator.parse_programming_task(prompt)
    
    assert spec.ui_required is True
    assert spec.execution_requested is True
    assert "workout" in spec.filename or "fitness" in spec.filename or "nutrition" in spec.filename
    assert not spec.filename.startswith("create_")
    
    spec = CodeGenerator.generate_solution(spec)
    content = spec.files[0].content
    assert len(content) > 500
    assert "TODO" not in content
    assert "<!DOCTYPE html>" in content


# =========================================================================
# ANTI-HARDCODING AND DYNAMIC SLUG VALIDATION
# =========================================================================

def test_dynamic_slug_extraction():
    """Verifies that filename derivation never produces long sentence filenames."""
    sentences = [
        ("Go to VS Code and create a fully functional calculator with a good UI and run it", "calculator.html"),
        ("Write a C++ binary search tree implementation", "binary_search_tree.cpp"),
        ("Create a real-time cryptocurrency portfolio tracker in React", "App.jsx"),
        ("Build an automated solar energy ROI calculator in Python", "solar_energy_roi.py"),
        ("Make a Java bank account management system", "BankAccountManager.java"),
    ]
    for sentence, expected_suffix in sentences:
        spec = CodeGenerator.parse_programming_task(sentence)
        assert len(spec.filename) < 40
        assert " " not in spec.filename
        assert "go_to" not in spec.filename.lower()
        assert not spec.filename.startswith("go_to_vs_code")


def test_no_false_success_when_runtime_missing(workspace_tmp):
    """Verifies that when runtime is unavailable, task marks BLOCKED honestly."""
    prompt = "Write a Go program for high frequency data stream processing and run it."
    planner = AgentPlanner(project_registry=ProjectRegistry(str(workspace_tmp / "projects.json")))
    state = planner.plan_task(prompt)
    
    assert state is not None
    assert any(s.action_type == "VERIFY_TOOLCHAIN" for s in state.steps)
    
    fs = FilesystemController(default_workspace=str(workspace_tmp))
    # Write the file on disk so file verification passes
    fs.write_file(state.steps[0].parameters["path"], "package main\nfunc main() {}\n")
    
    mock_tools = MagicMock()
    mock_tools.execute_tool.return_value = MagicMock(success=True, message="Success", data={"evidence": "verified"})
    
    mock_analyzer = MagicMock()
    mock_analyzer.verify_window.return_value = MagicMock(success=True, evidence="Visual Studio Code active")
    
    # Mock toolchain check returning False
    with patch.object(ToolchainManager, "is_toolchain_available", return_value=(False, None)):
        loop = ComputerAgentLoop(
            tool_engine=mock_tools,
            computer=MagicMock(),
            filesystem=fs,
            terminal=MagicMock(),
            browser=MagicMock(),
            apps=MagicMock(),
            screen_analyzer=mock_analyzer,
            projects=ProjectRegistry(str(workspace_tmp / "projects.json")),
        )
        success, msg = loop.execute_plan(state)
        
        assert success is False
        assert state.status == TaskStatus.BLOCKED
        assert "not installed" in msg or "blocked" in msg.lower()


def test_task_cancellation(workspace_tmp):
    """Verifies that user saying 'stop' or 'cancel' immediately stops the active task."""
    mgr = LaptopAgentManager(workspace_dir=str(workspace_tmp))
    # Create active task state
    mgr.active_task_state = TaskState(task_description="Running complex project build", status=TaskStatus.EXECUTING)
    
    handled, resp, _ = mgr.handle_command("Stop", lang="en")
    assert handled is True
    assert mgr.active_task_state.status == TaskStatus.CANCELLED
    assert "cancelled" in resp.lower()
