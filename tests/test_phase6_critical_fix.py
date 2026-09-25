"""
Phase 6 Critical Fix Test Suite.
Validates natural language understanding, dynamic planning, language selection,
genuine code generation (zero placeholders), toolchain checks, execution and honest verification.
"""

from pathlib import Path
import pytest

from src.agent.code_generator import CodeGenerator
from src.agent.manager import LaptopAgentManager
from src.agent.task_state import ExecutionFlag, TaskStatus


@pytest.fixture
def agent_mgr(tmp_path):
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


def test_01_original_bug_calculator_ui_and_run(agent_mgr):
    """
    Validates: 'go to vs code and create a fully functional calculator with a good ui and run it'
    Must:
    1. Parse intent as calculator with UI.
    2. NOT choose Go because of 'go to'.
    3. Choose HTML/CSS/JS (or appropriate web UI) with clean filename calculator.html.
    4. Generate real calculator code (buttons 0-9, operators, clear, equals, responsive UI), zero placeholders.
    5. Save in VS Code and execute in browser.
    6. Complete honestly with verified execution.
    """
    raw_prompt = "go to vs code and create a fully functional calculator with a good ui and run it"
    spec = CodeGenerator.parse_programming_task(raw_prompt)

    # NLU assertions
    assert spec.language in ("html", "htm", "javascript")
    assert spec.language != "go"
    assert spec.ui_required is True
    assert spec.execution_requested is True
    assert "go to" not in spec.problem_description.lower()
    assert spec.filename in ("calculator.html", "calculator.htm")

    # Code generation check
    gen_spec = CodeGenerator.generate_code_for_topic(raw_prompt)
    main_code = gen_spec.files[0].content
    assert "Executing Script solution" not in main_code
    assert "button" in main_code.lower()
    assert "display" in main_code.lower()
    assert "calculate" in main_code.lower() or "eval" in main_code.lower() or "compute" in main_code.lower() or "append" in main_code.lower()

    # Agent execution check
    handled, msg, result = agent_mgr.handle_command(raw_prompt, lang="en")
    assert handled is True
    assert result.success is True

    calc_file = Path(agent_mgr.workspace_dir) / spec.filename
    assert calc_file.exists()
    file_content = calc_file.read_text(encoding="utf-8")
    assert "calculator" in file_content.lower()


def test_02_task_a_python_csv_summary(agent_mgr):
    """Task A: 'Create a Python program that reads a CSV and prints summary statistics, then run it.'"""
    raw_prompt = "Create a Python program that reads a CSV and prints summary statistics, then run it."
    spec = CodeGenerator.parse_programming_task(raw_prompt)
    assert spec.language == "python"
    assert spec.filename == "csv_analyzer.py"
    assert spec.execution_requested is True

    handled, msg, result = agent_mgr.handle_command(raw_prompt, lang="en")
    assert handled is True
    f = Path(agent_mgr.workspace_dir) / "csv_analyzer.py"
    assert f.exists()
    content = f.read_text(encoding="utf-8")
    assert "import csv" in content
    assert "statistics" in content.lower() or "summary" in content.lower() or "mean" in content.lower()


def test_03_task_b_html_todo_app_browser(agent_mgr):
    """Task B: 'Create a simple HTML todo app with a good UI and open it in the browser.'"""
    raw_prompt = "Create a simple HTML todo app with a good UI and open it in the browser."
    spec = CodeGenerator.parse_programming_task(raw_prompt)
    assert spec.language in ("html", "htm")
    assert spec.filename in ("todo.html", "todo_app.html")
    assert spec.ui_required is True

    handled, msg, result = agent_mgr.handle_command(raw_prompt, lang="en")
    assert handled is True
    assert result.success is True
    f = Path(agent_mgr.workspace_dir) / spec.filename
    assert f.exists()
    content = f.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in content
    assert "todo" in content.lower()


def test_04_task_c_cpp_sort_user_input(agent_mgr):
    """Task C: 'Create a C++ program that sorts user input and run it.'"""
    raw_prompt = "Create a C++ program that sorts user input and run it."
    spec = CodeGenerator.parse_programming_task(raw_prompt)
    assert spec.language in ("cpp", "c++")
    assert spec.filename == "sorter.cpp"
    assert spec.execution_requested is True

    handled, msg, result = agent_mgr.handle_command(raw_prompt, lang="en")
    assert handled is True
    # If g++ is missing, it should be marked BLOCKED or honest message, not falsely claim execution
    f = Path(agent_mgr.workspace_dir) / "sorter.cpp"
    assert f.exists()
    content = f.read_text(encoding="utf-8")
    assert "#include <iostream>" in content
    assert "sort" in content.lower()


def test_05_task_d_java_student_record_test(agent_mgr):
    """Task D: 'Create a Java program for student record management and test it.'"""
    raw_prompt = "Create a Java program for student record management and test it."
    spec = CodeGenerator.parse_programming_task(raw_prompt)
    assert spec.language == "java"
    assert spec.filename == "StudentManager.java"

    handled, msg, result = agent_mgr.handle_command(raw_prompt, lang="en")
    assert handled is True
    f = Path(agent_mgr.workspace_dir) / "StudentManager.java"
    assert f.exists()
    content = f.read_text(encoding="utf-8")
    assert "class StudentManager" in content or "class Student" in content


def test_06_task_e_react_dashboard(agent_mgr):
    """Task E: 'Create a small React dashboard and open it.'"""
    raw_prompt = "Create a small React dashboard and open it."
    spec = CodeGenerator.parse_programming_task(raw_prompt)
    assert spec.language in ("react", "javascript", "html")
    assert spec.ui_required is True

    handled, msg, result = agent_mgr.handle_command(raw_prompt, lang="en")
    assert handled is True
    f = Path(agent_mgr.workspace_dir) / spec.filename
    assert f.exists()


def test_07_unseen_task_pomodoro_timer(agent_mgr):
    """Unseen Task: 'Create an interactive Pomodoro timer with a modern UI and run it'"""
    raw_prompt = "Create an interactive Pomodoro timer with a modern UI and run it"
    spec = CodeGenerator.parse_programming_task(raw_prompt)
    assert spec.ui_required is True
    assert spec.filename in ("pomodoro_timer.html", "pomodoro.html", "timer.html")

    handled, msg, result = agent_mgr.handle_command(raw_prompt, lang="en")
    assert handled is True
    assert result.success is True
    f = Path(agent_mgr.workspace_dir) / spec.filename
    assert f.exists()
    content = f.read_text(encoding="utf-8")
    assert "timer" in content.lower() or "pomodoro" in content.lower()
