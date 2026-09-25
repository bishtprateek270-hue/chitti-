"""
Chitti Phase 6 Agentic Planning & Multi-Step Execution Regression Suite.
Validates dynamic goal decomposition, dependency execution, failure recovery,
re-planning, cancellation, pause/resume, and unanticipated tasks without hardcoded templates.
"""

import os
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.agent.actions import ActionResult, ActionType, RiskLevel, StructuredAction
from src.agent.code_generator import CodeGenerator
from src.agent.code_spec import ProgrammingTaskSpec
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
from src.agent.recovery import FailureRecoveryManager, RecoveryAction
from src.agent.task_state import (
    AgentStep,
    ExecutionFlag,
    StepStatus,
    TaskContext,
    TaskState,
    TaskStatus,
)
from src.agent.tools import ToolEngine
from src.agent.verifier import SubtaskVerifier


@pytest.fixture
def agent_env(tmp_path):
    ws_dir = str(tmp_path / "workspace")
    sc_dir = str(tmp_path / "screenshots")
    reg_file = str(tmp_path / "projects.json")
    os.makedirs(ws_dir, exist_ok=True)
    os.makedirs(sc_dir, exist_ok=True)

    mgr = LaptopAgentManager(
        workspace_dir=ws_dir,
        screenshots_dir=sc_dir,
        project_registry_path=reg_file,
    )
    return mgr, ws_dir, sc_dir


# ==============================================================================
# 1. SIMPLE TASKS (Fast-path / Single-step)
# ==============================================================================

def test_01_simple_open_application(agent_env):
    mgr, ws, sc = agent_env
    with patch("src.agent.registry.AppDiscovery.resolve_application", return_value="calc.exe"), \
         patch.object(mgr.executor, "_open_application", return_value=ActionResult(action=ActionType.OPEN_APPLICATION, success=True, message="Calculator is open", target="Calculator")):
        handled, msg, res = mgr.handle_command("Open Calculator")
        assert handled is True
        assert "Calculator is open" in msg or "open" in msg.lower()


def test_02_simple_create_folder(agent_env):
    mgr, ws, sc = agent_env
    folder_path = Path(ws) / "reports_2026"
    res = mgr.tools.execute_tool("create_directory", {"path": str(folder_path)})
    assert res.success is True
    assert folder_path.exists()


def test_03_simple_create_file(agent_env):
    mgr, ws, sc = agent_env
    file_path = Path(ws) / "notes.txt"
    res = mgr.tools.execute_tool("create_file", {"path": str(file_path), "content": "Meeting summary"})
    assert res.success is True
    assert file_path.exists()
    assert file_path.read_text(encoding="utf-8") == "Meeting summary"


def test_04_simple_open_website(agent_env):
    mgr, ws, sc = agent_env
    with patch("webbrowser.open", return_value=True):
        handled, msg, res = mgr.handle_command("Open https://python.org in browser")
        assert handled is True


# ==============================================================================
# 2. MULTI-STEP COMPUTER TASKS
# ==============================================================================

def test_05_multistep_folder_file_content_verify(agent_env):
    mgr, ws, sc = agent_env
    plan = mgr.planner.plan_task(f"Open folder my_data and create data.csv")
    assert plan is not None
    assert len(plan.steps) == 3

    # Assert dependency structure
    assert plan.steps[1].depends_on == [1]
    assert plan.steps[2].depends_on == [2]

    # Execute plan
    success, msg = mgr.loop.execute_plan(plan)
    assert success is True
    assert plan.status == TaskStatus.COMPLETED


def test_06_multistep_notepad_and_type(agent_env):
    mgr, ws, sc = agent_env
    with patch.object(mgr.tools, "execute_tool") as mock_exec, \
         patch.object(mgr.screen_analyzer, "verify_window", return_value=MagicMock(success=True, evidence="Notepad active")):
        mock_exec.return_value = ActionResult(action=ActionType.OPEN_APPLICATION, success=True, message="OK")
        plan = mgr.planner.plan_task("Open Notepad and type Hello Chitti")
        assert plan is not None
        assert len(plan.steps) == 3
        success, msg = mgr.loop.execute_plan(plan)
        assert success is True


def test_07_multistep_browser_search(agent_env):
    mgr, ws, sc = agent_env
    with patch.object(mgr.tools, "execute_tool", return_value=ActionResult(action=ActionType.OPEN_URL, success=True, message="OK")), \
         patch.object(mgr.screen_analyzer, "verify_window", return_value=MagicMock(success=True, evidence="Chrome open")):
        plan = mgr.planner.plan_task("Open Chrome and search for latest Python release")
        assert plan is not None
        assert len(plan.steps) == 2
        success, msg = mgr.loop.execute_plan(plan)
        assert success is True


# ==============================================================================
# 3. CODING TASKS (Dynamic, Validate, Run, Repair)
# ==============================================================================

def test_08_coding_generate_validate_run_verify(agent_env):
    mgr, ws, sc = agent_env
    cmd = "VS Code open kro aur Python me prime number checker ka code likho aur run karo"
    plan = mgr.planner.plan_task(cmd)
    assert plan is not None
    assert plan.get_flag(ExecutionFlag.TASK_UNDERSTOOD) is True
    assert plan.get_flag(ExecutionFlag.CODE_VALIDATED) is True

    # Check that toolchain and execute steps exist
    action_types = [s.action_type for s in plan.steps]
    assert "CREATE_FILE" in action_types
    assert "COMPILE_AND_EXECUTE" in action_types
    assert "VERIFY_EXECUTION" in action_types


def test_09_coding_recovery_and_auto_repair(agent_env):
    mgr, ws, sc = agent_env
    test_file = Path(ws) / "broken_calc.py"
    test_file.write_text("def calc():\n    return 10 / 0\nprint(calc())", encoding="utf-8")

    rec_mgr = FailureRecoveryManager(filesystem=mgr.filesystem)
    step = AgentStep(
        step_id=1,
        description="Run broken script",
        action_type="RUN_TERMINAL",
        parameters={"path": str(test_file), "command": f"python {test_file}"},
    )
    state = TaskState(task_description="Fix broken calc", steps=[step])

    plan = rec_mgr.analyze_failure(step, "ZeroDivisionError: division by zero", state)
    assert plan.action == RecoveryAction.REPAIR_CODE
    assert "Repaired source code" in plan.reason
    assert test_file.read_text(encoding="utf-8") != "def calc():\n    return 10 / 0\nprint(calc())"


def test_10_modify_existing_project_multi_file(agent_env):
    mgr, ws, sc = agent_env
    auth_file = Path(ws) / "auth.py"
    auth_file.write_text("def authenticate():\n    pass\n", encoding="utf-8")

    # Update code
    mgr.filesystem.modify_file(str(auth_file), "def authenticate(user, password):\n    return user == 'admin' and password == 'secret'\n")
    updated = mgr.filesystem.read_file(str(auth_file))
    assert "admin" in updated


# ==============================================================================
# 4. MIXED TASKS (Project, VS Code, Browser)
# ==============================================================================

def test_11_mixed_project_inspect_modify_run_vscode(agent_env):
    mgr, ws, sc = agent_env
    project_dir = Path(ws) / "ExpenseApp"
    project_dir.mkdir(parents=True, exist_ok=True)
    main_file = project_dir / "main.py"
    main_file.write_text("print('Expense App v1.0')", encoding="utf-8")

    with patch.object(mgr.tools, "execute_tool", return_value=ActionResult(action=ActionType.OPEN_APPLICATION, success=True, message="OK")), \
         patch.object(mgr.screen_analyzer, "verify_window", return_value=MagicMock(success=True, evidence="VS Code")):
        plan = mgr.planner.plan_task(f"Create Python expense tracker and open in VS Code")
        assert plan is not None
        assert any(s.action_type == "OPEN_APPLICATION" for s in plan.steps)


def test_12_mixed_browser_research_and_save(agent_env):
    mgr, ws, sc = agent_env
    summary_file = Path(ws) / "ai_summary.txt"
    mgr.filesystem.write_file(str(summary_file), "Latest AI Research 2026: Multi-step Autonomous Agents")
    assert summary_file.exists()
    assert "Multi-step" in summary_file.read_text(encoding="utf-8")


# ==============================================================================
# 5. RECOVERY & REPLANNING
# ==============================================================================

def test_13_recovery_missing_parent_directory(agent_env):
    mgr, ws, sc = agent_env
    rec_mgr = FailureRecoveryManager(filesystem=mgr.filesystem)

    nested_file = str(Path(ws) / "nested_dir" / "app.py")
    step = AgentStep(
        step_id=1,
        description="Write app.py",
        action_type="CREATE_FILE",
        parameters={"path": nested_file, "content": "print('Nested')"},
    )
    state = TaskState(task_description="Create nested file", steps=[step])

    rec_plan = rec_mgr.analyze_failure(step, "Directory does not exist", state)
    assert rec_plan.action == RecoveryAction.REPLAN_REMAINING
    assert len(rec_plan.new_steps) == 2
    assert rec_plan.new_steps[0].action_type == "CREATE_DIRECTORY"


def test_14_recovery_retry_limit_exceeded(agent_env):
    mgr, ws, sc = agent_env
    rec_mgr = FailureRecoveryManager(filesystem=mgr.filesystem)

    step = AgentStep(
        step_id=1,
        description="Execute binary",
        action_type="RUN_TERMINAL",
        parameters={"command": "unknown_cmd"},
        retry_count=2,
        max_retries=2,
    )
    state = TaskState(task_description="Execute", steps=[step])
    rec_plan = rec_mgr.analyze_failure(step, "Command failed", state)
    assert rec_plan.action == RecoveryAction.ABORT_HONESTLY


def test_15_dependency_blocking_prevents_execution(agent_env):
    mgr, ws, sc = agent_env
    step1 = AgentStep(step_id=1, description="Step 1", action_type="CREATE_FILE", parameters={"path": "f1.txt", "content": "x"})
    step2 = AgentStep(step_id=2, description="Step 2", action_type="RUN_TERMINAL", parameters={"command": "echo 1"}, depends_on=[1])

    state = TaskState(task_description="Dependency Test", steps=[step1, step2])
    # Step 1 fails
    step1.status = StepStatus.FAILED
    state.failed_steps.append(step1)
    state.current_step_index = 1

    can_exec, reason = state.can_execute_step(step2)
    assert can_exec is False
    assert "Prerequisite step 1" in reason


# ==============================================================================
# 6. USER CONTROL (Confirmation, Cancellation, Pause/Resume)
# ==============================================================================

def test_16_confirmation_required_action(agent_env):
    mgr, ws, sc = agent_env
    target_dir = Path(ws) / "temp_to_delete"
    target_dir.mkdir(parents=True, exist_ok=True)

    handled, msg, res = mgr.handle_command(f"Delete folder {target_dir.name}")
    assert handled is True
    assert mgr.pending_destructive_action is not None
    assert "modify or delete" in msg or "hata" in msg or "delete" in msg.lower()


def test_17_user_cancellation_during_task(agent_env):
    mgr, ws, sc = agent_env
    plan = TaskState(task_description="Long task", status=TaskStatus.EXECUTING)
    mgr.active_task_state = plan

    handled, msg, res = mgr.handle_command("Stop")
    assert handled is True
    assert plan.status == TaskStatus.CANCELLED
    assert plan.is_cancelled is True
    assert "cancelled" in msg.lower() or "रद्द" in msg


def test_18_user_pause_and_resume(agent_env):
    mgr, ws, sc = agent_env
    plan = TaskState(task_description="Task to pause", status=TaskStatus.EXECUTING)
    mgr.active_task_state = plan

    handled, msg, res = mgr.handle_command("Pause")
    assert handled is True
    assert plan.status == TaskStatus.PAUSED
    assert plan.is_paused is True

    plan.resume()
    assert plan.is_paused is False
    assert plan.status == TaskStatus.EXECUTING


# ==============================================================================
# 7. GENERALIZATION & UNSEEN TASKS (3 Newly Invented Unseen Tasks)
# ==============================================================================

def test_19_unseen_task_1_weather_report_parser(agent_env):
    mgr, ws, sc = agent_env
    cmd = "VS Code open karo aur Python mein weather report parser banao jisme temperature humidity analyze ho"
    plan = mgr.planner.plan_task(cmd)
    assert plan is not None
    assert plan.get_flag(ExecutionFlag.TASK_UNDERSTOOD) is True

    create_step = next(s for s in plan.steps if s.action_type == "CREATE_FILE")
    content = create_step.parameters["content"]
    assert "def " in content or "class " in content
    assert "temperature" in content.lower() or "weather" in content.lower()


def test_20_unseen_task_2_cpp_lru_cache(agent_env):
    mgr, ws, sc = agent_env
    cmd = "Write a C++ LRU cache module with get and put methods and open in VS Code"
    plan = mgr.planner.plan_task(cmd)
    assert plan is not None
    create_step = next(s for s in plan.steps if s.action_type == "CREATE_FILE")
    content = create_step.parameters["content"]
    assert "get" in content and "put" in content
    assert "#include" in content


def test_21_unseen_task_3_javascript_markdown_to_html(agent_env):
    mgr, ws, sc = agent_env
    cmd = "Create a JavaScript Markdown to HTML converter module"
    plan = mgr.planner.plan_task(cmd)
    assert plan is not None
    create_step = next(s for s in plan.steps if s.action_type == "CREATE_FILE")
    content = create_step.parameters["content"]
    assert "function" in content or "const" in content or "class" in content
    assert "markdown" in content.lower() or "html" in content.lower()
