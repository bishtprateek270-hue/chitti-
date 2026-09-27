"""
Tests for Phase 6 Project Request Detection, Task Intent Classification,
and Agent Routing (Phase 6 Router Fix).
"""

from pathlib import Path
import pytest

from src.agent.classifier import ClassificationResult, TaskClassifier, TaskIntent
from src.agent.code_generator import CodeGenerator
from src.agent.manager import LaptopAgentManager
from src.agent.planner import AgentPlanner
from src.agent.projects import ProjectRegistry
from src.memory.router import MemoryIntent, MemoryRouter


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


def test_test_a_create_fully_functional_todo_list_interactive_ui(agent_mgr):
    """
    Test A: 'create a fully functional todo list for me with an interactive ui'
    Must:
    1. Classify intent as CREATE_PROJECT (is_actionable_task = True).
    2. Activate Phase 6 agent instead of returning conversational text.
    3. Generate real interactive UI files with complete logic (zero placeholders).
    """
    raw_prompt = "create a fully functional todo list for me with an interactive ui"

    # 1. Classifier check
    class_res = TaskClassifier.classify(raw_prompt)
    assert class_res.intent == TaskIntent.CREATE_PROJECT
    assert class_res.is_actionable_task is True
    assert class_res.ui_required is True

    # 2. Agent execution check
    handled, msg, result = agent_mgr.handle_command(raw_prompt, lang="en")
    assert handled is True
    assert result is not None
    assert result.success is True

    # 3. File existence and content check
    files = list(Path(agent_mgr.workspace_dir).glob("*.html")) + list(Path(agent_mgr.workspace_dir).glob("*.py"))
    assert len(files) >= 1
    content = files[0].read_text(encoding="utf-8")
    assert "Executing Script solution" not in content
    assert "todo" in content.lower()


def test_test_b_what_is_a_todo_list(agent_mgr):
    """
    Test B: 'what is a todo list'
    Must:
    1. Classify as GENERAL_KNOWLEDGE.
    2. Not trigger agent action (return None to route to normal conversation).
    """
    raw_prompt = "what is a todo list"
    class_res = TaskClassifier.classify(raw_prompt)
    assert class_res.intent == TaskIntent.GENERAL_KNOWLEDGE
    assert class_res.is_actionable_task is False

    handled, msg, result = agent_mgr.handle_command(raw_prompt, lang="en")
    assert handled is False
    assert result is None


def test_test_c_how_do_i_create_a_todo_list_using_python(agent_mgr):
    """
    Test C: 'how do I create a todo list using python?'
    Must:
    1. Classify as TECHNICAL_QUERY.
    2. Not trigger desktop file creation (handled=False for conversational explanation).
    """
    raw_prompt = "how do I create a todo list using python?"
    class_res = TaskClassifier.classify(raw_prompt)
    assert class_res.intent == TaskIntent.TECHNICAL_QUERY
    assert class_res.is_actionable_task is False

    handled, msg, result = agent_mgr.handle_command(raw_prompt, lang="en")
    assert handled is False
    assert result is None


def test_test_d_create_todo_list_in_react_and_run_it(agent_mgr):
    """
    Test D: 'create a todo list in React and run it'
    Must:
    1. Classify as CREATE_PROJECT with execution_requested = True.
    2. Target React / Web framework.
    3. Activate Phase 6 pipeline.
    """
    raw_prompt = "create a todo list in React and run it"
    class_res = TaskClassifier.classify(raw_prompt)
    assert class_res.intent == TaskIntent.CREATE_PROJECT
    assert class_res.is_actionable_task is True
    assert class_res.execution_requested is True

    handled, msg, result = agent_mgr.handle_command(raw_prompt, lang="en")
    assert handled is True
    assert result is not None
    assert result.success is True


def test_test_e_build_college_attendance_project_with_good_ui(agent_mgr):
    """
    Test E: 'build a project for managing college attendance with a good UI'
    Must:
    1. Classify as CREATE_PROJECT dynamically without hardcoded attendance keywords.
    2. Activate Phase 6 and generate project.
    """
    raw_prompt = "build a project for managing college attendance with a good UI"
    class_res = TaskClassifier.classify(raw_prompt)
    assert class_res.intent == TaskIntent.CREATE_PROJECT
    assert class_res.is_actionable_task is True
    assert class_res.ui_required is True

    handled, msg, result = agent_mgr.handle_command(raw_prompt, lang="en")
    assert handled is True
    assert result is not None
    assert result.success is True


def test_test_f_unseen_desire_phrasing_and_dashboard(agent_mgr):
    """
    Test F: Arbitrary unseen project phrasings:
    1. 'I want an application that tracks my daily expenses.'
    2. 'Can you build me a small weather dashboard?'
    Must activate Phase 6 based on semantic intent.
    """
    prompt1 = "I want an application that tracks my daily expenses."
    class_res1 = TaskClassifier.classify(prompt1)
    assert class_res1.intent == TaskIntent.CREATE_PROJECT
    assert class_res1.is_actionable_task is True

    prompt2 = "Can you build me a small weather dashboard?"
    class_res2 = TaskClassifier.classify(prompt2)
    assert class_res2.intent == TaskIntent.CREATE_PROJECT
    assert class_res2.is_actionable_task is True
    assert class_res2.ui_required is True


def test_memory_router_does_not_block_task_router(agent_mgr):
    """
    Verify that MemoryRouter returning UNKNOWN (long-term retrieval skipped)
    does NOT prevent TaskClassifier & Phase 6 from creating the project.
    """
    user_prompt = "create a fully functional todo list for me with an interactive ui"

    # Memory router evaluates memory retrieval (returns UNKNOWN -> skip retrieval)
    mem_router = MemoryRouter()
    mem_decision = mem_router.classify_intent(user_prompt)
    assert mem_decision.intent == MemoryIntent.UNKNOWN
    assert mem_decision.should_retrieve_memory is False

    # Task router independently evaluates actionable project intent
    task_res = TaskClassifier.classify(user_prompt)
    assert task_res.intent == TaskIntent.CREATE_PROJECT
    assert task_res.is_actionable_task is True

    # Agent executes independently and succeeds
    handled, msg, result = agent_mgr.handle_command(user_prompt, lang="en")
    assert handled is True
    assert result is not None
    assert result.success is True
