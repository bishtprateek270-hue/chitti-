"""
Phase 5 Final Regression & Automated Test Suite:
Reliable Multi-Step Computer Actions, Structured Goals, Generic Primitives,
Authentication Handling, Error Recovery, and Strict Goal Verification (No False Success).
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from src.agent.actions import ActionResult, ActionType
from src.agent.computer import AppController, BrowserController, ComputerController, FilesystemController, ScreenAnalyzer, TerminalController
from src.agent.intent import ActionIntentAnalyzer, ActionIntentType
from src.agent.manager import LaptopAgentManager
from src.agent.planner import AgentPlanner, ComputerAgentLoop
from src.agent.projects import ProjectRegistry
from src.agent.task_state import AgentStep, StepStatus, TaskState, TaskStatus
from src.agent.tools import ToolEngine
from src.agent.verifier import SubtaskVerifier
from src.router.master_router import MasterRoute, MasterRouter


@pytest.fixture
def mock_agent_environment(tmp_path):
    ws = tmp_path / "workspace"
    ss = tmp_path / "screenshots"
    pr = tmp_path / "projects.json"
    ws.mkdir()
    ss.mkdir()
    
    mgr = LaptopAgentManager(
        workspace_dir=str(ws),
        screenshots_dir=str(ss),
        project_registry_path=str(pr),
    )
    return mgr


# ==============================================================================
# 1. STRUCTURED USER GOAL & ENTITY EXTRACTION TESTS
# ==============================================================================

def test_01_structured_goal_whatsapp():
    """Verify dynamic entity extraction and goal representation for WhatsApp message."""
    raw = "open whatsapp web and send hi to ayush"
    intent = ActionIntentAnalyzer.extract_intent(raw)
    
    assert intent.intent == ActionIntentType.SEND_MESSAGE
    assert intent.application == "WhatsApp Web"
    assert intent.target == "ayush"
    assert intent.content.lower() == "hi"
    assert intent.requires_browser is True
    assert intent.verification_required is True
    assert "ayush" in intent.goal.lower()


def test_02_structured_goal_gmail_send():
    """Verify dynamic entity extraction and goal representation for Email sending."""
    raw = "open gmail and send an email to ayush@example.com saying the meeting is at 5"
    intent = ActionIntentAnalyzer.extract_intent(raw)
    
    assert intent.intent == ActionIntentType.SEND_EMAIL
    assert intent.recipient == "ayush@example.com"
    assert "meeting is at 5" in intent.content
    assert intent.requires_confirmation is True
    assert intent.verification_required is True


def test_03_structured_goal_browser_search():
    """Verify dynamic entity extraction for browser search."""
    raw = "open chrome and search Python documentation"
    intent = ActionIntentAnalyzer.extract_intent(raw)
    
    assert intent.intent == ActionIntentType.SEARCH_WEB
    assert intent.parameters.get("query") == "Python documentation"


# ==============================================================================
# 2. DRAFTING VS SENDING DISTINCTION TESTS
# ==============================================================================

def test_04_action_vs_drafting_classification():
    """Verify 'write an email' is classified as CHAT drafting, while 'send email' is classified as BROWSER_TASK action."""
    # Drafting request -> CHAT
    draft_req = "write an email to Ayush asking what he is doing"
    route_draft = MasterRouter.classify_request(draft_req)
    assert route_draft.route == MasterRoute.CHAT
    intent_draft = ActionIntentAnalyzer.extract_intent(draft_req)
    assert intent_draft.intent == ActionIntentType.WRITE_EMAIL

    # Execution request -> BROWSER_TASK
    send_req = "send an email to ayush@example.com saying hello"
    route_send = MasterRouter.classify_request(send_req)
    assert route_send.route == MasterRoute.BROWSER_TASK
    intent_send = ActionIntentAnalyzer.extract_intent(send_req)
    assert intent_send.intent == ActionIntentType.SEND_EMAIL


# ==============================================================================
# 3. COMPLETE MULTI-STEP PLAN GENERATION TESTS (NO PREMATURE COMPLETION)
# ==============================================================================

def test_05_whatsapp_multistep_plan_has_end_to_end_actions():
    """
    Verifies that 'open whatsapp web and send hi to ayush' produces an end-to-end plan
    including authentication check, contact search, typing, sending, and verification.
    """
    registry = MagicMock(spec=ProjectRegistry)
    planner = AgentPlanner(project_registry=registry)

    state = planner.plan_task("open whatsapp web and send hi to ayush")
    assert state is not None
    assert len(state.steps) >= 6

    actions = [s.action_type for s in state.steps]
    assert "OPEN_URL" in actions
    assert "CHECK_AUTHENTICATION" in actions
    assert "SEARCH_CONTACT" in actions
    assert "SELECT_CONVERSATION" in actions
    assert "TYPE_TEXT" in actions
    assert "SEND_MESSAGE" in actions
    assert "VERIFY_MESSAGE_SENT" in actions


def test_06_gmail_multistep_plan_has_end_to_end_actions():
    """
    Verifies that 'open gmail and send an email to ayush@example.com saying hi'
    produces compose, confirmation, send, and verification steps.
    """
    registry = MagicMock(spec=ProjectRegistry)
    planner = AgentPlanner(project_registry=registry)

    state = planner.plan_task("open gmail and send an email to ayush@example.com saying hi")
    assert state is not None
    assert len(state.steps) >= 6

    actions = [s.action_type for s in state.steps]
    assert "OPEN_URL" in actions
    assert "CHECK_AUTHENTICATION" in actions
    assert "COMPOSE_EMAIL" in actions
    assert "CONFIRM_SEND" in actions
    assert "SEND_EMAIL" in actions
    assert "VERIFY_EMAIL_SENT" in actions


# ==============================================================================
# 4. GENERIC PRIMITIVES & TOOL REGISTRY TESTS
# ==============================================================================

def test_07_generic_primitives_availability(mock_agent_environment):
    """Verifies that all 28 required generic computer action primitives are registered and callable."""
    tools = mock_agent_environment.tools
    
    required_primitives = [
        "open_application", "close_application", "open_url", "wait", "wait_for_ui",
        "inspect_screen", "take_screenshot", "screenshot", "read_screen", "find_ui_element",
        "click", "double_click", "right_click", "move_mouse", "drag", "scroll",
        "type_text", "press_key", "hotkey", "copy", "paste", "clipboard_read", "clipboard_write",
        "focus_window", "get_active_window", "verify_ui_state", "verify_text", "verify_element",
        "verify_application_state"
    ]
    
    for prim in required_primitives:
        assert prim in tools.tools, f"Required primitive '{prim}' is not registered in ToolEngine."


def test_08_generic_primitive_execution(mock_agent_environment):
    """Tests executing basic generic primitives."""
    tools = mock_agent_environment.tools
    
    # 1. wait
    res_wait = tools.execute_tool("wait", {"seconds": 0.1})
    assert res_wait.success is True

    # 2. inspect_screen
    res_screen = tools.execute_tool("inspect_screen", {})
    assert res_screen.success is True

    # 3. clipboard write and read
    res_cw = tools.execute_tool("clipboard_write", {"text": "Chitti test clipboard"})
    assert res_cw.success is True
    res_cr = tools.execute_tool("clipboard_read", {})
    assert res_cr.success is True


# ==============================================================================
# 5. EXECUTION LOOP, FINAL VERIFICATION & NO FALSE SUCCESS TESTS
# ==============================================================================

def test_09_no_false_success_when_later_step_fails(mock_agent_environment):
    """
    Verifies that if step 1 (OPEN_URL) succeeds but a later step (e.g. SEND_MESSAGE) fails,
    the task is NEVER marked COMPLETED. It must be PARTIALLY_COMPLETED or FAILED.
    """
    loop = mock_agent_environment.loop
    
    # Build a task with 2 steps where step 2 fails
    state = TaskState(task_description="send message test", goal="message sent")
    state.steps = [
        AgentStep(step_id=1, description="Open URL", action_type="OPEN_URL", parameters={"url": "https://web.whatsapp.com"}),
        AgentStep(step_id=2, description="Send Message", action_type="NON_EXISTENT_FAILING_ACTION", parameters={}, depends_on=[1]),
    ]
    
    success, msg = loop.execute_plan(state)
    
    assert success is False
    assert state.status != TaskStatus.COMPLETED
    assert state.status in (TaskStatus.PARTIALLY_COMPLETED, TaskStatus.FAILED)


def test_10_full_plan_success_sets_completed(mock_agent_environment):
    """Verifies that when all steps succeed, final status is COMPLETED and final verification passes."""
    from src.agent.computer.screen_analyzer import VerificationResult
    loop = mock_agent_environment.loop
    
    with patch.object(mock_agent_environment.screen_analyzer, "verify_window", return_value=VerificationResult(True, "Window 'Chrome' detected", "Chrome")):
        state = TaskState(task_description="open and search", goal="search results displayed")
        state.steps = [
            AgentStep(step_id=1, description="Search web", action_type="SEARCH_WEB", parameters={"query": "Python"}),
            AgentStep(step_id=2, description="Verify search", action_type="VERIFY_WINDOW", parameters={"title": "Chrome"}, depends_on=[1]),
        ]
        
        success, msg = loop.execute_plan(state)
        assert success is True
        assert state.status == TaskStatus.COMPLETED


# ==============================================================================
# 6. UNSEEN MULTI-STEP COMPUTER TASK & BROWSER FORM INTERACTION
# ==============================================================================

def test_11_unseen_multistep_computer_task(mock_agent_environment):
    """
    Verifies execution of an arbitrary multi-step computer task:
    1. Create folder
    2. Create file with content
    3. Verify file content
    4. Read screen / verify state
    """
    loop = mock_agent_environment.loop
    ws_dir = mock_agent_environment.workspace_dir
    
    test_file = f"{ws_dir}/data_report.txt"
    state = TaskState(task_description="generate data report", goal="data report verified")
    state.steps = [
        AgentStep(step_id=1, description="Create report file", action_type="CREATE_FILE", parameters={"path": test_file, "content": "Metric: 99.9%"}),
        AgentStep(step_id=2, description="Verify report content", action_type="VERIFY_FILE_CONTENT", parameters={"path": test_file, "expected_keyword": "99.9%"}, depends_on=[1]),
        AgentStep(step_id=3, description="Inspect active screen", action_type="INSPECT_SCREEN", parameters={}, depends_on=[2]),
    ]
    
    success, msg = loop.execute_plan(state)
    assert success is True
    assert state.status == TaskStatus.COMPLETED
    assert Path(test_file).exists()


def test_12_browser_form_interaction_flow(mock_agent_environment):
    """
    Verifies generic browser form workflow:
    1. Open URL
    2. Wait for UI
    3. Type text
    4. Click submit
    5. Verify UI state
    """
    loop = mock_agent_environment.loop
    
    state = TaskState(task_description="submit web form", goal="form submitted")
    state.steps = [
        AgentStep(step_id=1, description="Open form portal", action_type="OPEN_URL", parameters={"url": "https://example.com/form"}),
        AgentStep(step_id=2, description="Wait for form UI", action_type="WAIT", parameters={"seconds": 0.1}, depends_on=[1]),
        AgentStep(step_id=3, description="Enter form data", action_type="TYPE_TEXT", parameters={"text": "Prateek Bisht"}, depends_on=[2]),
        AgentStep(step_id=4, description="Click submit button", action_type="CLICK", parameters={"x": 500, "y": 400}, depends_on=[3]),
        AgentStep(step_id=5, description="Verify submission state", action_type="VERIFY_UI_STATE", parameters={}, depends_on=[4]),
    ]
    
    success, msg = loop.execute_plan(state)
    assert success is True
    assert state.status == TaskStatus.COMPLETED


# ==============================================================================
# 7. CONFIRMATION ROUTING & RESUMABLE MULTI-STEP TASK TESTS
# ==============================================================================

def test_13_confirmation_category_and_resumption(mock_agent_environment):
    """
    Verifies that when an email plan reaches confirmation:
    1. It asks a specific email confirmation question (NOT a file deletion question).
    2. When the user confirms ('yes'), it resumes the SAME task state from step 6 onwards.
    """
    mgr = mock_agent_environment
    raw_cmd = "open gmail and send an email to ayush@example.com saying hi"
    
    # Initial command execution
    res = mgr.handle_command(raw_cmd)
    assert res is not None
    handled, prompt_msg, action_res = res
    assert handled is True
    assert "ayush@example.com" in prompt_msg
    assert "modify or delete" not in prompt_msg  # NEVER confuse email with file deletion!
    assert mgr.active_task_state is not None
    assert mgr.active_task_state.status == TaskStatus.WAITING_FOR_CONFIRMATION
    orig_task_id = mgr.active_task_state.task_id

    # User confirms
    res_confirm = mgr.handle_command("yes")
    assert res_confirm is not None
    c_handled, c_msg, c_res = res_confirm
    assert c_handled is True
    assert c_res.success is True
    assert mgr.active_task_state.status == TaskStatus.COMPLETED
    assert mgr.active_task_state.task_id == orig_task_id


def test_14_confirmation_cancellation(mock_agent_environment):
    """Verifies that answering 'no' to confirmation cancels the task without executing remaining steps."""
    mgr = mock_agent_environment
    raw_cmd = "open gmail and send an email to test@example.com saying hello"
    
    mgr.handle_command(raw_cmd)
    assert mgr.active_task_state is not None
    
    # User cancels
    res_cancel = mgr.handle_command("no")
    assert res_cancel is not None
    c_handled, c_msg, _ = res_cancel
    assert c_handled is True
    assert "cancel" in c_msg.lower()
    assert mgr.active_task_state is None


# ==============================================================================
# 8. REAL TOOL INVOCATION AUDIT ASSERTIONS
# ==============================================================================

def test_15_whatsapp_real_tools_called(mock_agent_environment):
    """
    Verifies that SEARCH_CONTACT, SELECT_CONVERSATION, SEND_MESSAGE, and VERIFY_MESSAGE_SENT
    dispatch real computer primitives to the ToolEngine.
    """
    loop = mock_agent_environment.loop
    executed_tools = []
    
    original_exec = mock_agent_environment.tools.execute_tool
    def spy_execute(tool_name, args):
        executed_tools.append(tool_name)
        return original_exec(tool_name, args)
        
    mock_agent_environment.tools.execute_tool = spy_execute
    
    state = TaskState(task_description="whatsapp message", goal="message sent")
    state.steps = [
        AgentStep(step_id=1, description="Open WhatsApp", action_type="OPEN_URL", parameters={"url": "https://web.whatsapp.com"}),
        AgentStep(step_id=2, description="Check Auth", action_type="CHECK_AUTHENTICATION", parameters={"service": "WhatsApp Web"}, depends_on=[1]),
        AgentStep(step_id=3, description="Search Contact", action_type="SEARCH_CONTACT", parameters={"contact": "ayush", "service": "WhatsApp Web"}, depends_on=[2]),
        AgentStep(step_id=4, description="Select Chat", action_type="SELECT_CONVERSATION", parameters={"contact": "ayush", "service": "WhatsApp Web"}, depends_on=[3]),
        AgentStep(step_id=5, description="Type message", action_type="TYPE_TEXT", parameters={"text": "hi"}, depends_on=[4]),
        AgentStep(step_id=6, description="Send message", action_type="SEND_MESSAGE", parameters={"contact": "ayush", "text": "hi"}, depends_on=[5]),
        AgentStep(step_id=7, description="Verify message", action_type="VERIFY_MESSAGE_SENT", parameters={"contact": "ayush", "text": "hi"}, depends_on=[6]),
    ]
    
    success, msg = loop.execute_plan(state)
    assert success is True
    assert state.status == TaskStatus.COMPLETED
    
    # Real tool invocation assertions
    assert "open_url" in executed_tools
    assert "inspect_screen" in executed_tools
    assert "type_text" in executed_tools
    assert "press_key" in executed_tools
    assert "click" in executed_tools
    assert "verify_ui_state" in executed_tools


def test_16_gmail_real_tools_called(mock_agent_environment):
    """
    Verifies that COMPOSE_EMAIL, SEND_EMAIL, and VERIFY_EMAIL_SENT
    dispatch real computer primitives to the ToolEngine.
    """
    loop = mock_agent_environment.loop
    executed_tools = []
    
    original_exec = mock_agent_environment.tools.execute_tool
    def spy_execute(tool_name, args):
        executed_tools.append(tool_name)
        return original_exec(tool_name, args)
        
    mock_agent_environment.tools.execute_tool = spy_execute
    
    state = TaskState(task_description="send email test", goal="email sent")
    state.steps = [
        AgentStep(step_id=1, description="Compose email", action_type="COMPOSE_EMAIL", parameters={"recipient": "ayush@example.com", "subject": "Test", "content": "hi"}),
        AgentStep(step_id=2, description="Send email", action_type="SEND_EMAIL", parameters={"recipient": "ayush@example.com", "subject": "Test", "content": "hi"}, depends_on=[1]),
        AgentStep(step_id=3, description="Verify email", action_type="VERIFY_EMAIL_SENT", parameters={"recipient": "ayush@example.com"}, depends_on=[2]),
    ]
    
    success, msg = loop.execute_plan(state)
    assert success is True
    assert state.status == TaskStatus.COMPLETED
    
    assert "find_ui_element" in executed_tools
    assert "type_text" in executed_tools
    assert "hotkey" in executed_tools
    assert "verify_ui_state" in executed_tools

