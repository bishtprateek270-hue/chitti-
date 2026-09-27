"""
Test Suite for Chitti Unified Master Router.
Tests:
- Master Router semantic intent classification across all 19 categories.
- Decoupling of memory routing, computer agent routing, and conversational knowledge answering.
- Multilingual support (English, Hindi, Hinglish).
- Verifies no 'Not an actionable task' output for conversational queries.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.router.master_router import MasterRouter, MasterRoute, MasterRouteDecision
from src.main import ChittiController


# ==============================================================================
# 1. MASTER ROUTER CLASSIFICATION UNIT TESTS
# ==============================================================================

def test_master_route_personal_memory_english():
    """Test 1: Personal memory query in English."""
    prompts = [
        "what do you know about me",
        "what do u know about me",
        "what is my name",
        "tell me about my college",
        "who is my best friend",
    ]
    for p in prompts:
        res = MasterRouter.classify_request(p)
        assert res.route == MasterRoute.PERSONAL_MEMORY
        assert res.requires_memory is True
        assert res.requires_computer is False
        assert res.requires_phase5 is False
        assert res.requires_phase6 is False


def test_master_route_personal_memory_hindi():
    """Test 2: Personal memory query in Hindi/Hinglish."""
    prompts = [
        "mera naam kya hai",
        "mere baare mein kya jaante ho",
        "tumhe kisne banaya",
    ]
    for p in prompts:
        res = MasterRouter.classify_request(p)
        assert res.route == MasterRoute.PERSONAL_MEMORY
        assert res.requires_memory is True
        assert res.requires_computer is False


def test_master_route_self_identity():
    """Test 3: Self-identity and introduction query."""
    prompts = [
        "tell me about yourself",
        "who are you",
        "introduce yourself",
        "tum kaun ho",
        "apne baare me batao",
    ]
    for p in prompts:
        res = MasterRouter.classify_request(p)
        assert res.route == MasterRoute.SELF_IDENTITY
        assert res.requires_memory is False
        assert res.requires_computer is False


def test_master_route_technical_knowledge():
    """Test 4: Pure technical questions (ANN, linked list, error explanations)."""
    prompts = [
        "what is ANN",
        "what is a linked list",
        "explain convolutional neural networks",
        "how does backpropagation work",
        "why is my Python program giving this error?",
        "ANN kya hota hai",
    ]
    for p in prompts:
        res = MasterRouter.classify_request(p)
        assert res.route == MasterRoute.TECHNICAL_KNOWLEDGE
        assert res.requires_memory is False
        assert res.requires_computer is False
        assert res.requires_phase5 is False
        assert res.requires_phase6 is False


def test_master_route_general_knowledge():
    """Test 5: General world knowledge and factual questions."""
    prompts = [
        "what is the capital of France",
        "where is Mount Everest",
        "how many continents are there",
    ]
    for p in prompts:
        res = MasterRouter.classify_request(p)
        assert res.route == MasterRoute.GENERAL_KNOWLEDGE
        assert res.requires_memory is False
        assert res.requires_computer is False


def test_master_route_casual_conversation():
    """Test 6: Casual chat, greetings, small talk."""
    prompts = [
        "how are you",
        "hello",
        "hey chitti",
        "kaise ho",
        "kya haal hai",
    ]
    for p in prompts:
        res = MasterRouter.classify_request(p)
        assert res.route == MasterRoute.CHAT
        assert res.requires_memory is False
        assert res.requires_computer is False


def test_master_route_project_creation():
    """Test 7: Project creation requests (Phase 6 Agent)."""
    prompts = [
        "create a fully functional todo list with an interactive UI",
        "build a calculator with a good UI and run it",
        "make an expense tracker app for me",
        "mere liye ek simple expense tracker bana do",
        "create an ANN visualization project and run it",
    ]
    for p in prompts:
        res = MasterRouter.classify_request(p)
        assert res.route == MasterRoute.PROJECT_CREATION
        assert res.requires_memory is False
        assert res.requires_computer is True
        assert res.requires_phase6 is True


def test_master_route_computer_task():
    """Test 8: Direct computer / OS tasks (Phase 5)."""
    prompts = [
        "open VS Code",
        "launch Notepad",
        "take a screenshot",
        "open Downloads folder",
    ]
    for p in prompts:
        res = MasterRouter.classify_request(p)
        assert res.route in (MasterRoute.COMPUTER_TASK, MasterRoute.SYSTEM_TASK, MasterRoute.FILE_OPERATION)
        assert res.requires_computer is True
        assert res.requires_phase5 is True
        assert res.requires_phase6 is False


def test_master_route_browser_task():
    """Test 9: Browser & YouTube interaction tasks."""
    prompts = [
        "open Chrome and search for Python documentation",
        "go to youtube and play a Sonu Nigam song",
        "open https://python.org in browser",
    ]
    for p in prompts:
        res = MasterRouter.classify_request(p)
        assert res.route == MasterRoute.BROWSER_TASK
        assert res.requires_computer is True
        assert res.requires_phase5 is True


def test_master_route_project_execution():
    """Test 10: Project execution / debugging."""
    res_run = MasterRouter.classify_request("run my existing project")
    assert res_run.route == MasterRoute.PROJECT_EXECUTION
    assert res_run.requires_computer is True

    res_dbg = MasterRouter.classify_request("open VS Code, find the error in my project and fix it")
    assert res_dbg.route in (MasterRoute.PROJECT_DEBUGGING, MasterRoute.PROJECT_MODIFICATION)
    assert res_dbg.requires_computer is True


# ==============================================================================
# 2. INTEGRATION TESTS (CHITTI ASSISTANT PIPELINE)
# ==============================================================================

@patch("src.main.get_config")
@patch("src.main.get_llm")
@patch("src.main.get_stt")
@patch("src.main.get_tts")
@patch("src.main.MicrophoneManager")
def test_conversational_questions_do_not_output_actionable_error(
    mock_mic, mock_tts, mock_stt, mock_llm_factory, mock_cfg
):
    """
    Verifies that 'what is ANN', 'what do you know about me', and 'tell me about yourself'
    reach the LLM / Memory pipeline and NEVER produce 'Not an actionable task'.
    """
    mock_config_obj = MagicMock()
    mock_config_obj.log_level = "INFO"
    mock_config_obj.language.confidence_threshold = 0.6
    mock_config_obj.memory.enabled = False
    mock_config_obj.memory.top_k = 3
    mock_cfg.return_value = mock_config_obj

    mock_llm = MagicMock()
    mock_llm.generate_response.return_value = "Artificial Neural Networks (ANN) are computing systems inspired by biological neural networks."
    mock_llm_factory.return_value = mock_llm

    assistant = ChittiController()
    assistant.llm = mock_llm
    assistant.memory = None
    # Mock speak to capture speech outputs
    spoken = []
    assistant.speak = lambda text: spoken.append(text)

    # 1. Technical Query: 'what is ANN'
    assistant.process_user_input("what is ANN")
    assert len(spoken) > 0
    assert "Not an actionable task" not in spoken[-1]
    assert "Artificial Neural Networks" in spoken[-1]

    # 2. Personal Memory Query: 'what do u know about me'
    spoken.clear()
    mock_llm.generate_response.return_value = "I remember that you are Prateek and you created me."
    assistant.process_user_input("what do u know about me")
    assert len(spoken) > 0
    assert "Not an actionable task" not in spoken[-1]

    # 3. Identity Query: 'tell me about yourself'
    spoken.clear()
    mock_llm.generate_response.return_value = "I am Chitti, your personal AI desktop companion robot."
    assistant.process_user_input("tell me about yourself")
    assert len(spoken) > 0
    assert "Not an actionable task" not in spoken[-1]
