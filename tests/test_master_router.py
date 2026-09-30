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
from src.agent.intent import ActionIntentAnalyzer, ActionIntentType
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
        "who i am",
        "who am i",
        "do you know who i am",
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


def test_master_route_screen_vision():
    """Test: Screen vision grounding and error diagnosis queries."""
    prompts = [
        "screen pe kya hai batao",
        "meri screen dekho",
        "look at my screen",
        "what is on my screen",
        "explain the error on screen",
        "screen pe kya error hai",
    ]
    for p in prompts:
        res = MasterRouter.classify_request(p)
        assert res.route == MasterRoute.VISION_TASK, f"Failed for prompt: {p}"
        assert res.requires_computer is True
        assert res.requires_phase5 is True


def test_master_route_roadmap():
    """Test: Architecture roadmap queries."""
    prompts = [
        "roadmap k bare m batao",
        "roadmap kya hai",
        "tell me about the roadmap",
        "architecture roadmap",
        "next gen roadmap",
    ]
    for p in prompts:
        res = MasterRouter.classify_request(p)
        assert res.route == MasterRoute.TECHNICAL_KNOWLEDGE, f"Failed for prompt: {p}"
        assert res.confidence >= 0.90



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


# ==============================================================================
# 3. NEW BROWSER MESSAGING, TYPO TOLERANCE & MULTI-STEP PLANNING TESTS
# ==============================================================================

def test_master_route_whatsapp_and_messaging():
    """Verifies that messaging and WhatsApp requests are correctly classified as actions, including typos."""
    prompts = [
        "open whatsapp on web browser and message hi to ayush",
        "open whatsapp web and message hi to ayush",
        "opem whatsapp",
        "open watsapp",
        "whatsapp kholo",
        "whatsapp web kholo",
        "ayush ko hi message kar",
        "whatsapp par ayush ko hi bhejo",
        "search youtube for sonu nigam",
    ]
    for p in prompts:
        res = MasterRouter.classify_request(p)
        assert res.route in (MasterRoute.BROWSER_TASK, MasterRoute.COMPUTER_TASK), f"Failed for prompt: {p}"
        assert res.requires_computer is True
        assert res.requires_memory is False


def test_agent_planner_whatsapp_multistep_plan():
    """Verifies that AgentPlanner decomposes WhatsApp messaging requests into realistic multi-step plans."""
    from src.agent.planner import AgentPlanner
    from src.agent.projects import ProjectRegistry

    registry = MagicMock(spec=ProjectRegistry)
    planner = AgentPlanner(project_registry=registry)

    state = planner.plan_task("open whatsapp on web browser and message hi to ayush")
    assert state is not None
    assert len(state.steps) >= 5
    
    actions = [s.action_type for s in state.steps]
    assert "OPEN_URL" in actions
    assert "VERIFY_PAGE_LOADED" in actions
    assert "SEARCH_CONTACT" in actions
    assert "SELECT_CONVERSATION" in actions
    assert "SEND_MESSAGE" in actions
    assert "VERIFY_MESSAGE_SENT" in actions


# ==============================================================================
# 4. PROJECT CODE GENERATION ISOLATION TESTS (NO TEMPLATE CROSS-CONTAMINATION)
# ==============================================================================

def test_calculator_code_generation_has_no_todo_items():
    """Verifies that calculator web app generation contains ZERO todo/expense items or categories."""
    from src.agent.code_generator import CodeGenerator

    spec = CodeGenerator.generate_code_for_topic("create a calculator with a good UI and run it")
    code = spec.files[0].content

    # Must contain real calculator elements
    assert "calc-display" in code
    assert "appendNum" in code or "calculateResult" in code or "clearCalc" in code
    assert "grid" in code.lower()

    # Must NOT contain cross-contaminated Todo or Expense items
    assert "renderItems" not in code
    assert "addItem" not in code
    assert "toggleItem" not in code
    assert "deleteItem" not in code
    assert "category" not in code.lower()
    assert "searchQuery" not in code


def test_quiz_code_generation_is_domain_isolated():
    """Verifies that quiz app generation contains question engine and no todo items."""
    from src.agent.code_generator import CodeGenerator

    spec = CodeGenerator.generate_code_for_topic("create a simple quiz application with score tracking and UI")
    code = spec.files[0].content

    assert "score" in code.lower()
    assert "question" in code.lower()
    assert "renderItems" not in code
    assert "addItem" not in code


def test_weather_code_generation_is_domain_isolated():
    """Verifies that weather dashboard contains city search and temperature metrics without todo items."""
    from src.agent.code_generator import CodeGenerator

    spec = CodeGenerator.generate_code_for_topic("create a weather dashboard with clean UI")
    code = spec.files[0].content

    assert "weather" in code.lower()
    assert "temp" in code.lower()
    assert "humidity" in code.lower()
    assert "renderItems" not in code


# ==============================================================================
# 5. PHASE 3 FACE REGISTRATION & DATABASE REGRESSION TESTS
# ==============================================================================

def test_phase3_face_database_and_vision_manager_apis(tmp_path):
    """Verifies that FaceDatabase.list_all_faces and VisionManager.register_face_interactive work properly."""
    from src.vision.face_database import FaceDatabase
    from src.vision.vision_manager import VisionManager
    from src.vision.models import RecognizedPerson, DetectedObject

    # 1. Models compatibility properties
    p = RecognizedPerson(name="Prateek", confidence=0.98, is_known=True, bbox=(10, 10, 100, 100))
    assert p.identity == "Prateek"

    obj = DetectedObject(class_name="laptop", confidence=0.95, bbox=(20, 20, 200, 200))
    assert obj.label == "laptop"

    # 2. FaceDatabase list_all_faces
    db_file = tmp_path / "faces.db"
    db = FaceDatabase(db_path=str(db_file))
    db.register_or_update_face("Prateek", [0.1] * 128, sample_count=5)

    faces = db.list_all_faces()
    assert len(faces) == 1
    assert faces[0].name == "Prateek"
    assert faces[0].sample_count == 5

    # 3. VisionManager register_face_interactive method exists and is callable
    vm = VisionManager(face_db=db)
    assert hasattr(vm, "register_face_interactive")
    assert callable(vm.register_face_interactive)


# ==============================================================================
# 6. PHASE 5 ACTION EXECUTION ENGINE, ENTITY EXTRACTION & MULTI-STEP VERIFICATION
# ==============================================================================

def test_phase5_intent_extraction_and_routing_cases():
    """Verifies entity extraction, ActionIntent dataclass and MasterRouter for all 8 cases."""
    from src.agent.intent import ActionIntentAnalyzer, ActionIntentType
    from src.agent.planner import AgentPlanner
    from src.agent.projects import ProjectRegistry

    registry = MagicMock(spec=ProjectRegistry)
    planner = AgentPlanner(project_registry=registry)

    # Test 1: Simple browser action
    t1_text = "open youtube"
    t1_route = MasterRouter.classify_request(t1_text)
    assert t1_route.route == MasterRoute.BROWSER_TASK
    t1_intent = ActionIntentAnalyzer.extract_intent(t1_text)
    assert t1_intent.intent in (ActionIntentType.OPEN_URL, ActionIntentType.OPEN_APPLICATION)
    assert t1_intent.application in ("YouTube", "Browser")
    assert "youtube" in t1_intent.goal.lower() or "youtube" in str(t1_intent.parameters).lower()

    # Test 2: Browser multi-step action
    t2_text = "search youtube for sonu nigam"
    t2_route = MasterRouter.classify_request(t2_text)
    assert t2_route.route == MasterRoute.BROWSER_TASK
    t2_intent = ActionIntentAnalyzer.extract_intent(t2_text)
    assert t2_intent.intent in (ActionIntentType.SEARCH_WEB, ActionIntentType.PLAY_MEDIA)

    # Test 3: WhatsApp multi-step action
    t3_text = "open whatsapp web and send hi to ayush"
    t3_route = MasterRouter.classify_request(t3_text)
    assert t3_route.route == MasterRoute.BROWSER_TASK
    t3_intent = ActionIntentAnalyzer.extract_intent(t3_text)
    assert t3_intent.intent == ActionIntentType.SEND_MESSAGE
    assert t3_intent.application == "WhatsApp Web"
    assert t3_intent.target == "ayush"
    assert t3_intent.content.lower() == "hi"
    assert t3_intent.requires_browser is True
    assert t3_intent.verification_required is True

    # Multi-step plan verification for WhatsApp
    t3_plan = planner.plan_task(t3_text)
    assert t3_plan is not None
    actions = [s.action_type for s in t3_plan.steps]
    assert "OPEN_URL" in actions
    assert "CHECK_AUTHENTICATION" in actions
    assert "SEARCH_CONTACT" in actions
    assert "SELECT_CONVERSATION" in actions
    assert "SEND_MESSAGE" in actions
    assert "VERIFY_MESSAGE_SENT" in actions

    # Test 4: Email sending (action request, NOT drafting)
    t4_text1 = 'send "hi, what are you doing?" to ayusharyaa618@gmail.com'
    t4_route1 = MasterRouter.classify_request(t4_text1)
    assert t4_route1.route == MasterRoute.BROWSER_TASK
    t4_intent1 = ActionIntentAnalyzer.extract_intent(t4_text1)
    assert t4_intent1.intent == ActionIntentType.SEND_EMAIL
    assert t4_intent1.recipient == "ayusharyaa618@gmail.com"
    assert "hi, what are you doing?" in t4_intent1.content.lower()
    assert t4_intent1.requires_confirmation is True

    t4_text2 = "email hi, what are you doing?? message to ayusharyaa618@gmail.com from my side"
    t4_route2 = MasterRouter.classify_request(t4_text2)
    assert t4_route2.route == MasterRoute.BROWSER_TASK
    t4_intent2 = ActionIntentAnalyzer.extract_intent(t4_text2)
    assert t4_intent2.intent == ActionIntentType.SEND_EMAIL
    assert t4_intent2.recipient == "ayusharyaa618@gmail.com"

    # Multi-step plan verification for Email
    t4_plan = planner.plan_task(t4_text1)
    assert t4_plan is not None
    actions_email = [s.action_type for s in t4_plan.steps]
    assert "OPEN_URL" in actions_email
    assert "CHECK_AUTHENTICATION" in actions_email
    assert "COMPOSE_EMAIL" in actions_email
    assert "CONFIRM_SEND" in actions_email
    assert "SEND_EMAIL" in actions_email
    assert "VERIFY_EMAIL_SENT" in actions_email

    # Test 5: Email drafting (conversational drafting, NOT send action)
    t5_text = "write an email to Ayush asking what he is doing"
    t5_route = MasterRouter.classify_request(t5_text)
    assert t5_route.route == MasterRoute.CHAT
    t5_intent = ActionIntentAnalyzer.extract_intent(t5_text)
    assert t5_intent.intent == ActionIntentType.WRITE_EMAIL

    # Test 6: YouTube playback flow
    t6_text = "play a Sonu Nigam song on YouTube"
    t6_route = MasterRouter.classify_request(t6_text)
    assert t6_route.route == MasterRoute.BROWSER_TASK
    t6_intent = ActionIntentAnalyzer.extract_intent(t6_text)
    assert t6_intent.intent == ActionIntentType.PLAY_MEDIA
    t6_plan = planner.plan_task(t6_text)
    assert t6_plan is not None
    assert any(s.action_type == "PLAY_YOUTUBE" for s in t6_plan.steps)
    assert any(s.action_type == "VERIFY_PLAYBACK" for s in t6_plan.steps)

    # Test 7: Computer action
    t7_text = "open VS Code"
    t7_route = MasterRouter.classify_request(t7_text)
    assert t7_route.route == MasterRoute.COMPUTER_TASK
    t7_intent = ActionIntentAnalyzer.extract_intent(t7_text)
    assert t7_intent.intent == ActionIntentType.OPEN_APPLICATION
    assert t7_intent.application == "Visual Studio Code"

    # Test 8: Project action
    t8_text = "create a calculator with UI and run it"
    t8_route = MasterRouter.classify_request(t8_text)
    assert t8_route.route == MasterRoute.PROJECT_CREATION
    t8_intent = ActionIntentAnalyzer.extract_intent(t8_text)
    assert t8_intent.intent == ActionIntentType.CREATE_PROJECT


def test_typo_and_variation_application_routing():
    """Verify typo handling and natural phrasing variations for opening apps and sites."""
    # 1. "oprn yt"
    r1 = MasterRouter.classify_request("oprn yt")
    assert r1.route in (MasterRoute.BROWSER_TASK, MasterRoute.COMPUTER_TASK)
    i1 = ActionIntentAnalyzer.extract_intent("oprn yt")
    assert i1.intent == ActionIntentType.OPEN_URL
    assert "youtube" in i1.parameters.get("url", "").lower()

    # 2. "open you tube in any browser"
    r2 = MasterRouter.classify_request("open you tube in any browser")
    assert r2.route in (MasterRoute.BROWSER_TASK, MasterRoute.COMPUTER_TASK)
    i2 = ActionIntentAnalyzer.extract_intent("open you tube in any browser")
    assert i2.intent == ActionIntentType.OPEN_URL
    assert "youtube" in i2.parameters.get("url", "").lower()

    # 3. "oprn yt for me"
    r3 = MasterRouter.classify_request("oprn yt for me")
    assert r3.route in (MasterRoute.BROWSER_TASK, MasterRoute.COMPUTER_TASK)
    i3 = ActionIntentAnalyzer.extract_intent("oprn yt for me")
    assert i3.intent == ActionIntentType.OPEN_URL
    assert "youtube" in i3.parameters.get("url", "").lower()

    # 4. "open you tube"
    r4 = MasterRouter.classify_request("open you tube")
    assert r4.route in (MasterRoute.BROWSER_TASK, MasterRoute.COMPUTER_TASK)
    i4 = ActionIntentAnalyzer.extract_intent("open you tube")
    assert i4.intent == ActionIntentType.OPEN_URL
    assert "youtube" in i4.parameters.get("url", "").lower()



