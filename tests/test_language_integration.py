"""
Integration tests for Chitti Multilingual Understanding & Intelligence (Phase 4).
Tests controller processing with multilingual inputs, language switching, translation, and cross-lingual memory/vision integration.
"""

import pytest
from unittest.mock import MagicMock, patch
from src.main import ChittiController
from src.language.language_models import LanguageCode


@pytest.fixture
def mock_controller():
    with patch("src.main.get_stt"), patch("src.main.get_tts"), patch("src.main.get_llm"):
        controller = ChittiController()
        controller.llm = MagicMock()
        controller.llm.check_connection.return_value = True
        controller.llm.list_available_models.return_value = ["qwen2.5-coder:7b"]
        controller.llm.generate_response.return_value = "This is a mock response."
        controller.tts = MagicMock()
        controller.tts.speak.return_value = None
        controller.memory = MagicMock()
        controller.memory.handle_interaction.return_value = None
        controller.memory.recall.return_value = []
        controller.vision = MagicMock()
        controller.vision.handle_face_commands.return_value = None
        controller.vision.is_vision_query.return_value = False
        controller.translator = MagicMock()
        controller.translator.translate.return_value.translated_text = "Translated text."
        return controller


def test_controller_language_switch_command(mock_controller):
    mock_controller.process_user_input("From now on Hinglish mein answer karna")
    assert mock_controller.session_response_language == "hinglish"
    assert mock_controller.history.message_count > 0


def test_controller_translation_command(mock_controller):
    mock_controller.process_user_input("Translate this into Hindi: Machine learning is powerful")
    assert mock_controller.translator.translate.called


def test_controller_multilingual_vision_query(mock_controller):
    mock_controller.vision.analyze_frame.return_value.summary_text = "I see a laptop."
    mock_controller.vision.analyze_frame.return_value.known_people_names = []
    mock_controller.vision.format_vision_context_for_llm.return_value = "Vision context: Laptop"

    mock_controller.process_user_input("Chitti tu kya dekh raha hai?")
    assert mock_controller.vision.analyze_frame.called


def test_controller_cross_lingual_memory_recall(mock_controller):
    mock_controller.process_user_input("Mera preferred programming language kya hai?")
    # Should query memory
    assert mock_controller.memory.recall.called


def test_controller_language_guidance_injected_into_llm(mock_controller):
    mock_controller.process_user_input("Mujhe CNN architecture explain karo")
    messages = mock_controller.history.get_messages_for_llm()
    system_prompt = messages[0]["content"]
    assert "Hinglish" in system_prompt or "RESPONSE LANGUAGE" in system_prompt
