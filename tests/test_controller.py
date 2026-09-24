"""Tests for Chitti Main Application Controller."""

from unittest.mock import patch, MagicMock
from src.main import ChittiController
from src.brain.llm import LLMConnectionError


def test_controller_initialization():
    controller = ChittiController()
    with patch("src.main.MicrophoneManager.list_microphones", return_value=[]), \
         patch("src.main.get_tts", return_value=MagicMock()), \
         patch("src.main.get_stt", return_value=MagicMock()), \
         patch("src.main.get_llm") as mock_llm_factory:

        mock_llm = MagicMock()
        mock_llm.check_connection.return_value = True
        mock_llm.list_available_models.return_value = ["llama3.2"]
        mock_llm_factory.return_value = mock_llm

        controller.initialize()

        assert controller.llm is not None
        assert controller.history.is_empty


def test_controller_process_user_input_and_history():
    controller = ChittiController()
    mock_llm = MagicMock()
    mock_llm.generate_response.return_value = "Neural networks are computational models inspired by the brain."
    controller.llm = mock_llm

    mock_tts = MagicMock()
    controller.tts = mock_tts

    controller.process_user_input("Explain neural networks.")

    assert controller.history.message_count == 2
    mock_llm.generate_response.assert_called_once()
    mock_tts.speak.assert_called_once_with(
        "Neural networks are computational models inspired by the brain.",
        wait=True
    )


def test_controller_handles_llm_failure_gracefully():
    controller = ChittiController()
    mock_llm = MagicMock()
    mock_llm.generate_response.side_effect = LLMConnectionError("Ollama offline")
    controller.llm = mock_llm

    mock_tts = MagicMock()
    controller.tts = mock_tts

    controller.process_user_input("Hello Chitti")

    # Controller should not raise uncaught exception
    mock_tts.speak.assert_called_once()
    spoken_text = mock_tts.speak.call_args[0][0]
    assert "can't reach my local AI brain" in spoken_text.lower() or "ollama" in spoken_text.lower()
