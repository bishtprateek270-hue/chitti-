"""Tests for Chitti LLM Module and Error Handling."""

import pytest
from unittest.mock import patch, MagicMock
import requests

from src.brain.llm import (
    OllamaLLM,
    LLMError,
    LLMConnectionError,
    LLMTimeoutError,
    LLMModelNotFoundError
)
from src.config import LLMConfig


def test_ollama_generate_response_success():
    cfg = LLMConfig(base_url="http://localhost:11434", model="llama3.2")
    llm = OllamaLLM(cfg)

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "message": {"role": "assistant", "content": "Hello! I am Chitti."}
    }

    with patch("requests.post", return_value=mock_response) as mock_post:
        messages = [{"role": "user", "content": "Hi Chitti"}]
        result = llm.generate_response(messages)
        assert result == "Hello! I am Chitti."
        mock_post.assert_called_once()


def test_ollama_connection_error():
    cfg = LLMConfig(base_url="http://localhost:11434", model="llama3.2")
    llm = OllamaLLM(cfg)

    with patch("requests.post", side_effect=requests.exceptions.ConnectionError("Connection refused")):
        with pytest.raises(LLMConnectionError):
            llm.generate_response([{"role": "user", "content": "Hi"}])


def test_ollama_timeout_error():
    cfg = LLMConfig(base_url="http://localhost:11434", model="llama3.2")
    llm = OllamaLLM(cfg)

    with patch("requests.post", side_effect=requests.exceptions.Timeout("Request timed out")):
        with pytest.raises(LLMTimeoutError):
            llm.generate_response([{"role": "user", "content": "Hi"}])


def test_ollama_model_not_found():
    cfg = LLMConfig(base_url="http://localhost:11434", model="nonexistent_model")
    llm = OllamaLLM(cfg)

    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.text = "Model not found"

    with patch("requests.post", return_value=mock_response):
        with patch.object(llm, "list_available_models", return_value=["llama3.2"]):
            with pytest.raises(LLMModelNotFoundError):
                llm.generate_response([{"role": "user", "content": "Hi"}])


def test_ollama_check_connection():
    llm = OllamaLLM()
    mock_resp = MagicMock()
    mock_resp.status_code = 200

    with patch("requests.get", return_value=mock_resp):
        assert llm.check_connection() is True

    with patch("requests.get", side_effect=Exception("Failed")):
        assert llm.check_connection() is False


def test_ollama_ensure_running_success():
    cfg = LLMConfig(base_url="http://localhost:11434", model="llama3.2")
    llm = OllamaLLM(cfg)

    # First check fails, then subprocess starts, then second check succeeds
    with patch.object(llm, "check_connection", side_effect=[False, True]), \
         patch("shutil.which", return_value="C:\\fake\\ollama.exe"), \
         patch("subprocess.Popen", return_value=MagicMock()) as mock_popen:
        assert llm.ensure_running(wait_timeout=2.0) is True
        mock_popen.assert_called_once()


def test_ollama_self_heal_retry():
    cfg = LLMConfig(base_url="http://localhost:11434", model="llama3.2")
    llm = OllamaLLM(cfg)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"message": {"content": "Healed response"}}

    # First post raises ConnectionError, then ensure_running succeeds, then second post succeeds
    with patch("requests.post", side_effect=[requests.exceptions.ConnectionError("Refused"), mock_resp]), \
         patch.object(llm, "ensure_running", return_value=True) as mock_heal:
        result = llm.generate_response([{"role": "user", "content": "Hello"}])
        assert result == "Healed response"
        mock_heal.assert_called_once()

