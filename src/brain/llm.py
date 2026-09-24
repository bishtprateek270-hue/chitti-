"""
Chitti LLM Module.
Provides an abstract base interface and an Ollama implementation for local LLM inference.
"""

import json
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import requests

from src.config import LLMConfig, get_config
from src.brain.personality import ConversationHistory


class LLMError(Exception):
    """Base exception for LLM-related errors."""
    pass


class LLMConnectionError(LLMError):
    """Raised when unable to connect to the LLM backend (e.g., Ollama not running)."""
    pass


class LLMTimeoutError(LLMError):
    """Raised when the LLM request times out."""
    pass


class LLMModelNotFoundError(LLMError):
    """Raised when the specified model is not pulled or available."""
    pass


class BaseLLM(ABC):
    """Abstract Base Class for LLM providers."""

    @abstractmethod
    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        """
        Generates a response from the LLM given a structured messages list.
        Each message has 'role' and 'content' keys.
        """
        pass

    @abstractmethod
    def check_connection(self) -> bool:
        """Checks if the LLM backend is accessible and ready."""
        pass

    @abstractmethod
    def list_available_models(self) -> List[str]:
        """Lists models currently installed and available in the backend."""
        pass


class OllamaLLM(BaseLLM):
    """Ollama local inference provider."""

    def __init__(self, config: Optional[LLMConfig] = None):
        cfg = config or get_config().llm
        self.base_url = cfg.base_url.rstrip("/")
        self.model = cfg.model
        self.timeout = cfg.timeout_seconds
        self.chat_endpoint = f"{self.base_url}/api/chat"
        self.tags_endpoint = f"{self.base_url}/api/tags"

    def check_connection(self) -> bool:
        """Pings the Ollama server to check reachability."""
        try:
            resp = requests.get(self.base_url, timeout=3)
            return resp.status_code == 200
        except Exception:
            return False

    def list_available_models(self) -> List[str]:
        """Retrieves list of model names currently available in Ollama."""
        try:
            resp = requests.get(self.tags_endpoint, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                return [m.get("name", "") for m in data.get("models", [])]
            return []
        except Exception:
            return []

    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        """
        Sends the messages to Ollama /api/chat endpoint and returns the generated text.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
            }
        }

        try:
            response = requests.post(
                self.chat_endpoint,
                json=payload,
                timeout=self.timeout
            )
        except requests.exceptions.ConnectionError as e:
            raise LLMConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. Is Ollama running?"
            ) from e
        except requests.exceptions.Timeout as e:
            raise LLMTimeoutError(
                f"Ollama inference timed out after {self.timeout} seconds."
            ) from e
        except Exception as e:
            raise LLMError(f"Unexpected error communicating with Ollama: {e}") from e

        if response.status_code == 404:
            available = self.list_available_models()
            avail_str = f" Available models: {available}" if available else ""
            raise LLMModelNotFoundError(
                f"Model '{self.model}' not found in Ollama.{avail_str} "
                f"Run `ollama pull {self.model}` to download it."
            )

        if response.status_code != 200:
            raise LLMError(
                f"Ollama API returned HTTP {response.status_code}: {response.text}"
            )

        try:
            data = response.json()
            message = data.get("message", {})
            content = message.get("content", "").strip()
            if not content:
                return "I processed your request, but received an empty response."
            return content
        except Exception as e:
            raise LLMError(f"Failed to parse Ollama response: {e}") from e


def get_llm(config: Optional[LLMConfig] = None) -> BaseLLM:
    """Factory function returning the configured LLM engine."""
    return OllamaLLM(config)
