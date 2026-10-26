"""
Chitti LLM Module.
Provides an abstract base interface and an Ollama implementation for local LLM inference.
"""

import json
import os
import sys
import time
import shutil
import subprocess
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import requests

from src.config import LLMConfig, get_config
from src.brain.personality import ConversationHistory
from src.utils.logging import log_info, log_warning, log_error, log_chitti


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
    def generate_response(self, messages: List[Dict[str, str]], request_id: Optional[int] = None) -> str:
        """
        Generates a response from the LLM given a structured messages list.
        Each message has 'role' and 'content' keys.
        """
        pass

    @abstractmethod
    def generate_response_stream(
        self, messages: List[Dict[str, str]], request_id: Optional[int] = None
    ):
        """
        Streams response tokens/chunks incrementally from the LLM.
        Yields string chunks.
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
    """Ollama local inference provider with streaming and keep-alive optimization."""

    _auto_start_proc: Optional[subprocess.Popen] = None

    def __init__(self, config: Optional[LLMConfig] = None):
        cfg = config or get_config().llm
        self.base_url = cfg.base_url.rstrip("/")
        self.model = cfg.model
        self.timeout = cfg.timeout_seconds
        self.keep_alive = getattr(cfg, "keep_alive", "-1")
        self.chat_endpoint = f"{self.base_url}/api/chat"
        self.tags_endpoint = f"{self.base_url}/api/tags"

    @property
    def resolved_keep_alive(self):
        """Returns numeric -1 for permanent 24/7 in-memory retention, or custom duration."""
        val = str(self.keep_alive).strip().lower()
        if val in ("-1", "infinite", "forever", "permanent", "24/7", "24h"):
            return -1
        return self.keep_alive

    def preload_brain(self) -> bool:
        """Preloads the LLM brain model into GPU VRAM with indefinite keep-alive for 24/7 readiness."""
        try:
            payload = {
                "model": self.model,
                "keep_alive": self.resolved_keep_alive
            }
            resp = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=30)
            if resp.status_code == 200:
                log_info(f"[CHITTI][BRAIN] AI Brain model ({self.model}) successfully preloaded in VRAM for 24/7 operation.")
                return True
        except Exception as e:
            log_warning(f"[CHITTI][BRAIN] Brain preload notice: {e}")
        return False

    def check_connection(self) -> bool:
        """Pings the Ollama server to check reachability."""
        try:
            resp = requests.get(self.base_url, timeout=3)
            return resp.status_code == 200
        except Exception:
            return False

    def ensure_running(self, wait_timeout: float = 8.0) -> bool:
        """
        Verifies if Ollama is running. If not and base_url is localhost,
        attempts to auto-start `ollama serve` in the background and waits for reachability.
        """
        if self.check_connection():
            return True

        is_local = "localhost" in self.base_url or "127.0.0.1" in self.base_url
        if not is_local:
            return False

        ollama_bin = shutil.which("ollama")
        if not ollama_bin:
            candidates = [
                os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe"),
                r"C:\Program Files\Ollama\ollama.exe",
                r"C:\Program Files (x86)\Ollama\ollama.exe",
            ]
            for cand in candidates:
                if os.path.isfile(cand):
                    ollama_bin = cand
                    break

        if not ollama_bin:
            log_warning("Ollama executable was not found on PATH or standard install directories.")
            return False

        log_chitti("Ollama server is not running. Launching local Ollama service in background...")

        creationflags = 0
        if sys.platform == "win32":
            creationflags = subprocess.CREATE_NO_WINDOW

        try:
            proc = subprocess.Popen(
                [ollama_bin, "serve"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creationflags,
            )
            OllamaLLM._auto_start_proc = proc
        except Exception as e:
            log_error(f"Failed to auto-start Ollama service: {e}")
            return False

        start_t = time.time()
        while time.time() - start_t < wait_timeout:
            time.sleep(0.5)
            if self.check_connection():
                log_info("[CHITTI][BRAIN] Ollama background service connected successfully.")
                self.preload_brain()
                return True

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

    def _bound_messages(self, messages: List[Dict[str, str]], max_history: int = 12) -> List[Dict[str, str]]:
        """Ensures messages do not exceed max history while preserving the system prompt."""
        if not messages or len(messages) <= max_history:
            return messages

        system_msgs = [m for m in messages if m.get("role") == "system"]
        other_msgs = [m for m in messages if m.get("role") != "system"]
        trimmed_others = other_msgs[-max_history:]
        return system_msgs + trimmed_others

    def generate_response(self, messages: List[Dict[str, str]], request_id: Optional[int] = None) -> str:
        """
        Sends the messages to Ollama /api/chat endpoint and returns the generated text.
        Keeps model resident in memory using keep_alive.
        """
        from src.utils.metrics import get_metrics_collector
        metrics = get_metrics_collector()
        metrics.mark_stage(request_id, "llm_started")

        bounded = self._bound_messages(messages)
        payload = {
            "model": self.model,
            "messages": bounded,
            "stream": False,
            "keep_alive": self.resolved_keep_alive,
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
            if self.ensure_running(wait_timeout=6.0):
                try:
                    response = requests.post(
                        self.chat_endpoint,
                        json=payload,
                        timeout=self.timeout
                    )
                except requests.exceptions.ConnectionError as e2:
                    raise LLMConnectionError(
                        f"Cannot connect to Ollama at {self.base_url}. Is Ollama running?"
                    ) from e2
            else:
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
            metrics.mark_stage(request_id, "llm_first_token")
            metrics.mark_stage(request_id, "llm_completed")
            prof = metrics.get_profile(request_id)
            if prof:
                prof.token_count = data.get("eval_count", len(content.split()))
            if not content:
                return "I processed your request, but received an empty response."
            return content
        except Exception as e:
            raise LLMError(f"Failed to parse Ollama response: {e}") from e

    def generate_response_stream(
        self, messages: List[Dict[str, str]], request_id: Optional[int] = None
    ):
        """
        Streams response tokens incrementally from Ollama /api/chat.
        Yields chunk strings as they arrive.
        """
        from src.utils.metrics import get_metrics_collector
        from src.core.state_manager import get_state_manager

        metrics = get_metrics_collector()
        state_mgr = get_state_manager()
        metrics.mark_stage(request_id, "llm_started")

        bounded = self._bound_messages(messages)
        payload = {
            "model": self.model,
            "messages": bounded,
            "stream": True,
            "keep_alive": self.resolved_keep_alive,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
            }
        }

        try:
            response = requests.post(
                self.chat_endpoint,
                json=payload,
                stream=True,
                timeout=self.timeout
            )
        except requests.exceptions.ConnectionError as e:
            if self.ensure_running(wait_timeout=6.0):
                try:
                    response = requests.post(
                        self.chat_endpoint,
                        json=payload,
                        stream=True,
                        timeout=self.timeout
                    )
                except requests.exceptions.ConnectionError as e2:
                    raise LLMConnectionError(
                        f"Cannot connect to Ollama at {self.base_url}. Is Ollama running?"
                    ) from e2
            else:
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

        first_token_marked = False
        token_count = 0
        try:
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                # Invalidation check: if cancelled or stopped by user, stop stream
                if request_id is not None and not state_mgr.is_generation_valid(request_id):
                    break

                try:
                    chunk_data = json.loads(line)
                except Exception:
                    continue

                delta = chunk_data.get("message", {}).get("content", "")
                if delta:
                    if not first_token_marked:
                        first_token_marked = True
                        metrics.mark_stage(request_id, "llm_first_token")
                    token_count += 1
                    yield delta

                if chunk_data.get("done", False):
                    break
        finally:
            metrics.mark_stage(request_id, "llm_completed")
            prof = metrics.get_profile(request_id)
            if prof:
                prof.token_count = token_count
            response.close()


def get_llm(config: Optional[LLMConfig] = None) -> BaseLLM:
    """Factory function returning the configured LLM engine."""
    return OllamaLLM(config)

