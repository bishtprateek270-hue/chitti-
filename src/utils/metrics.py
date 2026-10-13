"""
Chitti Latency Instrumentation & Observability Metrics Module.
Measures high-precision monotonic timestamps across the entire request lifecycle:
audio capture -> STT -> intent resolution -> LLM (TTFT & total) -> tool execution -> TTS -> first audible response.
Calculates percentiles and provides structured [PERF] logging.
"""

import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
from src.utils.logging import log_chitti, log_debug


class ExecutionPath(str, Enum):
    PATH_A_SIMPLE = "PATH_A_SIMPLE"              # Simple deterministic command (Volume, Stop, App)
    PATH_B_CONVERSATION = "PATH_B_CONVERSATION"  # Normal conversational query
    PATH_C_COMPUTER = "PATH_C_COMPUTER"          # Computer-use actions (Open project, send message)
    PATH_D_CODING = "PATH_D_CODING"              # Coding and project execution
    PATH_E_COMPLEX = "PATH_E_COMPLEX"            # Multi-step planner & verification


@dataclass
class RequestLatencyProfile:
    """Detailed timing record for a single user request."""
    request_id: int
    execution_path: ExecutionPath = ExecutionPath.PATH_B_CONVERSATION
    created_at_mono: float = field(default_factory=time.perf_counter)
    
    # Monotonic timestamps (None if not reached)
    audio_capture_started: Optional[float] = None
    speech_started: Optional[float] = None
    speech_ended: Optional[float] = None
    transcription_started: Optional[float] = None
    transcription_completed: Optional[float] = None
    intent_resolved: Optional[float] = None
    llm_started: Optional[float] = None
    llm_first_token: Optional[float] = None
    llm_completed: Optional[float] = None
    tool_execution_started: Optional[float] = None
    tool_execution_completed: Optional[float] = None
    tts_started: Optional[float] = None
    first_audio_played: Optional[float] = None
    task_completed: Optional[float] = None

    # Metadata
    token_count: int = 0
    first_sentence_len: int = 0
    tool_name: Optional[str] = None
    success: bool = True
    cancelled: bool = False

    def mark(self, stage: str, timestamp: Optional[float] = None) -> float:
        """Records a timestamp for a given stage and logs structured [PERF] tag."""
        ts = timestamp if timestamp is not None else time.perf_counter()
        setattr(self, stage, ts)
        
        delta_from_start = ts - self.created_at_mono
        log_chitti(f"[PERF] {stage} (req_id={self.request_id}, t=+{delta_from_start:.3f}s)")
        return ts

    @property
    def audio_capture_duration(self) -> Optional[float]:
        if self.speech_started and self.speech_ended:
            return max(0.0, self.speech_ended - self.speech_started)
        return None

    @property
    def stt_latency(self) -> Optional[float]:
        if self.transcription_started and self.transcription_completed:
            return max(0.0, self.transcription_completed - self.transcription_started)
        return None

    @property
    def intent_resolution_latency(self) -> Optional[float]:
        if self.transcription_completed and self.intent_resolved:
            return max(0.0, self.intent_resolved - self.transcription_completed)
        return None

    @property
    def llm_time_to_first_token(self) -> Optional[float]:
        if self.llm_started and self.llm_first_token:
            return max(0.0, self.llm_first_token - self.llm_started)
        return None

    @property
    def llm_total_latency(self) -> Optional[float]:
        if self.llm_started and self.llm_completed:
            return max(0.0, self.llm_completed - self.llm_started)
        return None

    @property
    def tool_execution_latency(self) -> Optional[float]:
        if self.tool_execution_started and self.tool_execution_completed:
            return max(0.0, self.tool_execution_completed - self.tool_execution_started)
        return None

    @property
    def tts_startup_latency(self) -> Optional[float]:
        if self.tts_started and self.first_audio_played:
            return max(0.0, self.first_audio_played - self.tts_started)
        return None

    @property
    def time_to_first_audible_response(self) -> Optional[float]:
        """Total time from speech ended until the user hears the first spoken word."""
        start = self.speech_ended or self.created_at_mono
        if self.first_audio_played:
            return max(0.0, self.first_audio_played - start)
        return None

    @property
    def total_task_duration(self) -> Optional[float]:
        if self.task_completed:
            return max(0.0, self.task_completed - self.created_at_mono)
        return None

    def summarize(self) -> Dict[str, Any]:
        """Returns structured dictionary of key metrics."""
        return {
            "request_id": self.request_id,
            "path": self.execution_path.value,
            "audio_capture_duration_s": round(self.audio_capture_duration or 0.0, 3),
            "stt_latency_s": round(self.stt_latency or 0.0, 3),
            "intent_latency_s": round(self.intent_resolution_latency or 0.0, 3),
            "llm_ttft_s": round(self.llm_time_to_first_token or 0.0, 3),
            "llm_total_s": round(self.llm_total_latency or 0.0, 3),
            "tool_latency_s": round(self.tool_execution_latency or 0.0, 3),
            "tts_startup_s": round(self.tts_startup_latency or 0.0, 3),
            "time_to_first_response_s": round(self.time_to_first_audible_response or 0.0, 3),
            "total_duration_s": round(self.total_task_duration or 0.0, 3),
            "tokens": self.token_count,
            "success": self.success,
            "cancelled": self.cancelled,
        }


class MetricsCollector:
    """Thread-safe collector for request latency profiles with percentile aggregation."""

    def __init__(self):
        self._profiles: Dict[int, RequestLatencyProfile] = {}
        self._history: List[RequestLatencyProfile] = []
        self._lock = threading.Lock()

    def start_request(self, request_id: int, path: ExecutionPath = ExecutionPath.PATH_B_CONVERSATION) -> RequestLatencyProfile:
        with self._lock:
            prof = RequestLatencyProfile(request_id=request_id, execution_path=path)
            self._profiles[request_id] = prof
            return prof

    def get_profile(self, request_id: Optional[int]) -> Optional[RequestLatencyProfile]:
        if request_id is None:
            return None
        with self._lock:
            return self._profiles.get(request_id)

    def mark_stage(self, request_id: Optional[int], stage: str) -> Optional[float]:
        if request_id is None:
            return None
        with self._lock:
            prof = self._profiles.get(request_id)
            if prof:
                return prof.mark(stage)
        return None

    def finish_request(self, request_id: int, success: bool = True, cancelled: bool = False) -> Optional[RequestLatencyProfile]:
        with self._lock:
            prof = self._profiles.pop(request_id, None)
            if prof:
                prof.task_completed = time.perf_counter()
                prof.success = success
                prof.cancelled = cancelled
                prof.mark("task_completed", prof.task_completed)
                self._history.append(prof)
                # Keep history bounded to last 500 requests
                if len(self._history) > 500:
                    self._history.pop(0)
                return prof
        return None

    def get_percentiles(self, path: Optional[ExecutionPath] = None) -> Dict[str, Any]:
        """Calculates p50, p90, p95 for key metrics, optionally filtered by path."""
        with self._lock:
            records = [
                p for p in self._history
                if (path is None or p.execution_path == path) and not p.cancelled
            ]

        if not records:
            return {"count": 0, "p50": {}, "p90": {}, "p95": {}}

        def _calc_percentile(values: List[float], pct: float) -> float:
            if not values:
                return 0.0
            s = sorted(values)
            k = (len(s) - 1) * (pct / 100.0)
            f = int(k)
            c = min(f + 1, len(s) - 1)
            d0 = s[f] * (c - k)
            d1 = s[c] * (k - f)
            return round(d0 + d1, 3)

        ttft_vals = [p.llm_time_to_first_token for p in records if p.llm_time_to_first_token is not None]
        tt_first_resp_vals = [p.time_to_first_audible_response for p in records if p.time_to_first_audible_response is not None]
        total_vals = [p.total_task_duration for p in records if p.total_task_duration is not None]
        stt_vals = [p.stt_latency for p in records if p.stt_latency is not None]

        return {
            "count": len(records),
            "p50": {
                "ttft_s": _calc_percentile(ttft_vals, 50),
                "time_to_first_response_s": _calc_percentile(tt_first_resp_vals, 50),
                "stt_latency_s": _calc_percentile(stt_vals, 50),
                "total_duration_s": _calc_percentile(total_vals, 50),
            },
            "p90": {
                "ttft_s": _calc_percentile(ttft_vals, 90),
                "time_to_first_response_s": _calc_percentile(tt_first_resp_vals, 90),
                "stt_latency_s": _calc_percentile(stt_vals, 90),
                "total_duration_s": _calc_percentile(total_vals, 90),
            },
            "p95": {
                "ttft_s": _calc_percentile(ttft_vals, 95),
                "time_to_first_response_s": _calc_percentile(tt_first_resp_vals, 95),
                "stt_latency_s": _calc_percentile(stt_vals, 95),
                "total_duration_s": _calc_percentile(total_vals, 95),
            },
        }


# Global singleton instance
_METRICS_COLLECTOR = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Returns the singleton MetricsCollector instance."""
    return _METRICS_COLLECTOR
