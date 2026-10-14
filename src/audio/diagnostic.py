"""
Chitti Audio Diagnostics & Signal Telemetry Module (Phase 4 & Phase 9).
Provides opt-in diagnostic recording, non-spammy aggregated acoustic logging,
utterance accept/reject reasons, and temporary test audio preservation.
"""

from dataclasses import dataclass
import os
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional
import numpy as np

from src.utils.logging import log_debug, log_info, log_warn


DIAGNOSTICS_DIR = Path("diagnostics/audio")
MAX_DIAGNOSTIC_RECORDINGS = 5


def is_diagnostic_mode_enabled() -> bool:
    """Returns True if audio diagnostic mode is enabled via environment or config."""
    return os.getenv("CHITTI_AUDIO_DIAGNOSTICS", "false").lower() in ("true", "1", "yes")


def save_diagnostic_recording(
    audio: np.ndarray,
    label: str = "utterance",
    sample_rate: int = 16000,
) -> Optional[Path]:
    """
    Saves audio to a temporary diagnostic WAV file only if diagnostic mode is active.
    Maintains a rolling window of at most MAX_DIAGNOSTIC_RECORDINGS files and auto-cleans older ones.
    """
    if not is_diagnostic_mode_enabled() or audio is None or len(audio) == 0:
        return None

    try:
        DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)

        # Rotate/clean old files if count exceeds limit
        existing = sorted(DIAGNOSTICS_DIR.glob("*.wav"), key=lambda p: p.stat().st_mtime)
        while len(existing) >= MAX_DIAGNOSTIC_RECORDINGS:
            oldest = existing.pop(0)
            try:
                oldest.unlink(missing_ok=True)
            except Exception:
                pass

        timestamp = int(time.time() * 1000)
        file_path = DIAGNOSTICS_DIR / f"{label}_{timestamp}.wav"

        # Write 16-bit PCM WAV
        import wave
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        pcm_data = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()

        with wave.open(str(file_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_data)

        log_debug(f"[DIAGNOSTICS] Saved test recording ({len(audio)/sample_rate:.2f}s) to: {file_path}")
        return file_path
    except Exception as e:
        log_debug(f"[DIAGNOSTICS] Failed to save diagnostic audio: {e}")
        return None


def clear_diagnostic_recordings():
    """Explicitly deletes all temporary diagnostic test recordings."""
    if not DIAGNOSTICS_DIR.exists():
        return
    for wav_file in DIAGNOSTICS_DIR.glob("*.wav"):
        try:
            wav_file.unlink(missing_ok=True)
        except Exception:
            pass
    log_info("[DIAGNOSTICS] Diagnostic audio recordings cleared.")


@dataclass
class UtteranceDiagnosticLog:
    """Detailed summary of why a speech candidate was accepted or rejected."""
    timestamp: float
    duration_sec: float
    rms: float
    peak: float
    noise_floor: float
    snr_ratio: float
    is_voiced: bool
    voicing_score: float
    speech_band_ratio: float
    accepted: bool
    rejection_reason: str
    transcript: str = ""
    stt_duration_sec: float = 0.0


class AudioDiagnosticLogger:
    """
    Periodic aggregator to log acoustic health without printing a line for every silent frame.
    """

    def __init__(self, summary_interval_sec: float = 5.0):
        self.summary_interval_sec = summary_interval_sec
        self._last_summary_time = time.time()
        self._silent_frames = 0
        self._speech_frames = 0
        self._peak_rms = 0.0
        self._lock = threading.Lock()

    def record_frame(self, rms: float, is_speech: bool):
        """Records frame metric and logs periodic aggregate summary."""
        with self._lock:
            if is_speech:
                self._speech_frames += 1
            else:
                self._silent_frames += 1
            self._peak_rms = max(self._peak_rms, rms)

            now = time.time()
            if (now - self._last_summary_time) >= self.summary_interval_sec:
                if is_diagnostic_mode_enabled():
                    log_debug(
                        f"[AUDIO HEALTH] Frames: speech={self._speech_frames}, silence={self._silent_frames}, "
                        f"peak_rms={self._peak_rms:.4f}"
                    )
                self._silent_frames = 0
                self._speech_frames = 0
                self._peak_rms = 0.0
                self._last_summary_time = now

    def log_utterance(self, diag: UtteranceDiagnosticLog):
        """Logs an accepted or rejected speech candidate with detailed rationale."""
        if not is_diagnostic_mode_enabled():
            return

        status = "ACCEPTED" if diag.accepted else f"REJECTED ({diag.rejection_reason})"
        log_info(
            f"[AUDIO DIAGNOSTIC] Utterance {status}: dur={diag.duration_sec:.2f}s, "
            f"rms={diag.rms:.4f}, snr={diag.snr_ratio:.1f}x, voiced={diag.is_voiced} ({diag.voicing_score:.2f}), "
            f"speech_band={diag.speech_band_ratio:.2f}, text='{diag.transcript}'"
        )
