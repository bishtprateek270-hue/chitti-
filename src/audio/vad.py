"""
Chitti Voice Activity Detection (VAD) & Acoustic Signal Analysis Engine (Phase 3).
Provides multi-feature speech vs noise discrimination combining adaptive noise tracking,
harmonic pitch autocorrelation (voicing), zero-crossing analysis, spectral speech-band ratios,
and temporal hysteresis to eliminate false activations from fan noise, typing, hiss, and silence.
"""

from dataclasses import dataclass
import enum
from typing import Optional, Tuple
import numpy as np


class VADState(enum.Enum):
    """Temporal state of the Voice Activity Detector."""
    SILENCE = "SILENCE"
    SPEECH_ONSET = "SPEECH_ONSET"
    SPEECH = "SPEECH"
    SPEECH_HANGOVER = "SPEECH_HANGOVER"


@dataclass
class VADDecision:
    """Detailed diagnostic result for a single processed audio frame."""
    is_speech: bool
    state: VADState
    rms: float
    peak: float
    noise_floor: float
    snr_ratio: float
    is_voiced: bool
    voicing_score: float
    zcr: float
    speech_band_ratio: float
    is_stationary_noise: bool
    rejection_reason: str = ""


class AdaptiveNoiseEstimator:
    """
    Asymmetric rolling noise floor estimator.
    Adapts quickly downward during quiet periods, and extremely slowly upward to prevent
    continuous speech or long commands from raising the noise baseline.
    """

    def __init__(
        self,
        initial_floor: float = 0.010,
        min_floor: float = 0.0015,
        max_floor: float = 0.080,
        downward_alpha: float = 0.15,
        upward_alpha: float = 0.020,
    ):
        self.noise_floor = float(initial_floor)
        self.min_floor = float(min_floor)
        self.max_floor = float(max_floor)
        self.downward_alpha = float(downward_alpha)
        self.upward_alpha = float(upward_alpha)
        self.frame_count = 0

    def update(self, current_rms: float, is_speech_active: bool = False) -> float:
        """Updates noise estimate based on current frame RMS."""
        self.frame_count += 1
        current_rms = float(max(1e-6, current_rms))

        # Rapid initial calibration during first 25 frames (~1.25s) to learn true room baseline
        if self.frame_count <= 25 and not is_speech_active:
            cal_alpha = 0.35
            self.noise_floor = (1.0 - cal_alpha) * self.noise_floor + cal_alpha * current_rms
            self.noise_floor = float(np.clip(self.noise_floor, self.min_floor, self.max_floor))
            return self.noise_floor

        # While speech is actively detected, do NOT adapt upward
        if is_speech_active:
            if current_rms < self.noise_floor:
                self.noise_floor = (1.0 - self.downward_alpha) * self.noise_floor + self.downward_alpha * current_rms
            return self.noise_floor

        if current_rms < self.noise_floor:
            # Fast adaptation downward toward true quiet floor
            self.noise_floor = (1.0 - self.downward_alpha) * self.noise_floor + self.downward_alpha * current_rms
        else:
            # Smooth adaptation upward to handle changing room noise (e.g. AC/fan)
            self.noise_floor = (1.0 - self.upward_alpha) * self.noise_floor + self.upward_alpha * current_rms

        self.noise_floor = float(np.clip(self.noise_floor, self.min_floor, self.max_floor))
        return self.noise_floor

    def reset(self, initial_floor: float = 0.010):
        """Resets the noise baseline."""
        self.noise_floor = float(initial_floor)
        self.frame_count = 0


class VoiceActivityDetector:
    """
    Multi-feature, low-latency Voice Activity Detector (VAD).
    Combines:
    1. Adaptive SNR Gating (speech must be significantly louder than ambient room baseline)
    2. Voicing Detection via Pitch Autocorrelation (human vocal cord periodicity in 80-350Hz)
    3. Zero-Crossing Rate (ZCR) checking (rejects DC bias and pure low-frequency rumble)
    4. Speech-Band Energy Ratio (300-3400Hz telephony/speech formant concentration)
    5. Stationarity Detection (fan and machine noise is stationary; speech syllables modulate at 2-6Hz)
    6. Hysteresis Temporal Smoothing (distinct onset and hangover states)
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration: float = 0.05,  # 50ms chunks (800 samples @ 16kHz)
        snr_onset_ratio: float = 1.85,  # +5.3 dB above ambient noise floor
        snr_hangover_ratio: float = 1.30,  # +2.3 dB above ambient noise floor
        min_voice_rms: float = 0.005,  # Minimum absolute speech threshold
        onset_consecutive_frames: int = 2,  # 100ms of confirmed speech to trigger
        hangover_consecutive_frames: int = 14,  # 700ms of silence before ending speech
        min_voicing_score: float = 0.28,  # Minimum pitch autocorrelation peak
    ):
        self.sample_rate = sample_rate
        self.frame_duration = frame_duration
        self.chunk_size = int(sample_rate * frame_duration)

        self.snr_onset_ratio = snr_onset_ratio
        self.snr_hangover_ratio = snr_hangover_ratio
        self.min_voice_rms = min_voice_rms
        self.onset_consecutive_frames = onset_consecutive_frames
        self.hangover_consecutive_frames = hangover_consecutive_frames
        self.min_voicing_score = min_voicing_score

        self.noise_estimator = AdaptiveNoiseEstimator(initial_floor=0.010)
        self.state = VADState.SILENCE

        self._consecutive_speech_frames = 0
        self._consecutive_silence_frames = 0
        self._recent_rms_history: list = []
        self._history_len = 12  # ~600ms window for stationarity analysis

    def reset(self):
        """Resets the temporal state of the detector."""
        self.state = VADState.SILENCE
        self._consecutive_speech_frames = 0
        self._consecutive_silence_frames = 0
        self._recent_rms_history.clear()
        self.noise_estimator.reset()

    def process_frame(self, chunk: np.ndarray) -> VADDecision:
        """
        Analyzes a single 16kHz mono audio frame and returns a comprehensive VADDecision.
        """
        if chunk is None or len(chunk) == 0:
            return VADDecision(
                is_speech=False,
                state=self.state,
                rms=0.0,
                peak=0.0,
                noise_floor=self.noise_estimator.noise_floor,
                snr_ratio=0.0,
                is_voiced=False,
                voicing_score=0.0,
                zcr=0.0,
                speech_band_ratio=0.0,
                is_stationary_noise=False,
                rejection_reason="empty_frame",
            )

        # Ensure float32 1D
        if chunk.dtype != np.float32:
            chunk = chunk.astype(np.float32)
        if chunk.ndim > 1:
            chunk = chunk.flatten()

        # 1. Amplitude metrics
        peak = float(np.max(np.abs(chunk)))
        rms = float(np.sqrt(np.mean(chunk**2) + 1e-12))

        # Check for invalid signal (clipping or flatline)
        if peak >= 0.999:
            return self._build_decision(False, rms, peak, self.noise_estimator.noise_floor, 0.0, False, 0.0, 0.0, 0.0, False, "clipped_signal")
        if peak < 1e-5 or rms < 1e-6:
            self.noise_estimator.update(rms, is_speech_active=False)
            return self._build_decision(False, rms, peak, self.noise_estimator.noise_floor, 0.0, False, 0.0, 0.0, 0.0, False, "flatline_silence")

        # 2. Zero-Crossing Rate (ZCR)
        # Normal speech has moderate ZCR (~0.04 to ~0.35). High ZCR (>0.45) indicates white noise/hiss.
        signs = np.sign(chunk)
        signs[signs == 0] = 1
        zcr = float(np.mean(np.abs(np.diff(signs))) / 2.0)

        # 3. Voicing Analysis via Normalized Autocorrelation (human vocal pitch: 80Hz - 350Hz)
        voicing_score, is_voiced = self._compute_voicing(chunk)

        # 4. Spectral Speech-Band Energy Ratio (300Hz - 3400Hz vs Total Spectrum)
        speech_band_ratio = self._compute_speech_band_ratio(chunk)

        # 5. Stationarity Analysis (Fan/HVAC noise has steady RMS; speech has syllabic envelope variance)
        self._recent_rms_history.append(rms)
        if len(self._recent_rms_history) > self._history_len:
            self._recent_rms_history.pop(0)

        is_stationary_noise = False
        if len(self._recent_rms_history) >= self._history_len:
            history_arr = np.array(self._recent_rms_history, dtype=np.float32)
            coeff_of_variation = float(np.std(history_arr) / (np.mean(history_arr) + 1e-6))
            # Stationary fan noise has coefficient of variation < 0.12 even when moderately loud
            if coeff_of_variation < 0.10 and not is_voiced:
                is_stationary_noise = True

        # Update noise floor:
        # When human vocal cord periodicity is confirmed (is_voiced) or state is active speech/onset,
        # protect the noise floor from adapting upward into the speaker's voice!
        is_speech_like = (self.state in (VADState.SPEECH, VADState.SPEECH_ONSET, VADState.SPEECH_HANGOVER) or is_voiced)
        noise_floor = self.noise_estimator.update(rms, is_speech_active=is_speech_like)
        snr_ratio = float(rms / max(1e-6, noise_floor))

        # 6. Candidate Speech Evaluation
        # When pitch voicing is confirmed, allow lower onset SNR threshold since vocal chords are active
        req_onset_snr = (self.snr_onset_ratio * 0.88) if (is_voiced and voicing_score >= 0.40) else self.snr_onset_ratio
        has_voice_qualities = (is_voiced or speech_band_ratio >= 0.55 or (snr_ratio > 2.8 and zcr < 0.35))
        is_candidate_frame = (
            rms >= self.min_voice_rms
            and snr_ratio >= req_onset_snr
            and not is_stationary_noise
            and zcr <= 0.48
            and has_voice_qualities
        )

        rejection_reason = ""
        if not is_candidate_frame:
            if rms < self.min_voice_rms:
                rejection_reason = "below_min_voice_rms"
            elif snr_ratio < self.snr_onset_ratio:
                rejection_reason = "insufficient_snr"
            elif is_stationary_noise:
                rejection_reason = "stationary_fan_noise"
            elif zcr > 0.48:
                rejection_reason = "high_frequency_hiss"
            elif not has_voice_qualities:
                rejection_reason = "unvoiced_non_speech"

        # 7. Temporal State Machine with Hysteresis
        if self.state in (VADState.SILENCE, VADState.SPEECH_ONSET):
            if is_candidate_frame:
                added_frames = max(1, len(chunk) // self.chunk_size)
                self._consecutive_speech_frames += added_frames
                self._consecutive_silence_frames = 0
                if self._consecutive_speech_frames >= self.onset_consecutive_frames:
                    self.state = VADState.SPEECH
                else:
                    self.state = VADState.SPEECH_ONSET
            else:
                self._consecutive_speech_frames = 0
                self._consecutive_silence_frames += 1
                self.state = VADState.SILENCE

        elif self.state in (VADState.SPEECH, VADState.SPEECH_HANGOVER):
            # During ongoing speech, hangover condition allows brief quiet pauses but REQUIRES
            # vocal cord periodicity (voicing) or significant formant concentration so room hiss drops to silence.
            has_continuing_voice = (
                is_voiced
                or speech_band_ratio >= 0.50
                or (snr_ratio >= 2.2 and zcr < 0.38)
            )
            is_continuing_speech = (
                rms >= (self.min_voice_rms * 0.75)
                and snr_ratio >= self.snr_hangover_ratio
                and not is_stationary_noise
                and zcr <= 0.45
                and has_continuing_voice
            )

            if is_continuing_speech:
                self._consecutive_silence_frames = 0
                self.state = VADState.SPEECH
            else:
                self._consecutive_silence_frames += 1
                if self._consecutive_silence_frames >= self.hangover_consecutive_frames:
                    self.state = VADState.SILENCE
                    self._consecutive_speech_frames = 0
                else:
                    self.state = VADState.SPEECH_HANGOVER

        is_active_speech = (self.state in (VADState.SPEECH, VADState.SPEECH_HANGOVER))
        return self._build_decision(
            is_active_speech,
            rms,
            peak,
            noise_floor,
            snr_ratio,
            is_voiced,
            voicing_score,
            zcr,
            speech_band_ratio,
            is_stationary_noise,
            rejection_reason if not is_active_speech else "",
        )

    def _compute_voicing(self, chunk: np.ndarray) -> Tuple[float, bool]:
        """
        Computes pitch periodicity using normalized autocorrelation across human voice range (80-350Hz).
        """
        if len(chunk) < int(self.sample_rate * 0.025):
            return 0.0, False

        # Cap analysis window to at most 100ms (1600 samples @ 16kHz) for sub-millisecond execution
        max_samples = int(self.sample_rate * 0.10)
        if len(chunk) > max_samples:
            chunk = chunk[-max_samples:]

        min_lag = max(1, int(self.sample_rate / 350))  # ~45 samples @ 16kHz
        max_lag = min(len(chunk) - 1, int(self.sample_rate / 80))   # ~200 samples @ 16kHz
        if max_lag <= min_lag:
            return 0.0, False

        seg = chunk - np.mean(chunk)
        energy_sum = float(np.sum(seg**2))
        if energy_sum < 1e-7:
            return 0.0, False

        corr = np.correlate(seg, seg, mode="full")[len(seg) - 1 :] / energy_sum
        if len(corr) <= max_lag:
            return 0.0, False

        search_window = corr[min_lag:max_lag]
        max_score = float(np.max(search_window)) if len(search_window) > 0 else 0.0
        is_voiced = (max_score >= self.min_voicing_score)
        return max_score, is_voiced

    def _compute_speech_band_ratio(self, chunk: np.ndarray) -> float:
        """
        Calculates the ratio of energy in the 300Hz - 3400Hz speech frequency band.
        Uses fast FFT power spectrum.
        """
        if len(chunk) > 2048:
            chunk = chunk[-2048:]
        n = len(chunk)
        if n < 64:
            return 0.5

        # Compute FFT magnitude spectrum
        windowed = chunk * np.hanning(n)
        fft_vals = np.abs(np.fft.rfft(windowed))
        freqs = np.fft.rfftfreq(n, d=1.0 / self.sample_rate)

        total_power = float(np.sum(fft_vals**2) + 1e-12)
        # Speech band mask: 300Hz to 3400Hz
        band_mask = (freqs >= 300) & (freqs <= 3400)
        speech_power = float(np.sum(fft_vals[band_mask]**2))

        ratio = speech_power / total_power
        return float(np.clip(ratio, 0.0, 1.0))

    def _build_decision(
        self,
        is_speech: bool,
        rms: float,
        peak: float,
        noise_floor: float,
        snr_ratio: float,
        is_voiced: bool,
        voicing_score: float,
        zcr: float,
        speech_band_ratio: float,
        is_stationary_noise: bool,
        rejection_reason: str,
    ) -> VADDecision:
        return VADDecision(
            is_speech=is_speech,
            state=self.state,
            rms=rms,
            peak=peak,
            noise_floor=noise_floor,
            snr_ratio=snr_ratio,
            is_voiced=is_voiced,
            voicing_score=voicing_score,
            zcr=zcr,
            speech_band_ratio=speech_band_ratio,
            is_stationary_noise=is_stationary_noise,
            rejection_reason=rejection_reason,
        )
