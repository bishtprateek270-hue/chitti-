"""
Chitti Audio Utilities Module.
Provides hardware auto-discovery, persistent device caching, and fast linear resampling
to ensure 100% compatibility between Windows MME/WASAPI/DirectSound microphones and STT engines.
"""

from typing import Optional, Tuple
import numpy as np

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    sd = None
    HAS_SOUNDDEVICE = False

import math

try:
    from scipy.signal import resample_poly
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

_CACHED_INPUT_DEVICE: Optional[int] = None
_CACHED_SAMPLE_RATE: int = 16000


def resample_to_16k(audio: np.ndarray, orig_sr: int) -> np.ndarray:
    """
    Resamples 1D float32 audio array from orig_sr to 16000Hz.
    Uses high-quality polyphase FIR filtering (scipy.signal.resample_poly) with
    anti-aliasing low-pass filter to eliminate distortion, falling back to np.interp.
    """
    if audio is None or len(audio) == 0:
        return np.array([], dtype=np.float32)

    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    if orig_sr == 16000:
        return audio

    target_sr = 16000
    if HAS_SCIPY:
        try:
            g = math.gcd(int(target_sr), int(orig_sr))
            up = int(target_sr // g)
            down = int(orig_sr // g)
            resampled = resample_poly(audio, up, down).astype(np.float32)
            return resampled
        except Exception:
            pass

    # Fallback to linear interpolation if scipy is unavailable or fails
    num_target_samples = int(len(audio) * float(target_sr) / float(orig_sr))
    if num_target_samples <= 0:
        return np.array([], dtype=np.float32)

    orig_indices = np.linspace(0, len(audio) - 1, num=len(audio))
    target_indices = np.linspace(0, len(audio) - 1, num=num_target_samples)
    resampled = np.interp(target_indices, orig_indices, audio).astype(np.float32)
    return resampled


def get_best_input_device(preferred_index: Optional[int] = None, force_refresh: bool = False) -> Tuple[Optional[int], int]:
    """
    Finds the best working microphone device index and its native sample rate.
    Prioritizes the Windows system default input device (e.g. connected earbuds or headset).
    """
    global _CACHED_INPUT_DEVICE, _CACHED_SAMPLE_RATE

    if preferred_index is not None:
        if HAS_SOUNDDEVICE and sd:
            try:
                dev = sd.query_devices(preferred_index)
                return preferred_index, int(dev.get("default_samplerate", 16000))
            except Exception:
                pass
        return preferred_index, 16000

    if force_refresh:
        _CACHED_INPUT_DEVICE = None
        _CACHED_SAMPLE_RATE = 16000

    if not force_refresh and _CACHED_INPUT_DEVICE is not None:
        return _CACHED_INPUT_DEVICE, _CACHED_SAMPLE_RATE

    if not HAS_SOUNDDEVICE or not sd:
        return None, 16000

    try:
        # 1. Prioritize the Windows system default input device (e.g. built-in mic array or connected headset)
        indices_to_try = []
        try:
            default_idx = int(sd.default.device[0])
            if default_idx >= 0:
                indices_to_try.append(default_idx)
        except Exception:
            pass

        # 2. Check for DirectSound primary as fallback candidate
        devices = sd.query_devices()
        directsound_primary = None
        for idx, d in enumerate(devices):
            if d.get("max_input_channels", 0) > 0:
                name = d.get("name", "").lower()
                hostapi_idx = d.get("hostapi", -1)
                try:
                    api_name = sd.query_hostapis(hostapi_idx).get("name", "").lower()
                except Exception:
                    api_name = ""
                if "directsound" in api_name and "primary sound capture" in name:
                    directsound_primary = idx
                    break

        if directsound_primary is not None and directsound_primary not in indices_to_try:
            indices_to_try.append(directsound_primary)

        for dev_idx in indices_to_try:
            try:
                d = sd.query_devices(dev_idx)
                if d.get("max_input_channels", 0) > 0:
                    native_sr = int(d.get("default_samplerate", 44100))
                    # Test native sample rate first for rock-solid PortAudio stability
                    rates_to_test = [native_sr] if native_sr == 16000 else [native_sr, 16000, 44100, 48000]
                    for rate in rates_to_test:
                        try:
                            test_samples = int(rate * 0.04)
                            data = sd.rec(test_samples, samplerate=rate, channels=1, dtype="float32", device=dev_idx)
                            sd.wait()
                            rms = float(np.sqrt(np.mean(data**2)))
                            if np.isfinite(rms):
                                _CACHED_INPUT_DEVICE = dev_idx
                                _CACHED_SAMPLE_RATE = rate
                                return dev_idx, rate
                        except Exception:
                            continue
            except Exception:
                continue

        # 2. Search all physical input devices (skipping virtual mappers & stereo mix)
        devices = sd.query_devices()
        candidates = []
        for idx, d in enumerate(devices):
            if d.get("max_input_channels", 0) > 0:
                name = d.get("name", "").lower()
                # Skip virtual speakers, sound mapper, and stereo mix unless no alternative
                if any(k in name for k in ("speaker", "stereo mix", "output", "mapper")):
                    continue
                native_sr = int(d.get("default_samplerate", 44100))
                candidates.append((idx, native_sr, d.get("name", "")))

        for dev_idx, native_sr, d_name in candidates:
            rates_to_test = [16000] if 16000 == native_sr else [16000, native_sr, 44100, 48000]
            for rate in rates_to_test:
                try:
                    test_samples = int(rate * 0.04)
                    data = sd.rec(test_samples, samplerate=rate, channels=1, dtype="float32", device=dev_idx)
                    sd.wait()
                    rms = float(np.sqrt(np.mean(data**2)))
                    if np.isfinite(rms):
                        _CACHED_INPUT_DEVICE = dev_idx
                        _CACHED_SAMPLE_RATE = rate
                        return dev_idx, rate
                except Exception:
                    continue

        # 3. Fallback: try device index 0
        _CACHED_INPUT_DEVICE = 0
        _CACHED_SAMPLE_RATE = 16000
        return 0, 16000
    except Exception:
        _CACHED_INPUT_DEVICE = None
        _CACHED_SAMPLE_RATE = 16000
        return None, 16000
