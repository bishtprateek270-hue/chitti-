"""
Chitti Audio Utilities Module.
Provides hardware auto-discovery, persistent device caching, and fast linear resampling
to ensure 100% compatibility between Windows MME/WASAPI/WDM-KS microphones and STT engines.
"""

from typing import Optional, Tuple
import numpy as np

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    sd = None
    HAS_SOUNDDEVICE = False

_CACHED_INPUT_DEVICE: Optional[int] = None
_CACHED_SAMPLE_RATE: int = 16000


def resample_to_16k(audio: np.ndarray, orig_sr: int) -> np.ndarray:
    """
    Resamples 1D float32 audio array from orig_sr to 16000Hz using fast linear interpolation.
    """
    if audio is None or len(audio) == 0:
        return np.array([], dtype=np.float32)
    
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    if orig_sr == 16000:
        return audio

    target_sr = 16000
    num_target_samples = int(len(audio) * float(target_sr) / float(orig_sr))
    if num_target_samples <= 0:
        return np.array([], dtype=np.float32)

    orig_indices = np.linspace(0, len(audio) - 1, num=len(audio))
    target_indices = np.linspace(0, len(audio) - 1, num=num_target_samples)
    resampled = np.interp(target_indices, orig_indices, audio).astype(np.float32)
    return resampled


def get_best_input_device(preferred_index: Optional[int] = None) -> Tuple[Optional[int], int]:
    """
    Finds and caches the best working microphone device index by testing live RMS energy.
    Picks the real physical laptop microphone with highest signal clarity.
    """
    global _CACHED_INPUT_DEVICE, _CACHED_SAMPLE_RATE

    if preferred_index is not None:
        return preferred_index, 16000

    if _CACHED_INPUT_DEVICE is not None:
        return _CACHED_INPUT_DEVICE, _CACHED_SAMPLE_RATE

    if not HAS_SOUNDDEVICE or not sd:
        return None, 16000

    best_dev = None
    best_rate = 16000
    best_rms = -1.0

    try:
        devices = sd.query_devices()
        for idx, d in enumerate(devices):
            if d.get("max_input_channels", 0) > 0:
                name = d.get("name", "").lower()
                # Skip virtual speakers / stereo mix
                if "speaker" in name or "stereo mix" in name or "output" in name:
                    continue
                native_sr = int(d.get("default_samplerate", 16000))
                for rate in [native_sr, 16000, 44100, 48000]:
                    try:
                        test_samples = int(rate * 0.08)
                        data = sd.rec(test_samples, samplerate=rate, channels=1, dtype="float32", device=idx)
                        sd.wait()
                        rms = float(np.sqrt(np.mean(data**2)))
                        if np.isfinite(rms) and rms > best_rms:
                            best_rms = rms
                            best_dev = idx
                            best_rate = rate
                        break
                    except Exception:
                        continue

        if best_dev is not None:
            _CACHED_INPUT_DEVICE = best_dev
            _CACHED_SAMPLE_RATE = best_rate
            return best_dev, best_rate

        _CACHED_INPUT_DEVICE = None
        _CACHED_SAMPLE_RATE = 16000
        return None, 16000

    except Exception:
        _CACHED_INPUT_DEVICE = None
        _CACHED_SAMPLE_RATE = 16000
        return None, 16000
