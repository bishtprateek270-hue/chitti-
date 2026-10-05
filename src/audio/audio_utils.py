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
    Finds and caches the best working microphone device index safely.
    Uses the Windows system default input or standard WASAPI/DirectSound device.
    """
    global _CACHED_INPUT_DEVICE, _CACHED_SAMPLE_RATE

    if preferred_index is not None:
        return preferred_index, 16000

    if _CACHED_INPUT_DEVICE is not None:
        return _CACHED_INPUT_DEVICE, _CACHED_SAMPLE_RATE

    if not HAS_SOUNDDEVICE or not sd:
        return None, 16000

    try:
        # Check system default input device first
        def_idx = sd.default.device[0]
        if def_idx is not None and def_idx >= 0:
            dev = sd.query_devices(def_idx)
            if dev.get("max_input_channels", 0) > 0:
                sr = int(dev.get("default_samplerate", 16000))
                _CACHED_INPUT_DEVICE = def_idx
                _CACHED_SAMPLE_RATE = sr
                return def_idx, sr
    except Exception:
        pass

    try:
        # Fallback to finding first valid standard input device (excluding virtual speakers and WDM-KS)
        devices = sd.query_devices()
        for idx, d in enumerate(devices):
            if d.get("max_input_channels", 0) > 0:
                name = d.get("name", "").lower()
                hostapi = d.get("hostapi", 0)
                # Skip output/stereo mix and unstable WDM-KS exclusive devices (hostapi == 3)
                if "speaker" in name or "stereo mix" in name or "output" in name or hostapi == 3:
                    continue
                sr = int(d.get("default_samplerate", 16000))
                _CACHED_INPUT_DEVICE = idx
                _CACHED_SAMPLE_RATE = sr
                return idx, sr
    except Exception:
        pass

    _CACHED_INPUT_DEVICE = None
    _CACHED_SAMPLE_RATE = 16000
    return None, 16000
