"""
Chitti Microphone & Audio Capture Diagnostic Utility (Phase 2).
Inspects Windows audio devices, tests stream initialization, measures ambient energy,
detects overflows/underflows, and validates that microphone samples contain changing data.
"""

import sys
import time
from pathlib import Path
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import sounddevice as sd
except ImportError:
    print("[ERROR] sounddevice is not installed. Run `pip install sounddevice`.")
    sys.exit(1)

from src.audio.audio_utils import get_best_input_device
from src.config import get_config


def diagnose():
    print("=" * 70)
    print("           CHITTI MICROPHONE & AUDIO HARDWARE DIAGNOSTIC")
    print("=" * 70)

    # 1. Query all audio devices
    devices = sd.query_devices()
    default_input_idx, default_output_idx = sd.default.device
    print(f"\n[SYSTEM AUDIO SETTINGS]")
    print(f"  Windows Default Input Device Index : {default_input_idx}")
    print(f"  Windows Default Output Device Index: {default_output_idx}")

    print(f"\n[AVAILABLE INPUT DEVICES]")
    input_devices = []
    for idx, d in enumerate(devices):
        if d.get("max_input_channels", 0) > 0:
            is_default = (idx == default_input_idx)
            mark = " [DEFAULT]" if is_default else ""
            print(f"  ID {idx:2d}: {d.get('name', 'Unknown')}{mark}")
            print(f"         Channels: {d.get('max_input_channels')}, Native Rate: {d.get('default_samplerate')} Hz, Host API: {sd.query_hostapis(d.get('hostapi', 0)).get('name', 'N/A')}")
            input_devices.append((idx, d))

    if not input_devices:
        print("\n[CRITICAL ERROR] No audio input devices detected by PortAudio.")
        return False

    # 2. Test get_best_input_device resolution
    cfg = get_config().audio
    preferred_idx = cfg.microphone_index
    print(f"\n[DEVICE SELECTION LOGIC]")
    print(f"  Configured preferred index: {preferred_idx}")
    selected_idx, selected_sr = get_best_input_device(preferred_idx, force_refresh=True)
    selected_dev = devices[selected_idx] if selected_idx is not None and selected_idx < len(devices) else None
    dev_name = selected_dev.get("name", "Unknown") if selected_dev else "None"
    print(f"  Auto-selected device ID  : {selected_idx} ({dev_name})")
    print(f"  Selected sample rate     : {selected_sr} Hz")

    # 3. Test Supported Sample Rates on Selected Device
    print(f"\n[SUPPORTED SAMPLE RATES FOR DEVICE #{selected_idx}]")
    test_rates = [8000, 16000, 22050, 32000, 44100, 48000, 96000]
    supported_rates = []
    for sr in test_rates:
        try:
            sd.check_input_settings(device=selected_idx, samplerate=sr, channels=1, dtype="float32")
            print(f"  {sr:5d} Hz : SUPPORTED (float32, 1 channel)")
            supported_rates.append(sr)
        except Exception as e:
            print(f"  {sr:5d} Hz : UNSUPPORTED ({e})")

    # 4. Stream Initialization Test & Live Audio Capture
    print(f"\n[LIVE MICROPHONE CAPTURE TEST (2.0 seconds)]")
    chunk_duration = 0.05  # 50ms chunks
    blocksize = int(selected_sr * chunk_duration)
    captured_chunks = []
    overflow_count = 0
    underflow_count = 0

    def stream_callback(indata, frames, time_info, status):
        nonlocal overflow_count, underflow_count
        if status.input_overflow:
            overflow_count += 1
        if status.input_underflow:
            underflow_count += 1
        captured_chunks.append(indata[:, 0].copy())

    print(f"  Initializing sd.InputStream:")
    print(f"    - Device      : {selected_idx} ({dev_name})")
    print(f"    - Sample Rate : {selected_sr} Hz")
    print(f"    - Channels    : 1 (mono)")
    print(f"    - Dtype       : float32")
    print(f"    - Block Size  : {blocksize} samples ({chunk_duration * 1000:.1f}ms per frame)")

    try:
        with sd.InputStream(
            device=selected_idx,
            samplerate=selected_sr,
            channels=1,
            dtype="float32",
            blocksize=blocksize,
            callback=stream_callback,
        ):
            print("  [RECORDING] Streaming audio for 2 seconds... Please remain quiet or speak a word.")
            time.sleep(2.0)
    except Exception as e:
        print(f"  [ERROR] Failed to open audio stream: {e}")
        return False

    # 5. Analyze Captured Data
    if not captured_chunks:
        print("  [ERROR] No audio frames were received in callback! Stream is silent or hung.")
        return False

    all_samples = np.concatenate(captured_chunks)
    total_samples = len(all_samples)
    total_seconds = total_samples / float(selected_sr)
    rms = float(np.sqrt(np.mean(all_samples**2) + 1e-12))
    peak = float(np.max(np.abs(all_samples)))
    std = float(np.std(all_samples))
    min_val = float(np.min(all_samples))
    max_val = float(np.max(all_samples))
    zeros_fraction = float(np.mean(all_samples == 0.0))

    print(f"\n[DATA INTEGRITY & SIGNAL METRICS]")
    print(f"  Captured Duration   : {total_seconds:.2f} s ({len(captured_chunks)} frames, {total_samples} samples)")
    print(f"  Overflow Events     : {overflow_count}")
    print(f"  Underflow Events    : {underflow_count}")
    print(f"  Signal RMS Energy   : {rms:.6f}")
    print(f"  Peak Amplitude      : {peak:.6f}")
    print(f"  Min / Max Values    : {min_val:.6f} / {max_val:.6f}")
    print(f"  Standard Deviation  : {std:.6f}")
    print(f"  Zero-Sample Ratio   : {zeros_fraction * 100:.1f}%")

    # Diagnostic Assessments
    is_flatline = (std < 1e-6 or peak < 1e-5)
    is_clipped = (peak >= 0.99)
    is_valid_data = (not is_flatline and not is_clipped and np.all(np.isfinite(all_samples)))

    print(f"\n[DIAGNOSTIC STATUS]")
    if is_flatline:
        print("  STATUS: [FAIL] Audio stream received FLATLINE / ZERO data. Microphone may be muted or hardware access blocked.")
    elif is_clipped:
        print("  STATUS: [WARN] Audio signal is CLIPPED. Microphone gain is set too high in Windows.")
    elif is_valid_data:
        print(f"  STATUS: [PASS] Stream active and receiving valid, changing audio data.")
        if rms < 0.005:
            print(f"  AMBIENT NOISE: Clean silence/quiet room (RMS={rms:.5f}).")
        elif rms < 0.020:
            print(f"  AMBIENT NOISE: Moderate room sound / fan noise (RMS={rms:.5f}).")
        else:
            print(f"  AMBIENT NOISE: Loud ambient noise or active speech present (RMS={rms:.5f}).")

    print("=" * 70)
    return is_valid_data


if __name__ == "__main__":
    success = diagnose()
    sys.exit(0 if success else 1)
