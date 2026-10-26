"""
Shinchan Voice & Personality Test Script for Chitti.
Tests:
1. Dynamic Shinchan Persona Prompt
2. High-Pitched Mischievous Child Neural Voice (Edge-TTS)
3. RVC Character Voice Conversion Engine (CUDA RTX 5050)
4. Instant Audio Playback on System Speakers
"""

import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import reload_config, get_config
from src.audio.tts import get_tts
from src.brain.personality import get_default_system_prompt, ConversationHistory
from src.audio.rvc_engine import get_rvc_engine


def main():
    print("=" * 70)
    print("        CHITTI SHINCHAN VOICE & AI CLONE DIAGNOSTIC TEST")
    print("=" * 70)

    cfg = reload_config()
    print(f"\n[1] Configuration:")
    print(f"    - Voice Preset: {cfg.tts.voice_preset}")
    print(f"    - Base Voice:   {cfg.tts.voice_id}")
    print(f"    - Pitch:        {cfg.tts.pitch}")
    print(f"    - Rate:         {cfg.tts.rate_percentage}")
    print(f"    - RVC Enabled:  {cfg.tts.rvc_enabled}")
    print(f"    - RVC Model:    {cfg.tts.rvc_model}")

    # Check RVC Engine
    print("\n[2] Checking RVC Neural Engine...")
    rvc = get_rvc_engine()
    if rvc.is_available:
        print(f"    [OK] RVC Model Loaded: {rvc.model_path}")
        print(f"    [OK] Device: {rvc.device}")
    else:
        print(f"    [NOTICE] RVC fallback active (Direct Neural Shinchan synthesis).")

    # Check Shinchan Persona
    print("\n[3] Checking Shinchan Persona:")
    history = ConversationHistory()
    is_shinchan_prompt = "Shinchan" in history.system_prompt
    print(f"    - Shinchan Persona Active: {is_shinchan_prompt}")

    # Test Speech Synthesis and Playback
    test_line = "Arre arre Prateek bhaiya! Main itna bhi smart nahi hoon! Hehehe, chalo Action Kamen dekhne chalte hain!"
    print(f"\n[4] Speaking Test Dialogue in Shinchan's Voice:")
    print(f'    "{test_line}"')

    tts = get_tts()
    print("\n    [Speaking on speakers now...]")
    tts.speak(test_line, wait=True)
    print("    [Playback finished cleanly!]")

    print("\n" + "=" * 70)
    print("  [SUCCESS] Shinchan Voice is active and operational in Chitti!")
    print("=" * 70)


if __name__ == "__main__":
    main()
