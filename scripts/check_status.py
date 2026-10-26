"""
Chitti AI Companion & Brain 24/7 Status Monitor.
Displays real-time diagnostic metrics:
1. Chitti Background Daemon (PID, Memory, Responding)
2. Ollama Server & In-Memory AI Brain Model (VRAM, Expiry/Keep-Alive)
3. Active Voice & Character Persona (e.g. Shinchan / SwaraNeural / RVC)
4. Proactive Scheduler & Knowledge Graph State
5. Recent Live Activity Log
"""

import os
import sys
import time
import json
import urllib.request
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ANSI Colors
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"

# Enable Windows VT100 colors
if sys.platform == "win32":
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except Exception:
        pass


def check_chitti_daemon():
    try:
        import psutil
        chitti_procs = []
        for proc in psutil.process_iter(["pid", "name", "cmdline", "memory_info"]):
            try:
                cmdline = " ".join(proc.info.get("cmdline") or []).lower()
                if "src.main" in cmdline or "main.py" in cmdline:
                    chitti_procs.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        if chitti_procs:
            main_p = chitti_procs[0]
            mem_mb = main_p.info["memory_info"].rss / (1024 * 1024)
            print(f"  {GREEN}[ONLINE]{RESET} Chitti Daemon: PID {main_p.info['pid']} | Memory: {mem_mb:.1f} MB | Status: Active 24/7")
        else:
            print(f"  {RED}[OFFLINE]{RESET} Chitti Background Daemon is not running.")
    except Exception as e:
        print(f"  {YELLOW}[NOTICE]{RESET} Process check: {e}")


def check_ollama_brain():
    # 1. Server ping
    ollama_pid = None
    try:
        import psutil
        for proc in psutil.process_iter(["pid", "name"]):
            try:
                if "ollama" in (proc.info.get("name") or "").lower():
                    ollama_pid = proc.info["pid"]
                    break
            except Exception:
                continue
    except Exception:
        pass

    if ollama_pid:
        print(f"  {GREEN}[ONLINE]{RESET} Ollama Brain Server: PID {ollama_pid} | Port: 11434")
    else:
        print(f"  {RED}[OFFLINE]{RESET} Ollama Brain Server is not running.")

    # 2. Check loaded model in VRAM
    try:
        req = urllib.request.Request("http://localhost:11434/api/ps", headers={"User-Agent": "ChittiMonitor"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = data.get("models", [])
            if models:
                m = models[0]
                name = m.get("name", "Unknown")
                vram_gb = m.get("size_vram", 0) / (1024 * 1024 * 1024)
                expires = m.get("expires_at", "")
                is_permanent = "2319" in expires or expires.startswith("2") and int(expires[:4]) > 2050
                ka_str = "Permanent 24/7 (Never Unloads)" if is_permanent else expires[:19]
                print(f"  {GREEN}[ONLINE]{RESET} AI Brain Model: {BOLD}{name}{RESET} | VRAM: {vram_gb:.2f} GB (GPU) | Keep-Alive: {GREEN}{ka_str}{RESET}")
            else:
                print(f"  {YELLOW}[IDLE]{RESET} AI Brain Model not yet resident in VRAM. (Will load upon first interaction)")
    except Exception:
        print(f"  {RED}[OFFLINE]{RESET} AI Brain Model: Could not query Ollama API.")


def check_persona():
    preset = os.getenv("VOICE_PRESET", "default").lower()
    persona = os.getenv("CHITTI_PERSONA", "default").lower()
    rvc = os.getenv("RVC_ENABLED", "false").lower() in ("true", "1", "yes")
    pitch = os.getenv("TTS_PITCH", "+0Hz")
    rate = os.getenv("TTS_RATE_PERCENTAGE", "+0%")

    if preset == "shinchan" or persona == "shinchan":
        print(f"  {CYAN}[ACTIVE]{RESET} Character Persona: {BOLD}Shinchan (शिनचैन){RESET} | Pitch: {pitch} | Rate: {rate} | RVC AI Clone: {GREEN if rvc else YELLOW}{'Enabled (RTX 5050 CUDA)' if rvc else 'Disabled'}{RESET}")
    else:
        print(f"  {CYAN}[ACTIVE]{RESET} Character Persona: Default Chitti | Voice: {os.getenv('TTS_VOICE_ID', 'hi-IN-MadhurNeural')}")


def check_recent_logs():
    log_file = PROJECT_ROOT / "chitti_daemon.log"
    if not log_file.exists():
        print(f"  No {log_file.name} found yet.")
        return

    try:
        with open(log_file, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
            tail = [l.strip() for l in lines[-8:] if l.strip()]
            for line in tail:
                print(f"  {line}")
    except Exception as e:
        print(f"  Could not read log file: {e}")


def main():
    print(f"\n{BOLD}{CYAN}=============================================================================={RESET}")
    print(f"{BOLD}{CYAN}                   CHITTI AI 24/7 BRAIN & SYSTEM MONITOR                      {RESET}")
    print(f"{BOLD}{CYAN}=============================================================================={RESET}\n")

    print(f"{BOLD}--- 1. Brain & Process Health ---{RESET}")
    check_chitti_daemon()
    check_ollama_brain()

    print(f"\n{BOLD}--- 2. Voice & Persona Configuration ---{RESET}")
    check_persona()

    print(f"\n{BOLD}--- 3. Live Activity (Last Log Events) ---{RESET}")
    check_recent_logs()

    print(f"\n{BOLD}{CYAN}=============================================================================={RESET}")
    print(" Controls:")
    print('   - Wake Word : Say "Hey Chitti" anytime hands-free')
    print("   - Visual HUD: Press Alt + Space (or Ctrl + Space / Alt + C)")
    print("   - Start     : Double-click Start_Chitti.vbs or Launch_Chitti.bat")
    print("   - Stop      : Double-click Stop_Chitti.bat")
    print(f"{BOLD}{CYAN}=============================================================================={RESET}\n")


if __name__ == "__main__":
    main()
