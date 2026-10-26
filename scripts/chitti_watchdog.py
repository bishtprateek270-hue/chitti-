"""
Chitti 24/7 Watchdog Supervisor.
Periodically monitors Chitti background daemon and Ollama server.
If either service crashes or exits, automatically restarts it silently to ensure 24/7 uptime.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def is_proc_active(keyword: str) -> bool:
    """Checks if any process matching keyword is running."""
    try:
        import psutil
        for proc in psutil.process_iter(["name", "cmdline"]):
            try:
                cmdline = " ".join(proc.info.get("cmdline") or []).lower()
                name = (proc.info.get("name") or "").lower()
                if keyword.lower() in cmdline or keyword.lower() in name:
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except Exception:
        pass
    return False


def ensure_ollama():
    """Starts Ollama in background if not running."""
    if not is_proc_active("ollama"):
        try:
            create_flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            subprocess.Popen(["ollama", "serve"], creationflags=create_flags)
        except Exception:
            pass


def ensure_chitti():
    """Starts Chitti daemon if not running."""
    try:
        import psutil
        my_pid = os.getpid()
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                if proc.info["pid"] == my_pid:
                    continue
                cmdline = " ".join(proc.info.get("cmdline") or []).lower()
                if ("src.main" in cmdline or "main.py" in cmdline) and "--daemon" in cmdline:
                    return
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except Exception:
        pass

    # Launch Chitti via launcher batch file
    launcher = PROJECT_ROOT / "scripts" / "chitti_launcher.bat"
    if launcher.exists():
        create_flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        subprocess.Popen([str(launcher)], cwd=str(PROJECT_ROOT), creationflags=create_flags)


def ensure_brain_loaded():
    """Checks if LLM model is resident in Ollama VRAM; if not, triggers preload with keep_alive: -1."""
    try:
        import json
        import urllib.request
        with urllib.request.urlopen("http://localhost:11434/api/ps", timeout=3) as resp:
            data = json.loads(resp.read())
            models = [m.get("name", "") for m in data.get("models", [])]
            if not any("qwen2.5-coder" in m or "llama" in m for m in models):
                model_name = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
                req = urllib.request.Request(
                    "http://localhost:11434/api/generate",
                    data=json.dumps({"model": model_name, "keep_alive": -1}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                )
                urllib.request.urlopen(req, timeout=15)
    except Exception:
        pass


def main():
    while True:
        try:
            ensure_ollama()
            ensure_brain_loaded()
            ensure_chitti()
        except Exception:
            pass
        time.sleep(20)


if __name__ == "__main__":
    main()
