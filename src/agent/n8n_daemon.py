"""
Chitti 24/7 n8n Daemon & Watchdog Manager.
Spawns and monitors local n8n workflow server in the background,
keeping it alive 24/7 alongside Chitti with auto-recovery.
"""

import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from typing import Optional

from src.utils.logging import log_chitti, log_debug, log_error, log_info, log_warning


class N8NDaemonManager:
    """Manages the background execution and watchdog supervision of n8n."""

    def __init__(
        self,
        base_url: str = "http://localhost:5678",
        auto_restart: bool = True,
        check_interval: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.auto_restart = auto_restart
        self.check_interval = check_interval
        self._process: Optional[subprocess.Popen] = None
        self._watchdog_thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()

    def is_healthy(self) -> bool:
        """Checks if n8n is currently responding to HTTP requests."""
        for endpoint in (f"{self.base_url}/healthz", f"{self.base_url}/"):
            try:
                req = urllib.request.Request(endpoint, headers={"User-Agent": "Chitti-Monitor/1.0"})
                with urllib.request.urlopen(req, timeout=2.5) as resp:
                    if 200 <= resp.getcode() < 400:
                        return True
            except Exception:
                pass
        return False

    def _find_n8n_command(self) -> Optional[list]:
        """Detects the best way to launch n8n on the current machine."""
        # 1. Check for globally installed n8n executable
        n8n_cmd = shutil.which("n8n") or shutil.which("n8n.cmd")
        if n8n_cmd:
            return [n8n_cmd]

        # 2. Check for npx
        npx_cmd = shutil.which("npx") or shutil.which("npx.cmd")
        if npx_cmd:
            return [npx_cmd, "n8n"]

        # 3. Check for docker
        docker_cmd = shutil.which("docker") or shutil.which("docker.exe")
        if docker_cmd:
            return [
                docker_cmd, "run", "-d", "--name", "chitti-n8n-24x7",
                "-p", "5678:5678",
                "-v", f"{os.path.expanduser('~')}/.n8n:/home/node/.n8n",
                "docker.n8n.io/n8nio/n8n"
            ]

        return None

    def start_n8n(self) -> bool:
        """Spawns n8n in an independent background process."""
        if self.is_healthy():
            log_chitti(f"[n8n-daemon] 🌐 n8n is already active and healthy on {self.base_url}")
            return True

        cmd = self._find_n8n_command()
        if not cmd:
            log_warning("[n8n-daemon] Neither 'n8n', 'npx', nor 'docker' found on system PATH.")
            return False

        log_chitti(f"[n8n-daemon] 🚀 Launching 24/7 n8n server via: {' '.join(cmd)}")

        creationflags = 0
        if sys.platform == "win32":
            # DETACHED_PROCESS + CREATE_NO_WINDOW ensures no black cmd window pops up
            creationflags = subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS

        try:
            # We run via cmd /c on Windows to properly resolve batch/npm shims
            if sys.platform == "win32" and not cmd[0].endswith(".exe"):
                full_cmd = ["cmd.exe", "/c"] + cmd
            else:
                full_cmd = cmd

            self._process = subprocess.Popen(
                full_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                creationflags=creationflags,
                shell=False,
            )
            log_chitti(f"[n8n-daemon] ✓ Process started (PID: {self._process.pid}). Warming up...")
            return True
        except Exception as e:
            log_error(f"[n8n-daemon] Failed to spawn n8n process: {e}")
            return False

    def ensure_running_async(self):
        """Starts n8n and launches background watchdog thread."""
        with self._lock:
            if self._running:
                return
            self._running = True

        def _supervisor_loop():
            # Initial launch if not already online
            if not self.is_healthy():
                self.start_n8n()

            while self._running:
                time.sleep(self.check_interval)
                if not self._running:
                    break

                if self.auto_restart and not self.is_healthy():
                    log_chitti("[n8n-daemon] ⚠️ n8n stopped responding. Watchdog auto-restarting...")
                    self.start_n8n()

        self._watchdog_thread = threading.Thread(
            target=_supervisor_loop,
            name="N8NWatchdogSupervisor",
            daemon=True,
        )
        self._watchdog_thread.start()

    def stop(self):
        """Terminates watchdog and managed process."""
        self._running = False
        if self._process and self._process.poll() is None:
            try:
                self._process.terminate()
            except Exception:
                pass
            self._process = None
        log_debug("[n8n-daemon] Watchdog stopped.")
