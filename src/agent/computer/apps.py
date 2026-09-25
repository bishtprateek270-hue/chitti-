"""
Chitti Application Controller.
Discovers, launches, monitors, and terminates Windows desktop applications.
"""

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import psutil

from src.agent.registry import AppDiscovery
from src.utils.logging import log_debug, log_info, log_warn


@dataclass
class RunningApp:
    name: str
    pid: int
    exe: str
    memory_mb: float


class AppController:
    """Manages application lifecycle on Windows."""

    def __init__(self):
        pass

    def resolve_application(self, app_name: str) -> Optional[str]:
        """Resolves an application name to an executable path."""
        return AppDiscovery.resolve_app(app_name)

    def launch_application(self, app_name: str, args: Optional[List[str]] = None, cwd: Optional[str] = None) -> bool:
        """Launches an application by name or path."""
        exe_path = self.resolve_application(app_name)
        if not exe_path:
            # Check if it's already an absolute path
            p = Path(app_name)
            if p.exists() and p.is_file():
                exe_path = str(p)
            else:
                log_warn(f"Could not resolve application: {app_name}")
                return False

        try:
            cmd = [exe_path] + (args or [])
            is_shell = str(exe_path).lower().endswith((".cmd", ".bat"))
            subprocess.Popen(cmd, cwd=cwd, shell=is_shell)
            log_info(f"Launched application: {exe_path} with args: {args}")
            return True
        except Exception as e:
            log_warn(f"Failed to launch application {exe_path}: {e}")
            return False

    def list_running_applications(self) -> List[RunningApp]:
        """Returns a list of actively running desktop processes."""
        apps: List[RunningApp] = []
        for proc in psutil.process_iter(['pid', 'name', 'exe', 'memory_info']):
            try:
                info = proc.info
                if info['name'] and info['exe']:
                    mem_mb = info['memory_info'].rss / (1024 * 1024) if info['memory_info'] else 0.0
                    apps.append(RunningApp(
                        name=info['name'],
                        pid=info['pid'],
                        exe=info['exe'],
                        memory_mb=round(mem_mb, 1),
                    ))
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        return sorted(apps, key=lambda a: a.name.lower())

    def is_application_running(self, app_name: str) -> bool:
        """Checks if an application matching app_name is currently running."""
        query = app_name.lower().strip()
        for proc in psutil.process_iter(['name', 'exe']):
            try:
                name = (proc.info['name'] or "").lower()
                exe = (proc.info['exe'] or "").lower()
                if query in name or query in exe:
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        return False

    def close_application(self, app_name: str, force: bool = False) -> bool:
        """Terminates processes matching app_name."""
        query = app_name.lower().strip()
        closed_any = False

        for proc in psutil.process_iter(['pid', 'name', 'exe']):
            try:
                name = (proc.info['name'] or "").lower()
                exe = (proc.info['exe'] or "").lower()
                if query in name or query in exe:
                    p = psutil.Process(proc.info['pid'])
                    if force:
                        p.kill()
                    else:
                        p.terminate()
                    closed_any = True
                    log_info(f"Terminated process {proc.info['name']} (PID: {proc.info['pid']})")
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue

        return closed_any
