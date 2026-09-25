"""
Chitti Terminal Controller.
Executes shell and development commands with structured risk classification, output capture, and safety gates.
"""

import enum
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.utils.logging import log_debug, log_info, log_warn


class TerminalRiskLevel(str, enum.Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class TerminalResult:
    command: str
    exit_code: int
    stdout: str
    stderr: str
    elapsed_time_sec: float
    risk_level: TerminalRiskLevel

    @property
    def is_success(self) -> bool:
        return self.exit_code == 0

    @property
    def output(self) -> str:
        out = self.stdout.strip()
        err = self.stderr.strip()
        if out and err:
            return f"{out}\nErrors:\n{err}"
        return out or err or f"[Process exited with code {self.exit_code}]"


class TerminalController:
    """Safely executes terminal commands with risk analysis."""

    CRITICAL_PATTERNS = [
        r"(?i)\bformat\s+[a-z]:",
        r"(?i)\bdiskpart\b",
        r"(?i)\b(?:rmdir|rd|del|erase)\b.*[a-z]:\\",
        r"(?i)\bremove-item\s+.*-recurse.*(?:c:\\|windows)",
        r"(?i)\bset-mppreference\s+-disablerealtimemonitoring",
        r"(?i)\bnetsh\s+advfirewall\s+set\s+allprofiles\s+state\s+off",
        r"(?i)\bshutdown\s+/(?:s|r|p)\b",
    ]

    HIGH_PATTERNS = [
        r"(?i)\b(?:del|erase|rmdir|rd|rm)\b",
        r"(?i)\bremove-item\b",
        r"(?i)\bgit\s+reset\s+--hard\b",
        r"(?i)\bgit\s+clean\s+-[a-z]*f",
        r"(?i)\breg\s+(?:add|delete)\b",
        r"(?i)\bnet\s+user\b",
        r"(?i)\bsetx?\b",
    ]

    MEDIUM_PATTERNS = [
        r"(?i)\b(?:pip|conda|poetry)\s+install\b",
        r"(?i)\bnpm\s+(?:install|i|update|run|start)\b",
        r"(?i)\byarn\s+(?:add|install|run)\b",
        r"(?i)\bgit\s+(?:pull|push|commit|checkout|merge|rebase)\b",
        r"(?i)\bpython\s+\S+\.py\b",
        r"(?i)\bnode\s+\S+\.js\b",
    ]

    LOW_PATTERNS = [
        r"(?i)\b(?:python|node|npm|git|docker)\s+--version\b",
        r"(?i)\b(?:dir|ls|pwd|whoami|echo|type|cat|pytest|git\s+status|git\s+log|git\s+diff)\b",
    ]

    def __init__(self, default_cwd: Optional[str] = None):
        self.default_cwd = Path(default_cwd or os.getcwd()).resolve()

    def classify_risk(self, command: str) -> TerminalRiskLevel:
        """Classifies the execution risk level of a terminal command."""
        cmd = command.strip()

        # 1. Critical check
        if any(re.search(pat, cmd) for pat in self.CRITICAL_PATTERNS):
            return TerminalRiskLevel.CRITICAL

        # 2. High check
        if any(re.search(pat, cmd) for pat in self.HIGH_PATTERNS):
            return TerminalRiskLevel.HIGH

        # 3. Medium check
        if any(re.search(pat, cmd) for pat in self.MEDIUM_PATTERNS):
            return TerminalRiskLevel.MEDIUM

        # 4. Low check
        if any(re.search(pat, cmd) for pat in self.LOW_PATTERNS):
            return TerminalRiskLevel.LOW

        # Default fallback risk
        return TerminalRiskLevel.MEDIUM

    def is_destructive(self, command: str) -> bool:
        """Determines if a command requires user confirmation before running."""
        risk = self.classify_risk(command)
        return risk in (TerminalRiskLevel.HIGH, TerminalRiskLevel.CRITICAL)

    def execute_command(
        self,
        command: str,
        cwd: Optional[str] = None,
        timeout_sec: float = 30.0,
        env: Optional[Dict[str, str]] = None,
    ) -> TerminalResult:
        """Executes a command safely via powershell / cmd and captures output."""
        risk = self.classify_risk(command)
        work_dir = Path(cwd).resolve() if cwd else self.default_cwd
        if not work_dir.exists():
            work_dir.mkdir(parents=True, exist_ok=True)

        log_info(f"[TERMINAL] Running (Risk: {risk.value}): '{command}' in '{work_dir}'")

        start_time = time.time()
        try:
            # Use PowerShell for rich Windows terminal support
            process = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
                cwd=str(work_dir),
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                env={**os.environ, **(env or {})},
                errors="replace",
            )
            elapsed = time.time() - start_time
            log_info(f"[TERMINAL] Finished with exit code {process.returncode} in {elapsed:.2f}s")
            return TerminalResult(
                command=command,
                exit_code=process.returncode,
                stdout=process.stdout,
                stderr=process.stderr,
                elapsed_time_sec=elapsed,
                risk_level=risk,
            )
        except subprocess.TimeoutExpired as e:
            elapsed = time.time() - start_time
            log_warn(f"[TERMINAL] Command timed out after {timeout_sec}s: {command}")
            return TerminalResult(
                command=command,
                exit_code=-1,
                stdout=e.stdout or "" if isinstance(e.stdout, str) else "",
                stderr=f"Command timed out after {timeout_sec} seconds.",
                elapsed_time_sec=elapsed,
                risk_level=risk,
            )
        except Exception as e:
            elapsed = time.time() - start_time
            log_warn(f"[TERMINAL] Command execution failed: {e}")
            return TerminalResult(
                command=command,
                exit_code=-1,
                stdout="",
                stderr=str(e),
                elapsed_time_sec=elapsed,
                risk_level=risk,
            )
