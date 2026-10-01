"""
Chitti Multi-Language Code Execution Sandbox.
Provides isolated, timeout-guarded execution of code, scripts, and test suites
across Python, JavaScript/Node.js, TypeScript, C/C++, Rust, and Java.
"""

import os
import sys
import time
import shutil
import tempfile
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

from src.utils.logging import log_chitti, log_debug, log_info, log_warn, log_error


@dataclass
class ExecutionResult:
    """Represents the structured outcome of a sandboxed execution."""
    success: bool
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: float
    timed_out: bool = False
    command: List[str] = field(default_factory=list)
    language: str = "python"
    error_summary: str = ""

    @property
    def combined_output(self) -> str:
        """Returns stdout and stderr merged cleanly."""
        parts = []
        if self.stdout and self.stdout.strip():
            parts.append(self.stdout.strip())
        if self.stderr and self.stderr.strip():
            parts.append(self.stderr.strip())
        return "\n".join(parts)


class CodeExecutionSandbox:
    """
    Sandboxed execution engine for multi-language scripts and test suites.
    Safely captures stdout, stderr, execution time, and prevents runaway processes.
    """

    DEFAULT_TIMEOUT_SEC = 30.0

    LANGUAGE_RUNNERS = {
        "python": [sys.executable],
        "py": [sys.executable],
        "javascript": ["node"],
        "js": ["node"],
        "node": ["node"],
        "typescript": ["npx", "ts-node"],
        "ts": ["npx", "ts-node"],
        "rust": ["rustc"],
        "rs": ["rustc"],
        "cpp": ["g++"],
        "c": ["gcc"],
        "java": ["java"],
    }

    @classmethod
    def detect_language(cls, file_path: Union[str, Path]) -> str:
        """Infers programming language from file extension."""
        p = Path(file_path)
        ext = p.suffix.lower().lstrip(".")
        mapping = {
            "py": "python",
            "js": "javascript",
            "ts": "typescript",
            "jsx": "javascript",
            "tsx": "typescript",
            "cpp": "cpp",
            "cc": "cpp",
            "cxx": "cpp",
            "c": "c",
            "rs": "rust",
            "java": "java",
            "go": "go",
            "html": "html",
        }
        return mapping.get(ext, "python")

    @classmethod
    def execute_command(
        cls,
        cmd: List[str],
        cwd: Optional[Union[str, Path]] = None,
        timeout: float = DEFAULT_TIMEOUT_SEC,
        env: Optional[Dict[str, str]] = None,
        language: str = "generic",
    ) -> ExecutionResult:
        """Executes a command list with timeout protection and output capture."""
        start_time = time.perf_counter()
        work_dir = str(cwd) if cwd else os.getcwd()
        full_env = os.environ.copy()
        if env:
            full_env.update(env)

        # Force unbuffered Python output
        full_env["PYTHONUNBUFFERED"] = "1"
        full_env["PYTHONDONTWRITEBYTECODE"] = "1"

        try:
            log_debug(f"[SANDBOX] Running: {' '.join(cmd)} (cwd={work_dir}, timeout={timeout}s)")
            proc = subprocess.run(
                cmd,
                cwd=work_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
                env=full_env,
                encoding="utf-8",
                errors="replace",
            )
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            success = proc.returncode == 0
            err_sum = "" if success else (proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else f"Exit code {proc.returncode}")

            return ExecutionResult(
                success=success,
                exit_code=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
                duration_ms=duration_ms,
                timed_out=False,
                command=cmd,
                language=language,
                error_summary=err_sum,
            )

        except subprocess.TimeoutExpired as e:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            log_warn(f"[SANDBOX] Command timed out after {timeout}s: {' '.join(cmd)}")
            out = e.stdout if isinstance(e.stdout, str) else (e.stdout.decode("utf-8", errors="replace") if e.stdout else "")
            err = e.stderr if isinstance(e.stderr, str) else (e.stderr.decode("utf-8", errors="replace") if e.stderr else "")
            return ExecutionResult(
                success=False,
                exit_code=-1,
                stdout=out,
                stderr=err + f"\n[Execution timed out after {timeout} seconds]",
                duration_ms=duration_ms,
                timed_out=True,
                command=cmd,
                language=language,
                error_summary=f"Timed out after {timeout}s",
            )

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            log_error(f"[SANDBOX] Execution exception: {e}")
            return ExecutionResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr=f"Execution error: {str(e)}",
                duration_ms=duration_ms,
                timed_out=False,
                command=cmd,
                language=language,
                error_summary=str(e),
            )

    @classmethod
    def run_script(
        cls,
        file_path: Union[str, Path],
        language: Optional[str] = None,
        args: Optional[List[str]] = None,
        cwd: Optional[Union[str, Path]] = None,
        timeout: float = DEFAULT_TIMEOUT_SEC,
    ) -> ExecutionResult:
        """Executes a standalone source script."""
        p = Path(file_path).resolve()
        if not p.exists():
            return ExecutionResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr=f"Script file not found: {p}",
                duration_ms=0.0,
                timed_out=False,
                command=[str(p)],
                language=language or "unknown",
                error_summary="File not found",
            )

        lang = (language or cls.detect_language(p)).lower()
        effective_cwd = cwd or p.parent
        extra_args = args or []

        # 1. Python Execution
        if lang in ("python", "py"):
            cmd = [sys.executable, str(p)] + extra_args
            return cls.execute_command(cmd, cwd=effective_cwd, timeout=timeout, language="python")

        # 2. Node.js / JavaScript Execution
        elif lang in ("javascript", "js", "node"):
            node_exec = shutil.which("node") or "node"
            cmd = [node_exec, str(p)] + extra_args
            return cls.execute_command(cmd, cwd=effective_cwd, timeout=timeout, language="javascript")

        # 3. TypeScript Execution
        elif lang in ("typescript", "ts"):
            npx_exec = shutil.which("npx") or "npx"
            cmd = [npx_exec, "ts-node", str(p)] + extra_args
            return cls.execute_command(cmd, cwd=effective_cwd, timeout=timeout, language="typescript")

        # 4. Compiled C/C++ Execution
        elif lang in ("cpp", "c"):
            compiler = "g++" if lang == "cpp" else "gcc"
            if not shutil.which(compiler):
                return ExecutionResult(
                    success=False,
                    exit_code=-1,
                    stdout="",
                    stderr=f"Compiler '{compiler}' not found on system PATH.",
                    duration_ms=0.0,
                    language=lang,
                )
            bin_path = p.with_suffix(".exe" if sys.platform == "win32" else "")
            compile_cmd = [compiler, str(p), "-o", str(bin_path)]
            comp_res = cls.execute_command(compile_cmd, cwd=effective_cwd, timeout=timeout, language=lang)
            if not comp_res.success:
                return comp_res
            # Run the compiled binary
            run_cmd = [str(bin_path)] + extra_args
            run_res = cls.execute_command(run_cmd, cwd=effective_cwd, timeout=timeout, language=lang)
            # Cleanup binary
            try:
                if bin_path.exists():
                    bin_path.unlink()
            except Exception:
                pass
            return run_res

        # Fallback to direct Python runner
        cmd = [sys.executable, str(p)] + extra_args
        return cls.execute_command(cmd, cwd=effective_cwd, timeout=timeout, language=lang)

    @classmethod
    def run_tests(
        cls,
        target_path: Union[str, Path],
        framework: Optional[str] = None,
        cwd: Optional[Union[str, Path]] = None,
        timeout: float = 45.0,
        extra_pytest_args: Optional[List[str]] = None,
    ) -> ExecutionResult:
        """
        Executes automated test suites across Pytest, Unittest, Jest, etc.
        """
        target = Path(target_path).resolve()
        effective_cwd = cwd or (target if target.is_dir() else target.parent)
        fw = (framework or "pytest").lower()

        if fw in ("pytest", "py"):
            args = extra_pytest_args or ["-v", "--tb=short"]
            cmd = [sys.executable, "-m", "pytest", str(target)] + args
            return cls.execute_command(cmd, cwd=effective_cwd, timeout=timeout, language="python")

        elif fw in ("unittest", "unit"):
            cmd = [sys.executable, "-m", "unittest", "discover", "-s", str(effective_cwd), "-p", target.name if target.is_file() else "test*.py"]
            return cls.execute_command(cmd, cwd=effective_cwd, timeout=timeout, language="python")

        elif fw in ("jest", "npm"):
            npm_exec = shutil.which("npm") or "npm"
            cmd = [npm_exec, "test", "--", str(target)]
            return cls.execute_command(cmd, cwd=effective_cwd, timeout=timeout, language="javascript")

        # Fallback default to pytest
        cmd = [sys.executable, "-m", "pytest", str(target), "-v"]
        return cls.execute_command(cmd, cwd=effective_cwd, timeout=timeout, language="python")

    @classmethod
    def run_code_in_memory(
        cls,
        code: str,
        language: str = "python",
        timeout: float = DEFAULT_TIMEOUT_SEC,
    ) -> ExecutionResult:
        """Executes a temporary code snippet in an isolated scratch file."""
        lang = language.lower()
        suffix = ".py" if lang in ("python", "py") else (".js" if lang in ("javascript", "js") else ".txt")
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False, encoding="utf-8") as tmp:
            tmp.write(code)
            tmp_path = Path(tmp.name)

        try:
            res = cls.run_script(tmp_path, language=lang, timeout=timeout)
            return res
        finally:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass
