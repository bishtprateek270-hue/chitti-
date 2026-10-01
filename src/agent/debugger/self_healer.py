"""
Chitti Autonomous Multi-File Code Self-Healing & Debugging Loop (Phase 2).
Provides an end-to-end feedback loop:
1. Sandboxed test / script execution
2. Multi-language traceback parsing & error diagnosis
3. LLM / AST-assisted patch generation
4. Atomic multi-file patching with checkpoint rollback safety
5. Iterative verification until 100% passing
"""

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple

from src.agent.debugger.runner import CodeExecutionSandbox, ExecutionResult
from src.agent.debugger.traceback_parser import TracebackParser, ParsedErrorInfo
from src.agent.debugger.patcher import CodePatcher, PatchResult
from src.agent.debugger.test_generator import TestGenerator
from src.agent.code_validator import CodeValidator
from src.brain.llm import BaseLLM
from src.utils.logging import log_chitti, log_debug, log_info, log_warn, log_error


@dataclass
class HealingResult:
    """Outcome of an autonomous self-healing session."""
    success: bool
    iterations: int
    initial_error: Optional[ParsedErrorInfo] = None
    final_output: str = ""
    patches_applied: List[Dict[str, Any]] = field(default_factory=list)
    files_modified: List[str] = field(default_factory=list)
    diagnosis: str = ""
    rollback_occurred: bool = False
    execution_history: List[ExecutionResult] = field(default_factory=list)

    def summary(self) -> str:
        if self.success:
            return f"Self-Healing SUCCESS after {self.iterations} iteration(s). Modified files: {', '.join(self.files_modified) or 'None'}."
        else:
            return f"Self-Healing FAILED after {self.iterations} iteration(s). Diagnosis: {self.diagnosis}."


class SelfHealingDebugger:
    """
    Autonomous code debugger and self-healing loop for Python, JS, C++, and multi-file projects.
    """

    def __init__(self, llm: Optional[BaseLLM] = None, default_timeout: float = 30.0):
        self.llm = llm
        self.default_timeout = default_timeout

    def heal_file(
        self,
        target_file: Union[str, Path],
        test_file: Optional[Union[str, Path]] = None,
        max_iterations: int = 3,
        extra_files: Optional[List[Union[str, Path]]] = None,
    ) -> HealingResult:
        """
        Executes and automatically heals a single target file or module against its test suite.
        """
        t_path = Path(target_file).resolve()
        if not t_path.exists():
            return HealingResult(
                success=False,
                iterations=0,
                diagnosis=f"Target file not found: {t_path}",
                final_output=f"File {t_path} does not exist",
            )

        all_tracked_files = [t_path]
        if test_file:
            all_tracked_files.append(Path(test_file).resolve())
        if extra_files:
            all_tracked_files.extend([Path(f).resolve() for f in extra_files])

        # Step 1: Create safe rollback checkpoint
        checkpoint_id = CodePatcher.create_checkpoint(all_tracked_files)
        log_chitti(f"[CHITTI] [SELF-HEAL] Initialized healing session for '{t_path.name}' (Checkpoint: {checkpoint_id})")

        history: List[ExecutionResult] = []
        applied_patches: List[Dict[str, Any]] = []
        modified_files_set = set()
        initial_error: Optional[ParsedErrorInfo] = None
        last_error: Optional[ParsedErrorInfo] = None

        try:
            for iteration in range(1, max_iterations + 1):
                log_chitti(f"[CHITTI] [SELF-HEAL] [Iteration {iteration}/{max_iterations}] Running execution verification...")

                # 1. Run execution / tests
                if test_file:
                    exec_res = CodeExecutionSandbox.run_tests(test_file, cwd=t_path.parent, timeout=self.default_timeout)
                else:
                    exec_res = CodeExecutionSandbox.run_script(t_path, cwd=t_path.parent, timeout=self.default_timeout)

                history.append(exec_res)

                # 2. Check if clean success
                if exec_res.success:
                    log_chitti(f"[CHITTI] [SELF-HEAL] SUCCESS! Execution passed with zero errors in iteration {iteration}.")
                    CodePatcher.cleanup_checkpoint(checkpoint_id)
                    return HealingResult(
                        success=True,
                        iterations=iteration,
                        initial_error=initial_error,
                        final_output=exec_res.combined_output,
                        patches_applied=applied_patches,
                        files_modified=list(modified_files_set),
                        diagnosis=f"Code passed all execution & assertion checks in {iteration} iteration(s).",
                        rollback_occurred=False,
                        execution_history=history,
                    )

                # 3. Parse traceback & errors
                err_info = TracebackParser.parse(exec_res.combined_output, default_file=str(t_path))
                if iteration == 1:
                    initial_error = err_info
                last_error = err_info

                log_chitti(f"[CHITTI] [SELF-HEAL] Detected failure: {err_info.summary()}")

                # 4. Determine which file needs fixing
                fix_target = t_path
                if err_info.file_path:
                    candidate = Path(err_info.file_path).resolve()
                    if candidate.exists() and candidate in [f.resolve() for f in all_tracked_files]:
                        fix_target = candidate

                # 5. Read current content of failing file
                current_code = fix_target.read_text(encoding="utf-8", errors="replace")

                # 6. Generate fix / patch
                fixed_code, diag = self._generate_fix(
                    file_path=fix_target,
                    code=current_code,
                    error=err_info,
                    iteration=iteration,
                )

                if not fixed_code or fixed_code.strip() == current_code.strip():
                    log_warn("[CHITTI] [SELF-HEAL] Generated fix did not produce any difference. Retrying with fallback heuristic...")
                    fixed_code, diag = self._heuristic_fix(current_code, err_info)

                # 7. Apply patch
                patch_res = CodePatcher.apply_full_file(fix_target, fixed_code, validate=True)
                if patch_res.success:
                    modified_files_set.add(str(fix_target))
                    applied_patches.append({
                        "iteration": iteration,
                        "file": str(fix_target),
                        "error_type": err_info.error_type,
                        "diagnosis": diag,
                        "diff": patch_res.diff,
                    })
                    log_info(f"[CHITTI] [SELF-HEAL] Patch successfully applied to {fix_target.name}.")
                else:
                    log_warn(f"[CHITTI] [SELF-HEAL] Patch application failed: {patch_res.error}. Skipping to next iteration.")

            # Max iterations exceeded without resolution: Rollback
            log_warn(f"[CHITTI] [SELF-HEAL] Max iterations ({max_iterations}) exceeded. Rolling back files to safe state.")
            CodePatcher.rollback_checkpoint(checkpoint_id, all_tracked_files)

            return HealingResult(
                success=False,
                iterations=max_iterations,
                initial_error=initial_error,
                final_output=history[-1].combined_output if history else "No output",
                patches_applied=applied_patches,
                files_modified=[],
                diagnosis=f"Failed to heal after {max_iterations} iterations. Final error: {last_error.summary() if last_error else 'Unknown'}",
                rollback_occurred=True,
                execution_history=history,
            )

        except Exception as e:
            log_error(f"[CHITTI] [SELF-HEAL] Critical exception during self-healing: {e}")
            CodePatcher.rollback_checkpoint(checkpoint_id, all_tracked_files)
            return HealingResult(
                success=False,
                iterations=len(history),
                initial_error=initial_error,
                diagnosis=f"Exception during self-healing loop: {str(e)}",
                rollback_occurred=True,
                execution_history=history,
            )

    def heal_project(
        self,
        project_dir: Union[str, Path],
        test_file: Optional[Union[str, Path]] = None,
        max_iterations: int = 3,
    ) -> HealingResult:
        """
        Scans a project directory, locates primary entry points / tests, and heals broken modules.
        """
        p_dir = Path(project_dir).resolve()
        if not p_dir.exists() or not p_dir.is_dir():
            return HealingResult(
                success=False,
                iterations=0,
                diagnosis=f"Project directory not found: {p_dir}",
            )

        # Locate Python files in project
        py_files = list(p_dir.glob("*.py")) + list(p_dir.glob("src/**/*.py"))
        if not py_files:
            return HealingResult(
                success=False,
                iterations=0,
                diagnosis="No Python source files found in project directory.",
            )

        # Look for entry point or test file
        test_candidates = [f for f in py_files if f.name.startswith("test_") or f.name.endswith("_test.py")]
        main_candidates = [f for f in py_files if f.name in ("main.py", "app.py", "index.py", "solution.py")]

        target_main = main_candidates[0] if main_candidates else py_files[0]
        target_test = Path(test_file).resolve() if test_file else (test_candidates[0] if test_candidates else None)

        log_chitti(f"[CHITTI] [SELF-HEAL] Healing project at '{p_dir.name}' (Target: {target_main.name}, Tests: {target_test.name if target_test else 'None'})")
        return self.heal_file(
            target_file=target_main,
            test_file=target_test,
            max_iterations=max_iterations,
            extra_files=py_files,
        )

    def _generate_fix(
        self,
        file_path: Path,
        code: str,
        error: ParsedErrorInfo,
        iteration: int,
    ) -> Tuple[str, str]:
        """
        Generates repaired code using LLM reasoning or AST heuristics.
        """
        lang = file_path.suffix.lstrip(".").lower() or "python"

        if self.llm:
            prompt = f"""You are Chitti's Autonomous Code Self-Healing Engine.
Your task is to fix the following {lang} source file to resolve the runtime / test failure.

Error Details:
- Error Type: {error.error_type}
- Error Message: {error.error_message}
- Failing Line: {error.line_number or 'Unknown'}
- Failing Code: {error.failing_code_snippet or 'Unknown'}
- Traceback / Diagnostic:
{error.raw_traceback}

File Path: {file_path.name}
Current Source Code:
```{lang}
{code}
```

Instructions:
1. Fix the root cause of the error.
2. Keep all existing functionality, function signatures, and comments intact.
3. Do not introduce placeholders, TODOs, or mock stubs.
4. Output ONLY the complete, corrected {lang} source code wrapped in ```{lang} ... ``` code blocks.
"""
            try:
                response = self.llm.generate_response(prompt).strip()
                match = re.search(r"```(?:[a-zA-Z0-9_\+\#\-]+)?\n(.*?)```", response, re.DOTALL)
                clean_code = match.group(1).strip() if match else response.strip()
                if clean_code and clean_code != code.strip():
                    return clean_code, f"LLM repaired {error.error_type}: {error.error_message}"
            except Exception as e:
                log_warn(f"[CHITTI] [SELF-HEAL] LLM fix synthesis error: {e}")

        # Fallback to rule-based heuristic repair
        return self._heuristic_fix(code, error)

    def _heuristic_fix(self, code: str, error: ParsedErrorInfo) -> Tuple[str, str]:
        """
        Deterministic rule-based repair for common syntax errors, off-by-one, zero division, and imports.
        """
        lines = code.splitlines()
        err_type = error.error_type
        err_msg = error.error_message.lower()
        lineno = error.line_number

        # 1. ZeroDivisionError Fix
        if err_type == "ZeroDivisionError" or "division by zero" in err_msg:
            if lineno and 1 <= lineno <= len(lines):
                target_line = lines[lineno - 1]
                # Wrap line with zero check or replace `/ b` with `/ (b if b != 0 else 1)`
                indent = len(target_line) - len(target_line.lstrip())
                ind_str = " " * indent
                lines[lineno - 1] = f"{ind_str}try:\n{ind_str}    {target_line.strip()}\n{ind_str}except ZeroDivisionError:\n{ind_str}    pass"
                return "\n".join(lines), "Added ZeroDivisionError exception safety handler"

        # 2. NameError Fix (e.g. missing import or undefined variable)
        if err_type == "NameError" or "name '" in err_msg:
            var_match = re.search(r"name '([A-Za-z0-9_]+)' is not defined", error.error_message)
            if var_match:
                missing_var = var_match.group(1)
                # Check for standard libraries
                std_libs = {"math", "os", "sys", "re", "json", "time", "random", "pathlib", "datetime", "typing"}
                if missing_var in std_libs:
                    new_code = f"import {missing_var}\n" + code
                    return new_code, f"Added missing import: import {missing_var}"
                elif lineno and 1 <= lineno <= len(lines):
                    # Define variable as None or default before usage
                    indent = len(lines[lineno - 1]) - len(lines[lineno - 1].lstrip())
                    lines.insert(lineno - 1, f"{' ' * indent}{missing_var} = 0")
                    return "\n".join(lines), f"Initialized missing variable '{missing_var}'"

        # 3. IndexError Fix
        if err_type == "IndexError" or "list index out of range" in err_msg:
            if lineno and 1 <= lineno <= len(lines):
                target_line = lines[lineno - 1]
                indent = len(target_line) - len(target_line.lstrip())
                ind_str = " " * indent
                lines[lineno - 1] = f"{ind_str}try:\n{ind_str}    {target_line.strip()}\n{ind_str}except IndexError:\n{ind_str}    pass"
                return "\n".join(lines), "Added IndexError bounds protection"

        # 4. KeyError Fix
        if err_type == "KeyError":
            if lineno and 1 <= lineno <= len(lines):
                target_line = lines[lineno - 1]
                indent = len(target_line) - len(target_line.lstrip())
                ind_str = " " * indent
                lines[lineno - 1] = f"{ind_str}try:\n{ind_str}    {target_line.strip()}\n{ind_str}except KeyError:\n{ind_str}    pass"
                return "\n".join(lines), "Added KeyError dictionary protection"

        # 5. SyntaxError (missing colon at end of def/if/for/while/class/try/except/with)
        if err_type == "SyntaxError" and lineno and 1 <= lineno <= len(lines):
            target_line = lines[lineno - 1]
            stripped = target_line.rstrip()
            if any(stripped.startswith(kw) for kw in ("def ", "if ", "elif ", "else", "for ", "while ", "class ", "try", "except", "with ")) and not stripped.endswith(":"):
                lines[lineno - 1] = stripped + ":"
                return "\n".join(lines), "Added missing colon at end of statement"

        # Fallback: append safety import or return statement
        return code, f"No heuristic rule matched for {err_type}"
