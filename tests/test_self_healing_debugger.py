"""
Comprehensive Test Suite for Phase 2: Autonomous Multi-File Code Self-Healing & Debugging Loop.
Tests:
- CodeExecutionSandbox (script running, pytest execution, timeout safety)
- TracebackParser (Python tracebacks, pytest failures, C++ compiler diagnostics, heuristic fixes)
- CodePatcher (checkpoint backups, atomic rollbacks, line replacements, syntax validation)
- TestGenerator (AST function extraction, automated pytest synthesis)
- SelfHealingDebugger (iterative feedback loop, bug repair, multi-file healing, rollback safety)
- Agent Intent & Planner integration
"""

import os
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.agent.actions import ActionType, StructuredAction
from src.agent.debugger import (
    CodeExecutionSandbox,
    CodePatcher,
    ExecutionResult,
    HealingResult,
    ParsedErrorInfo,
    PatchResult,
    SelfHealingDebugger,
    TestGenerator,
    TracebackParser,
)
from src.agent.intent import ActionIntentAnalyzer, ActionIntentType
from src.agent.planner import AgentPlanner
from src.agent.projects import ProjectRegistry
from src.agent.tools import ToolEngine
from src.agent.executor import ActionExecutor


class TestSelfHealingDebuggerPhase2:
    """Test suite for Phase 2 Self-Healing Debugger components."""

    # =========================================================================
    # 1. CODE EXECUTION SANDBOX TESTS
    # =========================================================================

    def test_sandbox_run_python_script_success(self, tmp_path):
        """Verify successful Python script execution in sandbox."""
        script = tmp_path / "hello.py"
        script.write_text('print("Hello from Chitti Sandbox")', encoding="utf-8")

        res = CodeExecutionSandbox.run_script(script)
        assert res.success is True
        assert res.exit_code == 0
        assert "Hello from Chitti Sandbox" in res.stdout
        assert res.duration_ms > 0
        assert res.timed_out is False

    def test_sandbox_run_python_script_failure(self, tmp_path):
        """Verify failing Python script execution captures stderr and non-zero exit code."""
        script = tmp_path / "broken.py"
        script.write_text('raise ValueError("Custom calculation failure")', encoding="utf-8")

        res = CodeExecutionSandbox.run_script(script)
        assert res.success is False
        assert res.exit_code != 0
        assert "ValueError: Custom calculation failure" in res.stderr
        assert res.timed_out is False

    def test_sandbox_timeout_handling(self, tmp_path):
        """Verify that infinite loops are killed when timeout expires."""
        script = tmp_path / "loop.py"
        script.write_text('import time\nwhile True:\n    time.sleep(0.1)', encoding="utf-8")

        res = CodeExecutionSandbox.run_script(script, timeout=1.0)
        assert res.success is False
        assert res.timed_out is True
        assert "timed out" in res.combined_output.lower()

    def test_sandbox_run_in_memory_code(self):
        """Verify running code snippet directly in-memory."""
        code = 'x = 10 + 20\nprint(f"Result: {x}")'
        res = CodeExecutionSandbox.run_code_in_memory(code, language="python")
        assert res.success is True
        assert "Result: 30" in res.stdout

    # =========================================================================
    # 2. TRACEBACK PARSER TESTS
    # =========================================================================

    def test_traceback_parser_syntax_error(self):
        """Verify parsing of Python SyntaxError with line number and snippet."""
        sample_output = """
  File "calculator.py", line 14
    def add(a, b)
                 ^
SyntaxError: expected ':'
"""
        err = TracebackParser.parse(sample_output)
        assert err.error_type == "SyntaxError"
        assert err.line_number == 14
        assert "expected ':'" in err.error_message or "Invalid syntax" in err.error_message
        assert "colon" in err.suggested_fix_summary.lower()

    def test_traceback_parser_zero_division(self):
        """Verify parsing of runtime ZeroDivisionError."""
        sample_output = """
Traceback (most recent call last):
  File "math_ops.py", line 8, in divide
    return a / b
ZeroDivisionError: division by zero
"""
        err = TracebackParser.parse(sample_output)
        assert err.error_type == "ZeroDivisionError"
        assert err.line_number == 8
        assert "division by zero" in err.error_message
        assert "zero" in err.suggested_fix_summary.lower()

    def test_traceback_parser_pytest_failure(self):
        """Verify parsing of Pytest assertion failure."""
        sample_output = """
============================= test session starts =============================
FAILED tests/test_calc.py::test_addition - AssertionError: assert 15 == 20
============================= 1 failed in 0.05s ===============================
"""
        err = TracebackParser.parse(sample_output)
        assert err.is_test_failure is True
        assert "test_calc.py" in err.file_path
        assert "Assertion" in err.error_type or "assert" in err.error_message

    def test_traceback_parser_cpp_compilation_error(self):
        """Verify parsing of GCC/Clang C++ compiler diagnostics."""
        sample_output = "src/main.cpp:25:10: error: 'vector' was not declared in this scope"
        err = TracebackParser.parse(sample_output)
        assert err.is_compilation_error is True
        assert err.line_number == 25
        assert err.column_number == 10
        assert "vector" in err.error_message

    # =========================================================================
    # 3. CODE PATCHER & ROLLBACK TESTS
    # =========================================================================

    def test_patcher_checkpoint_and_rollback(self, tmp_path):
        """Verify creating checkpoint, editing file, and successfully rolling back."""
        f1 = tmp_path / "module.py"
        f1.write_text('def original():\n    return "original"\n', encoding="utf-8")

        # Create checkpoint
        cp_id = CodePatcher.create_checkpoint([f1])
        assert cp_id is not None

        # Modify file
        f1.write_text('def corrupted():\n    raise RuntimeError()\n', encoding="utf-8")
        assert "corrupted" in f1.read_text(encoding="utf-8")

        # Rollback
        rolled_back = CodePatcher.rollback_checkpoint(cp_id, [f1])
        assert rolled_back is True
        assert "original" in f1.read_text(encoding="utf-8")

        # Cleanup
        CodePatcher.cleanup_checkpoint(cp_id)

    def test_patcher_apply_line_replacement(self, tmp_path):
        """Verify line replacement in source code."""
        f1 = tmp_path / "app.py"
        f1.write_text("x = 1\ny = 2\nz = x + y\nprint(z)\n", encoding="utf-8")

        res = CodePatcher.apply_line_replacement(f1, start_line=2, end_line=2, replacement_text="y = 20\n")
        assert res.success is True
        new_content = f1.read_text(encoding="utf-8")
        assert "y = 20" in new_content

    # =========================================================================
    # 4. TEST GENERATOR TESTS
    # =========================================================================

    def test_test_generator_python_ast(self, tmp_path):
        """Verify extracting functions via AST and synthesizing pytest test suite."""
        code = """
def calculate_tax(amount: float, rate: float) -> float:
    return amount * rate

def format_currency(val: float) -> str:
    return f"${val:.2f}"
"""
        src_file = tmp_path / "finance.py"
        src_file.write_text(code, encoding="utf-8")
        test_file = tmp_path / "test_finance.py"

        suite = TestGenerator.generate_tests_for_file(src_file, output_test_file=test_file)
        assert suite.test_count >= 2
        assert "calculate_tax" in suite.target_functions
        assert "format_currency" in suite.target_functions
        assert test_file.exists()
        assert "def test_calculate_tax" in test_file.read_text(encoding="utf-8")

    # =========================================================================
    # 5. AUTONOMOUS SELF-HEALING LOOP TESTS
    # =========================================================================

    def test_self_healing_fixes_zero_division(self, tmp_path):
        """Verify that SelfHealingDebugger autonomously catches ZeroDivisionError and repairs it."""
        buggy_code = """
def safe_divide(a, b):
    return a / b

# Execution check
result = safe_divide(100, 0)
"""
        script = tmp_path / "division_bug.py"
        script.write_text(buggy_code, encoding="utf-8")

        healer = SelfHealingDebugger(llm=None)
        res = healer.heal_file(target_file=script, max_iterations=3)

        assert res.success is True
        assert res.iterations >= 1
        assert len(res.patches_applied) > 0
        assert script.name in res.files_modified[0] or str(script) in res.files_modified[0]

        # Verify fixed script runs with zero errors
        verify_run = CodeExecutionSandbox.run_script(script)
        assert verify_run.success is True

    def test_self_healing_fixes_missing_import(self, tmp_path):
        """Verify that SelfHealingDebugger autonomously fixes missing standard library import."""
        buggy_code = """
def compute_hypotenuse(a, b):
    return math.sqrt(a*a + b*b)

res = compute_hypotenuse(3, 4)
"""
        script = tmp_path / "geometry.py"
        script.write_text(buggy_code, encoding="utf-8")

        healer = SelfHealingDebugger(llm=None)
        res = healer.heal_file(target_file=script, max_iterations=3)

        assert res.success is True
        fixed_content = script.read_text(encoding="utf-8")
        assert "import math" in fixed_content

        # Verify execution
        verify_run = CodeExecutionSandbox.run_script(script)
        assert verify_run.success is True

    def test_self_healing_with_mock_llm_patch(self, tmp_path):
        """Verify LLM-driven repair in self-healing loop."""
        buggy_code = """
def multiply_by_two(val):
    return val + 1  # Bug
"""
        script = tmp_path / "multiplier.py"
        script.write_text(buggy_code, encoding="utf-8")

        test_code = """
import pytest
from multiplier import multiply_by_two

def test_multiplier():
    assert multiply_by_two(5) == 10
"""
        test_file = tmp_path / "test_multiplier.py"
        test_file.write_text(test_code, encoding="utf-8")

        # Mock LLM that generates the corrected function
        mock_llm = MagicMock()
        mock_llm.generate_response.return_value = """```python
def multiply_by_two(val):
    return val * 2
```"""

        healer = SelfHealingDebugger(llm=mock_llm)
        res = healer.heal_file(target_file=script, test_file=test_file, max_iterations=3)

        assert res.success is True
        assert "val * 2" in script.read_text(encoding="utf-8")

    # =========================================================================
    # 6. AGENT INTENT, PLANNER & TOOL ENGINE INTEGRATION
    # =========================================================================

    def test_agent_intent_extraction_for_self_healing(self):
        """Verify natural language requests classify as SELF_HEAL_CODE and RUN_CODE_TESTS."""
        r1 = ActionIntentAnalyzer.extract_intent("run the tests in this folder and fix whatever is broken")
        assert r1.intent == ActionIntentType.SELF_HEAL_CODE

        r2 = ActionIntentAnalyzer.extract_intent("self heal my project in data/workspace")
        assert r2.intent == ActionIntentType.SELF_HEAL_CODE

        r3 = ActionIntentAnalyzer.extract_intent("run pytest on tests/test_calc.py")
        assert r3.intent == ActionIntentType.RUN_CODE_TESTS

    def test_agent_planner_creates_self_healing_plan(self):
        """Verify AgentPlanner decomposes self-healing into structured multi-step plan."""
        registry = MagicMock(spec=ProjectRegistry)
        planner = AgentPlanner(project_registry=registry)

        state = planner.plan_task("run the tests in this folder and fix whatever is broken")
        assert state is not None
        actions = [s.action_type for s in state.steps]
        assert "PREPARE_DEBUG_ENVIRONMENT" in actions
        assert "SELF_HEAL_CODE" in actions
        assert "VERIFY_HEALED_STATE" in actions

    def test_tool_engine_run_code_tests_tool(self, tmp_path):
        """Verify ToolEngine execute_tool for run_code_tests."""
        test_script = tmp_path / "test_simple.py"
        test_script.write_text("def test_ok():\n    assert 1 + 1 == 2\n", encoding="utf-8")

        mock_comp = MagicMock()
        mock_fs = MagicMock()
        mock_term = MagicMock()
        mock_br = MagicMock()
        mock_apps = MagicMock()
        mock_scr = MagicMock()
        mock_proj = MagicMock()

        engine = ToolEngine(mock_comp, mock_fs, mock_term, mock_br, mock_apps, mock_scr, mock_proj)
        res = engine.execute_tool("run_code_tests", {"target": str(test_script)})
        assert res.success is True
        assert "PASSED" in res.message
