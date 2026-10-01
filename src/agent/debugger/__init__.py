"""
Chitti Autonomous Multi-File Code Self-Healing & Debugging Package (Phase 2).
Provides execution sandboxing, traceback parsing, automated test generation,
AST-aware patching, and iterative self-healing feedback loops.
"""

from src.agent.debugger.runner import CodeExecutionSandbox, ExecutionResult
from src.agent.debugger.traceback_parser import TracebackParser, ParsedErrorInfo
from src.agent.debugger.patcher import CodePatcher, PatchResult
from src.agent.debugger.test_generator import TestGenerator, GeneratedTestSuite
from src.agent.debugger.self_healer import SelfHealingDebugger, HealingResult

__all__ = [
    "CodeExecutionSandbox",
    "ExecutionResult",
    "TracebackParser",
    "ParsedErrorInfo",
    "CodePatcher",
    "PatchResult",
    "TestGenerator",
    "GeneratedTestSuite",
    "SelfHealingDebugger",
    "HealingResult",
]
