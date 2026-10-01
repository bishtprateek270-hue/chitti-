"""
Chitti Multi-Language Traceback & Compiler Error Parser.
Extracts structured error types, file paths, line numbers, failing code lines,
and failure root causes from Python, JavaScript, Pytest, and C/C++ tracebacks.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Union


@dataclass
class StackFrame:
    file_path: str
    line_number: int
    function_name: str = ""
    code_line: str = ""


@dataclass
class ParsedErrorInfo:
    """Structured representation of a runtime error or test failure."""
    error_type: str
    error_message: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    column_number: Optional[int] = None
    failing_code_snippet: Optional[str] = None
    stack_frames: List[StackFrame] = field(default_factory=list)
    raw_traceback: str = ""
    is_compilation_error: bool = False
    is_test_failure: bool = False
    suggested_fix_summary: str = ""

    def summary(self) -> str:
        loc = f" in {Path(self.file_path).name}:{self.line_number}" if self.file_path and self.line_number else ""
        return f"{self.error_type}{loc}: {self.error_message}"


class TracebackParser:
    """
    Parses compiler diagnostics, runtime tracebacks, and test runner outputs.
    """

    PYTHON_ERROR_TYPES = [
        "SyntaxError", "IndentationError", "TabError", "NameError", "TypeError",
        "ValueError", "IndexError", "KeyError", "AttributeError", "ImportError",
        "ModuleNotFoundError", "ZeroDivisionError", "FileNotFoundError",
        "PermissionError", "AssertionError", "RuntimeError", "RecursionError",
        "UnboundLocalError", "NotImplementedError", "TimeoutError",
    ]

    @classmethod
    def parse(cls, output_text: str, default_file: Optional[str] = None) -> ParsedErrorInfo:
        """Parses output text to extract structured error details."""
        if not output_text or not output_text.strip():
            return ParsedErrorInfo(
                error_type="UnknownError",
                error_message="No error output recorded.",
                raw_traceback="",
            )

        text = output_text.strip()

        # 1. Check for Pytest test failure
        pytest_match = re.search(r"(?i)FAILED\s+([^\s:]+)(?:::([^\s]+))?\s+-\s+([A-Za-z0-9_]+)?(?::\s*(.*))?", text)
        if pytest_match:
            test_file = pytest_match.group(1)
            test_fn = pytest_match.group(2) or ""
            err_type = pytest_match.group(3) or "TestAssertionFailure"
            err_msg = pytest_match.group(4) or f"Test '{test_fn}' failed."

            # Look deeper into pytest traceback for assertion specifics
            sub_err = cls._parse_python_traceback(text, default_file or test_file)
            if sub_err.line_number:
                sub_err.is_test_failure = True
                if not sub_err.error_message:
                    sub_err.error_message = err_msg
                return sub_err

            return ParsedErrorInfo(
                error_type=err_type,
                error_message=err_msg,
                file_path=test_file,
                is_test_failure=True,
                raw_traceback=text,
                suggested_fix_summary=cls._suggest_fix(err_type, err_msg),
            )

        # 2. Check for Python Traceback
        if "Traceback (most recent call last):" in text or any(f"{et}:" in text for et in cls.PYTHON_ERROR_TYPES):
            py_err = cls._parse_python_traceback(text, default_file)
            if py_err.error_type != "UnknownError":
                return py_err

        # 3. Check for JavaScript / Node.js Error Trace
        if re.search(r"(?:TypeError|ReferenceError|SyntaxError|RangeError):", text) or "at " in text:
            js_err = cls._parse_javascript_traceback(text, default_file)
            if js_err.error_type != "UnknownError":
                return js_err

        # 4. Check for GCC / Clang / C++ Compiler Errors
        # Format: file.cpp:12:5: error: ...
        c_match = re.search(r"([^\s:]+\.[a-zA-Z0-9_]+):(\d+):(\d+):\s*(?:fatal\s+)?error:\s*(.*)", text)
        if c_match:
            f_path = c_match.group(1)
            l_num = int(c_match.group(2))
            c_num = int(c_match.group(3))
            msg = c_match.group(4).strip()
            return ParsedErrorInfo(
                error_type="CompilationError",
                error_message=msg,
                file_path=f_path,
                line_number=l_num,
                column_number=c_num,
                is_compilation_error=True,
                raw_traceback=text,
                suggested_fix_summary=cls._suggest_fix("CompilationError", msg),
            )

        # 5. Check for Rustc Compiler Errors
        rust_match = re.search(r"error(?:\[E\d+\])?:\s*(.*)\n\s*-->\s*([^:]+):(\d+):(\d+)", text)
        if rust_match:
            msg = rust_match.group(1).strip()
            f_path = rust_match.group(2).strip()
            l_num = int(rust_match.group(3))
            c_num = int(rust_match.group(4))
            return ParsedErrorInfo(
                error_type="RustcCompilationError",
                error_message=msg,
                file_path=f_path,
                line_number=l_num,
                column_number=c_num,
                is_compilation_error=True,
                raw_traceback=text,
                suggested_fix_summary=cls._suggest_fix("RustcCompilationError", msg),
            )

        # 6. Fallback generic parsing
        last_line = [l.strip() for l in text.splitlines() if l.strip()][-1] if text else "Unknown Error"
        return ParsedErrorInfo(
            error_type="RuntimeError",
            error_message=last_line,
            file_path=default_file,
            raw_traceback=text,
            suggested_fix_summary=cls._suggest_fix("RuntimeError", last_line),
        )

    @classmethod
    def _parse_python_traceback(cls, text: str, default_file: Optional[str] = None) -> ParsedErrorInfo:
        """Extracts frames, line numbers, and error name/message from Python output."""
        frames: List[StackFrame] = []
        
        # Extract all stack frames: File "path", line 12, in func\n  code
        frame_matches = re.finditer(
            r'File\s+"([^"]+)",\s+line\s+(\d+)(?:,\s+in\s+([^\n]+))?(?:\n\s+(.+))?',
            text
        )
        for m in frame_matches:
            fpath = m.group(1)
            lineno = int(m.group(2))
            fn = m.group(3) or ""
            code_line = (m.group(4) or "").strip()
            frames.append(StackFrame(file_path=fpath, line_number=lineno, function_name=fn, code_line=code_line))

        # Extract Error Type & Message at end of traceback
        # e.g., "TypeError: unsupported operand type(s)..."
        err_type = "RuntimeError"
        err_msg = ""
        
        # Check standard error line
        err_match = re.search(r"\n?([A-Za-z0-9_]+Error|[A-Za-z0-9_]+Exception):\s*(.*)", text)
        if err_match:
            err_type = err_match.group(1)
            err_msg = err_match.group(2).strip()
        elif "AssertionError" in text:
            err_type = "AssertionError"
            ass_msg = re.search(r"AssertionError(?::\s*(.*))?", text)
            err_msg = ass_msg.group(1).strip() if ass_msg and ass_msg.group(1) else "Assertion failed"

        # If SyntaxError with caret indicator (^)
        caret_match = re.search(r'File\s+"([^"]+)",\s+line\s+(\d+)\n\s*(.+)\n\s*(\^)', text)
        if caret_match:
            syntax_file = caret_match.group(1)
            syntax_line = int(caret_match.group(2))
            syntax_code = caret_match.group(3).strip()
            return ParsedErrorInfo(
                error_type="SyntaxError",
                error_message=err_msg or "Invalid syntax",
                file_path=syntax_file,
                line_number=syntax_line,
                failing_code_snippet=syntax_code,
                stack_frames=frames,
                raw_traceback=text,
                suggested_fix_summary=cls._suggest_fix("SyntaxError", err_msg),
            )

        # Most relevant frame is usually the last user frame
        target_file = default_file
        target_line = None
        target_code = None

        if frames:
            # Pick the last frame that doesn't point inside standard library if possible
            last_frame = frames[-1]
            target_file = last_frame.file_path
            target_line = last_frame.line_number
            target_code = last_frame.code_line

        return ParsedErrorInfo(
            error_type=err_type,
            error_message=err_msg or (text.splitlines()[-1] if text else "Unknown failure"),
            file_path=target_file,
            line_number=target_line,
            failing_code_snippet=target_code,
            stack_frames=frames,
            raw_traceback=text,
            is_test_failure="AssertionError" in err_type,
            suggested_fix_summary=cls._suggest_fix(err_type, err_msg),
        )

    @classmethod
    def _parse_javascript_traceback(cls, text: str, default_file: Optional[str] = None) -> ParsedErrorInfo:
        """Parses Node.js / JavaScript runtime traces."""
        # e.g., TypeError: Cannot read properties of undefined (reading 'x')
        #       at foo (/path/to/file.js:15:9)
        err_match = re.search(r"([A-Za-z0-9_]+Error):\s*(.*)", text)
        err_type = err_match.group(1) if err_match else "JavaScriptError"
        err_msg = err_match.group(2).strip() if err_match else ""

        frame_match = re.search(r"at\s+(?:([^\s(]+)\s+)?\(?([^\s:()]+):(\d+):(\d+)\)?", text)
        f_path = default_file
        l_num = None
        c_num = None
        if frame_match:
            f_path = frame_match.group(2)
            l_num = int(frame_match.group(3))
            c_num = int(frame_match.group(4))

        return ParsedErrorInfo(
            error_type=err_type,
            error_message=err_msg or "JavaScript runtime exception",
            file_path=f_path,
            line_number=l_num,
            column_number=c_num,
            raw_traceback=text,
            suggested_fix_summary=cls._suggest_fix(err_type, err_msg),
        )

    @classmethod
    def _suggest_fix(cls, error_type: str, message: str) -> str:
        """Provides heuristic guidance on how to fix common errors."""
        msg = (message or "").lower()
        if error_type == "ZeroDivisionError":
            return "Add a guard check to ensure divisor is non-zero before performing division."
        elif error_type in ("NameError", "UnboundLocalError"):
            return "Verify that the variable or imported identifier is defined and in scope before usage."
        elif error_type in ("ImportError", "ModuleNotFoundError"):
            return "Ensure the required module is installed and package import path is correct."
        elif error_type == "TypeError":
            if "unsupported operand" in msg:
                return "Convert or cast variables to compatible types before binary operation."
            elif "missing" in msg and "argument" in msg:
                return "Check function signature and provide all required positional/keyword arguments."
            elif "'nonetype'" in msg:
                return "Add null/None check before calling methods or accessing attributes."
            return "Verify argument types match function signature."
        elif error_type in ("IndexError", "KeyError"):
            return "Add length bounds check or use safe .get() accessor before indexing."
        elif error_type == "AttributeError":
            return "Verify object instance is not None and attribute name is spelled correctly."
        elif error_type == "SyntaxError":
            return "Fix syntax anomaly, missing colon, unmatched parentheses, or indentation."
        elif error_type in ("AssertionError", "TestAssertionFailure"):
            return "Update function return value or calculation logic to satisfy expected test assertion."
        elif error_type == "CompilationError":
            return "Fix compilation syntax, missing headers, or type mismatches."
        return "Inspect failing line and add error handling or fix logic flow."
