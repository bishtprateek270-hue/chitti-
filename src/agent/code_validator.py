"""
Chitti Code Quality & Semantic Validator.
Validates syntax, structural completeness, requirement coverage, and detects placeholders across languages.
"""

import ast
import re
from dataclasses import dataclass, field
from typing import List, Optional

from src.brain.llm import BaseLLM
from src.utils.logging import log_debug, log_info, log_warn


@dataclass
class ValidationReport:
    valid: bool
    issues: List[str] = field(default_factory=list)
    requirements_covered: bool = True
    language: str = "python"


class CodeValidator:
    """Validates code syntax, structural integrity, and completeness across arbitrary programming languages."""

    PLACEHOLDER_PATTERNS = [
        r"(?i)\bTODO\b",
        r"(?i)\bFIXME\b",
        r"(?i)\bplaceholder\b",
        r"(?i)\bnot implemented\b",
        r"(?i)\bpass\s*#\s*implement\b",
        r"(?i)\bthrow new NotImplementedException\b",
        r"(?i)\bunimplemented!\(\)",
    ]

    @classmethod
    def validate_code(
        cls,
        code: str,
        language: str = "python",
        expected_markers: Optional[List[str]] = None,
        llm: Optional[BaseLLM] = None,
    ) -> ValidationReport:
        """Performs static & structural checks on generated code."""
        issues: List[str] = []
        lang = language.lower().strip()
        markers = expected_markers or []

        # 1. Non-empty check
        if not code or not code.strip():
            return ValidationReport(valid=False, issues=["Code content is completely empty."], requirements_covered=False, language=lang)

        # 2. Check for lazy placeholders
        for pat in cls.PLACEHOLDER_PATTERNS:
            if re.search(pat, code):
                issues.append(f"Code contains unresolved placeholder pattern: '{pat}'")

        # 3. Language-specific syntax / structure checks
        if lang in ("python", "py"):
            try:
                tree = ast.parse(code)
                # Check for empty functions/classes with only `pass` or `...`
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                            issues.append(f"Function '{node.name}' has empty body ('pass').")
            except SyntaxError as e:
                issues.append(f"Python syntax error at line {e.lineno}: {e.msg}")

        elif lang in ("cpp", "c", "java", "javascript", "typescript", "rust", "csharp"):
            # Bracket balance check
            open_braces = code.count("{")
            close_braces = code.count("}")
            if open_braces != close_braces:
                issues.append(f"Mismatched curly braces in {lang} code: {open_braces} open vs {close_braces} close.")

            open_parens = code.count("(")
            close_parens = code.count(")")
            if open_parens != close_parens:
                issues.append(f"Mismatched parentheses in {lang} code: {open_parens} open vs {close_parens} close.")

            # Check language-specific essential keywords
            if lang in ("cpp", "c") and "#include" not in code:
                issues.append(f"C/C++ code is missing standard library #include header.")
            elif lang == "java" and "class " not in code:
                issues.append(f"Java code is missing class definition.")
            elif lang == "rust" and "fn " not in code:
                issues.append(f"Rust code is missing function definitions.")

        # 4. Marker presence check
        missing_markers = []
        for m in markers:
            if m.lower() not in code.lower():
                missing_markers.append(m)

        if missing_markers and len(missing_markers) == len(markers):
            issues.append(f"Code is missing all expected structural markers: {missing_markers}")

        is_valid = len(issues) == 0
        return ValidationReport(
            valid=is_valid,
            issues=issues,
            requirements_covered=len(missing_markers) < len(markers),
            language=lang,
        )
