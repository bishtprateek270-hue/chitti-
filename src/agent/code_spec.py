"""
Chitti Programming Task Specification.
Represents general-purpose coding tasks, target languages, multi-file projects, and execution requirements.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CodeFileSpec:
    """Represents a single file in a generated project."""
    path: str
    content: str
    description: str = ""
    expected_markers: List[str] = field(default_factory=list)


@dataclass
class ProgrammingTaskSpec:
    """Represents a structured, general-purpose programming task."""
    task_type: str = "CODE_CREATION"  # CODE_CREATION, PROJECT_CREATION, CODE_MODIFICATION, DEBUG_FIX
    language: str = "python"
    framework: Optional[str] = None
    project_type: str = "single_file"  # "single_file" | "multi_file"
    problem_description: str = ""
    requirements: List[str] = field(default_factory=list)
    filename: str = "main.py"
    files: List[CodeFileSpec] = field(default_factory=list)
    execution_requested: bool = False
    application: str = "Visual Studio Code"
    expected_markers: List[str] = field(default_factory=list)
    expected_symbol: str = "main"
    entry_point: str = "main.py"
    toolchain_cmd: Optional[str] = None
    compile_cmd: Optional[str] = None
    run_cmd: Optional[str] = None
    raw_input: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_type": self.task_type,
            "language": self.language,
            "framework": self.framework,
            "project_type": self.project_type,
            "problem_description": self.problem_description,
            "requirements": self.requirements,
            "filename": self.filename,
            "files": [{"path": f.path, "description": f.description} for f in self.files],
            "execution_requested": self.execution_requested,
            "application": self.application,
            "expected_markers": self.expected_markers,
            "expected_symbol": self.expected_symbol,
        }
