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
    filename: Optional[str] = None
    description: str = ""
    expected_markers: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.filename and self.path:
            self.filename = self.path


@dataclass
class ProgrammingTaskSpec:
    """Represents a structured, general-purpose programming and project building task."""
    task_type: str = "PROJECT_CREATION"  # PROJECT_CREATION, CODE_CREATION, CODE_MODIFICATION, DEBUG_FIX
    project_name: str = "project"
    language: str = "python"
    framework: Optional[str] = None
    project_type: str = "single_file"  # "web_app", "management_system", "rest_api", "automation_script", "desktop_gui", "ml_project", "algorithm", "multi_file"
    problem_description: str = ""
    requirements: List[str] = field(default_factory=list)
    filename: str = "main.py"
    files: List[CodeFileSpec] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    ui_required: bool = False
    execution_requested: bool = False
    application: str = "Visual Studio Code"
    expected_markers: List[str] = field(default_factory=list)
    expected_symbol: str = "main"
    entry_point: str = "main.py"
    target_directory: Optional[str] = None
    toolchain_cmd: Optional[str] = None
    compile_cmd: Optional[str] = None
    run_cmd: Optional[str] = None
    test_cmd: Optional[str] = None
    verification_strategy: str = "cli_output"  # "browser_ui", "cli_output", "http_endpoint", "gui_window", "file_output"
    is_existing_project: bool = False
    raw_input: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_type": self.task_type,
            "project_name": self.project_name,
            "language": self.language,
            "framework": self.framework,
            "project_type": self.project_type,
            "problem_description": self.problem_description,
            "requirements": self.requirements,
            "filename": self.filename,
            "files": [{"path": f.path, "description": f.description} for f in self.files],
            "dependencies": self.dependencies,
            "ui_required": self.ui_required,
            "execution_requested": self.execution_requested,
            "application": self.application,
            "expected_markers": self.expected_markers,
            "expected_symbol": self.expected_symbol,
            "entry_point": self.entry_point,
            "verification_strategy": self.verification_strategy,
            "is_existing_project": self.is_existing_project,
        }
