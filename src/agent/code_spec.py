"""
Chitti Programming Task & Project Specification (Phase 6).
Represents general-purpose coding tasks, target languages, multi-file projects,
dynamic requirements extraction, and execution/verification configurations.
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
class ProjectSpecification:
    """
    Structured project specification representing a dynamic project build plan.
    Generated dynamically for every project from natural language requirements.
    """
    project_name: str = "project"
    project_type: str = "single_file"  # "web_app", "management_system", "rest_api", "automation_script", "desktop_gui", "ml_project", "multi_file", "single_file"
    goal: str = ""
    language: str = "python"
    framework: Optional[str] = None
    ui_required: bool = False
    requirements: List[str] = field(default_factory=list)
    features: List[str] = field(default_factory=list)
    data_storage: Optional[str] = None  # "localStorage", "json_file", "in_memory", "sqlite"
    dependencies: List[str] = field(default_factory=list)
    directory: Optional[str] = None
    files: List[CodeFileSpec] = field(default_factory=list)
    entry_point: str = "main.py"
    run_command: Optional[str] = None
    test_command: Optional[str] = None
    build_command: Optional[str] = None
    execution_requested: bool = False
    verification_strategy: str = "cli_output"  # "browser_ui", "cli_output", "http_endpoint", "gui_window"
    technology_choice_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_name": self.project_name,
            "project_type": self.project_type,
            "goal": self.goal,
            "language": self.language,
            "framework": self.framework,
            "ui_required": self.ui_required,
            "requirements": self.requirements,
            "features": self.features,
            "data_storage": self.data_storage,
            "dependencies": self.dependencies,
            "directory": self.directory,
            "files": [{"path": f.path, "description": f.description} for f in self.files],
            "entry_point": self.entry_point,
            "run_command": self.run_command,
            "test_command": self.test_command,
            "build_command": self.build_command,
            "execution_requested": self.execution_requested,
            "verification_strategy": self.verification_strategy,
            "technology_choice_reason": self.technology_choice_reason,
        }


@dataclass
class ProgrammingTaskSpec:
    """
    Represents a structured, general-purpose programming and project building task.
    Compatible with both single-file operations and full ProjectSpecification.
    """
    task_type: str = "PROJECT_CREATION"  # PROJECT_CREATION, CODE_CREATION, CODE_MODIFICATION, DEBUG_FIX
    project_name: str = "project"
    language: str = "python"
    framework: Optional[str] = None
    project_type: str = "single_file"  # "web_app", "management_system", "rest_api", "automation_script", "desktop_gui", "ml_project", "algorithm", "multi_file"
    problem_description: str = ""
    requirements: List[str] = field(default_factory=list)
    features: List[str] = field(default_factory=list)
    data_storage: Optional[str] = None
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
    run_command: Optional[str] = None
    test_cmd: Optional[str] = None
    test_command: Optional[str] = None
    build_cmd: Optional[str] = None
    build_command: Optional[str] = None
    verification_strategy: str = "cli_output"  # "browser_ui", "cli_output", "http_endpoint", "gui_window", "file_output"
    technology_choice_reason: str = ""
    is_existing_project: bool = False
    raw_input: str = ""

    def __post_init__(self):
        if not self.run_command and self.run_cmd:
            self.run_command = self.run_cmd
        elif not self.run_cmd and self.run_command:
            self.run_cmd = self.run_command
        if not self.test_command and self.test_cmd:
            self.test_command = self.test_cmd
        elif not self.test_cmd and self.test_command:
            self.test_cmd = self.test_command
        if not self.build_command and self.build_cmd:
            self.build_command = self.build_cmd
        elif not self.build_cmd and self.build_command:
            self.build_cmd = self.build_command

    def to_project_spec(self) -> ProjectSpecification:
        return ProjectSpecification(
            project_name=self.project_name,
            project_type=self.project_type,
            goal=self.problem_description,
            language=self.language,
            framework=self.framework,
            ui_required=self.ui_required,
            requirements=self.requirements,
            features=self.features,
            data_storage=self.data_storage,
            dependencies=self.dependencies,
            directory=self.target_directory,
            files=self.files,
            entry_point=self.entry_point or self.filename,
            run_command=self.run_command or self.run_cmd,
            test_command=self.test_command or self.test_cmd,
            build_command=self.build_command or self.build_cmd,
            execution_requested=self.execution_requested,
            verification_strategy=self.verification_strategy,
            technology_choice_reason=self.technology_choice_reason,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_type": self.task_type,
            "project_name": self.project_name,
            "language": self.language,
            "framework": self.framework,
            "project_type": self.project_type,
            "problem_description": self.problem_description,
            "requirements": self.requirements,
            "features": self.features,
            "data_storage": self.data_storage,
            "filename": self.filename,
            "files": [{"path": f.path, "description": f.description} for f in self.files],
            "dependencies": self.dependencies,
            "ui_required": self.ui_required,
            "execution_requested": self.execution_requested,
            "application": self.application,
            "expected_markers": self.expected_markers,
            "expected_symbol": self.expected_symbol,
            "entry_point": self.entry_point,
            "run_command": self.run_command or self.run_cmd,
            "test_command": self.test_command or self.test_cmd,
            "build_command": self.build_command or self.build_cmd,
            "verification_strategy": self.verification_strategy,
            "technology_choice_reason": self.technology_choice_reason,
            "is_existing_project": self.is_existing_project,
        }
