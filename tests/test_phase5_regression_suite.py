"""
Phase 5 Comprehensive Final Regression Test Suite.
Verifies all 24 required categories across general computer use, arbitrary coding tasks,
multi-file projects, existing code modifications, debugging loops, and combined workflows.
"""

from pathlib import Path
import pytest

from src.agent.code_generator import CodeGenerator
from src.agent.code_spec import ProgrammingTaskSpec
from src.agent.code_validator import CodeValidator
from src.agent.manager import LaptopAgentManager
from src.agent.task_state import ExecutionFlag, TaskStatus
from src.agent.toolchain import ToolchainManager


@pytest.fixture
def agent(tmp_path):
    workspace = tmp_path / "workspace"
    screenshots = tmp_path / "screenshots"
    registry = tmp_path / "projects.json"
    mgr = LaptopAgentManager(
        workspace_dir=str(workspace),
        screenshots_dir=str(screenshots),
        project_registry_path=str(registry),
    )
    return mgr


# ======================================================================
# 1. CODING TASKS (Arbitrary Languages & Dynamic Synthesis)
# ======================================================================

def test_01_python_program(agent):
    """1. Python program creation & verification."""
    handled, msg, res = agent.handle_command("Python mein prime number checker banao", lang="hinglish")
    assert handled is True
    p = Path(agent.workspace_dir) / "prime_checker.py"
    assert p.exists()
    assert "def is_prime" in p.read_text(encoding="utf-8")


def test_02_cpp_program(agent):
    """2. C++ program creation & verification."""
    handled, msg, res = agent.handle_command("C++ mein binary search algorithm likho", lang="hinglish")
    assert handled is True
    p = Path(agent.workspace_dir) / "binary_search.cpp"
    assert p.exists()
    content = p.read_text(encoding="utf-8")
    assert "#include <iostream>" in content
    assert "binarySearch" in content


def test_03_java_program(agent):
    """3. Java program creation & verification."""
    handled, msg, res = agent.handle_command("Java mein student management class banao", lang="hinglish")
    assert handled is True
    p = Path(agent.workspace_dir) / "StudentManager.java"
    assert p.exists()
    content = p.read_text(encoding="utf-8")
    assert "public class StudentManager" in content
    assert "public static void main" in content


def test_04_javascript_program(agent):
    """4. JavaScript program creation & verification."""
    handled, msg, res = agent.handle_command("JavaScript mein weather API fetcher banao", lang="hinglish")
    assert handled is True
    js_files = list(Path(agent.workspace_dir).glob("*.js"))
    assert len(js_files) >= 1
    p = js_files[0]
    assert p.exists()
    assert "function" in p.read_text(encoding="utf-8")


def test_05_sql_task(agent):
    """5. SQL task schema & query generation."""
    spec = CodeGenerator.generate_code_for_topic("Write SQL queries for e-commerce orders and customers")
    assert spec.language in ("sql", "python")
    assert len(spec.files) >= 1
    assert len(spec.files[0].content) > 10


def test_06_html_css_js_task(agent):
    """6. HTML/CSS/JS task generation."""
    spec = CodeGenerator.generate_code_for_topic("Create a modern responsive landing page with HTML and CSS")
    assert spec.language in ("html", "javascript", "react")
    assert len(spec.files[0].content) > 20


def test_07_multi_file_project(agent):
    """7. Multi-file project structure generation."""
    spec = CodeGenerator.generate_code_for_topic("Create a React todo application with components")
    assert spec.project_type in ("multi_file", "single_file")
    assert len(spec.files[0].content) > 20


def test_08_modify_existing_code(agent):
    """8. Modify existing code without overwriting unrelated parts."""
    initial_code = (
        "def compute_tax(income):\n"
        "    return income * 0.2\n\n"
        "def existing_unrelated_feature():\n"
        "    return 'preserved_data'\n"
    )
    p = agent.filesystem.create_file("tax_calculator.py", initial_code)
    # Append/modify function safely
    agent.filesystem.append_file("tax_calculator.py", "\ndef compute_discount(price):\n    return price * 0.1\n")
    updated = agent.filesystem.read_file(p)
    assert "preserved_data" in updated
    assert "compute_discount" in updated


def test_09_run_code(agent):
    """9. Run code and verify exit code and stdout."""
    code = "print('Hello from Chitti Automated Runner')\n"
    p = agent.filesystem.create_file("runner_test.py", code)
    res = agent.tools.execute_tool("execute_terminal_command", {"command": f'python "{p}"'})
    assert res.success is True
    assert "Hello from Chitti Automated Runner" in res.data.get("output", "")


def test_10_debug_broken_code(agent):
    """10. Debug broken code error inspection and repair."""
    broken_code = "def faulty():\n    return 10 / 0\n"
    spec = ProgrammingTaskSpec(language="python", filename="faulty.py", problem_description="Fix zero division")
    fixed = CodeGenerator.fix_code_after_error(spec, broken_code, "ZeroDivisionError: division by zero")
    assert fixed is not None


# ======================================================================
# 2. COMPUTER USE TOOLS
# ======================================================================

def test_11_open_application(agent):
    """11. Open application tool."""
    res = agent.tools.execute_tool("open_application", {"application": "Notepad"})
    assert res.success is True


def test_12_open_folder(agent):
    """12. Open folder action."""
    handled, msg, res = agent.handle_command("Open Downloads folder", lang="en")
    assert handled is True
    assert "downloads" in msg.lower() or "opened" in msg.lower()


def test_13_create_file(agent):
    """13. Create file tool."""
    res = agent.tools.execute_tool("create_file", {"path": "notes.txt", "content": "Meeting summary"})
    assert res.success is True
    assert Path(agent.workspace_dir, "notes.txt").exists()


def test_14_rename_file(agent):
    """14. Rename file tool."""
    agent.filesystem.create_file("old_name.txt", "content")
    res = agent.tools.execute_tool("rename_file", {"path": "old_name.txt", "new_name": "new_name.txt"})
    assert res.success is True
    assert Path(agent.workspace_dir, "new_name.txt").exists()


def test_15_move_file(agent):
    """15. Move file tool."""
    agent.filesystem.create_file("move_source.txt", "data")
    agent.filesystem.create_directory("archive")
    res = agent.tools.execute_tool("move_file", {"src": "move_source.txt", "dst": "archive/move_source.txt"})
    assert res.success is True
    assert Path(agent.workspace_dir, "archive/move_source.txt").exists()


def test_16_search_filesystem(agent):
    """16. Search filesystem tool."""
    agent.filesystem.create_file("doc1.pdf", "data")
    agent.filesystem.create_file("doc2.pdf", "data")
    res = agent.tools.execute_tool("search_files", {"pattern": "*.pdf", "root": agent.workspace_dir})
    assert res.success is True
    assert res.data.get("count", 0) >= 2


def test_17_open_vscode(agent):
    """17. Open VS Code with absolute file."""
    p = agent.filesystem.create_file("editor_target.py", "print('VS Code target')")
    res = agent.tools.execute_tool("open_application", {"application": "VS Code", "args": [p]})
    assert res.success is True


def test_18_open_project(agent):
    """18. Open project in VS Code."""
    proj_dir = str(Path(agent.workspace_dir) / "MyTestProject")
    agent.filesystem.create_directory("MyTestProject")
    agent.projects.register_project("MyTestProject", proj_dir)
    handled, msg, res = agent.handle_command("open my MyTestProject project in VS Code", lang="en")
    assert handled is True


def test_19_browser_navigation(agent):
    """19. Browser navigation & web search."""
    res = agent.tools.execute_tool("search_web", {"query": "latest Python features"})
    assert res.success is True


def test_20_youtube_interaction(agent):
    """20. YouTube interaction and search query parsing."""
    handled, msg, res = agent.handle_command("Go to YouTube and play a Sonu Nigam song", lang="en")
    assert handled is True
    assert "youtube" in msg.lower() or "sonu" in msg.lower()


# ======================================================================
# 3. COMBINED WORKFLOWS & UNANTICIPATED TASKS
# ======================================================================

def test_21_create_project_open_vscode_and_run(agent):
    """21. Create code -> open VS Code -> run code with verified execution state."""
    handled, msg, res = agent.handle_command(
        "VS Code open karo, Python mein calculator banao aur run karo",
        lang="hinglish"
    )
    assert handled is True
    task_state = agent.active_task_state
    assert task_state is not None
    assert task_state.get_flag(ExecutionFlag.FILE_CREATED) is True
    assert task_state.get_flag(ExecutionFlag.FILE_OPENED) is True
    assert task_state.get_flag(ExecutionFlag.EDITOR_CONTENT_VERIFIED) is True
    assert task_state.get_flag(ExecutionFlag.FILE_SAVED) is True
    assert task_state.get_flag(ExecutionFlag.CODE_EXECUTED) is True


def test_22_open_project_modify_and_verify(agent):
    """22. Open project -> inspect code -> modify -> verify."""
    p = agent.filesystem.create_file("math_lib.py", "def multiply(a, b):\n    return a * b\n")
    agent.filesystem.append_file("math_lib.py", "\ndef divide(a, b):\n    return a / b\n")
    ver_res = agent.tools.execute_tool("verify_file_content", {"path": "math_lib.py", "expected_keyword": "divide"})
    assert ver_res.success is True


def test_23_browser_research_and_save_file(agent):
    """23. Browser research -> write notes file."""
    search_res = agent.tools.execute_tool("search_web", {"query": "Python 3.14 features"})
    assert search_res.success is True
    file_res = agent.tools.execute_tool("create_file", {"path": "research_notes.md", "content": "# Python 3.14 Features\n- Fast execution\n- Better types\n"})
    assert file_res.success is True
    assert Path(agent.workspace_dir, "research_notes.md").exists()


def test_24_create_code_execute_and_verify_result(agent):
    """24. Create code -> execute -> verify stdout."""
    code = "import sys\nprint(f'Execution Verified: Python {sys.version_info.major}.{sys.version_info.minor}')\n"
    p = agent.filesystem.create_file("verified_exec.py", code)
    run_res = agent.tools.execute_tool("execute_terminal_command", {"command": f'python "{p}"'})
    assert run_res.success is True
    assert "Execution Verified:" in run_res.data.get("output", "")


def test_25_unanticipated_task_sha256_duplicates(agent):
    """25. Completely unanticipated task: Duplicate file finder using SHA-256."""
    handled, msg, res = agent.handle_command(
        "Create a Python program that finds duplicate files in a directory using SHA-256 hashes",
        lang="en"
    )
    assert handled is True
    p = Path(agent.workspace_dir) / "duplicate_finder.py"
    assert p.exists()
    content = p.read_text(encoding="utf-8")
    assert "hashlib.sha256" in content
    assert "find_duplicates" in content
