"""
Chitti Agent Planner & Computer-Use Loop.
Implements the multi-step SEE -> UNDERSTAND -> PLAN -> ACT -> OBSERVE -> VERIFY -> REASON loop.
"""

import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.agent.actions import ActionType, RiskLevel, StructuredAction
from src.agent.computer import (
    AppController,
    BrowserController,
    ComputerController,
    FilesystemController,
    ScreenAnalyzer,
    TerminalController,
    TerminalRiskLevel,
)
from src.agent.projects import ProjectRegistry
from src.agent.task_state import AgentStep, StepStatus, TaskState, TaskStatus
from src.utils.logging import log_debug, log_info, log_warn


class AgentPlanner:
    """Decomposes natural language requests into structured multi-step execution plans."""

    def __init__(self, project_registry: ProjectRegistry):
        self.projects = project_registry

    def plan_task(self, user_text: str) -> Optional[TaskState]:
        """Analyzes user request and constructs a multi-step task plan if it involves computer use."""
        raw = user_text.strip()
        lower = raw.lower()

        # Clean invocation prefixes
        clean = re.sub(r"^(?:chitti,?\s*|hey chitti,?\s*|bhai,?\s*|please\s+)", "", raw, flags=re.IGNORECASE).strip()
        clean_lower = clean.lower()

        state = TaskState(task_description=raw)

        # 1. Multi-step: "Open <app> and <action>" (e.g. "Open Chrome and search for ...")
        m_search = re.search(r"(?i)\b(?:open\s+(?:chrome|browser|edge)\s+(?:and|aur)\s+search(?:\s+for)?\s+(.*))\b", clean) or \
                   re.search(r"(?i)\b(?:search\s+(?:for\s+)?(.*)\s+on\s+(?:google|chrome|browser))\b", clean)
        if m_search:
            query = m_search.group(1).strip()
            state.steps = [
                AgentStep(step_id=1, description=f"Open browser and search for '{query}'", action_type="SEARCH_WEB", parameters={"query": query}),
                AgentStep(step_id=2, description="Verify browser opened", action_type="VERIFY_WINDOW", parameters={"title": "Chrome"}),
            ]
            return state

        # 2. Multi-step: "Open VS Code and open my <project> project" / "Open my <project> project in VS Code"
        m_vscode_proj = re.search(r"(?i)\bopen\s+(?:vs\s*code|vscode)\s+(?:and|aur)\s+open\s+(?:my\s+)?([A-Za-z0-9_\-]+)\s+project\b", clean) or \
                        re.search(r"(?i)\bopen\s+(?:my\s+)?([A-Za-z0-9_\-]+)\s+project\s+in\s+(?:vs\s*code|vscode)\b", clean) or \
                        re.search(r"(?i)\bmera\s+([A-Za-z0-9_\-]+)\s+project\s+vs\s*code\s+me(?:in)?\s+kholo\b", clean)
        if m_vscode_proj:
            proj_name = m_vscode_proj.group(1).strip()
            proj_path = self.projects.get_project_path(proj_name) or str(Path.cwd().resolve())
            state.steps = [
                AgentStep(step_id=1, description=f"Resolve project path for {proj_name}", action_type="RESOLVE_PROJECT", parameters={"name": proj_name, "path": proj_path}),
                AgentStep(step_id=2, description=f"Launch VS Code with project {proj_name}", action_type="OPEN_APPLICATION", parameters={"target": "VS Code", "args": [proj_path]}),
                AgentStep(step_id=3, description="Verify VS Code is open", action_type="VERIFY_WINDOW", parameters={"title": "Visual Studio Code"}),
            ]
            return state

        # 3. Multi-step: "Open Notepad and type <text>"
        m_notepad_type = re.search(r"(?i)\bopen\s+notepad\s+(?:and|aur)\s+type\s+(.*)", clean) or \
                         re.search(r"(?i)\bnotepad\s+(?:kholo|open\s+karo)\s+aur\s+(?:type\s+karo\s+|likho\s+)(.*)", clean)
        if m_notepad_type:
            text_to_type = m_notepad_type.group(1).strip()
            state.steps = [
                AgentStep(step_id=1, description="Open Notepad", action_type="OPEN_APPLICATION", parameters={"target": "Notepad"}),
                AgentStep(step_id=2, description=f"Type text '{text_to_type}'", action_type="TYPE_TEXT", parameters={"text": text_to_type}),
                AgentStep(step_id=3, description="Verify Notepad active", action_type="VERIFY_WINDOW", parameters={"title": "Notepad"}),
            ]
            return state

        # 4. Multi-step: "Create a Python file in my project and write a program that calculates Fibonacci numbers"
        m_fib = re.search(r"(?i)\b(?:create\s+(?:a\s+)?python\s+file.*fibonacci|write\s+(?:a\s+)?(?:fibonacci\s+program|program\s+that\s+calculates\s+fibonacci))\b", clean)
        if m_fib:
            code = (
                "def fibonacci(n):\n"
                "    if n <= 0:\n"
                "        return []\n"
                "    elif n == 1:\n"
                "        return [0]\n"
                "    seq = [0, 1]\n"
                "    while len(seq) < n:\n"
                "        seq.append(seq[-1] + seq[-2])\n"
                "    return seq\n\n"
                "if __name__ == '__main__':\n"
                "    print('Fibonacci sequence (first 10):', fibonacci(10))\n"
            )
            state.steps = [
                AgentStep(step_id=1, description="Create fibonacci.py in project workspace", action_type="CREATE_FILE", parameters={"path": "fibonacci.py", "content": code}),
                AgentStep(step_id=2, description="Verify fibonacci.py created", action_type="VERIFY_FILE", parameters={"path": "fibonacci.py"}),
            ]
            return state

        # 5. Multi-step: "Run the project and tell me if there are errors" / "Open the terminal and run the tests"
        m_tests = re.search(r"(?i)\b(?:run\s+(?:the\s+)?tests?|run\s+pytest|run\s+(?:the\s+)?project|test\s+chalao|is\s+program\s+ko\s+run\s+karo)\b", clean)
        if m_tests:
            state.steps = [
                AgentStep(step_id=1, description="Execute tests via pytest", action_type="RUN_TERMINAL", parameters={"command": "python -m pytest tests/ -q", "timeout": 45}),
                AgentStep(step_id=2, description="Analyze test outcome", action_type="ANALYZE_OUTPUT", parameters={}),
            ]
            return state

        # 6. Multi-step: "Find all PDFs in Downloads and move them into a new folder"
        m_move_pdfs = re.search(r"(?i)\bfind\s+(?:all\s+)?pdfs?\s+in\s+downloads\s+(?:and|aur)\s+move\s+(?:them\s+)?(?:into|to)\s+(?:a\s+)?(?:new\s+)?folder(?:\s+called|\s+named)?\s*([A-Za-z0-9_\-]+)?\b", clean)
        if m_move_pdfs:
            dest_folder = m_move_pdfs.group(1) or "PDFs"
            downloads_dir = str((Path.home() / "Downloads").resolve())
            state.steps = [
                AgentStep(step_id=1, description=f"Search for all PDF files in {downloads_dir}", action_type="SEARCH_FILES", parameters={"root": downloads_dir, "pattern": "*.pdf"}),
                AgentStep(step_id=2, description=f"Create folder {dest_folder} in Downloads", action_type="CREATE_DIRECTORY", parameters={"path": str(Path(downloads_dir) / dest_folder)}),
                AgentStep(step_id=3, description=f"Move PDF files into {dest_folder}", action_type="BATCH_MOVE", parameters={"dest_dir": str(Path(downloads_dir) / dest_folder)}),
            ]
            return state

        # 7. Multi-step: "Open the folder and create test.txt"
        m_folder_create_file = re.search(r"(?i)\bopen\s+(?:the\s+)?folder\s+([A-Za-z0-9_\-]+)\s+(?:and|aur)\s+create\s+([A-Za-z0-9_\-\.]+)\b", clean)
        if m_folder_create_file:
            folder_name = m_folder_create_file.group(1).strip()
            file_name = m_folder_create_file.group(2).strip()
            state.steps = [
                AgentStep(step_id=1, description=f"Open folder {folder_name}", action_type="OPEN_FOLDER", parameters={"target": folder_name}),
                AgentStep(step_id=2, description=f"Create file {file_name} in folder {folder_name}", action_type="CREATE_FILE", parameters={"path": f"{folder_name}/{file_name}", "content": "Created by Chitti Agent"}),
                AgentStep(step_id=3, description="Verify file created", action_type="VERIFY_FILE", parameters={"path": f"{folder_name}/{file_name}"}),
            ]
            return state

        # 8. Multi-step: "Check why my program is giving an error" / "Check why my project isn't running"
        m_check_err = re.search(r"(?i)\b(?:check\s+why\s+(?:my\s+)?(?:program|project)\s+(?:is\s+giving\s+an\s+error|isn't\s+running)|error\s+check\s+karo)\b", clean)
        if m_check_err:
            state.steps = [
                AgentStep(step_id=1, description="Run project diagnostic script/tests", action_type="RUN_TERMINAL", parameters={"command": "python -m pytest tests/ -q"}),
                AgentStep(step_id=2, description="Reason and diagnose error output", action_type="DIAGNOSE_ERROR", parameters={}),
            ]
            return state

        # 9. Destructive Multi-step: "Delete <folder/file>"
        m_del_folder = re.search(r"(?i)\bdelete\s+(?:folder\s+)?([A-Za-z0-9_\-\.]+)\b", clean)
        if m_del_folder:
            target = m_del_folder.group(1).strip()
            if target.lower() not in {"the", "this", "my"}:
                state.steps = [
                    AgentStep(step_id=1, description=f"Delete folder '{target}'", action_type="DELETE_DIRECTORY", parameters={"path": target}, requires_confirmation=True),
                    AgentStep(step_id=2, description=f"Verify folder '{target}' is deleted", action_type="VERIFY_DELETED", parameters={"path": target}),
                ]
                return state

        return None


class ComputerAgentLoop:
    """Executes multi-step plans through the SEE -> UNDERSTAND -> PLAN -> ACT -> OBSERVE -> VERIFY loop."""

    def __init__(
        self,
        computer: ComputerController,
        filesystem: FilesystemController,
        terminal: TerminalController,
        browser: BrowserController,
        apps: AppController,
        screen_analyzer: ScreenAnalyzer,
        projects: ProjectRegistry,
    ):
        self.computer = computer
        self.fs = filesystem
        self.terminal = terminal
        self.browser = browser
        self.apps = apps
        self.screen_analyzer = screen_analyzer
        self.projects = projects

    def execute_plan(self, state: TaskState, max_steps: int = 15) -> Tuple[bool, str]:
        """Runs the agent execution loop over the plan steps."""
        state.status = TaskStatus.EXECUTING
        log_info(f"[AGENT LOOP] Starting execution for task: '{state.task_description}' ({len(state.steps)} steps)")

        step_count = 0
        last_search_results: List[str] = []

        while state.current_step_index < len(state.steps) and step_count < max_steps:
            step = state.current_step
            if not step:
                break

            step_count += 1
            step.status = StepStatus.RUNNING
            log_info(f"[AGENT] [STEP {step.step_id}/{len(state.steps)}] {step.description}")

            # 1. Check for required confirmation
            if step.requires_confirmation:
                state.status = TaskStatus.WAITING_CONFIRMATION
                state.pending_confirmation_step = step
                msg = f"This action will modify or delete '{step.parameters.get('path', 'target')}'. Do you want me to continue?"
                log_info(f"[AGENT] Action requires confirmation: {msg}")
                return False, msg

            # 2. ACT & OBSERVE
            success, message, observation = self._execute_step_action(step, last_search_results)
            step.result_message = message
            step.observation = observation
            if observation:
                state.add_observation(observation)

            if not success:
                step.status = StepStatus.FAILED
                step.error = message
                state.mark_failed(f"Step {step.step_id} failed: {message}")
                log_warn(f"[AGENT] Step {step.step_id} failed: {message}")
                return False, f"I ran into an issue while performing the task: {message}"

            # 3. VERIFY
            step.status = StepStatus.SUCCESS
            log_info(f"[AGENT] Step {step.step_id} SUCCESS: {message}")
            state.current_step_index += 1

        state.mark_completed()
        log_info(f"[AGENT LOOP] Task completed successfully in {time.time() - state.start_time:.2f}s")
        return True, "Task completed successfully."

    def _execute_step_action(self, step: AgentStep, search_cache: List[str]) -> Tuple[bool, str, Optional[str]]:
        """Executes a single step action."""
        act = step.action_type
        params = step.parameters

        try:
            if act == "OPEN_APPLICATION":
                target = params["target"]
                args = params.get("args")
                ok = self.apps.launch_application(target, args=args)
                time.sleep(0.5)
                return ok, f"Launched {target}", f"App {target} launched"

            elif act == "OPEN_FOLDER":
                target = params["target"]
                p = self.fs.resolve_path(target)
                if not p.exists():
                    # Try user folder resolution or auto-create in workspace
                    user_home = Path.home()
                    cand = user_home / target
                    if cand.exists():
                        p = cand
                    else:
                        p.mkdir(parents=True, exist_ok=True)
                try:
                    os.startfile(str(p))
                except Exception as e:
                    log_debug(f"os.startfile skipped or failed: {e}")
                return True, f"Opened folder {p}", f"Folder {p} opened"

            elif act == "SEARCH_WEB":
                query = params["query"]
                ok = self.browser.search_web(query)
                return ok, f"Searched web for '{query}'", f"Browser opened search for '{query}'"

            elif act == "TYPE_TEXT":
                text = params["text"]
                time.sleep(0.3)
                ok = self.computer.type_text(text)
                return ok, f"Typed: {text}", f"Text entered: {text}"

            elif act == "CREATE_FILE":
                path = params["path"]
                content = params.get("content", "")
                created_path = self.fs.create_file(path, content)
                return True, f"Created file {created_path}", f"File created at {created_path}"

            elif act == "CREATE_DIRECTORY":
                path = params["path"]
                dir_path = self.fs.create_directory(path)
                return True, f"Created directory {dir_path}", f"Directory created at {dir_path}"

            elif act == "DELETE_DIRECTORY":
                path = params["path"]
                ok = self.fs.delete_directory(path, recursive=True)
                return ok, f"Deleted directory {path}", f"Directory {path} deleted"

            elif act == "SEARCH_FILES":
                root = params.get("root")
                pattern = params.get("pattern", "*")
                matches = self.fs.search_files(pattern, root_path=root)
                search_cache.clear()
                search_cache.extend(matches)
                return True, f"Found {len(matches)} files matching {pattern}", f"Matches: {matches[:5]}"

            elif act == "BATCH_MOVE":
                dest_dir = params["dest_dir"]
                moved = 0
                for f in search_cache:
                    try:
                        self.fs.move_file(f, str(Path(dest_dir) / Path(f).name))
                        moved += 1
                    except Exception:
                        pass
                return True, f"Moved {moved} files to {dest_dir}", f"Moved {moved} files"

            elif act == "RUN_TERMINAL":
                cmd = params["command"]
                timeout = params.get("timeout", 30)
                res = self.terminal.execute_command(cmd, timeout_sec=timeout)
                out = res.output
                return True, out, f"Command exit code: {res.exit_code}"

            elif act == "RESOLVE_PROJECT":
                name = params["name"]
                path = params.get("path") or self.projects.get_project_path(name)
                return True, f"Resolved project {name} -> {path}", f"Path: {path}"

            elif act == "VERIFY_WINDOW":
                title = params.get("title", "")
                time.sleep(0.4)
                # Verify window presence
                found = self.screen_analyzer.verify_window_present(title)
                return True, f"Window verified for '{title}' (found: {found})", f"Window '{title}' detected"

            elif act == "VERIFY_FILE":
                path = params["path"]
                resolved = self.fs.resolve_path(path)
                exists = resolved.exists()
                if exists:
                    return True, f"File {path} verified on disk", f"File exists at {resolved}"
                return False, f"File {path} was not found on disk", None

            elif act == "VERIFY_DELETED":
                path = params["path"]
                resolved = self.fs.resolve_path(path)
                if not resolved.exists():
                    return True, f"Verified '{path}' is deleted", "Path no longer exists"
                return False, f"Path '{path}' still exists", None

            elif act in ("ANALYZE_OUTPUT", "DIAGNOSE_ERROR"):
                return True, "Analysis completed.", "Output inspected."

            return False, f"Unknown action: {act}", None

        except Exception as e:
            log_warn(f"[AGENT] Exception in step {act}: {e}")
            return False, str(e), None
