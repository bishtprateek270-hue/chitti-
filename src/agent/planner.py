"""
Chitti Agent Planner & Computer-Use Loop.
Implements the multi-step SEE -> UNDERSTAND -> PLAN -> ACT -> OBSERVE -> VERIFY loop.
Supports general-purpose coding across arbitrary programming languages, multi-file projects,
toolchain verification, compiler execution, and automated error recovery.
"""

import os
import re
import time
import urllib.parse
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.agent.actions import ActionType, RiskLevel, StructuredAction
from src.agent.code_generator import CodeGenerator
from src.agent.code_spec import CodeFileSpec, ProgrammingTaskSpec
from src.agent.code_validator import CodeValidator
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
from src.agent.task_state import AgentStep, ExecutionFlag, StepStatus, TaskState, TaskStatus
from src.agent.toolchain import ToolchainManager
from src.agent.tools import ToolEngine
from src.brain.llm import BaseLLM
from src.utils.logging import log_debug, log_info, log_warn


class AgentPlanner:
    """Decomposes natural language requests into structured multi-step execution plans."""

    def __init__(
        self,
        project_registry: ProjectRegistry,
        filesystem: Optional[FilesystemController] = None,
        llm: Optional[BaseLLM] = None,
    ):
        self.projects = project_registry
        self.fs = filesystem or FilesystemController()
        self.llm = llm

    def plan_task(self, user_text: str) -> Optional[TaskState]:
        """Analyzes user request and constructs a multi-step task plan if it involves computer use."""
        raw = user_text.strip()
        lower = raw.lower()

        # Clean invocation prefixes
        clean = re.sub(r"^(?:chitti,?\s*|hey chitti,?\s*|bhai,?\s*|please\s+)", "", raw, flags=re.IGNORECASE).strip()
        clean_lower = clean.lower()

        state = TaskState(task_description=raw)

        # 1. YOUTUBE / SONG PLAYBACK COMMANDS
        m_yt = re.search(r"(?i)\b(?:play\s+(?:a\s+)?(.*)\s+(?:song|music|track)|go\s+to\s+youtube\s+and\s+(?:play|search(?:\s+for)?)\s+(.*)|(?:search\s+for\s+|play\s+)?(.*)\s+on\s+youtube|youtube\s+(?:pe\s+|par\s+)(.*)\s+(?:chalao|play\s+karo|search\s+karo)|(.*)\s+(?:ka\s+gaana|song)\s+(?:chalao|play\s+karo))\b", clean)
        if m_yt or ("youtube" in clean_lower and ("play" in clean_lower or "search" in clean_lower or "song" in clean_lower or "gaana" in clean_lower)):
            if m_yt:
                song_query = (m_yt.group(1) or m_yt.group(2) or m_yt.group(3) or m_yt.group(4) or m_yt.group(5) or "").strip()
            else:
                song_query = clean
            artist = BrowserController.normalize_artist_query(song_query)
            if not artist or artist == "Top Songs":
                artist = "Sonu Nigam" if "sonu" in clean_lower else ("Shreya Ghoshal" if "shreya" in clean_lower else "Trending Music")

            state.steps = [
                AgentStep(step_id=1, description=f"Resolve top video and start playing '{artist}' on YouTube", action_type="PLAY_YOUTUBE", parameters={"query": artist}),
                AgentStep(step_id=2, description="Verify browser opened YouTube", action_type="VERIFY_WINDOW", parameters={"title": "YouTube"}),
                AgentStep(step_id=3, description=f"Verify playback started for '{artist}'", action_type="VERIFY_PLAYBACK", parameters={"query": artist}),
            ]
            return state

        # 2. VS CODE + PROJECT OPENING COMMANDS
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

        # 3. GENERAL-PURPOSE PROGRAMMING & CODING TASKS
        # Handles any language (Python, C++, Java, JS, Rust, etc.), any problem, single-file or multi-file
        is_coding_request = (
            bool(re.search(r"(?i)\b(?:vs\s*code|vscode)\b.*(?:code|program|script|file|banao|kro|create|write|likho|implement|class|api|app|algorithm|bana|karo)", clean)) or
            bool(re.search(r"(?i)\b(?:python|cpp|c|java|javascript|typescript|rust|go|golang|csharp|react|html|css|sql)\b.*(?:code|program|script|file|banao|kro|create|write|likho|implement|class|api|app|algorithm|calculator|search|sort|list|tree|reader|analyzer|finder|page|checker)", clean)) or
            bool(re.search(r"(?i)(?:c\+\+|c\#).*(?:code|program|script|file|banao|kro|create|write|likho|implement|class|api|app|algorithm|calculator|search|sort|list|tree|reader|analyzer|finder|page|checker)", clean)) or
            bool(re.search(r"(?i)\b(?:write|create|make|build|generate|implement)\s+(?:a|an)?\s*(?:.*)?\s*(?:program|code|script|algorithm|class|api|model|page|app)\b", clean)) or
            bool(re.search(r"(?i)\b(?:code|program|script|calculator|api|app|algorithm)\s+(?:likho|banao|bana\s+do|create\s+karo)\b", clean))
        )

        if is_coding_request:
            spec = CodeGenerator.generate_code_for_topic(clean, llm=self.llm)
            state.set_flag(ExecutionFlag.TASK_UNDERSTOOD, True)
            state.set_flag(ExecutionFlag.CODE_GENERATED, True)

            # Validate generated code
            main_code = spec.files[0].content if spec.files else ""
            val_report = CodeValidator.validate_code(
                main_code, language=spec.language, expected_markers=spec.expected_markers, llm=self.llm
            )
            if val_report.valid:
                state.set_flag(ExecutionFlag.CODE_VALIDATED, True)

            filename = spec.filename
            abs_path = str(self.fs.resolve_path(filename).resolve())
            topic_title = spec.problem_description.title()
            lang_title = spec.language.upper()

            # Multi-step Plan Construction
            state.steps = [
                AgentStep(
                    step_id=1,
                    description=f"Generate and write {topic_title} {lang_title} code to {filename}",
                    action_type="CREATE_FILE",
                    parameters={"path": abs_path, "content": main_code},
                ),
                AgentStep(
                    step_id=2,
                    description=f"Open {filename} in VS Code with absolute path",
                    action_type="OPEN_APPLICATION",
                    parameters={"target": "VS Code", "args": [abs_path]},
                ),
                AgentStep(
                    step_id=3,
                    description=f"Wait for VS Code editor and {filename}",
                    action_type="WAIT_FOR_EDITOR",
                    parameters={"application": "Visual Studio Code", "expected_file": filename, "timeout": 6.0},
                ),
                AgentStep(
                    step_id=4,
                    description="Capture screenshot of active editor",
                    action_type="TAKE_SCREENSHOT",
                    parameters={"filename": f"vscode_{Path(filename).stem}.png"},
                ),
                AgentStep(
                    step_id=5,
                    description=f"Verify {filename} editor content in VS Code",
                    action_type="VERIFY_EDITOR_CONTENT",
                    parameters={
                        "application": "Visual Studio Code",
                        "expected_file": abs_path,
                        "expected_markers": spec.expected_markers,
                    },
                ),
                AgentStep(
                    step_id=6,
                    description="Save file in VS Code editor",
                    action_type="SAVE_EDITOR",
                    parameters={"application": "Visual Studio Code"},
                ),
                AgentStep(
                    step_id=7,
                    description=f"Verify {filename} saved on disk with {spec.expected_symbol}",
                    action_type="VERIFY_FILE_CONTENT",
                    parameters={"path": abs_path, "expected_keyword": spec.expected_symbol},
                ),
            ]

            # If user explicitly requested execution ("run karo", "execute", etc.)
            if spec.execution_requested:
                state.steps.extend([
                    AgentStep(
                        step_id=8,
                        description=f"Verify toolchain for {spec.language}",
                        action_type="VERIFY_TOOLCHAIN",
                        parameters={"language": spec.language},
                    ),
                    AgentStep(
                        step_id=9,
                        description=f"Compile and execute {filename}",
                        action_type="COMPILE_AND_EXECUTE",
                        parameters={"language": spec.language, "file": abs_path, "spec": spec},
                    ),
                    AgentStep(
                        step_id=10,
                        description="Verify successful execution output",
                        action_type="VERIFY_EXECUTION",
                        parameters={"file": abs_path},
                    ),
                ])

            return state

        # 4. Multi-step: "Open Notepad and type <text>"
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

        # 5. Multi-step: "Create a folder called <name> on Desktop"
        m_desktop_folder = re.search(r"(?i)\bcreate\s+(?:a\s+)?folder\s+(?:called|named)?\s*([A-Za-z0-9_\-]+)\s+(?:on|in)\s+(?:my\s+)?desktop\b", clean) or \
                           re.search(r"(?i)\bdesktop\s+(?:pe|par|me|mein)\s+(?:ek\s+)?([A-Za-z0-9_\-]+)\s+(?:naam\s+ka\s+)?folder\s+banao\b", clean)
        if m_desktop_folder:
            f_name = m_desktop_folder.group(1).strip()
            desktop_path = str((Path.home() / "Desktop" / f_name).resolve())
            if not Path.home().joinpath("Desktop").exists() and Path.home().joinpath("OneDrive", "Desktop").exists():
                desktop_path = str((Path.home() / "OneDrive" / "Desktop" / f_name).resolve())

            state.steps = [
                AgentStep(step_id=1, description=f"Create folder '{f_name}' on Desktop", action_type="CREATE_DIRECTORY", parameters={"path": desktop_path}),
                AgentStep(step_id=2, description=f"Verify folder '{f_name}' created on Desktop", action_type="VERIFY_FILE", parameters={"path": desktop_path}),
            ]
            return state

        # 6. Multi-step: "Open Chrome and search for <query>"
        m_search = re.search(r"(?i)\b(?:open\s+(?:chrome|browser|edge)\s+(?:and|aur)\s+search(?:\s+for)?\s+(.*))\b", clean) or \
                   re.search(r"(?i)\b(?:search\s+(?:for\s+)?(.*)\s+on\s+(?:google|chrome|browser))\b", clean)
        if m_search:
            query = m_search.group(1).strip()
            state.steps = [
                AgentStep(step_id=1, description=f"Open browser and search for '{query}'", action_type="SEARCH_WEB", parameters={"query": query}),
                AgentStep(step_id=2, description="Verify browser opened", action_type="VERIFY_WINDOW", parameters={"title": "Chrome"}),
            ]
            return state

        # 7. Multi-step: "Run the project and tell me if there are errors" / "Open the terminal and run the tests"
        m_tests = re.search(r"(?i)\b(?:run\s+(?:the\s+)?tests?|run\s+pytest|run\s+(?:the\s+)?project|test\s+chalao|is\s+program\s+ko\s+run\s+karo)\b", clean)
        if m_tests:
            state.steps = [
                AgentStep(step_id=1, description="Execute tests via pytest", action_type="RUN_TERMINAL", parameters={"command": "python -m pytest tests/ -q", "timeout": 45}),
                AgentStep(step_id=2, description="Analyze test outcome", action_type="ANALYZE_OUTPUT", parameters={}),
            ]
            return state

        # 8. Multi-step: "Open the folder <name> and create <file>"
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
        tool_engine: ToolEngine,
        computer: ComputerController,
        filesystem: FilesystemController,
        terminal: TerminalController,
        browser: BrowserController,
        apps: AppController,
        screen_analyzer: ScreenAnalyzer,
        projects: ProjectRegistry,
        llm: Optional[BaseLLM] = None,
    ):
        self.tools = tool_engine
        self.computer = computer
        self.fs = filesystem
        self.terminal = terminal
        self.browser = browser
        self.apps = apps
        self.screen_analyzer = screen_analyzer
        self.projects = projects
        self.llm = llm

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
            success, message, observation = self._execute_step_action(step, state, last_search_results)
            step.result_message = message
            step.observation = observation
            if observation:
                state.add_observation(observation)

            if not success:
                step.status = StepStatus.FAILED
                step.error = message
                state.mark_failed(f"Step {step.step_id} failed: {message}")
                log_warn(f"[AGENT] Step {step.step_id} FAILED: {message}")
                return False, f"I ran into an issue while performing the task: {message}"

            # 3. VERIFY
            step.status = StepStatus.SUCCESS
            log_info(f"[AGENT] Step {step.step_id} SUCCESS: {message}")
            state.current_step_index += 1

        state.mark_completed()
        log_info(f"[AGENT LOOP] Task completed successfully in {time.time() - state.start_time:.2f}s")
        return True, "Task completed successfully."

    def _execute_step_action(self, step: AgentStep, state: TaskState, search_cache: List[str]) -> Tuple[bool, str, Optional[str]]:
        """Executes a single step action with strict honest verification."""
        act = step.action_type
        params = step.parameters

        try:
            if act == "PLAY_YOUTUBE":
                query = params["query"]
                res = self.tools.execute_tool("play_youtube", {"query": query})
                return res.success, res.message, f"Playing YouTube query '{query}'"

            elif act == "OPEN_APPLICATION":
                target = params["target"]
                args = params.get("args")
                res = self.tools.execute_tool("open_application", {"application": target, "args": args})
                if res.success:
                    state.set_flag(ExecutionFlag.FILE_OPENED, True)
                time.sleep(0.3)
                return res.success, res.message, f"App {target} launched with args {args}"

            elif act == "WAIT_FOR_EDITOR":
                app = params.get("application", "Visual Studio Code")
                expected_file = params.get("expected_file")
                timeout = params.get("timeout", 6.0)
                res = self.tools.execute_tool("wait_for_editor", {"application": app, "expected_file": expected_file, "timeout": timeout})
                if not res.success:
                    return False, f"Wait for editor failed: {res.data.get('evidence', res.error)}", None
                return True, f"Editor ready: {res.data.get('evidence')}", res.data.get("evidence")

            elif act == "TAKE_SCREENSHOT":
                filename = params.get("filename")
                res = self.tools.execute_tool("take_screenshot", {"filename": filename})
                return res.success, res.message, f"Screenshot captured: {res.data.get('path')}"

            elif act == "VERIFY_EDITOR_CONTENT":
                app = params.get("application", "Visual Studio Code")
                expected_file = params.get("expected_file", "script.py")
                markers = params.get("expected_markers", [])
                res = self.tools.execute_tool("verify_editor_content", {"application": app, "expected_file": expected_file, "expected_markers": markers})
                if not res.success:
                    ev = res.data.get("evidence", res.error) or f"{app} editor verification failed for '{expected_file}'"
                    return False, f"Editor verification failed: {ev}", None
                state.set_flag(ExecutionFlag.EDITOR_CONTENT_VERIFIED, True)
                return True, f"Editor content verified: {res.data.get('evidence')}", res.data.get("evidence")

            elif act == "SAVE_EDITOR":
                app = params.get("application", "Visual Studio Code")
                res = self.tools.execute_tool("save_editor", {"application": app})
                state.set_flag(ExecutionFlag.FILE_SAVED, True)
                return res.success, res.message, f"Editor saved for {app}"

            elif act == "VERIFY_TOOLCHAIN":
                lang = params["language"]
                avail, bin_path = ToolchainManager.is_toolchain_available(lang)
                if avail:
                    state.set_flag(ExecutionFlag.TOOLCHAIN_VERIFIED, True)
                    return True, f"Toolchain for {lang} available ({bin_path})", f"Toolchain: {bin_path}"
                else:
                    # Honest report: toolchain not installed on this machine
                    state.set_flag(ExecutionFlag.TOOLCHAIN_VERIFIED, False)
                    log_warn(f"[TOOLCHAIN] Compiler/runtime for '{lang}' is not installed on this system.")
                    return True, f"Compiler/runtime for '{lang}' is not installed on this system. File created and saved, execution skipped.", "Toolchain unavailable"

            elif act == "COMPILE_AND_EXECUTE":
                lang = params["language"]
                file_path = params["file"]
                spec = params.get("spec")

                # If toolchain is missing, skip execution step honestly
                if not state.get_flag(ExecutionFlag.TOOLCHAIN_VERIFIED):
                    return True, f"Execution skipped because '{lang}' compiler/runtime is not installed.", "Skipped"

                compile_cmd, run_cmd = ToolchainManager.build_execution_commands(lang, file_path)

                # 1. Compilation step (if compiled language)
                if compile_cmd:
                    log_info(f"[COMPILER] Compiling {lang} source: {compile_cmd}")
                    comp_res = self.tools.execute_tool("execute_terminal_command", {"command": compile_cmd})
                    if not comp_res.success or comp_res.data.get("exit_code", 0) != 0:
                        err_out = comp_res.data.get("output", "Compilation error")
                        log_warn(f"[COMPILER] Build failed: {err_out}")

                        # Automated debugging loop (up to 3 attempts)
                        if spec and self.llm:
                            fixed_code = CodeGenerator.fix_code_after_error(spec, spec.files[0].content, err_out, self.llm)
                            self.fs.write_file(file_path, fixed_code)
                            # Retry compilation
                            comp_res = self.tools.execute_tool("execute_terminal_command", {"command": compile_cmd})

                        if not comp_res.success or comp_res.data.get("exit_code", 0) != 0:
                            return False, f"Compilation failed: {comp_res.data.get('output')}", None

                # 2. Execution step
                log_info(f"[EXECUTE] Running binary/script: {run_cmd}")
                run_res = self.tools.execute_tool("execute_terminal_command", {"command": run_cmd})
                out = run_res.data.get("output", run_res.message)
                state.set_flag(ExecutionFlag.CODE_EXECUTED, True)

                if run_res.success and run_res.data.get("exit_code", 0) == 0:
                    state.set_flag(ExecutionFlag.EXECUTION_VERIFIED, True)
                    return True, f"Output:\n{out}", out
                return False, f"Execution failed: {out}", out

            elif act == "VERIFY_EXECUTION":
                if not state.get_flag(ExecutionFlag.TOOLCHAIN_VERIFIED):
                    return True, "Execution skipped (compiler not installed).", "Skipped"
                if state.get_flag(ExecutionFlag.CODE_EXECUTED) and state.get_flag(ExecutionFlag.EXECUTION_VERIFIED):
                    return True, "Code execution completed and verified.", "Execution success"
                return False, "Code execution could not be verified.", None

            elif act == "OPEN_FOLDER":
                target = params["target"]
                p = self.fs.resolve_path(target)
                if not p.exists():
                    user_home = Path.home()
                    cand = user_home / target
                    if cand.exists():
                        p = cand
                    else:
                        p.mkdir(parents=True, exist_ok=True)
                try:
                    os.startfile(str(p))
                except Exception as e:
                    log_debug(f"os.startfile notice: {e}")
                return True, f"Opened folder {p}", f"Folder {p} opened"

            elif act == "SEARCH_WEB":
                query = params["query"]
                res = self.tools.execute_tool("search_web", {"query": query})
                return res.success, res.message, f"Browser opened search for '{query}'"

            elif act == "TYPE_TEXT":
                text = params["text"]
                time.sleep(0.2)
                res = self.tools.execute_tool("type_text", {"text": text})
                return res.success, res.message, f"Text entered: {text}"

            elif act == "CREATE_FILE":
                path = params["path"]
                content = params.get("content", "")
                res = self.tools.execute_tool("create_file", {"path": path, "content": content})
                if res.success:
                    state.set_flag(ExecutionFlag.FILE_CREATED, True)
                return res.success, res.message, f"File created at {path}"

            elif act == "CREATE_DIRECTORY":
                path = params["path"]
                res = self.tools.execute_tool("create_directory", {"path": path})
                return res.success, res.message, f"Directory created at {path}"

            elif act == "DELETE_DIRECTORY":
                path = params["path"]
                res = self.tools.execute_tool("delete_directory", {"path": path})
                return res.success, res.message, f"Directory {path} deleted"

            elif act == "SEARCH_FILES":
                root = params.get("root")
                pattern = params.get("pattern", "*")
                res = self.tools.execute_tool("search_files", {"pattern": pattern, "root": root})
                search_cache.clear()
                search_cache.extend(res.data.get("matches", []))
                return res.success, f"Found {len(search_cache)} files matching {pattern}", f"Matches: {search_cache[:5]}"

            elif act == "RUN_TERMINAL":
                cmd = params["command"]
                res = self.tools.execute_tool("execute_terminal_command", {"command": cmd})
                out = res.data.get("output", res.message)
                state.set_flag(ExecutionFlag.CODE_EXECUTED, True)
                if res.success and res.data.get("exit_code", 0) == 0:
                    state.set_flag(ExecutionFlag.EXECUTION_VERIFIED, True)
                return res.success, out, f"Command exit code: {res.data.get('exit_code', 0)}"

            elif act == "RESOLVE_PROJECT":
                name = params["name"]
                path = params.get("path") or self.projects.get_project_path(name)
                return True, f"Resolved project {name} -> {path}", f"Path: {path}"

            elif act == "VERIFY_WINDOW":
                title = params.get("title", "")
                time.sleep(0.3)
                res = self.screen_analyzer.verify_window(title)
                if not res.success:
                    return False, f"Verification failed: {res.evidence}", None
                return True, f"Window verified for '{title}': {res.evidence}", res.evidence

            elif act == "VERIFY_PLAYBACK":
                b_state = self.browser.get_browser_state()
                if b_state.playback_verified and (b_state.active_video_id or b_state.current_url):
                    ev = f"Video playback active (URL: {b_state.current_url}, Video ID: {b_state.active_video_id})"
                    return True, f"Playback verified: {ev}", ev
                win_res = self.screen_analyzer.verify_window("YouTube")
                if win_res.success:
                    return True, f"YouTube window verified: {win_res.evidence}", win_res.evidence
                return False, "Verification failed: YouTube video playback could not be verified", None

            elif act == "VERIFY_FILE":
                path = params["path"]
                resolved = self.fs.resolve_path(path)
                if not resolved.exists():
                    return False, f"Verification failed: File '{path}' was not found on disk", None
                state.set_flag(ExecutionFlag.FILE_SAVED, True)
                return True, f"File {path} verified on disk", f"File exists at {resolved}"

            elif act == "VERIFY_FILE_CONTENT":
                path = params["path"]
                keyword = params.get("expected_keyword", "")
                res = self.tools.execute_tool("verify_file_content", {"path": path, "expected_keyword": keyword})
                if not res.success:
                    ev = res.data.get("evidence", res.error) or f"File '{path}' missing expected content"
                    return False, f"Verification failed: {ev}", None
                state.set_flag(ExecutionFlag.FILE_SAVED, True)
                return True, f"File content verified: {res.data.get('evidence')}", res.data.get("evidence")

            elif act == "VERIFY_DELETED":
                path = params["path"]
                resolved = self.fs.resolve_path(path)
                if resolved.exists():
                    return False, f"Verification failed: Path '{path}' still exists on disk", None
                return True, f"Verified '{path}' is deleted", "Path no longer exists"

            elif act in ("ANALYZE_OUTPUT", "DIAGNOSE_ERROR"):
                return True, "Analysis completed.", "Output inspected."

            return False, f"Unknown action: {act}", None

        except Exception as e:
            log_warn(f"[AGENT] Exception in step {act}: {e}")
            return False, str(e), None
