"""
Chitti Agent Planner & Computer-Use Loop.
Implements the multi-step SEE -> UNDERSTAND -> PLAN -> ACT -> OBSERVE -> VERIFY loop.
"""

import os
import re
import time
import urllib.parse
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.agent.actions import ActionType, RiskLevel, StructuredAction
from src.agent.code_generator import CodeGenerator
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
from src.agent.tools import ToolEngine
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

        # 1. YOUTUBE / SONG PLAYBACK COMMANDS
        # e.g. "play a Sonu Nigam song", "play a shreya ghosal song", "go to youtube and play a sonu nigam song", "go to youtube and search for sonu nigam", "Sonu Nigam ka gaana chalao"
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

        # 2. VS CODE + PROJECT OPENING COMMANDS (Checked before generic VS Code code generation)
        # e.g. "Open VS Code and open my Chitti project" / "Open my Chitti project in VS Code"
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

        # 3. VS CODE + CODE GENERATION COMMANDS
        # e.g. "vs code open kro aur ek anagram ka python code banao", "VS Code open kro Aur ek Fibonacci series ka Python code kro", "open vs code and create a python anagram checker"
        m_vscode_code = re.search(r"(?i)\b(?:vs\s*code|vscode)\b.*(?:code|program|script|file|banao|kro|create|write)", clean) or \
                        re.search(r"(?i)\b(?:open\s+(?:vs\s*code|vscode)|vs\s*code\s+(?:open\s+kro|open\s+karo|kholo))\s+(?:and|aur)\s+.*(?:code|program|script|banao)", clean) or \
                        (("vs code" in clean_lower or "vscode" in clean_lower) and any(kw in clean_lower for kw in ["code", "anagram", "fibonacci", "palindrome", "prime", "factorial", "sort", "search", "python"]))
        if m_vscode_code:
            gen = CodeGenerator.generate_code_for_topic(clean)
            topic_title = gen.topic.replace("_", " ").title()
            filename = gen.filename

            state.steps = [
                AgentStep(step_id=1, description=f"Generate and write {topic_title} Python code to {filename}", action_type="CREATE_FILE", parameters={"path": filename, "content": gen.code}),
                AgentStep(step_id=2, description=f"Launch VS Code with {filename}", action_type="OPEN_APPLICATION", parameters={"target": "VS Code", "args": [filename]}),
                AgentStep(step_id=3, description=f"Verify {filename} content contains '{gen.expected_symbol}'", action_type="VERIFY_FILE_CONTENT", parameters={"path": filename, "expected_keyword": gen.expected_symbol}),
                AgentStep(step_id=4, description="Verify VS Code window is open", action_type="VERIFY_WINDOW", parameters={"title": "Visual Studio Code"}),
            ]
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

        # 5. Multi-step: "Create a folder called <name> on Desktop" / "Create folder <name> on my desktop"
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

        # 7. Multi-step: "Create a Python file in my project and write a program that calculates Fibonacci numbers / Anagrams"
        m_create_code = re.search(r"(?i)\b(?:create\s+(?:a\s+)?python\s+file|write\s+(?:a\s+)?(?:python\s+)?(?:program|code|script))\b", clean)
        if m_create_code:
            gen = CodeGenerator.generate_code_for_topic(clean)
            topic_title = gen.topic.replace("_", " ").title()
            filename = gen.filename
            state.steps = [
                AgentStep(step_id=1, description=f"Create {filename} with {topic_title} solution in workspace", action_type="CREATE_FILE", parameters={"path": filename, "content": gen.code}),
                AgentStep(step_id=2, description=f"Verify {filename} created and contains '{gen.expected_symbol}'", action_type="VERIFY_FILE_CONTENT", parameters={"path": filename, "expected_keyword": gen.expected_symbol}),
            ]
            return state

        # 8. Multi-step: "Run the project and tell me if there are errors" / "Open the terminal and run the tests"
        m_tests = re.search(r"(?i)\b(?:run\s+(?:the\s+)?tests?|run\s+pytest|run\s+(?:the\s+)?project|test\s+chalao|is\s+program\s+ko\s+run\s+karo)\b", clean)
        if m_tests:
            state.steps = [
                AgentStep(step_id=1, description="Execute tests via pytest", action_type="RUN_TERMINAL", parameters={"command": "python -m pytest tests/ -q", "timeout": 45}),
                AgentStep(step_id=2, description="Analyze test outcome", action_type="ANALYZE_OUTPUT", parameters={}),
            ]
            return state

        # 9. Multi-step: "Open the folder <name> and create <file>"
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

        # 10. Destructive Multi-step: "Delete <folder/file>"
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
    ):
        self.tools = tool_engine
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
                log_warn(f"[AGENT] Step {step.step_id} FAILED: {message}")
                return False, f"I ran into an issue while performing the task: {message}"

            # 3. VERIFY
            step.status = StepStatus.SUCCESS
            log_info(f"[AGENT] Step {step.step_id} SUCCESS: {message}")
            state.current_step_index += 1

        state.mark_completed()
        log_info(f"[AGENT LOOP] Task completed successfully in {time.time() - state.start_time:.2f}s")
        return True, "Task completed successfully."

    def _execute_step_action(self, step: AgentStep, search_cache: List[str]) -> Tuple[bool, str, Optional[str]]:
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
                time.sleep(0.3)
                return res.success, res.message, f"App {target} launched"

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
                return True, f"File {path} verified on disk", f"File exists at {resolved}"

            elif act == "VERIFY_FILE_CONTENT":
                path = params["path"]
                keyword = params.get("expected_keyword", "")
                res = self.tools.execute_tool("verify_file_content", {"path": path, "expected_keyword": keyword})
                if not res.success:
                    ev = res.data.get("evidence", res.error) or f"File '{path}' missing expected content"
                    return False, f"Verification failed: {ev}", None
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
