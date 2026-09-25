"""
Chitti Computer Tool Registry & Tool Engine.
Exposes standardized, executable tools for both the LLM agent and deterministic planners.
"""

import json
import os
import subprocess
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

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
from src.utils.logging import log_debug, log_info, log_warn


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Callable[..., Dict[str, Any]]
    is_destructive: bool = False


@dataclass
class ToolExecutionResult:
    tool: str
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    message: str = ""
    error: Optional[str] = None
    verification: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool": self.tool,
            "success": self.success,
            "data": self.data,
            "message": self.message,
            "error": self.error,
            "verification": self.verification,
        }


class ToolEngine:
    """Central registry and execution dispatcher for all laptop computer-use tools."""

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
        self.tools: Dict[str, ToolDefinition] = {}
        self._register_all_tools()

    def _register(self, name: str, description: str, parameters: Dict[str, Any], handler: Callable[..., Dict[str, Any]], is_destructive: bool = False):
        self.tools[name] = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            handler=handler,
            is_destructive=is_destructive,
        )

    def _register_all_tools(self):
        # 1. APPLICATION TOOLS
        self._register(
            "open_application",
            "Launches a Windows desktop application by name (e.g. 'Chrome', 'VS Code', 'Notepad', 'Calculator').",
            {"application": {"type": "string", "description": "Name or path of the application"}},
            self._tool_open_application,
        )
        self._register(
            "close_application",
            "Closes a running desktop application by name.",
            {"application": {"type": "string", "description": "Name of the application to terminate"}},
            self._tool_close_application,
        )

        # 2. BROWSER & YOUTUBE TOOLS
        self._register(
            "open_url",
            "Opens a URL in the user's default web browser.",
            {"url": {"type": "string", "description": "The URL to open"}},
            self._tool_open_url,
        )
        self._register(
            "search_web",
            "Performs a web search in the default browser.",
            {"query": {"type": "string", "description": "Search query string"}},
            self._tool_search_web,
        )
        self._register(
            "play_youtube",
            "Resolves, opens, and starts playing a requested YouTube song, artist, or video.",
            {"query": {"type": "string", "description": "Song name, artist, or video to play"}},
            self._tool_play_youtube,
        )

        # 3. WINDOW & SCREEN TOOLS
        self._register(
            "take_screenshot",
            "Captures a screenshot of the user's screen and saves it to data/screenshots.",
            {"filename": {"type": "string", "description": "Optional custom filename", "optional": True}},
            self._tool_take_screenshot,
        )
        self._register(
            "list_windows",
            "Lists all currently visible desktop windows with titles and coordinates.",
            {},
            self._tool_list_windows,
        )
        self._register(
            "focus_window",
            "Brings a specific window to the foreground by title query.",
            {"title": {"type": "string", "description": "Window title substring"}},
            self._tool_focus_window,
        )
        self._register(
            "verify_window",
            "Verifies that an application or window is running and visible.",
            {"title": {"type": "string", "description": "Window title substring"}},
            self._tool_verify_window,
        )
        self._register(
            "wait_for_editor",
            "Polls and waits until an application and target file window is ready.",
            {
                "application": {"type": "string", "description": "Application name e.g. 'Visual Studio Code'"},
                "expected_file": {"type": "string", "description": "Target filename e.g. 'anagram.py'", "optional": True},
                "timeout": {"type": "number", "description": "Timeout in seconds", "optional": True},
            },
            self._tool_wait_for_editor,
        )
        self._register(
            "verify_editor_content",
            "Verifies that the target application's active editor contains expected code markers via GUI / UI inspection.",
            {
                "application": {"type": "string", "description": "Application name e.g. 'Visual Studio Code'"},
                "expected_file": {"type": "string", "description": "Expected file name"},
                "expected_markers": {"type": "array", "items": {"type": "string"}, "description": "List of expected code marker strings"},
            },
            self._tool_verify_editor_content,
        )
        self._register(
            "save_editor",
            "Saves the active editor file via hotkey Ctrl+S.",
            {"application": {"type": "string", "description": "Application name e.g. 'Visual Studio Code'", "optional": True}},
            self._tool_save_editor,
        )

        # 4. MOUSE & KEYBOARD TOOLS
        self._register(
            "type_text",
            "Types text into the currently active input field or window.",
            {"text": {"type": "string", "description": "Text string to type"}},
            self._tool_type_text,
        )
        self._register(
            "press_key",
            "Presses a single keyboard key (e.g. 'enter', 'esc', 'tab', 'backspace').",
            {"key": {"type": "string", "description": "Key name"}},
            self._tool_press_key,
        )
        self._register(
            "hotkey",
            "Executes a keyboard shortcut combination (e.g. ['ctrl', 'c'], ['ctrl', 'v'], ['alt', 'tab']).",
            {"keys": {"type": "array", "items": {"type": "string"}, "description": "Keys to press together"}},
            self._tool_hotkey,
        )
        self._register(
            "click",
            "Clicks at specified (x, y) coordinates or at the current mouse position.",
            {
                "x": {"type": "integer", "description": "X coordinate", "optional": True},
                "y": {"type": "integer", "description": "Y coordinate", "optional": True},
                "button": {"type": "string", "description": "'left' or 'right'", "optional": True},
            },
            self._tool_click,
        )

        # 5. FILESYSTEM TOOLS
        self._register(
            "create_file",
            "Creates a new text or code file with specified content.",
            {
                "path": {"type": "string", "description": "File path (relative to workspace or absolute)"},
                "content": {"type": "string", "description": "File contents", "optional": True},
            },
            self._tool_create_file,
        )
        self._register(
            "read_file",
            "Reads content from a text or code file.",
            {"path": {"type": "string", "description": "File path to read"}},
            self._tool_read_file,
        )
        self._register(
            "write_file",
            "Writes or overwrites content in a file.",
            {
                "path": {"type": "string", "description": "File path to write"},
                "content": {"type": "string", "description": "New file content"},
            },
            self._tool_write_file,
        )
        self._register(
            "verify_file_content",
            "Verifies that a file exists and contains expected keywords or code symbols.",
            {
                "path": {"type": "string", "description": "File path to check"},
                "expected_keyword": {"type": "string", "description": "Expected function name or content keyword"},
            },
            self._tool_verify_file_content,
        )
        self._register(
            "create_directory",
            "Creates a new directory/folder.",
            {"path": {"type": "string", "description": "Folder path to create"}},
            self._tool_create_directory,
        )
        self._register(
            "delete_file",
            "Deletes a file from the laptop.",
            {"path": {"type": "string", "description": "File path to delete"}},
            self._tool_delete_file,
            is_destructive=True,
        )
        self._register(
            "delete_directory",
            "Deletes a folder and its contents.",
            {"path": {"type": "string", "description": "Folder path to delete"}},
            self._tool_delete_directory,
            is_destructive=True,
        )
        self._register(
            "search_files",
            "Searches for files matching a glob pattern (e.g. '*.py', '*.pdf').",
            {
                "pattern": {"type": "string", "description": "Glob pattern"},
                "root": {"type": "string", "description": "Directory to search in", "optional": True},
            },
            self._tool_search_files,
        )

        # 6. TERMINAL TOOLS
        self._register(
            "execute_terminal_command",
            "Runs a terminal / PowerShell command and captures stdout/stderr/exit code.",
            {
                "command": {"type": "string", "description": "Command string to run"},
                "cwd": {"type": "string", "description": "Working directory", "optional": True},
            },
            self._tool_execute_terminal,
        )

    # ---------------------------------------------------------
    # TOOL HANDLERS
    # ---------------------------------------------------------

    def _tool_open_application(self, application: str, args: Optional[List[str]] = None) -> Dict[str, Any]:
        log_info(f"[TOOL] open_application -> {application}")
        ok = self.apps.launch_application(application, args=args)
        if ok:
            time.sleep(0.5)
            return {"success": True, "application": application, "message": f"Successfully launched {application}"}
        return {"success": False, "application": application, "error": f"Could not find or launch {application}"}

    def _tool_close_application(self, application: str) -> Dict[str, Any]:
        log_info(f"[TOOL] close_application -> {application}")
        ok = self.apps.close_application(application)
        return {"success": ok, "application": application, "message": f"Closed application {application}"}

    def _tool_open_url(self, url: str, browser: Optional[str] = None) -> Dict[str, Any]:
        log_info(f"[TOOL] open_url -> {url}")
        ok = self.browser.open_url(url, browser=browser)
        return {"success": ok, "url": url, "message": f"Opened {url} in browser"}

    def _tool_search_web(self, query: str) -> Dict[str, Any]:
        log_info(f"[TOOL] search_web -> {query}")
        ok = self.browser.search_web(query)
        return {"success": ok, "query": query, "message": f"Searched web for '{query}'"}

    def _tool_play_youtube(self, query: str) -> Dict[str, Any]:
        log_info(f"[TOOL] play_youtube -> {query}")
        ok, msg, data = self.browser.search_and_play_youtube(query)
        return {
            "success": ok,
            "query": query,
            "message": msg,
            **data,
        }

    def _tool_take_screenshot(self, filename: Optional[str] = None) -> Dict[str, Any]:
        log_info("[TOOL] take_screenshot")
        path = self.computer.take_screenshot(filename=filename)
        return {"success": True, "path": path, "message": f"Screenshot saved to {path}"}

    def _tool_list_windows(self) -> Dict[str, Any]:
        windows = self.computer.list_windows()
        titles = [w.title for w in windows if w.title]
        return {"success": True, "windows": titles, "count": len(titles)}

    def _tool_focus_window(self, title: str) -> Dict[str, Any]:
        ok = self.computer.focus_window(title)
        return {"success": ok, "title": title, "message": f"Focused window matching '{title}'"}

    def _tool_verify_window(self, title: str) -> Dict[str, Any]:
        res = self.screen_analyzer.verify_window(title)
        return {"success": res.success, "evidence": res.evidence, "target": title}

    def _tool_wait_for_editor(self, application: str = "Visual Studio Code", expected_file: Optional[str] = None, timeout: float = 6.0) -> Dict[str, Any]:
        log_info(f"[TOOL] wait_for_editor -> App: {application}, File: {expected_file}")
        res = self.screen_analyzer.wait_for_window_and_file(application=application, expected_file=expected_file, timeout_sec=timeout)
        return {"success": res.success, "evidence": res.evidence, "application": application, "file": expected_file}

    def _tool_verify_editor_content(self, application: str = "Visual Studio Code", expected_file: str = "script.py", expected_markers: Optional[List[str]] = None) -> Dict[str, Any]:
        log_info(f"[TOOL] verify_editor_content -> App: {application}, File: {expected_file}, Markers: {expected_markers}")
        resolved = self.fs.resolve_path(expected_file)
        target_path = str(resolved) if resolved.exists() else expected_file
        return self.screen_analyzer.verify_editor_content(application=application, expected_file=target_path, expected_markers=expected_markers)

    def _tool_save_editor(self, application: str = "Visual Studio Code") -> Dict[str, Any]:
        log_info(f"[TOOL] save_editor -> App: {application}")
        ok = self.screen_analyzer.save_editor(application=application)
        return {"success": ok, "application": application, "message": f"Saved active file in {application}"}

    def _tool_type_text(self, text: str) -> Dict[str, Any]:
        ok = self.computer.type_text(text)
        return {"success": ok, "text": text, "message": f"Typed text: {text}"}

    def _tool_press_key(self, key: str) -> Dict[str, Any]:
        ok = self.computer.press_key(key)
        return {"success": ok, "key": key}

    def _tool_hotkey(self, keys: List[str]) -> Dict[str, Any]:
        ok = self.computer.hotkey(*keys)
        return {"success": ok, "keys": keys}

    def _tool_click(self, x: Optional[int] = None, y: Optional[int] = None, button: str = "left") -> Dict[str, Any]:
        ok = self.computer.click(x=x, y=y, button=button)
        return {"success": ok, "x": x, "y": y, "button": button}

    def _tool_create_file(self, path: str, content: str = "") -> Dict[str, Any]:
        log_info(f"[TOOL] create_file -> {path}")
        created_path = self.fs.create_file(path, content=content)
        return {"success": True, "path": created_path, "message": f"Created file at {created_path}"}

    def _tool_read_file(self, path: str) -> Dict[str, Any]:
        content = self.fs.read_file(path)
        return {"success": True, "path": path, "content": content}

    def _tool_write_file(self, path: str, content: str) -> Dict[str, Any]:
        p = self.fs.write_file(path, content)
        return {"success": True, "path": p, "message": f"Wrote content to {p}"}

    def _tool_verify_file_content(self, path: str, expected_keyword: str) -> Dict[str, Any]:
        resolved = self.fs.resolve_path(path)
        if not resolved.exists():
            return {"success": False, "evidence": f"File '{path}' does not exist on disk.", "path": str(resolved)}
        content = self.fs.read_file(str(resolved))
        if expected_keyword.lower() in content.lower():
            log_info(f"[VERIFY] File content verified: '{expected_keyword}' found in {path}")
            return {"success": True, "evidence": f"File exists and contains '{expected_keyword}'.", "path": str(resolved)}
        log_warn(f"[VERIFY] File content verification FAILED: '{expected_keyword}' not found in {path}")
        return {"success": False, "evidence": f"File exists but does not contain '{expected_keyword}'.", "path": str(resolved)}

    def _tool_create_directory(self, path: str) -> Dict[str, Any]:
        p = self.fs.create_directory(path)
        return {"success": True, "path": p, "message": f"Created directory at {p}"}

    def _tool_delete_file(self, path: str) -> Dict[str, Any]:
        ok = self.fs.delete_file(path)
        return {"success": ok, "path": path, "message": f"Deleted file {path}"}

    def _tool_delete_directory(self, path: str) -> Dict[str, Any]:
        ok = self.fs.delete_directory(path, recursive=True)
        return {"success": ok, "path": path, "message": f"Deleted directory {path}"}

    def _tool_search_files(self, pattern: str, root: Optional[str] = None) -> Dict[str, Any]:
        matches = self.fs.search_files(pattern, root_path=root)
        return {"success": True, "pattern": pattern, "matches": matches, "count": len(matches)}

    def _tool_execute_terminal(self, command: str, cwd: Optional[str] = None) -> Dict[str, Any]:
        res = self.terminal.execute_command(command, cwd=cwd)
        return {
            "success": res.is_success,
            "exit_code": res.exit_code,
            "output": res.output,
            "elapsed_sec": res.elapsed_time_sec,
        }

    # ---------------------------------------------------------
    # DISPATCHER
    # ---------------------------------------------------------

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> ToolExecutionResult:
        """Executes a tool by name with arguments."""
        tool_def = self.tools.get(tool_name)
        if not tool_def:
            log_warn(f"Tool '{tool_name}' is not registered.")
            return ToolExecutionResult(
                tool=tool_name,
                success=False,
                error=f"Unknown tool: {tool_name}",
            )

        log_info(f"[AGENT] Tool call: {tool_name} | Args: {arguments}")
        try:
            res_dict = tool_def.handler(**arguments)
            success = res_dict.get("success", True)
            msg = res_dict.get("message", "Executed successfully.")
            err = res_dict.get("error")
            log_info(f"[AGENT] Result: {'SUCCESS' if success else 'FAILED'}")
            return ToolExecutionResult(
                tool=tool_name,
                success=success,
                data=res_dict,
                message=msg,
                error=err,
            )
        except Exception as e:
            log_warn(f"[AGENT] Tool execution error in {tool_name}: {e}")
            return ToolExecutionResult(
                tool=tool_name,
                success=False,
                error=str(e),
                message=f"Failed to execute {tool_name}: {e}",
            )
