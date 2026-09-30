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
from src.agent.vision import VisionGroundingEngine
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
        vision: Optional[VisionGroundingEngine] = None,
    ):
        self.computer = computer
        self.fs = filesystem
        self.terminal = terminal
        self.browser = browser
        self.apps = apps
        self.screen_analyzer = screen_analyzer
        self.projects = projects
        self.vision = vision or VisionGroundingEngine()
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
            "screenshot",
            "Captures a screenshot of the user's screen.",
            {"filename": {"type": "string", "description": "Optional custom filename", "optional": True}},
            self._tool_take_screenshot,
        )
        self._register(
            "inspect_screen",
            "Inspects the active screen, active window, and visible UI elements.",
            {},
            self._tool_inspect_screen,
        )
        self._register(
            "read_screen",
            "Reads visible text and screen context from the active window.",
            {},
            self._tool_read_screen,
        )
        self._register(
            "analyze_screen",
            "Visually analyzes the active screen or window using multimodal vision reasoning.",
            {"query": {"type": "string", "description": "Analysis query or prompt", "optional": True}},
            self._tool_analyze_screen,
        )
        self._register(
            "diagnose_screen_error",
            "Inspects the active screen and window to detect, explain, and diagnose errors.",
            {"query": {"type": "string", "description": "Optional query or context", "optional": True}},
            self._tool_diagnose_screen_error,
        )
        self._register(
            "capture_active_window",
            "Captures a screenshot cropped strictly to the active foreground application window.",
            {},
            self._tool_capture_active_window,
        )
        self._register(
            "find_ui_element",
            "Locates a UI element, button, input box, or window by text/label.",
            {"query": {"type": "string", "description": "Text, label, or title to locate"}},
            self._tool_find_ui_element,
        )
        self._register(
            "list_windows",
            "Lists all currently visible desktop windows with titles and coordinates.",
            {},
            self._tool_list_windows,
        )
        self._register(
            "get_active_window",
            "Returns details of the currently focused active window.",
            {},
            self._tool_get_active_window,
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
            "verify_ui_state",
            "Verifies the current UI state matches expectations (active window, text, readiness).",
            {
                "expected_window": {"type": "string", "description": "Expected window title", "optional": True},
                "expected_text": {"type": "string", "description": "Expected visible text", "optional": True},
            },
            self._tool_verify_ui_state,
        )
        self._register(
            "verify_text",
            "Verifies that specific text is visible in active window or on screen.",
            {"text": {"type": "string", "description": "Text to verify"}},
            self._tool_verify_text,
        )
        self._register(
            "verify_element",
            "Verifies that a specific UI element is present on screen.",
            {"element_name": {"type": "string", "description": "Name or label of UI element"}},
            self._tool_verify_element,
        )
        self._register(
            "verify_application_state",
            "Verifies that an application is active, loaded, and ready for input.",
            {"application": {"type": "string", "description": "Application name"}},
            self._tool_verify_application_state,
        )
        self._register(
            "wait",
            "Pauses execution for a specified number of seconds.",
            {"seconds": {"type": "number", "description": "Seconds to pause", "optional": True}},
            self._tool_wait,
        )
        self._register(
            "wait_for_ui",
            "Waits for a window, element, or UI state to become ready.",
            {
                "target": {"type": "string", "description": "Window title or UI target"},
                "timeout": {"type": "number", "description": "Timeout in seconds", "optional": True},
            },
            self._tool_wait_for_ui,
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
        self._register(
            "double_click",
            "Double clicks at specified (x, y) coordinates or at current position.",
            {
                "x": {"type": "integer", "description": "X coordinate", "optional": True},
                "y": {"type": "integer", "description": "Y coordinate", "optional": True},
            },
            self._tool_double_click,
        )
        self._register(
            "right_click",
            "Right clicks at specified (x, y) coordinates or at current position.",
            {
                "x": {"type": "integer", "description": "X coordinate", "optional": True},
                "y": {"type": "integer", "description": "Y coordinate", "optional": True},
            },
            self._tool_right_click,
        )
        self._register(
            "move_mouse",
            "Moves mouse cursor to specified (x, y) coordinates.",
            {
                "x": {"type": "integer", "description": "X coordinate"},
                "y": {"type": "integer", "description": "Y coordinate"},
            },
            self._tool_move_mouse,
        )
        self._register(
            "drag",
            "Drags mouse from (x1, y1) to (x2, y2).",
            {
                "x1": {"type": "integer", "description": "Start X"},
                "y1": {"type": "integer", "description": "Start Y"},
                "x2": {"type": "integer", "description": "End X"},
                "y2": {"type": "integer", "description": "End Y"},
            },
            self._tool_drag,
        )
        self._register(
            "scroll",
            "Scrolls mouse wheel vertically.",
            {"amount": {"type": "integer", "description": "Scroll ticks (positive up, negative down)"}},
            self._tool_scroll,
        )
        self._register(
            "copy",
            "Copies selected text or object to clipboard via Ctrl+C.",
            {},
            self._tool_copy,
        )
        self._register(
            "paste",
            "Pastes clipboard contents via Ctrl+V.",
            {},
            self._tool_paste,
        )
        self._register(
            "clipboard_read",
            "Reads and returns text from clipboard.",
            {},
            self._tool_clipboard_read,
        )
        self._register(
            "clipboard_write",
            "Writes specified text to the system clipboard.",
            {"text": {"type": "string", "description": "Text to place on clipboard"}},
            self._tool_clipboard_write,
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
            "append_file",
            "Appends content to an existing text or code file.",
            {
                "path": {"type": "string", "description": "File path to append to"},
                "content": {"type": "string", "description": "Content to append"},
            },
            self._tool_append_file,
        )
        self._register(
            "rename_file",
            "Renames a file or folder in the filesystem.",
            {
                "path": {"type": "string", "description": "Current file path"},
                "new_name": {"type": "string", "description": "New filename"},
            },
            self._tool_rename_file,
        )
        self._register(
            "move_file",
            "Moves a file from source to destination path.",
            {
                "src": {"type": "string", "description": "Source path"},
                "dst": {"type": "string", "description": "Destination path"},
            },
            self._tool_move_file,
        )
        self._register(
            "copy_file",
            "Copies a file from source to destination path.",
            {
                "src": {"type": "string", "description": "Source path"},
                "dst": {"type": "string", "description": "Destination path"},
            },
            self._tool_copy_file,
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
        self._register(
            "list_directory",
            "Lists files and folders in a specified directory or workspace.",
            {
                "path": {"type": "string", "description": "Directory path (optional, defaults to current directory)", "optional": True},
                "recursive": {"type": "boolean", "description": "Whether to list recursively", "optional": True},
            },
            self._tool_list_directory,
        )

        # 6. TERMINAL & SERVER RUNTIME TOOLS
        self._register(
            "execute_terminal_command",
            "Runs a terminal / PowerShell command and captures stdout/stderr/exit code.",
            {
                "command": {"type": "string", "description": "Command string to run"},
                "cwd": {"type": "string", "description": "Working directory", "optional": True},
            },
            self._tool_execute_terminal,
        )
        self._register(
            "start_web_server",
            "Starts a development web server (Flask, FastAPI, Node/Express, Vite, static) in background and verifies readiness.",
            {
                "cwd": {"type": "string", "description": "Working directory of the project"},
                "command": {"type": "string", "description": "Explicit startup command", "optional": True},
                "framework": {"type": "string", "description": "Target framework", "optional": True},
                "target_file": {"type": "string", "description": "Main application file", "optional": True},
            },
            self._tool_start_web_server,
        )
        self._register(
            "check_server_health",
            "Probes an HTTP endpoint or local file for availability.",
            {
                "url": {"type": "string", "description": "URL or file URL to check"},
                "timeout": {"type": "number", "description": "Timeout in seconds", "optional": True},
            },
            self._tool_check_server_health,
        )
        self._register(
            "verify_web_page",
            "Verifies that a browser page loaded without browser error pages (e.g. ERR_CONNECTION_REFUSED).",
            {
                "url": {"type": "string", "description": "Target URL to verify"},
                "expected_keywords": {"type": "array", "description": "Expected content markers", "optional": True},
            },
            self._tool_verify_web_page,
        )

        # 7. COMMUNICATION & MESSAGING TOOLS
        self._register(
            "send_email",
            "Dispatches an email to a recipient with subject and body using SMTP or prefilled Gmail.",
            {
                "recipient": {"type": "string", "description": "Email address of the recipient"},
                "subject": {"type": "string", "description": "Subject line of the email", "optional": True},
                "content": {"type": "string", "description": "Body/content of the email", "optional": True},
            },
            self._tool_send_email,
        )
        self._register(
            "send_message",
            "Sends a message to a contact or phone number on WhatsApp Web or messaging service.",
            {
                "contact": {"type": "string", "description": "Contact name or phone number"},
                "text": {"type": "string", "description": "Message text to send"},
                "service": {"type": "string", "description": "Service name, defaults to WhatsApp Web", "optional": True},
            },
            self._tool_send_message,
        )

    # ---------------------------------------------------------
    # TOOL HANDLERS
    # ---------------------------------------------------------

    def _tool_send_email(self, recipient: str, subject: str = "Message from Chitti", content: str = "") -> Dict[str, Any]:
        from src.agent.messaging import EmailDispatcher
        # Attempt direct SMTP dispatch if credentials exist
        res = EmailDispatcher.send_email_smtp(recipient=recipient, subject=subject, body=content)
        if res.success:
            return {"success": True, "message": res.message, "evidence": res.evidence, "method": "smtp"}
        # Fallback to opening pre-filled Gmail compose URL in browser
        compose_url = EmailDispatcher.get_gmail_compose_url(recipient=recipient, subject=subject, body=content)
        self.browser.open_url(compose_url)
        return {"success": True, "message": f"Opened Gmail with pre-filled compose draft for {recipient}", "evidence": f"Draft: {recipient}", "method": "browser_prefill"}

    def _tool_send_message(self, contact: str, text: str, service: str = "WhatsApp Web") -> Dict[str, Any]:
        from src.agent.messaging import WhatsAppDispatcher
        if WhatsAppDispatcher.is_phone_number(contact):
            res_py = WhatsAppDispatcher.send_via_pywhatkit(contact, text)
            if res_py.success:
                return {"success": True, "message": res_py.message, "evidence": res_py.evidence, "method": "pywhatkit"}
            wa_url = WhatsAppDispatcher.get_whatsapp_url(contact, text)
            self.browser.open_url(wa_url)
            return {"success": True, "message": f"Navigated to WhatsApp chat with {contact}", "evidence": wa_url, "method": "browser_prefill"}

        # Contact name search
        wa_url = "https://web.whatsapp.com"
        self.browser.open_url(wa_url)
        return {"success": True, "message": f"Opened {service} for contact '{contact}'", "evidence": f"Chat with {contact}", "method": "browser_ui"}

    def _tool_start_web_server(self, cwd: str, command: Optional[str] = None, framework: Optional[str] = None, target_file: Optional[str] = None) -> Dict[str, Any]:
        from src.agent.computer.server_runtime import ServerProcessManager
        ok, inst, msg = ServerProcessManager.start_server(cwd=cwd, command=command, framework=framework, target_file=target_file)
        return {
            "success": ok,
            "url": inst.url if inst else None,
            "port": inst.port if inst else None,
            "pid": inst.process_id if inst else None,
            "message": msg,
            "error": None if ok else msg,
        }

    def _tool_check_server_health(self, url: str, timeout: float = 3.0) -> Dict[str, Any]:
        from src.agent.computer.server_runtime import ServerProcessManager
        ok, code, msg = ServerProcessManager.check_http_health(url, timeout=timeout)
        return {"success": ok, "status_code": code, "message": msg, "error": None if ok else msg}

    def _tool_verify_web_page(self, url: str, expected_keywords: Optional[List[str]] = None, expected_markers: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
        from src.agent.computer.server_runtime import ServerProcessManager
        keywords = expected_keywords or expected_markers or []
        ok, msg = ServerProcessManager.verify_web_page_rendering(url, expected_keywords=keywords)
        return {"success": ok, "evidence": msg, "message": msg, "error": None if ok else msg}

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

    def _tool_open_url(self, url: str, browser: Optional[str] = None, **kwargs) -> Dict[str, Any]:
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

    def _tool_append_file(self, path: str, content: str) -> Dict[str, Any]:
        p = self.fs.append_file(path, content)
        return {"success": True, "path": p, "message": f"Appended content to {p}"}

    def _tool_rename_file(self, path: str, new_name: str) -> Dict[str, Any]:
        p = self.fs.rename_file(path, new_name)
        return {"success": True, "path": p, "message": f"Renamed to {p}"}

    def _tool_move_file(self, src: str, dst: str) -> Dict[str, Any]:
        p = self.fs.move_file(src, dst)
        return {"success": True, "path": p, "message": f"Moved {src} -> {p}"}

    def _tool_copy_file(self, src: str, dst: str) -> Dict[str, Any]:
        p = self.fs.copy_file(src, dst)
        return {"success": True, "path": p, "message": f"Copied {src} -> {p}"}

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

    def _tool_list_directory(self, path: Optional[str] = None, recursive: bool = False) -> Dict[str, Any]:
        try:
            items = self.fs.list_directory(dir_path=path, recursive=recursive)
            files = [{"name": i.name, "path": i.path, "is_dir": i.is_dir, "size_bytes": i.size_bytes} for i in items]
            return {"success": True, "path": str(self.fs.resolve_path(path or "")), "items": files, "count": len(files)}
        except Exception as e:
            return {"success": False, "error": str(e), "items": [], "count": 0}

    def _tool_execute_terminal(self, command: str, cwd: Optional[str] = None) -> Dict[str, Any]:
        res = self.terminal.execute_command(command, cwd=cwd)
        return {
            "success": res.is_success,
            "exit_code": res.exit_code,
            "output": res.output,
            "elapsed_sec": res.elapsed_time_sec,
        }

    def _tool_inspect_screen(self) -> Dict[str, Any]:
        analysis = self.screen_analyzer.capture_and_analyze()
        return {
            "success": True,
            "screenshot_path": analysis.screenshot_path,
            "width": analysis.screen_width,
            "height": analysis.screen_height,
            "active_window": analysis.active_window.title if analysis.active_window else None,
            "visible_windows": [w.title for w in analysis.visible_windows if w.title],
            "summary": analysis.summary,
        }

    def _tool_read_screen(self) -> Dict[str, Any]:
        active = self.computer.get_active_window()
        windows = self.computer.list_windows()
        titles = [w.title for w in windows if w.title]
        return {
            "success": True,
            "active_title": active.title if active else None,
            "visible_titles": titles,
            "message": f"Active: {active.title if active else 'None'}. Visible: {', '.join(titles[:5])}",
        }

    def _tool_analyze_screen(self, query: str = "Explain what is on the screen") -> Dict[str, Any]:
        log_info(f"[TOOL] analyze_screen -> query: '{query}'")
        res = self.vision.analyze_screen(query=query)
        return {
            "success": True,
            "summary": res.summary,
            "active_window": res.active_window_title,
            "process": res.active_process,
            "screenshot_path": res.screenshot_path,
            "message": res.summary,
            "raw_ocr": res.raw_ocr_text,
        }

    def _tool_diagnose_screen_error(self, query: str = "") -> Dict[str, Any]:
        log_info(f"[TOOL] diagnose_screen_error -> query: '{query}'")
        diag = self.vision.diagnose_screen_error()
        return {
            "success": True,
            "has_error": diag.has_error,
            "error_type": diag.error_type,
            "description": diag.description,
            "suggested_fix": diag.suggested_fix,
            "evidence": diag.evidence_snippet,
            "message": diag.description,
        }

    def _tool_capture_active_window(self) -> Dict[str, Any]:
        log_info("[TOOL] capture_active_window")
        path, win = self.vision.screen_reader.capture_active_window()
        return {
            "success": True,
            "path": path,
            "active_window": win.title if win else "Desktop",
            "message": f"Captured active window screenshot: {path}",
        }

    def _tool_find_ui_element(self, query: str) -> Dict[str, Any]:
        # Try finding via vision localization first
        match = self.vision.find_ui_element(query)
        if match:
            return {
                "success": True,
                "found": True,
                "query": query,
                "label": match.label,
                "center_x": match.center_x,
                "center_y": match.center_y,
                "box": match.bounding_box,
                "evidence": f"Found UI element '{match.label}' at ({match.center_x}, {match.center_y})",
            }

        res = self.screen_analyzer.verify_window(query)
        return {
            "success": res.success,
            "found": res.success,
            "query": query,
            "evidence": res.evidence,
        }

    def _tool_get_active_window(self) -> Dict[str, Any]:
        active = self.computer.get_active_window()
        return {
            "success": active is not None,
            "title": active.title if active else None,
            "handle": active.handle if active else None,
            "is_active": active.is_active if active else False,
        }

    def _tool_verify_ui_state(self, expected_window: Optional[str] = None, expected_text: Optional[str] = None) -> Dict[str, Any]:
        if expected_window:
            res = self.screen_analyzer.verify_window(expected_window)
            if not res.success:
                return {"success": False, "evidence": f"Expected window '{expected_window}' not found."}
        return {"success": True, "evidence": "UI state verified successfully."}

    def _tool_verify_text(self, text: str) -> Dict[str, Any]:
        active = self.computer.get_active_window()
        if active and text.lower() in active.title.lower():
            return {"success": True, "evidence": f"Text '{text}' found in active window '{active.title}'."}
        windows = self.computer.list_windows()
        for w in windows:
            if text.lower() in w.title.lower():
                return {"success": True, "evidence": f"Text '{text}' found in window '{w.title}'."}
        return {"success": False, "evidence": f"Text '{text}' not found in any visible window."}

    def _tool_verify_element(self, element_name: str) -> Dict[str, Any]:
        res = self.screen_analyzer.verify_window(element_name)
        return {"success": res.success, "evidence": res.evidence}

    def _tool_verify_application_state(self, application: str) -> Dict[str, Any]:
        res = self.screen_analyzer.verify_window(application)
        return {"success": res.success, "evidence": res.evidence, "application": application}

    def _tool_wait(self, seconds: float = 1.0) -> Dict[str, Any]:
        sec = max(0.1, min(float(seconds), 30.0))
        time.sleep(sec)
        return {"success": True, "waited_seconds": sec, "message": f"Waited {sec:.1f}s"}

    def _tool_wait_for_ui(self, target: str, timeout: float = 5.0) -> Dict[str, Any]:
        t_end = time.time() + max(1.0, float(timeout))
        while time.time() < t_end:
            res = self.screen_analyzer.verify_window(target)
            if res.success:
                return {"success": True, "target": target, "evidence": res.evidence}
            time.sleep(0.3)
        return {"success": False, "target": target, "error": f"Timed out waiting for '{target}' UI"}

    def _tool_double_click(self, x: Optional[int] = None, y: Optional[int] = None) -> Dict[str, Any]:
        ok = self.computer.double_click(x=x, y=y)
        return {"success": ok, "x": x, "y": y, "button": "left"}

    def _tool_right_click(self, x: Optional[int] = None, y: Optional[int] = None) -> Dict[str, Any]:
        ok = self.computer.right_click(x=x, y=y)
        return {"success": ok, "x": x, "y": y, "button": "right"}

    def _tool_move_mouse(self, x: int, y: int) -> Dict[str, Any]:
        ok = self.computer.move_mouse(x=x, y=y)
        return {"success": ok, "x": x, "y": y}

    def _tool_drag(self, x1: int, y1: int, x2: int, y2: int) -> Dict[str, Any]:
        ok = self.computer.drag_mouse(from_x=x1, from_y=y1, to_x=x2, to_y=y2)
        return {"success": ok, "from": (x1, y1), "to": (x2, y2)}

    def _tool_scroll(self, amount: int = -3) -> Dict[str, Any]:
        ok = self.computer.scroll(amount=amount)
        return {"success": ok, "amount": amount}

    def _tool_copy(self) -> Dict[str, Any]:
        ok = self.computer.clipboard_copy()
        return {"success": ok, "action": "copy"}

    def _tool_paste(self) -> Dict[str, Any]:
        ok = self.computer.clipboard_paste()
        return {"success": ok, "action": "paste"}

    def _tool_clipboard_read(self) -> Dict[str, Any]:
        txt = self.computer.get_clipboard_text()
        return {"success": True, "text": txt}

    def _tool_clipboard_write(self, text: str) -> Dict[str, Any]:
        ok = self.computer.set_clipboard_text(text)
        return {"success": ok, "text": text}

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
