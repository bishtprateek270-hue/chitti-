"""
Tests for Phase 6 Web Application Runtime, Process Management, HTTP Health Checks,
and Browser Verification (Phase 6 Fix).
"""

import json
import socket
import tempfile
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import pytest

from src.agent.code_generator import CodeGenerator
from src.agent.code_spec import CodeFileSpec, ProgrammingTaskSpec, ProjectSpecification
from src.agent.computer import BrowserController, FilesystemController, ServerInstance, ServerProcessManager
from src.agent.planner import AgentPlanner, ComputerAgentLoop
from src.agent.projects import ProjectRegistry
from src.agent.task_state import AgentStep, ExecutionFlag, StepStatus, TaskContext, TaskState, TaskStatus
from src.agent.tools import ToolEngine
from src.agent.verifier import SubtaskVerifier


class MockHTTPHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/error":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(b"<html><body><h1>This site can't be reached</h1><p>ERR_CONNECTION_REFUSED</p></body></html>")
        elif self.path == "/app":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(b"<html><body><h1>Expense Tracker</h1><div id='expense-form'>Form Ready</div></body></html>")
        else:
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(b"<html><body><h1>Chitti Web App</h1></body></html>")

    def log_message(self, format, *args):
        pass  # Quiet logging in tests


def test_free_port_discovery_and_conflict_handling():
    """Verify ServerProcessManager dynamically detects ports and avoids conflicts."""
    port1 = ServerProcessManager.find_free_port(start_port=5000)
    assert isinstance(port1, int)
    assert 1024 <= port1 <= 65535

    # Bind a socket to port1 and verify find_free_port picks a different port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", port1))
        s.listen(1)
        port2 = ServerProcessManager.find_free_port(start_port=port1)
        assert port2 != port1
        assert not ServerProcessManager.is_port_in_use(port2)


def test_dynamic_server_config_detection_for_multiple_stacks():
    """Verify ServerProcessManager detects configurations across Node, Flask, FastAPI, and HTML."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # 1. Static HTML
        html_file = tmp / "index.html"
        html_file.write_text("<!DOCTYPE html><html><body><h1>Dashboard</h1></body></html>", encoding="utf-8")
        fw, cmd, port = ServerProcessManager.detect_server_config(str(tmp), filename="index.html")
        assert fw == "static_html"
        assert "http.server" in cmd

        # 2. Flask Python
        py_flask = tmp / "app.py"
        py_flask.write_text("from flask import Flask\napp = Flask(__name__)\n@app.route('/')\ndef home(): return 'OK'", encoding="utf-8")
        fw, cmd, port = ServerProcessManager.detect_server_config(str(tmp), filename="app.py")
        assert fw == "flask"
        assert "app.py" in cmd

        # 3. FastAPI Python
        py_fast = tmp / "main.py"
        py_fast.write_text("from fastapi import FastAPI\napp = FastAPI()\n@app.get('/')\ndef root(): return {'status': 'ok'}", encoding="utf-8")
        fw, cmd, port = ServerProcessManager.detect_server_config(str(tmp), filename="main.py")
        assert fw == "fastapi"
        assert "uvicorn" in cmd

        # 4. Node / Vite project
        pkg_vite = tmp / "package.json"
        pkg_vite.write_text(json.dumps({"scripts": {"dev": "vite"}, "dependencies": {"vite": "^4.0.0"}}), encoding="utf-8")
        fw, cmd, port = ServerProcessManager.detect_server_config(str(tmp))
        assert fw == "vite"
        assert "vite" in cmd


def test_http_health_check_on_live_and_unreachable_endpoints():
    """Verify health checker handles both live servers, file URLs, and unreachable ports."""
    # 1. Unreachable port check (must fail gracefully and NOT hang)
    unreachable_port = ServerProcessManager.find_free_port(9900)
    healthy, code, msg = ServerProcessManager.check_http_health(f"http://127.0.0.1:{unreachable_port}", timeout=0.5)
    assert healthy is False
    assert code == 0

    # 2. Local file check
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        f.write(b"<!DOCTYPE html><html><head><title>App</title></head><body><h1>Working</h1></body></html>")
        f_path = f.name

    try:
        file_url = f"file:///{f_path.replace('\\', '/')}"
        healthy, code, msg = ServerProcessManager.check_http_health(file_url)
        assert healthy is True
        assert code == 200
    finally:
        Path(f_path).unlink(missing_ok=True)


def test_browser_error_page_detection_prevents_false_success():
    """Verify that browser error pages like 'This site can't be reached' are detected as failures."""
    # Spin up a lightweight local mock server returning an error page
    server_port = ServerProcessManager.find_free_port(8765)
    server = HTTPServer(("127.0.0.1", server_port), MockHTTPHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    try:
        # Error URL
        error_url = f"http://127.0.0.1:{server_port}/error"
        loaded, ev = ServerProcessManager.verify_web_page_rendering(error_url)
        assert loaded is False
        assert "error page detected" in ev.lower()

        # Valid App URL
        app_url = f"http://127.0.0.1:{server_port}/app"
        loaded, ev = ServerProcessManager.verify_web_page_rendering(app_url, expected_keywords=["Expense Tracker"])
        assert loaded is True
        assert "loaded successfully" in ev.lower()
    finally:
        server.shutdown()


def test_planner_generates_comprehensive_web_lifecycle():
    """Verify planner dynamically creates start_server, health_check, open_url, and UI verification steps."""
    planner = AgentPlanner(project_registry=ProjectRegistry())
    task_state = planner.plan_task("create a simple expense tracker with a good UI and run it")

    assert task_state is not None
    assert len(task_state.steps) >= 5
    action_types = [s.action_type for s in task_state.steps]

    # Verify execution pipeline contains all required verification steps
    assert "CREATE_FILE" in action_types
    assert "CHECK_SERVER_HEALTH" in action_types or "START_SERVER" in action_types
    assert "OPEN_URL" in action_types
    assert "VERIFY_PAGE_LOADED" in action_types
    assert "VERIFY_UI" in action_types
    assert "VERIFY_FUNCTIONALITY" in action_types


def test_browser_controller_file_url_normalization():
    """Ensure BrowserController never corrupts file:/// URLs into https://file:///."""
    bc = BrowserController()
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        f.write(b"<html><body><h1>Test</h1></body></html>")
        f_path = f.name

    try:
        file_url = f"file:///{f_path.replace('\\', '/')}"
        success = bc.open_url(file_url, site_name="Test App")
        assert success is True
        assert bc.state.current_url == file_url
        assert not bc.state.current_url.startswith("https://file:///")
    finally:
        Path(f_path).unlink(missing_ok=True)
