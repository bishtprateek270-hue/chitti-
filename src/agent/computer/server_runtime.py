"""
Chitti Web Application Server Runtime & Process Manager (Phase 6).
Provides managed execution of local web servers (Flask, FastAPI, Node/Express, Vite, Next.js, static HTTP),
dynamic port conflict resolution, URL discovery from stdout, readiness polling, HTTP health checks,
and browser error page diagnostics.
"""

import os
import re
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.utils.logging import log_debug, log_info, log_warn


@dataclass
class ServerInstance:
    """Represents a running local development/backend server process."""
    process_id: int
    command: str
    working_directory: str
    framework: str
    port: int
    url: str
    process: Optional[subprocess.Popen] = None
    stdout_lines: List[str] = field(default_factory=list)
    stderr_lines: List[str] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    status: str = "STARTING"  # "STARTING", "READY", "FAILED", "STOPPED"
    error_message: Optional[str] = None


class ServerProcessManager:
    """
    Manages development and application web servers across arbitrary web stacks.
    Handles startup, readiness probing, port detection, background process supervision,
    and graceful shutdown.
    """

    # Common HTTP error signatures returned by browsers or proxy layers
    BROWSER_ERROR_PATTERNS = [
        r"(?i)this site can'?t be reached",
        r"(?i)err_connection_refused",
        r"(?i)err_connection_reset",
        r"(?i)err_empty_response",
        r"(?i)err_connection_timed_out",
        r"(?i)err_name_not_resolved",
        r"(?i)404\s+not\s+found",
        r"(?i)500\s+internal\s+server\s+error",
        r"(?i)502\s+bad\s+gateway",
        r"(?i)503\s+service\s+unavailable",
    ]

    _active_servers: Dict[int, ServerInstance] = {}

    @classmethod
    def is_port_in_use(cls, port: int, host: str = "127.0.0.1") -> bool:
        """Checks if a given TCP port on host is currently bound."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            return s.connect_ex((host, port)) == 0

    @classmethod
    def find_free_port(cls, start_port: int = 5000, max_attempts: int = 100) -> int:
        """Finds the next available unused TCP port on localhost."""
        for port in range(start_port, start_port + max_attempts):
            if not cls.is_port_in_use(port):
                return port
        # Fallback to OS-assigned ephemeral port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    @classmethod
    def check_http_health(cls, url: str, timeout: float = 3.0) -> Tuple[bool, int, str]:
        """
        Executes a real HTTP GET request or file check to verify service availability.
        Returns: (is_healthy, status_code, message)
        """
        target = url.strip()

        # 1. Local File URL Check
        if target.startswith("file:///"):
            file_path = target.replace("file:///", "")
            # Handle Windows drive letters correctly
            if re.match(r"^[a-zA-Z]:", file_path):
                p = Path(file_path)
            else:
                p = Path("/" + file_path) if not file_path.startswith("/") else Path(file_path)

            if p.exists() and p.is_file():
                try:
                    content = p.read_text(encoding="utf-8", errors="ignore")
                    if len(content) > 10:
                        return True, 200, f"Local file verified ({p.name}, {len(content)} bytes)"
                except Exception as e:
                    return False, 500, f"Failed to read local file: {e}"
            return False, 404, f"File does not exist: {file_path}"

        # 2. HTTP / HTTPS Network Check
        if not target.startswith(("http://", "https://")):
            target = f"http://{target}"

        try:
            req = urllib.request.Request(
                target,
                headers={"User-Agent": "Chitti-Runtime-HealthChecker/1.0", "Accept": "*/*"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                code = resp.getcode()
                if code in (200, 201, 204, 301, 302, 304):
                    return True, code, f"HTTP {code} OK"
                return True, code, f"HTTP {code} Received"
        except urllib.error.HTTPError as e:
            # 404 or 500 on an API root can still indicate the server process is alive and responding
            if e.code in (404, 405, 401, 403):
                return True, e.code, f"Server reachable (HTTP {e.code})"
            return False, e.code, f"Server returned error HTTP {e.code}: {e.reason}"
        except urllib.error.URLError as e:
            return False, 0, f"Connection failed: {e.reason}"
        except Exception as e:
            return False, 0, f"Health check failed: {e}"

    @classmethod
    def detect_server_config(cls, cwd: str, filename: Optional[str] = None) -> Tuple[str, str, int]:
        """
        Dynamically detects web framework, appropriate startup command, and target port.
        Returns: (framework, command, default_port)
        """
        p_cwd = Path(cwd).resolve()
        package_json = p_cwd / "package.json"

        # 1. Node / React / Vite / Next.js
        if package_json.exists():
            try:
                import json
                with open(package_json, "r", encoding="utf-8") as f:
                    pkg = json.load(f)
                scripts = pkg.get("scripts", {})
                deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}

                if "next" in deps:
                    return "nextjs", "npm run dev", cls.find_free_port(3000)
                elif "vite" in deps:
                    port = cls.find_free_port(5173)
                    return "vite", f"npx vite --port {port}", port
                elif "react-scripts" in deps:
                    return "react", "npm start", cls.find_free_port(3000)
                elif "express" in deps or "server.js" in scripts.get("start", ""):
                    main_file = pkg.get("main", "server.js")
                    port = cls.find_free_port(3000)
                    return "express", f"node {main_file}", port
                elif "dev" in scripts:
                    return "node_dev", "npm run dev", cls.find_free_port(3000)
                elif "start" in scripts:
                    return "node_start", "npm start", cls.find_free_port(3000)
            except Exception as e:
                log_debug(f"[SERVER_DETECT] package.json parse notice: {e}")

        # 2. Python Backend Frameworks (Flask / FastAPI)
        if filename:
            target_file = (p_cwd / filename).name if (p_cwd / filename).exists() else filename
        else:
            py_files = list(p_cwd.glob("*.py"))
            target_file = py_files[0].name if py_files else "app.py"

        file_content = ""
        full_path = p_cwd / target_file
        if full_path.exists():
            try:
                file_content = full_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                pass

        if "fastapi" in file_content.lower() or "uvicorn" in file_content.lower():
            port = cls.find_free_port(8000)
            mod_name = Path(target_file).stem
            return "fastapi", f"uvicorn {mod_name}:app --port {port} --host 127.0.0.1", port

        if "flask" in file_content.lower():
            port = cls.find_free_port(5000)
            return "flask", f"python {target_file}", port

        # 3. Static Web Project (HTML/CSS/JS / standalone JSX/TSX)
        if (filename and filename.endswith((".html", ".htm", ".jsx", ".tsx", ".js", ".mjs"))) or "<!doctype html" in file_content.lower() or "<html" in file_content.lower():
            port = cls.find_free_port(8080)
            return "static_html", f"python -m http.server {port}", port

        # Default fallback
        port = cls.find_free_port(8000)
        return "generic_python", f"python {target_file}", port

    @classmethod
    def start_server(
        cls,
        cwd: str,
        command: Optional[str] = None,
        framework: Optional[str] = None,
        target_file: Optional[str] = None,
        readiness_timeout: float = 15.0,
    ) -> Tuple[bool, Optional[ServerInstance], str]:
        """
        Launches development web server in background, captures logs, detects URL,
        and polls HTTP health until ready.
        """
        auto_fw, auto_cmd, auto_port = cls.detect_server_config(cwd, filename=target_file)
        fw = framework or auto_fw
        cmd = command or auto_cmd
        port = auto_port
        if (auto_fw == "static_html" or not (Path(cwd) / "package.json").exists()) and target_file and target_file.lower() not in ("index.html", "index.htm"):
            url = f"http://localhost:{port}/{target_file}"
        else:
            url = f"http://localhost:{port}"

        log_info(f"[SERVER] Launching {fw} server: '{cmd}' in {cwd} (Port: {port})")

        try:
            process = subprocess.Popen(
                cmd,
                cwd=cwd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except Exception as e:
            err = f"Failed to start server process: {e}"
            log_warn(f"[SERVER] {err}")
            return False, None, err

        instance = ServerInstance(
            process_id=process.pid,
            command=cmd,
            working_directory=cwd,
            framework=fw,
            port=port,
            url=url,
            process=process,
            status="STARTING",
        )
        cls._active_servers[process.pid] = instance

        # Spawn asynchronous log reader threads
        def read_pipe(pipe, line_list):
            try:
                for line in iter(pipe.readline, ""):
                    if line:
                        line_clean = line.strip()
                        line_list.append(line_clean)
                        # Detect dynamic URL announcement in logs
                        m = re.search(r"(?i)(?:running on|local:|listening on|url:)\s*(https?://[^\s\)]+)", line_clean)
                        if m:
                            detected_url = m.group(1).rstrip("/")
                            instance.url = detected_url
                            log_debug(f"[SERVER LOG] Detected dynamic URL: {detected_url}")
                pipe.close()
            except Exception:
                pass

        t_out = threading.Thread(target=read_pipe, args=(process.stdout, instance.stdout_lines), daemon=True)
        t_err = threading.Thread(target=read_pipe, args=(process.stderr, instance.stderr_lines), daemon=True)
        t_out.start()
        t_err.start()

        # Readiness & Health Check Loop
        start_wait = time.time()
        while time.time() - start_wait < readiness_timeout:
            # Check if process terminated prematurely
            ret = process.poll()
            if ret is not None:
                err_details = "\n".join(instance.stderr_lines[-5:]) or "\n".join(instance.stdout_lines[-5:]) or f"Process exited with code {ret}"
                instance.status = "FAILED"
                instance.error_message = err_details
                log_warn(f"[SERVER FAILED] Process exited prematurely: {err_details}")
                return False, instance, f"Server process stopped unexpectedly: {err_details}"

            # Probe HTTP health
            healthy, code, msg = cls.check_http_health(instance.url, timeout=1.0)
            if healthy:
                instance.status = "READY"
                log_info(f"[SERVER READY] {fw} server ready at {instance.url} ({msg}) in {time.time() - start_wait:.2f}s")
                return True, instance, f"Server running and healthy at {instance.url}"

            time.sleep(0.5)

        # Timeout reached without positive health check
        err_details = "\n".join(instance.stderr_lines[-5:]) or "Server readiness timeout exceeded without reachable HTTP endpoint."
        instance.status = "FAILED"
        instance.error_message = err_details
        log_warn(f"[SERVER TIMEOUT] {err_details}")
        return False, instance, f"Server failed to become healthy within {readiness_timeout}s: {err_details}"

    @classmethod
    def stop_server(cls, pid: int) -> bool:
        """Terminates an active server process."""
        inst = cls._active_servers.pop(pid, None)
        if inst and inst.process:
            try:
                inst.process.terminate()
                inst.process.wait(timeout=2.0)
                inst.status = "STOPPED"
                log_info(f"[SERVER] Stopped process {pid}")
                return True
            except Exception:
                try:
                    inst.process.kill()
                    inst.status = "STOPPED"
                    return True
                except Exception:
                    pass
        return False

    @classmethod
    def stop_all_servers(cls) -> None:
        """Cleanly halts all spawned background servers."""
        pids = list(cls._active_servers.keys())
        for pid in pids:
            cls.stop_server(pid)

    @classmethod
    def verify_web_page_rendering(
        cls,
        url: str,
        expected_keywords: Optional[List[str]] = None,
        timeout: float = 3.0,
    ) -> Tuple[bool, str]:
        """
        Verifies that a target URL or local HTML page actually loads valid content
        and is NOT displaying a browser error page.
        """
        target = url.strip()

        # 1. Local HTML Verification
        if target.startswith("file:///"):
            file_path = target.replace("file:///", "")
            if not re.match(r"^[a-zA-Z]:", file_path) and not file_path.startswith("/"):
                file_path = "/" + file_path
            p = Path(file_path)
            if not p.exists():
                return False, f"Browser verification failed: file '{file_path}' does not exist on disk."
            try:
                content = p.read_text(encoding="utf-8", errors="ignore")
                if len(content.strip()) < 20:
                    return False, "Browser verification failed: rendered page content is empty."
                if expected_keywords:
                    for kw in expected_keywords:
                        if kw.lower() not in content.lower():
                            return False, f"Browser verification failed: expected UI marker '{kw}' was not found in rendered page."
                return True, f"Rendered page verified successfully ({p.name}, {len(content)} bytes)."
            except Exception as e:
                return False, f"Failed to read rendered HTML: {e}"

        # 2. Network URL Verification
        healthy, code, msg = cls.check_http_health(target, timeout=timeout)
        if not healthy:
            return False, f"Browser error page detected: server at '{target}' is unreachable ({msg})."

        try:
            req = urllib.request.Request(target, headers={"User-Agent": "Chitti-PageVerifier/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8", errors="ignore")

                # Detect common browser error pages
                for err_pat in cls.BROWSER_ERROR_PATTERNS:
                    if re.search(err_pat, body):
                        return False, f"Browser error page detected: '{err_pat}' found in response."

                # Verify expected keywords if provided
                if expected_keywords:
                    for kw in expected_keywords:
                        if kw.lower() not in body.lower():
                            return False, f"Browser verification failed: expected keyword '{kw}' not found in rendered response."

                return True, f"Browser page loaded successfully at {target} (HTTP {code})."
        except Exception as e:
            return False, f"Browser connection error: {e}"
