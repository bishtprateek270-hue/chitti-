"""
Chitti Inbound Notification & Webhook Server for n8n.
Runs a lightweight, zero-dependency HTTP server on a background thread so n8n
(or any external webhook) can push proactive notifications, events, and commands directly into Chitti.
"""

import json
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Callable, Dict, Optional

from src.utils.logging import log_chitti, log_debug, log_error, log_info, log_warning


class N8NNotificationHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler for inbound n8n notifications and triggers."""

    # Handlers injected by the server
    notification_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    command_callback: Optional[Callable[[str], None]] = None
    memory_callback: Optional[Callable[[str], None]] = None

    def _send_json_response(self, status_code: int, data: Dict[str, Any]):
        try:
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
            self.end_headers()
            self.wfile.write(body)
        except Exception as e:
            log_debug(f"[n8n-server] Error sending response: {e}")

    def do_OPTIONS(self):
        """Handle CORS pre-flight requests."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_GET(self):
        """Handle Health Check and Info GET requests."""
        clean_path = self.path.split("?")[0].rstrip("/")
        if clean_path in ("", "/health", "/chitti/health", "/api/health"):
            self._send_json_response(200, {
                "status": "online",
                "service": "Chitti Inbound Automation Server",
                "companion": "Chitti",
                "creator": "Prateek Singh Bisht",
                "endpoints": [
                    "POST /chitti/notify - Send proactive audio & HUD notification",
                    "POST /chitti/command - Execute natural language command",
                    "POST /chitti/memory - Save fact to long-term memory",
                    "GET /health - Server health status"
                ]
            })
        else:
            self._send_json_response(404, {"error": "Endpoint not found", "path": self.path})

    def do_POST(self):
        """Handle incoming webhooks from n8n."""
        clean_path = self.path.split("?")[0].rstrip("/")
        content_length = int(self.headers.get("Content-Length", 0))

        if content_length > 0:
            raw_body = self.rfile.read(content_length).decode("utf-8", errors="replace")
            try:
                payload = json.loads(raw_body) if raw_body.strip() else {}
            except Exception as e:
                self._send_json_response(400, {"error": f"Invalid JSON payload: {e}"})
                return
        else:
            payload = {}

        # 1. NOTIFICATION ENDPOINT: /chitti/notify or /notify
        if clean_path in ("/chitti/notify", "/notify", "/api/notify"):
            message = payload.get("message") or payload.get("text") or payload.get("content") or ""
            title = payload.get("title", "n8n Alert")
            speak = payload.get("speak", True)
            show_hud = payload.get("show_hud", True)
            severity = payload.get("severity", "info")

            if not message:
                self._send_json_response(400, {"error": "Missing 'message' field in notification payload"})
                return

            log_chitti(f"[n8n-inbound] 🔔 Inbound Notification from n8n: [{title}] {message}")

            cb = getattr(self.server, "notification_callback", None) or N8NNotificationHandler.notification_callback
            if cb:
                try:
                    cb({
                        "title": title,
                        "message": message,
                        "speak": speak,
                        "show_hud": show_hud,
                        "severity": severity,
                        "raw_payload": payload,
                    })
                except Exception as e:
                    log_error(f"[n8n-inbound] Error executing notification callback: {e}")

            self._send_json_response(200, {
                "success": True,
                "status": "received",
                "message": f"Notification delivered to Chitti.",
            })
            return

        # 2. COMMAND ENDPOINT: /chitti/command or /command
        if clean_path in ("/chitti/command", "/command", "/api/command"):
            command = payload.get("command") or payload.get("text") or payload.get("prompt") or ""
            if not command:
                self._send_json_response(400, {"error": "Missing 'command' field in payload"})
                return

            log_chitti(f"[n8n-inbound] ⚡ Remote Command from n8n: '{command}'")

            cb = getattr(self.server, "command_callback", None) or N8NNotificationHandler.command_callback
            if cb:
                try:
                    cb(command)
                except Exception as e:
                    log_error(f"[n8n-inbound] Error executing command callback: {e}")

            self._send_json_response(200, {
                "success": True,
                "status": "queued",
                "command": command,
                "message": f"Command forwarded to Chitti executive pipeline.",
            })
            return

        # 3. MEMORY ENDPOINT: /chitti/memory or /memory
        if clean_path in ("/chitti/memory", "/memory", "/api/memory"):
            text = payload.get("text") or payload.get("memory") or payload.get("content") or ""
            if not text:
                self._send_json_response(400, {"error": "Missing 'text' field in payload"})
                return

            log_chitti(f"[n8n-inbound] 🧠 Long-Term Memory Injection: '{text}'")

            cb = getattr(self.server, "memory_callback", None) or N8NNotificationHandler.memory_callback
            if cb:
                try:
                    cb(text)
                except Exception as e:
                    log_error(f"[n8n-inbound] Error executing memory callback: {e}")

            self._send_json_response(200, {
                "success": True,
                "status": "saved",
                "message": "Memory entry recorded in Chitti's database.",
            })
            return

        # Unrecognized endpoint
        self._send_json_response(404, {"error": "Endpoint not found", "path": self.path})

    def log_message(self, format, *args):
        """Suppress default stdout web server spam; log via Chitti logger at debug level."""
        log_debug(f"[n8n-server] {format % args}")


class N8NInboundServer:
    """Manages the background HTTP server for receiving n8n notifications and triggers."""

    def __init__(
        self,
        port: int = 8765,
        host: str = "0.0.0.0",
        notification_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        command_callback: Optional[Callable[[str], None]] = None,
        memory_callback: Optional[Callable[[str], None]] = None,
    ):
        self.port = port
        self.host = host
        self.notification_callback = notification_callback
        self.command_callback = command_callback
        self.memory_callback = memory_callback
        self.httpd: Optional[HTTPServer] = None
        self.thread: Optional[threading.Thread] = None
        self._running = False

    def start(self):
        """Starts the inbound HTTP listener on a background daemon thread."""
        if self._running:
            return

        # Inject callbacks into handler class
        N8NNotificationHandler.notification_callback = self.notification_callback
        N8NNotificationHandler.command_callback = self.command_callback
        N8NNotificationHandler.memory_callback = self.memory_callback

        try:
            self.httpd = HTTPServer((self.host, self.port), N8NNotificationHandler)
            self.httpd.notification_callback = self.notification_callback
            self.httpd.command_callback = self.command_callback
            self.httpd.memory_callback = self.memory_callback
            self._running = True
            self.thread = threading.Thread(
                target=self.httpd.serve_forever,
                name=f"N8NInboundServer-{self.port}",
                daemon=True,
            )
            self.thread.start()
            log_chitti(f"[n8n] 🌐 Inbound Webhook Listener running on http://localhost:{self.port} (endpoints: /chitti/notify, /chitti/command)")
        except Exception as e:
            log_warning(f"[n8n] Could not start inbound server on port {self.port}: {e}")
            self._running = False

    def stop(self):
        """Stops the inbound HTTP listener."""
        if not self._running:
            return
        self._running = False
        if self.httpd:
            try:
                self.httpd.shutdown()
                self.httpd.server_close()
            except Exception as e:
                log_debug(f"[n8n] Error closing server: {e}")
            self.httpd = None
        log_debug("[n8n] Inbound server stopped.")

    @property
    def is_running(self) -> bool:
        return self._running
