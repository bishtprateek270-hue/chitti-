"""
Unit tests for Chitti n8n Workflow Automation Integration.
Tests N8NClient, N8NInboundServer, ToolEngine registration, and notification callbacks.
"""

import json
import time
import urllib.request
import urllib.error
import pytest
from unittest.mock import MagicMock, patch

from src.agent.n8n_client import N8NClient
from src.agent.n8n_server import N8NInboundServer
from src.agent.tools import ToolEngine


class TestN8NClient:
    """Tests for outbound n8n HTTP / Webhook client."""

    def test_client_initialization_defaults(self):
        client = N8NClient()
        assert client.base_url == "http://localhost:5678"
        assert client.webhook_prefix == "http://localhost:5678/webhook"

    def test_check_health_offline(self):
        client = N8NClient(base_url="http://127.0.0.1:59999")  # Unused port
        health = client.check_health()
        assert health["status"] == "offline"
        assert health["reachable"] is False
        assert "not reachable" in health["message"]

    @patch("urllib.request.urlopen")
    def test_check_health_online(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.getcode.return_value = 200
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        client = N8NClient(base_url="http://localhost:5678")
        health = client.check_health()
        assert health["status"] == "online"
        assert health["reachable"] is True

    @patch("urllib.request.urlopen")
    def test_trigger_webhook_success(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.getcode.return_value = 200
        mock_resp.read.return_value = json.dumps({"status": "received", "job_id": "abc-123"}).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        client = N8NClient()
        res = client.trigger_webhook("task-runner", payload={"action": "test"})
        assert res["success"] is True
        assert res["status_code"] == 200
        assert res["data"] == {"status": "received", "job_id": "abc-123"}

    @patch("urllib.request.urlopen")
    def test_trigger_webhook_http_error(self, mock_urlopen):
        err = urllib.error.HTTPError(
            url="http://localhost:5678/webhook/test",
            code=404,
            msg="Not Found",
            hdrs={},
            fp=MagicMock(read=lambda: b'{"error":"Workflow not found"}')
        )
        mock_urlopen.side_effect = err

        client = N8NClient()
        res = client.trigger_webhook("test")
        assert res["success"] is False
        assert res["status_code"] == 404
        assert "404" in res["message"]

    def test_list_workflows_without_key(self):
        client = N8NClient(api_key=None)
        res = client.list_workflows()
        assert res["success"] is False
        assert "N8N_API_KEY is not configured" in res["message"]


class TestN8NInboundServer:
    """Tests for inbound HTTP webhook listener server."""

    @pytest.fixture
    def test_server(self):
        received_notifications = []
        received_commands = []
        received_memories = []

        def on_notify(data):
            received_notifications.append(data)

        def on_cmd(cmd):
            received_commands.append(cmd)

        def on_mem(mem):
            received_memories.append(mem)

        # Use ephemeral high-range test port
        test_port = 8799
        server = N8NInboundServer(
            port=test_port,
            host="127.0.0.1",
            notification_callback=on_notify,
            command_callback=on_cmd,
            memory_callback=on_mem,
        )
        server.start()
        time.sleep(0.1)

        yield {
            "server": server,
            "port": test_port,
            "notifications": received_notifications,
            "commands": received_commands,
            "memories": received_memories,
        }

        server.stop()
        time.sleep(0.05)

    def test_health_endpoint(self, test_server):
        port = test_server["port"]
        url = f"http://127.0.0.1:{port}/health"

        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            assert resp.getcode() == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data["status"] == "online"
            assert data["companion"] == "Chitti"

    def test_inbound_notification(self, test_server):
        port = test_server["port"]
        url = f"http://127.0.0.1:{port}/chitti/notify"
        payload = {
            "title": "GitHub",
            "message": "PR #42 merged successfully",
            "speak": True,
            "show_hud": True,
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            assert resp.getcode() == 200
            res = json.loads(resp.read().decode("utf-8"))
            assert res["success"] is True

        assert len(test_server["notifications"]) == 1
        notif = test_server["notifications"][0]
        assert notif["title"] == "GitHub"
        assert notif["message"] == "PR #42 merged successfully"
        assert notif["speak"] is True

    def test_inbound_remote_command(self, test_server):
        port = test_server["port"]
        url = f"http://127.0.0.1:{port}/chitti/command"
        payload = {"command": "open vs code"}

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            assert resp.getcode() == 200
            res = json.loads(resp.read().decode("utf-8"))
            assert res["success"] is True

        assert test_server["commands"] == ["open vs code"]

    def test_inbound_memory_save(self, test_server):
        port = test_server["port"]
        url = f"http://127.0.0.1:{port}/chitti/memory"
        payload = {"text": "Project presentation is on Monday"}

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            assert resp.getcode() == 200
            res = json.loads(resp.read().decode("utf-8"))
            assert res["success"] is True

        assert test_server["memories"] == ["Project presentation is on Monday"]


class TestToolEngineN8NRegistration:
    """Verifies that n8n tools are registered in Chitti's ToolEngine."""

    def test_n8n_tools_registered(self):
        mock_computer = MagicMock()
        mock_fs = MagicMock()
        mock_term = MagicMock()
        mock_browser = MagicMock()
        mock_apps = MagicMock()
        mock_screen = MagicMock()
        mock_projects = MagicMock()

        engine = ToolEngine(
            computer=mock_computer,
            filesystem=mock_fs,
            terminal=mock_term,
            browser=mock_browser,
            apps=mock_apps,
            screen_analyzer=mock_screen,
            projects=mock_projects,
        )

        assert "n8n_trigger_webhook" in engine.tools
        assert "n8n_check_health" in engine.tools
        assert "n8n_list_workflows" in engine.tools

        # Check parameter schema for webhook
        webhook_tool = engine.tools["n8n_trigger_webhook"]
        assert "endpoint" in webhook_tool.parameters
        assert webhook_tool.parameters["endpoint"]["type"] == "string"
