"""
Unit tests for Chitti 24/7 N8NDaemonManager.
"""

from unittest.mock import MagicMock, patch
import pytest

from src.agent.n8n_daemon import N8NDaemonManager


class TestN8NDaemonManager:
    def test_daemon_initialization(self):
        daemon = N8NDaemonManager(base_url="http://localhost:5678", check_interval=15.0)
        assert daemon.base_url == "http://localhost:5678"
        assert daemon.check_interval == 15.0
        assert daemon.auto_restart is True

    @patch("urllib.request.urlopen")
    def test_is_healthy_online(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.getcode.return_value = 200
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        daemon = N8NDaemonManager()
        assert daemon.is_healthy() is True

    @patch("urllib.request.urlopen")
    def test_is_healthy_offline(self, mock_urlopen):
        mock_urlopen.side_effect = Exception("Connection refused")
        daemon = N8NDaemonManager()
        assert daemon.is_healthy() is False

    @patch.object(N8NDaemonManager, "is_healthy", return_value=True)
    def test_start_when_already_healthy(self, mock_health):
        daemon = N8NDaemonManager()
        assert daemon.start_n8n() is True

    @patch("shutil.which")
    def test_find_n8n_command_global(self, mock_which):
        mock_which.side_effect = lambda cmd: "C:\\npm\\n8n.cmd" if "n8n" in cmd else None
        daemon = N8NDaemonManager()
        cmd = daemon._find_n8n_command()
        assert cmd == ["C:\\npm\\n8n.cmd"]

    @patch("shutil.which")
    def test_find_n8n_command_npx_fallback(self, mock_which):
        def custom_which(cmd):
            if "npx" in cmd:
                return "C:\\nodejs\\npx.cmd"
            return None

        mock_which.side_effect = custom_which
        daemon = N8NDaemonManager()
        cmd = daemon._find_n8n_command()
        assert cmd == ["C:\\nodejs\\npx.cmd", "n8n"]
