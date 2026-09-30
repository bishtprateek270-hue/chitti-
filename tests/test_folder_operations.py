"""
Tests for Folder Opening and Navigation across all phrasing variations.
Verifies that folder opening requests are recognized, resolved, planned, and executed cleanly.
"""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.agent.intent import ActionIntentAnalyzer, ActionIntentType
from src.agent.parser import ActionParser, ActionType
from src.agent.registry import FolderDiscovery
from src.agent.manager import LaptopAgentManager
from src.router.master_router import MasterRouter, MasterRoute


class TestFolderDiscovery:
    def test_resolve_standard_aliases(self):
        downloads = FolderDiscovery.resolve_folder_path("downloads")
        assert downloads is not None
        assert "downloads" in str(downloads).lower() or downloads.exists()

        desktop = FolderDiscovery.resolve_folder_path("desktop")
        assert desktop is not None
        assert "desktop" in str(desktop).lower() or desktop.exists()

        documents = FolderDiscovery.resolve_folder_path("documents")
        assert documents is not None

    def test_resolve_generic_and_prefixed_requests(self):
        # "a folder", "the folder", "my folder", "folder"
        res1 = FolderDiscovery.resolve_folder_path("a folder")
        assert res1 is not None

        res2 = FolderDiscovery.resolve_folder_path("the folder")
        assert res2 is not None

        res3 = FolderDiscovery.resolve_folder_path("folder")
        assert res3 is not None

        res4 = FolderDiscovery.resolve_folder_path("my downloads folder")
        assert res4 is not None

    def test_resolve_workspace_and_subdirectories(self):
        res = FolderDiscovery.resolve_folder_path("src")
        assert res is not None
        assert res.name == "src"


class TestFolderIntentAndRouting:
    @pytest.mark.parametrize("phrase", [
        "open a folder",
        "open the folder",
        "open folder",
        "open downloads folder",
        "open folder downloads",
        "open downloads",
        "downloads folder kholo",
        "folder kholo",
        "open my documents folder",
        "open desktop folder",
        "open my folder",
    ])
    def test_master_router_classifies_folder_requests(self, phrase):
        decision = MasterRouter.classify_request(phrase)
        assert decision.requires_computer or decision.requires_phase5
        assert decision.route in (MasterRoute.FILE_OPERATION, MasterRoute.COMPUTER_TASK)

    @pytest.mark.parametrize("phrase, expected_target_sub", [
        ("open a folder", "desktop"),
        ("open folder downloads", "downloads"),
        ("open downloads folder", "downloads"),
        ("downloads folder kholo", "downloads"),
        ("open desktop", "desktop"),
        ("folder kholo", "desktop"),
    ])
    def test_intent_extraction(self, phrase, expected_target_sub):
        intent = ActionIntentAnalyzer.extract_intent(phrase)
        assert intent.intent == ActionIntentType.OPEN_FOLDER
        assert intent.target is not None

    @pytest.mark.parametrize("phrase", [
        "open a folder",
        "open the folder",
        "open folder",
        "open downloads folder",
        "open folder downloads",
        "open downloads",
        "downloads folder kholo",
        "folder kholo",
    ])
    def test_parser_recognizes_folder_action(self, phrase):
        action = ActionParser.parse_command(phrase)
        assert action is not None
        assert action.action == ActionType.OPEN_FOLDER


class TestLaptopAgentFolderExecution:
    @patch("os.startfile")
    @patch("subprocess.Popen")
    def test_agent_handles_open_a_folder(self, mock_popen, mock_startfile):
        agent = LaptopAgentManager()
        success, resp_msg, res = agent.handle_command("open a folder", lang="en")
        assert success is True
        assert "folder" in resp_msg.lower() or "desktop" in resp_msg.lower() or "opened" in resp_msg.lower()

    @patch("os.startfile")
    @patch("subprocess.Popen")
    def test_agent_handles_open_downloads_folder(self, mock_popen, mock_startfile):
        agent = LaptopAgentManager()
        success, resp_msg, res = agent.handle_command("open downloads folder", lang="en")
        assert success is True
        assert "downloads" in resp_msg.lower() or "opened" in resp_msg.lower()

    @patch("os.startfile")
    @patch("subprocess.Popen")
    def test_agent_handles_hinglish_folder_kholo(self, mock_popen, mock_startfile):
        agent = LaptopAgentManager()
        success, resp_msg, res = agent.handle_command("downloads folder kholo", lang="hinglish")
        assert success is True
        assert "open" in resp_msg.lower() or "khol" in resp_msg.lower() or "kar diya" in resp_msg.lower()
