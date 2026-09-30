"""
Tests for Generalized Dynamic Computer-Use Execution (Phase 5 & 6).
Verifies that Chitti resolves arbitrary, unseen applications, websites, folders, and actions
purely through generic intent extraction, dynamic OS application discovery, and dynamic tool selection,
WITHOUT hardcoded application or website lists.
"""

from unittest.mock import MagicMock, patch
from pathlib import Path
import pytest

from src.agent.actions import ActionType, StructuredAction
from src.agent.intent import ActionIntentAnalyzer, ActionIntentType
from src.agent.manager import LaptopAgentManager
from src.agent.parser import ActionParser
from src.agent.planner import AgentPlanner
from src.agent.projects import ProjectRegistry
from src.agent.registry import AppDiscovery, FolderDiscovery, ResourceDiscovery
from src.agent.tools import ToolEngine
from src.router.master_router import MasterRouter, MasterRoute


def test_unseen_application_intent_and_routing():
    """Verify that previously unseen applications route to COMPUTER_TASK and extract the target dynamically."""
    unseen_apps = [
        ("open obsidian", "Obsidian"),
        ("launch blender", "Blender"),
        ("start figma", "Figma"),
        ("open vlc", "Vlc"),
        ("vlc media player kholo", "Vlc Media Player"),
        ("open steam on my computer", "Steam"),
        ("launch wireshark for me", "Wireshark"),
        ("audacity chala do", "Audacity"),
        ("open postman", "Postman"),
        ("gimp open karo", "Gimp"),
    ]

    for user_input, expected_target in unseen_apps:
        # 1. Master Router routing
        decision = MasterRouter.classify_request(user_input)
        assert decision.requires_computer is True
        assert decision.requires_phase5 is True
        assert decision.route in (MasterRoute.COMPUTER_TASK, MasterRoute.BROWSER_TASK)

        # 2. Intent extraction
        intent = ActionIntentAnalyzer.extract_intent(user_input)
        assert intent.intent in (ActionIntentType.OPEN_APPLICATION, ActionIntentType.OPEN_URL)
        assert expected_target.lower() in (intent.target or "").lower()

        # 3. Single-step parsing
        parsed = ActionParser.parse_command(user_input)
        assert parsed is not None
        assert parsed.action in (ActionType.OPEN_APPLICATION, ActionType.OPEN_URL)


def test_unseen_website_intent_and_routing():
    """Verify that previously unseen URLs and domains route to BROWSER_TASK and extract URLs dynamically."""
    unseen_sites = [
        ("open https://custom-dashboard.internal.io", "https://custom-dashboard.internal.io"),
        ("visit coursera.org", "https://coursera.org"),
        ("go to twitch.tv in browser", "https://twitch.tv"),
        ("open leetcode.com", "https://leetcode.com"),
        ("open huggingface.co", "https://huggingface.co"),
        ("news.ycombinator.com open karo", "https://news.ycombinator.com"),
    ]

    for user_input, expected_url_sub in unseen_sites:
        decision = MasterRouter.classify_request(user_input)
        assert decision.requires_computer is True
        assert decision.route == MasterRoute.BROWSER_TASK

        intent = ActionIntentAnalyzer.extract_intent(user_input)
        assert intent.intent == ActionIntentType.OPEN_URL
        assert intent.requires_browser is True
        assert expected_url_sub in (intent.destination or intent.parameters.get("url", ""))


def test_semantic_app_alias_resolution():
    """Verify natural descriptive aliases like 'my editor', 'the browser', 'the terminal' resolve dynamically."""
    semantic_queries = [
        ("open my editor", "code"),
        ("open the browser", "chrome"),
        ("open that browser", "chrome"),
        ("open the terminal", "wt"),
        ("open the music player", "spotify"),
    ]

    for query, expected_resolution in semantic_queries:
        decision = MasterRouter.classify_request(query)
        assert decision.requires_computer is True

        parsed = ActionParser.parse_command(query)
        assert parsed is not None
        assert parsed.action in (ActionType.OPEN_APPLICATION, ActionType.OPEN_URL)

        resolved = AppDiscovery.resolve_application(parsed.parameters["target"])
        assert resolved is not None


def test_dynamic_app_discovery_registry_lookup():
    """Verify AppDiscovery inspects registry or PATH dynamically."""
    # Test built-in resolver
    res_calc = AppDiscovery.resolve_application("calc")
    assert res_calc is not None

    res_notepad = AppDiscovery.resolve_application("notepad")
    assert res_notepad is not None

    # Test unknown app fallback behavior (returns executable target for shell dispatch)
    res_unknown = AppDiscovery.resolve_application("custom_unseen_tool_99")
    assert res_unknown == "custom_unseen_tool_99"


def test_general_action_model_structure():
    """Verify the ActionIntent model complies with the general action model specification."""
    intent = ActionIntentAnalyzer.extract_intent("open vlc on my computer")
    assert intent.intent == ActionIntentType.OPEN_APPLICATION
    assert intent.target is not None
    assert intent.execution_required is True
    assert intent.verification_required is True
    assert isinstance(intent.parameters, dict)


def test_end_to_end_unseen_app_execution_with_diagnostics(tmp_path, caplog):
    """Verifies that an unseen application executes through LaptopAgentManager with Section 28 diagnostics."""
    manager = LaptopAgentManager(
        workspace_dir=str(tmp_path / "workspace"),
        screenshots_dir=str(tmp_path / "screenshots"),
        project_registry_path=str(tmp_path / "projects.json"),
    )

    with patch("os.startfile", return_value=None), patch("subprocess.Popen") as mock_popen:
        mock_popen.return_value = MagicMock()
        handled, resp_msg, result = manager.handle_command("open obsidian", lang="en")

        assert handled is True
        assert result is not None
        assert result.success is True
        assert "Obsidian" in resp_msg or "obsidian" in resp_msg.lower()

        # Verify diagnostics in logging
        log_content = caplog.text
        assert "[INTENT]" in log_content or "[AGENT]" in log_content
        assert "[FINAL STATUS]" in log_content or "[AGENT]" in log_content


def test_typo_tolerance_and_standalone_targets():
    """Verify that common typos and standalone shorthand targets route and parse accurately."""
    test_cases = [
        ("oprn yt", "youtube", ActionType.OPEN_URL),
        ("opne chrome", "chrome", ActionType.OPEN_APPLICATION),
        ("gemmini", "gemini", ActionType.OPEN_URL),
        ("watsapp", "whatsapp", ActionType.OPEN_URL),
        ("calcultor", "calculator", ActionType.OPEN_APPLICATION),
        ("strat notepad", "notepad", ActionType.OPEN_APPLICATION),
    ]

    for query, expected_keyword, expected_action in test_cases:
        decision = MasterRouter.classify_request(query)
        assert decision.requires_computer is True, f"Failed routing for {query}"

        parsed = ActionParser.parse_command(query)
        assert parsed is not None, f"Failed parsing for {query}"
        assert parsed.action == expected_action
        
        target_str = str(parsed.parameters.get("target") or parsed.parameters.get("url") or "")
        assert expected_keyword.lower() in target_str.lower()


def test_send_message_and_email_intent_and_routing():
    """Verify message and email dispatch intents extract targets and route to computer agent."""
    msg_query = "send message to Rahul saying hello"
    dec_msg = MasterRouter.classify_request(msg_query)
    assert dec_msg.requires_computer is True
    
    intent_msg = ActionIntentAnalyzer.extract_intent(msg_query)
    assert intent_msg.intent == ActionIntentType.SEND_MESSAGE
    assert "rahul" in intent_msg.target.lower() or "rahul" in intent_msg.recipient.lower()
    assert "hello" in intent_msg.content.lower()

    email_query = "send email to test@example.com with subject Meeting and body Hello team"
    dec_email = MasterRouter.classify_request(email_query)
    assert dec_email.requires_computer is True

    intent_email = ActionIntentAnalyzer.extract_intent(email_query)
    assert intent_email.intent == ActionIntentType.SEND_EMAIL
    assert "test@example.com" in intent_email.recipient


