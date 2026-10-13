"""
Tests for Web Resource and Application Routing Fixes.
Verifies that:
1. 'HD open YouTube', 'open application YouTube', and 'open YouTube' are properly recognized as OPEN_URL tasks.
2. Web services without local desktop executables are safely re-routed to browser URLs.
3. LanguageNormalizer does not prepend redundant 'application' keywords.
4. ActionValidator validates web destinations without failing on missing .exe files.
"""

from src.agent.actions import ActionType, StructuredAction, RiskLevel
from src.agent.parser import ActionParser
from src.agent.validator import ActionValidator
from src.agent.registry import AppDiscovery
from src.language.normalizer import LanguageNormalizer
from src.agent.manager import LaptopAgentManager


def test_action_parser_web_destinations():
    for cmd in ["HD open YouTube", "open application YouTube", "open YouTube", "open the youtube website"]:
        act = ActionParser.parse_command(cmd)
        assert act is not None, f"Failed to parse: {cmd}"
        assert act.action == ActionType.OPEN_URL
        assert "youtube" in act.parameters["url"].lower()


def test_language_normalizer_no_redundant_application_word():
    norm = LanguageNormalizer()
    res = norm.parse_intent("HD open YouTube")
    assert "Open YouTube" in res.normalized_text
    assert "Open application YouTube" not in res.normalized_text


def test_validator_reroutes_web_targets():
    # If an action arrives as OPEN_APPLICATION with 'application YouTube' or 'YouTube'
    raw_act = StructuredAction(
        action=ActionType.OPEN_APPLICATION,
        parameters={"target": "application YouTube"},
        risk_level=RiskLevel.LOW,
    )
    valid, reason, validated = ActionValidator.validate(raw_act)
    assert valid is True
    assert validated.action == ActionType.OPEN_URL
    assert "youtube.com" in validated.parameters["url"]


def test_app_discovery_does_not_return_dummy_for_web_services():
    # YouTube has no local .exe on Windows and belongs to DEFAULT_URL_MAP
    resolved = AppDiscovery.resolve_application("youtube")
    assert resolved is None, f"Expected None for web service 'youtube', got {resolved}"


def test_manager_handles_hd_open_youtube(monkeypatch):
    mgr = LaptopAgentManager()
    opened_urls = []
    
    import webbrowser
    monkeypatch.setattr(webbrowser, "open", lambda url, *args, **kwargs: opened_urls.append(url) or True)
    
    success, msg, res = mgr.handle_command("HD open YouTube")
    assert success is True
    assert "YouTube" in msg or "browser" in msg
    assert any("youtube.com" in u for u in opened_urls)
