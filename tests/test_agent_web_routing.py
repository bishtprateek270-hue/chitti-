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
    
    import subprocess
    def mock_popen(cmd, *args, **kwargs):
        if isinstance(cmd, list) and len(cmd) > 1 and any("http" in str(x) for x in cmd):
            for x in cmd:
                if "http" in str(x):
                    opened_urls.append(str(x))
        elif isinstance(cmd, str) and "http" in cmd:
            opened_urls.append(cmd)
        class DummyProc:
            pid = 9999
            def poll(self): return 0
        return DummyProc()
    monkeypatch.setattr(subprocess, "Popen", mock_popen)
    
    success, msg, res = mgr.handle_command("HD open YouTube")
    assert success is True
    assert "YouTube" in msg or "browser" in msg
    assert any("youtube.com" in u for u in opened_urls)


def test_open_arbitrary_websites_by_name_and_domain():
    """Verifies that queries asking to open websites or URLs resolve properly to OPEN_URL."""
    from src.agent.intent import ActionIntentAnalyzer, ActionIntentType
    from src.agent.parser import ActionParser
    from src.router.master_router import MasterRouter, MasterRoute

    cases = [
        ("open website google", "https://www.google.com"),
        ("open website stackoverflow", "https://stackoverflow.com"),
        ("open website cricbuzz", "https://www.cricbuzz.com"),
        ("open the website of netflix", "https://www.netflix.com"),
        ("open website https://news.ycombinator.com", "https://news.ycombinator.com"),
        ("open cricbuzz on browser", "https://www.cricbuzz.com"),
        ("open website wikipedia", "https://www.wikipedia.org"),
        ("open nytimes.com on browser", "https://nytimes.com"),
        ("browser me flipkart website kholo", "https://www.flipkart.com"),
        ("open website", "https://www.google.com"),
        ("open website leetcode", "https://leetcode.com"),
        ("open flipkart in browser", "https://www.flipkart.com"),
        ("open swiggy", "https://www.swiggy.com"),
        ("open zomato", "https://www.zomato.com"),
        ("open website nytimes", "https://www.nytimes.com"),
        ("open amazon.in", "https://amazon.in"),
    ]

    for query, expected_url in cases:
        intent = ActionIntentAnalyzer.extract_intent(query)
        assert intent.intent == ActionIntentType.OPEN_URL, f"Failed intent for: {query}"
        assert intent.destination == expected_url, f"Expected {expected_url}, got {intent.destination} for {query}"

        parsed = ActionParser.parse_command(query)
        assert parsed is not None, f"Failed parse for: {query}"
        assert parsed.action == ActionType.OPEN_URL, f"Failed parsed action for: {query}"
        assert parsed.parameters["url"] == expected_url, f"Expected {expected_url}, got {parsed.parameters['url']} for {query}"


def test_manager_opens_websites_on_browser_command(monkeypatch):
    """Verifies that LaptopAgentManager opens websites without errors."""
    mgr = LaptopAgentManager()
    opened_urls = []

    import webbrowser
    monkeypatch.setattr(webbrowser, "open", lambda url, *args, **kwargs: opened_urls.append(url) or True)

    import subprocess
    def mock_popen(cmd, *args, **kwargs):
        if isinstance(cmd, list) and len(cmd) > 1 and any("http" in str(x) for x in cmd):
            for x in cmd:
                if "http" in str(x):
                    opened_urls.append(str(x))
        elif isinstance(cmd, str) and "http" in cmd:
            opened_urls.append(cmd)
        class DummyProc:
            pid = 9999
            def poll(self): return 0
        return DummyProc()
    monkeypatch.setattr(subprocess, "Popen", mock_popen)

    for cmd, expected_domain in [
        ("open website cricbuzz", "cricbuzz.com"),
        ("browser me flipkart website kholo", "flipkart.com"),
        ("open website leetcode", "leetcode.com"),
    ]:
        opened_urls.clear()
        handled, msg, res = mgr.handle_command(cmd)
        assert handled is True
        assert any(expected_domain in u for u in opened_urls), f"Failed to open {expected_domain} for {cmd}"


def test_open_browser_applications_routes_to_open_application():
    """Verifies that queries asking to open desktop browser applications route to OPEN_APPLICATION, not OPEN_URL."""
    from src.agent.intent import ActionIntentAnalyzer, ActionIntentType
    from src.agent.parser import ActionParser

    cases = [
        ("open brave", "Brave"),
        ("open brave browser", "Brave"),
        ("open chrome", "Google Chrome"),
        ("open chrome browser", "Google Chrome"),
        ("open edge", "Microsoft Edge"),
        ("open edge browser", "Microsoft Edge"),
        ("open firefox", "Firefox"),
        ("open firefox browser", "Firefox"),
    ]

    for query, expected_app in cases:
        intent = ActionIntentAnalyzer.extract_intent(query)
        assert intent.intent == ActionIntentType.OPEN_APPLICATION, f"Expected OPEN_APPLICATION for {query}, got {intent.intent}"
        assert intent.application == expected_app, f"Expected {expected_app}, got {intent.application} for {query}"

        act = ActionParser.parse_command(query)
        assert act is not None
        assert act.action == ActionType.OPEN_APPLICATION



