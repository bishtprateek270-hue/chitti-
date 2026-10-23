"""
Comprehensive Test Suite for Platform-Targeted Browser Search,
Semantic Windows UI Automation, and Floating HUD Lifecycle Integration.
"""

from unittest.mock import MagicMock, patch
import pytest

from src.agent.intent import ActionIntentAnalyzer, ActionIntentType
from src.agent.registry import ResourceDiscovery
from src.agent.computer.controller import ComputerController, UIElementInfo, WindowInfo
from src.agent.actions import ActionType, StructuredAction
from src.agent.executor import ActionExecutor
from src.agent.planner import AgentPlanner
from src.agent.manager import LaptopAgentManager
from src.ui.visualizer import HUDMode, HUDState
from src.ui.hud_overlay import FloatingHUD


class TestPlatformSearch:
    """Validates platform-targeted and compound web search resolution."""

    @pytest.mark.parametrize(
        "query,expected_platform,expected_domain,expected_query",
        [
            ("open google and search for latest AI news", "google", "google.com", "latest AI news"),
            ("open youtube and search for python tutorial", "youtube", "youtube.com", "python tutorial"),
            ("open amazon and search for wireless headphones", "amazon", "amazon", "wireless headphones"),
            ("open github and search for chitti assistant", "github", "github.com", "chitti assistant"),
            ("open wikipedia and search for quantum mechanics", "wikipedia", "wikipedia.org", "quantum mechanics"),
            ("open flipkart and search for gaming laptop", "flipkart", "flipkart.com", "gaming laptop"),
            ("open reddit and search for local llms", "reddit", "reddit.com", "local llms"),
            ("open spotify and search for lofi beats", "spotify", "spotify.com", "lofi beats"),
            ("open chrome and search for weather today", "google", "google.com", "weather today"),
            ("search for deep learning on google", "google", "google.com", "deep learning"),
            ("search python tutorial on youtube", "youtube", "youtube.com", "python tutorial"),
            ("search mechanical keyboard on amazon", "amazon", "amazon", "mechanical keyboard"),
            ("search chitti on github", "github", "github.com", "chitti"),
            ("google latest artificial intelligence updates", "google", "google.com", "latest artificial intelligence updates"),
            ("google pe search karo latest AI news", "google", "google.com", "latest AI news"),
            ("youtube pe search karo python tutorial", "youtube", "youtube.com", "python tutorial"),
        ],
    )
    def test_compound_and_platform_search_resolution(self, query, expected_platform, expected_domain, expected_query):
        intent = ActionIntentAnalyzer.extract_intent(query)
        assert intent.intent == ActionIntentType.SEARCH_WEB
        assert intent.parameters.get("platform") == expected_platform
        assert expected_domain in intent.destination
        assert intent.target == expected_query
        assert "and search for" not in intent.target

    def test_planner_generates_search_web_steps(self):
        from src.agent.projects import ProjectRegistry
        from src.agent.computer import FilesystemController
        planner = AgentPlanner(project_registry=ProjectRegistry(), filesystem=FilesystemController())
        intent = ActionIntentAnalyzer.extract_intent("open amazon and search for mechanical keyboard")
        state = planner.plan_task("open amazon and search for mechanical keyboard")
        assert len(state.steps) >= 2
        assert state.steps[0].action_type == "SEARCH_WEB"
        assert state.steps[0].parameters.get("platform") == "amazon"
        assert "mechanical keyboard" in state.steps[0].parameters.get("query")
        assert "amazon" in state.steps[0].parameters.get("url")


class TestSemanticUIAutomation:
    """Validates Windows UI element discovery and native control clicking."""

    def test_find_element_by_name_heuristic_fallback(self):
        ctrl = ComputerController()
        mock_win = WindowInfo(
            title="Calculator",
            handle=12345,
            left=100,
            top=100,
            width=400,
            height=600,
            is_active=True,
            is_minimized=False,
        )
        with patch.object(ctrl, "get_active_window", return_value=mock_win), \
             patch.object(ctrl, "find_ui_elements", return_value=[]):
            elem = ctrl.find_element_by_name("Close")
            assert elem is not None
            assert elem.control_type == "button"
            assert elem.center_x > mock_win.left

    def test_find_ui_elements_native_win32_mock(self):
        ctrl = ComputerController()
        mock_elem = UIElementInfo(
            name="Submit",
            control_type="button",
            handle=999,
            left=200,
            top=300,
            width=80,
            height=30,
            center_x=240,
            center_y=315,
        )
        with patch.object(ctrl, "find_ui_elements", return_value=[mock_elem]):
            res = ctrl.find_element_by_name("Submit")
            assert res is not None
            assert res.name == "Submit"
            assert res.center_x == 240
            assert res.center_y == 315

    def test_click_element_by_name_success(self):
        ctrl = ComputerController()
        mock_elem = UIElementInfo(
            name="Search",
            control_type="edit",
            handle=888,
            left=150,
            top=100,
            width=200,
            height=40,
            center_x=250,
            center_y=120,
        )
        with patch.object(ctrl, "find_element_by_name", return_value=mock_elem), \
             patch.object(ctrl, "click", return_value=True):
            clicked = ctrl.click_element_by_name("Search")
            assert clicked is True

    def test_executor_handles_click_ui_element(self):
        action = StructuredAction(
            action=ActionType.CLICK_UI_ELEMENT,
            parameters={"element": "Submit"},
        )
        with patch("src.agent.computer.ComputerController.click_element_by_name", return_value=True):
            result = ActionExecutor.execute(action)
            assert result.success is True
            assert "Submit" in result.message


class TestHUDLifecycleIntegration:
    """Validates real-time HUD status updates during multi-step laptop execution."""

    def test_hud_progress_notifications_during_task(self):
        hud = FloatingHUD()
        agent = LaptopAgentManager(hud=hud)

        # Plan and mock execute a command
        with patch.object(agent.loop, "execute_plan") as mock_exec:
            mock_exec.return_value = (True, "Tasks completed")
            success, msg, act_res = agent.handle_command("open calculator")
            assert success is True
            # Verify hud received mode changes
            assert not hud._queue.empty()
            modes = []
            while not hud._queue.empty():
                item = hud._queue.get()
                if item[0] == "set_mode":
                    modes.append(item[1])
            assert HUDMode.THINKING in modes
            assert HUDMode.SUCCESS in modes
