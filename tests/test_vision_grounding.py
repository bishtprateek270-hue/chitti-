"""
Tests for Phase 1: Chitti Real-Time Screen & Vision Grounding Engine.
Verifies:
1. ScreenReader screenshot capture and active window metadata extraction.
2. OCRAnalyzer text parsing, UI element detection, and error signal extraction.
3. VisionGroundingEngine multimodal visual reasoning, error diagnosis, and UI element localization.
4. ActionIntentAnalyzer intent detection for English, Hindi, and Hinglish vision queries.
5. AgentPlanner multi-step vision plan construction and tool execution loop.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.agent.actions import ActionType, StructuredAction
from src.agent.executor import ActionExecutor
from src.agent.intent import ActionIntentAnalyzer, ActionIntentType
from src.agent.manager import LaptopAgentManager
from src.agent.planner import AgentPlanner, TaskState, TaskStatus
from src.agent.projects import ProjectRegistry
from src.agent.vision import (
    DetectedUIElement,
    ErrorDiagnosisResult,
    OCRAnalyzer,
    ScreenReader,
    UIElementMatch,
    VisionAnalysisResult,
    VisionGroundingEngine,
    WindowRect,
)
from src.brain.llm import BaseLLM


class MockLLM(BaseLLM):
    """Mock LLM for vision reasoning tests."""
    def generate_response(self, messages):
        return "The active window shows Visual Studio Code with a Python SyntaxError on line 12."

    def check_connection(self):
        return True

    def list_available_models(self):
        return ["mock-vision-model"]


class TestVisionGroundingEngine(unittest.TestCase):
    """Comprehensive test suite for Phase 1 vision capabilities."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name
        self.reader = ScreenReader(output_dir=self.output_dir)
        self.ocr = OCRAnalyzer()
        self.llm = MockLLM()
        self.engine = VisionGroundingEngine(
            screen_reader=self.reader,
            ocr_analyzer=self.ocr,
            llm=self.llm,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    # -------------------------------------------------------------
    # 1. SCREEN READER TESTS
    # -------------------------------------------------------------
    def test_screen_reader_capture_full_screen(self):
        path = self.reader.capture_full_screen()
        self.assertTrue(os.path.exists(path))
        self.assertTrue(path.endswith(".png"))

    def test_screen_reader_get_active_window(self):
        active_win = self.reader.get_active_window()
        self.assertIsNotNone(active_win)
        self.assertIsInstance(active_win.title, str)
        self.assertGreaterEqual(active_win.width, 0)
        self.assertGreaterEqual(active_win.height, 0)

    def test_screen_reader_capture_active_window(self):
        path, win = self.reader.capture_active_window()
        self.assertTrue(os.path.exists(path))
        self.assertIsNotNone(win)

    def test_screen_reader_crop_region(self):
        full_path = self.reader.capture_full_screen()
        crop_path = self.reader.crop_region(full_path, (10, 10, 200, 200))
        self.assertTrue(os.path.exists(crop_path))

    def test_screen_reader_encode_base64(self):
        path = self.reader.capture_full_screen()
        b64 = ScreenReader.encode_base64(path)
        self.assertIsInstance(b64, str)
        self.assertGreater(len(b64), 50)

    # -------------------------------------------------------------
    # 2. OCR & UI ELEMENT ANALYZER TESTS
    # -------------------------------------------------------------
    def test_ocr_parse_ui_elements(self):
        mock_screen_text = (
            "Visual Studio Code - main.py\n"
            "Traceback (most recent call last):\n"
            "  File \"main.py\", line 42, in <module>\n"
            "SyntaxError: invalid syntax\n"
            "Save\n"
            "Cancel\n"
            "Run Code\n"
        )
        elements = self.ocr.parse_ui_elements(mock_screen_text)
        self.assertGreater(len(elements), 0)

        error_elems = [e for e in elements if e.element_type == "error"]
        self.assertGreater(len(error_elems), 0)
        self.assertTrue(any("SyntaxError" in e.text for e in error_elems))

        button_elems = [e for e in elements if e.element_type == "button"]
        self.assertGreater(len(button_elems), 0)
        button_texts = [b.text for b in button_elems]
        self.assertTrue(any("Save" in t or "Cancel" in t or "Run" in t for t in button_texts))

    def test_ocr_extract_error_details(self):
        error_context = (
            "Running python script...\n"
            "Traceback (most recent call last):\n"
            "  File \"app/server.py\", line 88, in handle_request\n"
            "ModuleNotFoundError: No module named 'requests'\n"
        )
        details = self.ocr.extract_error_details(error_context)
        self.assertIsNotNone(details)
        self.assertTrue(details["has_error"])
        self.assertIn("ModuleNotFoundError", details["error_text"])
        self.assertEqual(details["file"], "app/server.py")
        self.assertEqual(details["line"], 88)

    # -------------------------------------------------------------
    # 3. MULTIMODAL VISION GROUNDING ENGINE TESTS
    # -------------------------------------------------------------
    def test_vision_engine_analyze_screen(self):
        result = self.engine.analyze_screen(query="Tell me what is open")
        self.assertIsInstance(result, VisionAnalysisResult)
        self.assertTrue(os.path.exists(result.screenshot_path))
        self.assertIsNotNone(result.summary)
        self.assertGreater(len(result.summary), 5)

    def test_vision_engine_diagnose_screen_error(self):
        # Inject mock error text into OCR analyzer
        with patch.object(self.ocr, "extract_text_from_image", return_value="File \"test.py\", line 15\nTypeError: unsupported operand type"):
            diag = self.engine.diagnose_screen_error()
            self.assertTrue(diag.has_error)
            self.assertIn("TypeError", diag.error_type)
            self.assertEqual(diag.file_path, "test.py")
            self.assertEqual(diag.line_number, 15)

    def test_vision_engine_find_ui_element(self):
        mock_ui_text = "File Edit View\n[Submit Order]\nSettings"
        with patch.object(self.ocr, "extract_text_from_image", return_value=mock_ui_text):
            match = self.engine.find_ui_element("Submit Order")
            self.assertIsNotNone(match)
            self.assertIn("Submit Order", match.label)
            self.assertGreater(match.center_x, 0)
            self.assertGreater(match.center_y, 0)

    # -------------------------------------------------------------
    # 4. ACTION INTENT RECOGNITION (ENGLISH, HINDI, HINGLISH)
    # -------------------------------------------------------------
    def test_intent_diagnose_screen_error(self):
        queries = [
            "look at my screen and tell me what is wrong",
            "explain the error on my screen",
            "why is my code failing on screen",
            "meri screen dekho aur error batao",
            "screen pe kya error hai batao",
            "read the error on screen and fix it",
        ]
        for q in queries:
            intent = ActionIntentAnalyzer.extract_intent(q)
            self.assertEqual(intent.intent, ActionIntentType.DIAGNOSE_SCREEN_ERROR, f"Failed for query: '{q}'")

    def test_intent_analyze_screen(self):
        queries = [
            "look at my screen",
            "what is on my screen",
            "summarize what is on my screen",
            "explain what is on my screen",
            "meri screen dekho",
            "screen pe kya hai batao",
            "screen summarize karo",
        ]
        for q in queries:
            intent = ActionIntentAnalyzer.extract_intent(q)
            self.assertEqual(intent.intent, ActionIntentType.ANALYZE_SCREEN, f"Failed for query: '{q}'")

    def test_intent_find_ui_element(self):
        queries = [
            "find the Submit button on screen",
            "locate Search input on screen",
            "screen pe Save button dhoondo",
        ]
        for q in queries:
            intent = ActionIntentAnalyzer.extract_intent(q)
            self.assertEqual(intent.intent, ActionIntentType.FIND_UI_ELEMENT, f"Failed for query: '{q}'")

    # -------------------------------------------------------------
    # 5. AGENT PLANNER & EXECUTOR VISION WORKFLOW
    # -------------------------------------------------------------
    def test_planner_creates_vision_plans(self):
        projects = MagicMock(spec=ProjectRegistry)
        planner = AgentPlanner(project_registry=projects)

        # 1. Error Diagnosis Plan
        state_diag = planner.plan_task("look at my screen and explain the error")
        self.assertIsNotNone(state_diag)
        self.assertEqual(len(state_diag.steps), 2)
        self.assertEqual(state_diag.steps[0].action_type, "CAPTURE_ACTIVE_WINDOW")
        self.assertEqual(state_diag.steps[1].action_type, "DIAGNOSE_SCREEN_ERROR")

        # 2. Screen Analysis Plan
        state_analysis = planner.plan_task("what is on my screen")
        self.assertIsNotNone(state_analysis)
        self.assertEqual(len(state_analysis.steps), 2)
        self.assertEqual(state_analysis.steps[0].action_type, "CAPTURE_SCREEN")
        self.assertEqual(state_analysis.steps[1].action_type, "ANALYZE_SCREEN")

        # 3. Find UI Element Plan
        state_find = planner.plan_task("find the submit button on screen")
        self.assertIsNotNone(state_find)
        self.assertEqual(len(state_find.steps), 2)
        self.assertEqual(state_find.steps[0].action_type, "CAPTURE_ACTIVE_WINDOW")
        self.assertEqual(state_find.steps[1].action_type, "FIND_UI_ELEMENT")

    def test_action_executor_vision_actions(self):
        # 1. ANALYZE_SCREEN
        action_analysis = StructuredAction(
            action=ActionType.ANALYZE_SCREEN,
            parameters={"query": "Explain active window"},
        )
        res1 = ActionExecutor.execute(action_analysis)
        self.assertTrue(res1.success)
        self.assertIn("summary", res1.data)

        # 2. DIAGNOSE_SCREEN_ERROR
        action_diag = StructuredAction(
            action=ActionType.DIAGNOSE_SCREEN_ERROR,
            parameters={"query": "Check for errors"},
        )
        res2 = ActionExecutor.execute(action_diag)
        self.assertTrue(res2.success)
        self.assertIn("has_error", res2.data)

        # 3. FIND_UI_ELEMENT
        action_find = StructuredAction(
            action=ActionType.FIND_UI_ELEMENT,
            parameters={"element": "VS Code"},
        )
        res3 = ActionExecutor.execute(action_find)
        self.assertTrue(res3.success)

    def test_manager_vision_response_formatting(self):
        manager = LaptopAgentManager()
        state = TaskState(task_description="look at my screen and explain error", goal="Diagnose screen error")
        from src.agent.planner import AgentStep
        state.steps = [
            AgentStep(step_id=1, description="Capture active window", action_type="CAPTURE_ACTIVE_WINDOW"),
            AgentStep(step_id=2, description="Diagnose screen error", action_type="DIAGNOSE_SCREEN_ERROR"),
        ]

        resp = manager._format_multistep_response(
            "look at my screen and explain error",
            state,
            success=True,
            raw_msg="Detected Python SyntaxError in main.py on line 12. Missing colon at end of statement.",
            lang="en",
        )
        self.assertIn("SyntaxError in main.py", resp)
        self.assertNotIn("WhatsApp", resp)
        self.assertNotIn("Email", resp)


if __name__ == "__main__":
    unittest.main()
