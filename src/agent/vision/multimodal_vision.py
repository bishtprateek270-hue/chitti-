"""
Chitti Multimodal Vision Grounding Engine.
Synthesizes visual screen captures, active window metadata, OCR extraction,
and Multimodal LLM reasoning to explain screens, diagnose errors, and locate UI elements.
"""

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.agent.vision.ocr_analyzer import DetectedUIElement, OCRAnalyzer
from src.agent.vision.screen_reader import ScreenReader, WindowRect
from src.brain.llm import BaseLLM
from src.utils.logging import log_debug, log_info, log_warn


@dataclass
class UIElementMatch:
    """Represents a matched on-screen UI element with coordinates."""
    label: str
    element_type: str
    center_x: int
    center_y: int
    bounding_box: Tuple[int, int, int, int]
    confidence: float = 0.9


@dataclass
class ErrorDiagnosisResult:
    """Represents an automated diagnosis of an on-screen error."""
    has_error: bool
    error_type: str
    description: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    suggested_fix: str = ""
    evidence_snippet: str = ""


@dataclass
class VisionAnalysisResult:
    """Represents the complete visual and multimodal analysis of the screen."""
    screenshot_path: str
    active_window_title: str
    active_process: str
    summary: str
    detected_elements: List[DetectedUIElement] = field(default_factory=list)
    error_diagnosis: Optional[ErrorDiagnosisResult] = None
    raw_ocr_text: str = ""


class VisionGroundingEngine:
    """Core engine for screen understanding, error diagnosis, and visual grounding."""

    def __init__(
        self,
        screen_reader: Optional[ScreenReader] = None,
        ocr_analyzer: Optional[OCRAnalyzer] = None,
        llm: Optional[BaseLLM] = None,
    ):
        self.screen_reader = screen_reader or ScreenReader()
        self.ocr_analyzer = ocr_analyzer or OCRAnalyzer()
        self.llm = llm

    def analyze_screen(
        self,
        query: str = "Explain what is on the screen",
        screenshot_path: Optional[str] = None,
        focus_active_window: bool = True,
    ) -> VisionAnalysisResult:
        """
        Captures the screen/active window and generates an intelligent visual analysis.
        """
        active_rect: Optional[WindowRect] = None
        if not screenshot_path:
            if focus_active_window:
                screenshot_path, active_rect = self.screen_reader.capture_active_window()
            else:
                screenshot_path = self.screen_reader.capture_full_screen()
                active_rect = self.screen_reader.get_active_window()
        else:
            active_rect = self.screen_reader.get_active_window()

        active_title = active_rect.title if active_rect else "Active Application"
        active_proc = active_rect.process_name if active_rect else "System"

        # Extract textual signals from image
        ocr_text = self.ocr_analyzer.extract_text_from_image(screenshot_path)
        detected_elements = self.ocr_analyzer.parse_ui_elements(ocr_text)

        # Check for errors on screen
        error_info = self.ocr_analyzer.extract_error_details(ocr_text)
        diagnosis: Optional[ErrorDiagnosisResult] = None
        if error_info and error_info.get("has_error"):
            diagnosis = ErrorDiagnosisResult(
                has_error=True,
                error_type=error_info.get("error_text", "Runtime Error"),
                description=f"Detected error in {active_title}: {error_info.get('error_text')}",
                file_path=error_info.get("file"),
                line_number=error_info.get("line"),
                suggested_fix="Check the traceback details and verify required modules/dependencies.",
                evidence_snippet=error_info.get("raw_context", ""),
            )

        # Generate intelligent summary using LLM if available, or structured reasoning
        summary = self._synthesize_summary(query, active_title, active_proc, ocr_text, diagnosis)

        log_info(f"[VISION ENGINE] Analysis completed for '{active_title}': {summary[:120]}...")

        return VisionAnalysisResult(
            screenshot_path=screenshot_path,
            active_window_title=active_title,
            active_process=active_proc,
            summary=summary,
            detected_elements=detected_elements,
            error_diagnosis=diagnosis,
            raw_ocr_text=ocr_text,
        )

    def diagnose_screen_error(self, screenshot_path: Optional[str] = None) -> ErrorDiagnosisResult:
        """
        Specifically focuses on finding and debugging errors visible on the active display.
        """
        analysis = self.analyze_screen(
            query="Analyze the error on screen and provide a diagnosis and fix",
            screenshot_path=screenshot_path,
            focus_active_window=True,
        )

        if analysis.error_diagnosis and analysis.error_diagnosis.has_error:
            return analysis.error_diagnosis

        # If no explicit error traceback detected, inspect active window context
        win_title = analysis.active_window_title.lower()
        if "error" in win_title or "fail" in win_title or "exception" in win_title:
            return ErrorDiagnosisResult(
                has_error=True,
                error_type="Window Error Alert",
                description=f"Active window '{analysis.active_window_title}' indicates an application error or warning dialog.",
                suggested_fix="Review application alert and confirm actions.",
                evidence_snippet=analysis.active_window_title,
            )

        return ErrorDiagnosisResult(
            has_error=False,
            error_type="None",
            description=f"No error tracebacks or crash alerts detected in '{analysis.active_window_title}'. The application appears to be running normally.",
            suggested_fix="",
            evidence_snippet="",
        )

    def find_ui_element(
        self,
        element_query: str,
        screenshot_path: Optional[str] = None,
    ) -> Optional[UIElementMatch]:
        """
        Locates an on-screen interactive control (e.g. button, input, tab) by text or description.
        """
        analysis = self.analyze_screen(screenshot_path=screenshot_path, focus_active_window=True)
        clean_query = element_query.lower().strip()

        for elem in analysis.detected_elements:
            if clean_query in elem.text.lower() or elem.text.lower() in clean_query:
                cx, cy = elem.center
                return UIElementMatch(
                    label=elem.text,
                    element_type=elem.element_type,
                    center_x=cx,
                    center_y=cy,
                    bounding_box=elem.box,
                    confidence=elem.confidence,
                )

        return None

    def summarize_active_window(self, screenshot_path: Optional[str] = None) -> str:
        """
        Returns a concise natural-language summary of what is visible in the foreground window.
        """
        analysis = self.analyze_screen(
            query="Summarize what is open in the active window",
            screenshot_path=screenshot_path,
            focus_active_window=True,
        )
        return analysis.summary

    def _synthesize_summary(
        self,
        query: str,
        window_title: str,
        process_name: str,
        ocr_text: str,
        diagnosis: Optional[ErrorDiagnosisResult],
    ) -> str:
        """
        Produces a concise, context-aware summary.
        """
        if self.llm:
            try:
                prompt = (
                    f"You are Chitti's Vision Grounding Engine. Analyze the active window on screen.\n"
                    f"User Query: {query}\n"
                    f"Active Window: {window_title} (Process: {process_name})\n"
                    f"On-Screen OCR Text Snippet:\n{ocr_text[:800]}\n"
                    f"Provide a clear, 2-3 sentence summary of what is happening on screen and any errors found."
                )
                resp = self.llm.generate_response([{"role": "user", "content": prompt}])
                if resp and len(resp.strip()) > 10:
                    return resp.strip()
            except Exception as e:
                log_debug(f"[VISION ENGINE] LLM synthesis notice: {e}")

        # Deterministic heuristic summary
        if diagnosis and diagnosis.has_error:
            loc_str = f" in {diagnosis.file_path} (line {diagnosis.line_number})" if diagnosis.file_path and diagnosis.line_number else ""
            return f"Active window '{window_title}' shows an error: {diagnosis.error_type}{loc_str}. {diagnosis.suggested_fix}"

        if ocr_text:
            snippet = ocr_text.replace("\n", " ")[:140]
            return f"Currently viewing '{window_title}'. On-screen content: {snippet}..."

        return f"Currently active on '{window_title}' ({process_name}). The display is open and responsive."
