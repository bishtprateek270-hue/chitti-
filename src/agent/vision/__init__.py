"""
Chitti Vision Grounding Engine (Phase 1).
Provides real-time screen capture, active-window isolation, visual OCR parsing,
multimodal visual reasoning, error diagnosis, and UI element localization.
"""

from src.agent.vision.screen_reader import ScreenReader, WindowRect
from src.agent.vision.ocr_analyzer import OCRAnalyzer, DetectedUIElement
from src.agent.vision.multimodal_vision import (
    VisionGroundingEngine,
    VisionAnalysisResult,
    ErrorDiagnosisResult,
    DocumentSummaryResult,
    UIElementMatch,
)

__all__ = [
    "ScreenReader",
    "WindowRect",
    "OCRAnalyzer",
    "DetectedUIElement",
    "VisionGroundingEngine",
    "VisionAnalysisResult",
    "ErrorDiagnosisResult",
    "DocumentSummaryResult",
    "UIElementMatch",
]
