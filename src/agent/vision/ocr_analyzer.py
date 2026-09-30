"""
Chitti OCR Analyzer & Visual Element Extractor.
Extracts on-screen text, error tracebacks, interactive buttons, inputs, and dialogs.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.utils.logging import log_debug, log_info, log_warn

try:
    from PIL import Image
    HAS_PIL = True
except Exception:
    HAS_PIL = False


@dataclass
class DetectedUIElement:
    """Represents a localized UI element detected on the display."""
    element_type: str  # "button", "input", "error", "dialog", "heading", "text"
    text: str
    box: Tuple[int, int, int, int]  # (x, y, width, height)
    confidence: float = 1.0

    @property
    def center(self) -> Tuple[int, int]:
        """Returns the click coordinate (x, y) center of the element."""
        x, y, w, h = self.box
        return (x + w // 2, y + h // 2)


class OCRAnalyzer:
    """Extracts text, errors, code snippets, and UI controls from screen captures."""

    COMMON_ERROR_PATTERNS = [
        r"(?i)\b(?:SyntaxError|NameError|TypeError|ValueError|IndexError|KeyError|AttributeError|ImportError|ModuleNotFoundError|FileNotFoundError|ZeroDivisionError|RuntimeError|RecursionError|IndentationError|TabError)\b.*",
        r"(?i)\b(?:Traceback\s+\(most\s+recent\s+call\s+last\):).*",
        r"(?i)\b(?:Error|Exception|Fatal|Critical|Failed|FAILED|ERR!|error TS\d+|NullPointerException|Segmentation fault)\b.*",
        r"(?i)\b(?:Cannot find module|is not defined|unexpected token|Uncaught [A-Za-z0-9_]+Error)\b.*",
        r"(?i)\b(?:exit code [1-9]\d*|command failed with exit status \d+)\b.*",
    ]

    BUTTON_PATTERNS = [
        r"(?i)\b(?:OK|Cancel|Close|Apply|Save|Submit|Continue|Next|Back|Finish|Yes|No|Retry|Ignore|Run|Debug|Play|Pause|Send)\b",
    ]

    def __init__(self):
        self._ocr_engine = None
        self._init_engine()

    def _init_engine(self):
        """Attempts to initialize local OCR engine if installed (pytesseract or easyocr)."""
        try:
            import pytesseract
            self._ocr_engine = pytesseract
            log_debug("[VISION] Local pytesseract engine initialized.")
        except Exception:
            self._ocr_engine = None

    def extract_text_from_image(self, image_path: str) -> str:
        """
        Extracts raw textual content from the given screenshot.
        Uses pytesseract if available, else returns structured inspection signals.
        """
        if self._ocr_engine and HAS_PIL:
            try:
                img = Image.open(image_path)
                text = self._ocr_engine.image_to_string(img)
                return text.strip()
            except Exception as e:
                log_warn(f"[VISION] OCR image_to_string error: {e}")

        # Fallback text extraction simulation for test/mock environments
        return ""

    def parse_ui_elements(self, raw_text: str, image_width: int = 1920, image_height: int = 1080) -> List[DetectedUIElement]:
        """
        Parses detected text into categorized UI elements (buttons, errors, text lines).
        """
        elements: List[DetectedUIElement] = []
        lines = [line.strip() for line in raw_text.split("\n") if line.strip()]

        for idx, line in enumerate(lines):
            # Estimate vertical position across document
            approx_y = int((idx / max(1, len(lines))) * image_height * 0.8) + 50
            approx_x = 100
            approx_w = min(image_width - 150, len(line) * 9)
            approx_h = 24

            # Check if this line contains an error signal
            is_error = any(re.search(pat, line) for pat in self.COMMON_ERROR_PATTERNS)
            if is_error:
                elements.append(DetectedUIElement(
                    element_type="error",
                    text=line,
                    box=(approx_x, approx_y, approx_w, approx_h),
                    confidence=0.95,
                ))
                continue

            # Check for button labels
            if len(line.split()) <= 3 and any(re.search(pat, line) for pat in self.BUTTON_PATTERNS):
                elements.append(DetectedUIElement(
                    element_type="button",
                    text=line,
                    box=(approx_x, approx_y, approx_w, approx_h),
                    confidence=0.90,
                ))
                continue

            elements.append(DetectedUIElement(
                element_type="text",
                text=line,
                box=(approx_x, approx_y, approx_w, approx_h),
                confidence=0.80,
            ))

        return elements

    def extract_error_details(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extracts specific error type, file, line number, and summary from on-screen text.
        """
        for pat in self.COMMON_ERROR_PATTERNS:
            m = re.search(pat, text)
            if m:
                error_snippet = m.group(0).strip()
                
                # Extract file and line number if present (e.g. File "main.py", line 42)
                m_loc = re.search(r"(?i)File\s+[\"']([^\"']+)[\"'],\s+line\s+(\d+)", text)
                file_name = m_loc.group(1) if m_loc else None
                line_no = int(m_loc.group(2)) if m_loc else None

                return {
                    "has_error": True,
                    "error_text": error_snippet,
                    "file": file_name,
                    "line": line_no,
                    "raw_context": text[:500],
                }

        return None
