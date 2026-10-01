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
        self._easyocr_cls = None
        self._easyocr_reader = None
        self._init_engine()

    def _init_engine(self):
        """Attempts to initialize local OCR engine if installed (pytesseract or easyocr)."""
        try:
            import pytesseract
            self._ocr_engine = pytesseract
            log_debug("[VISION] Local pytesseract module found.")
        except Exception:
            self._ocr_engine = None

        try:
            import easyocr
            self._easyocr_cls = easyocr
            log_debug("[VISION] Local EasyOCR module found.")
        except Exception:
            self._easyocr_cls = None

    def filter_chitti_terminal_noise(self, raw_text: str) -> str:
        """
        Filters out Chitti's own console prompts, banners, router logs, and terminal echoes
        so that self-reflection feedback loops are avoided.
        """
        if not raw_text:
            return ""
        lines = raw_text.split("\n")
        clean: List[str] = []
        skip_patterns = [
            r"^(?:\[CHITTI\]|\[MASTER ROUTER\]|\[THINKING\]|\[VISION\]|\[HUD\]|\[DOCUMENT SUMMARY\])",
            r"^(?:\[Ready\] Press ENTER|You \(typed\):|You:)",
            r"^\+[-=]+\+",
            r"^\|\s*C\s*H\s*I\s*T\s*T\s*I\s*\|",
            r"^\|\s*Personal AI Desktop Companion Robot",
            r"^\|\s*Phase \d+:",
            r"^(?:Route:|Confidence:|Memory retrieval:|Computer agent:)",
        ]
        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue
            if any(re.search(pat, line_str, re.IGNORECASE) for pat in skip_patterns):
                continue
            clean.append(line)
        return "\n".join(clean)

    def extract_text_from_image(self, image_path: str) -> str:
        """
        Extracts raw textual content from the given screenshot.
        Uses pytesseract or EasyOCR with intelligent fallbacks and de-noising.
        """
        # 1. Try pytesseract if available and binary working
        if self._ocr_engine and HAS_PIL:
            try:
                img = Image.open(image_path)
                text = self._ocr_engine.image_to_string(img)
                if text and text.strip():
                    return self.filter_chitti_terminal_noise(text.strip())
            except Exception as e:
                log_debug(f"[VISION] Pytesseract fallback: {e}")

        # 2. Try EasyOCR for robust local neural OCR
        if self._easyocr_cls:
            try:
                if self._easyocr_reader is None:
                    self._easyocr_reader = self._easyocr_cls.Reader(["en"], gpu=False, verbose=False)
                res = self._easyocr_reader.readtext(image_path)
                lines = [item[1].strip() for item in res if item and len(item) > 1 and item[1].strip()]
                text = "\n".join(lines)
                if text.strip():
                    return self.filter_chitti_terminal_noise(text.strip())
            except Exception as e:
                log_debug(f"[VISION] EasyOCR extraction error: {e}")

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

    def extract_document_structure(self, raw_text: str, window_title: str = "") -> Dict[str, Any]:
        """
        Cleans and extracts structured document sections (headings, bullets, paragraphs) from on-screen text.
        """
        lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
        clean_lines: List[str] = []
        headings: List[str] = []
        bullet_points: List[str] = []

        # Filter out common IDE / OS menu bar clutter
        clutter_patterns = [
            r"^(?:(?:File|Edit|Selection|View|Go|Run|Terminal|Help|Window|Format|Tools)\s*)+$",
            r"^(?:Problems|Output|Debug Console|Terminal|Ports)$",
            r"^(?:http[s]?://|www\.)",
            r"^(?:Ln\s+\d+,\s+Col\s+\d+|Spaces:\s+\d+|UTF-8|CRLF)",
        ]

        for line in lines:
            if any(re.match(p, line, re.IGNORECASE) for p in clutter_patterns):
                continue
            clean_lines.append(line)

            # Detect headings (# Heading, or short title-case lines)
            if re.match(r"^#{1,4}\s+", line) or (len(line) < 60 and line.endswith(":") and len(line.split()) < 8):
                headings.append(re.sub(r"^#{1,4}\s+", "", line).rstrip(":").strip())
            # Detect bullet points
            elif re.match(r"^[\*\-•]\s+", line) or re.match(r"^\d+\.\s+", line):
                bullet_points.append(re.sub(r"^[\*\-•\d\.]+\s*", "", line).strip())

        # Clean document title from window title
        doc_title = window_title
        for suffix in [" - Visual Studio Code", " - VS Code", " - Google Chrome", " - Microsoft Edge", " - Adobe Acrobat Reader", " - Notepad", " - Word"]:
            doc_title = doc_title.replace(suffix, "")
        doc_title = doc_title.strip() or "On-Screen Document"

        # Detect document format
        doc_type = "Document"
        lower_title = window_title.lower()
        lower_clean = doc_title.lower()
        if lower_clean.endswith(".pdf") or ".pdf" in lower_title or "acrobat" in lower_title or "pdf" in lower_title:
            doc_type = "PDF Document"
        elif lower_clean.endswith((".md", ".markdown")) or ".md" in lower_title or "markdown" in lower_title:
            doc_type = "Markdown Document"
        elif lower_clean.endswith((".py", ".js", ".ts", ".cpp", ".java", ".html", ".css", ".json", ".env")) or any(ext in lower_title for ext in [".py", ".js", ".ts", ".cpp", ".java", ".html", ".css", ".json"]):
            doc_type = "Source Code Document"
        elif lower_clean.endswith((".docx", ".doc")) or ".doc" in lower_title or "word" in lower_title:
            doc_type = "Word Document"
        elif "chrome" in lower_title or "edge" in lower_title or "browser" in lower_title or "firefox" in lower_title:
            doc_type = "Web Page / Article"
        elif "notepad" in lower_title or lower_clean.endswith(".txt") or ".txt" in lower_title:
            doc_type = "Text Document"

        word_count = sum(len(line.split()) for line in clean_lines)

        return {
            "title": doc_title,
            "document_type": doc_type,
            "headings": headings,
            "bullet_points": bullet_points,
            "clean_lines": clean_lines,
            "word_count": word_count,
            "clean_text": "\n".join(clean_lines),
        }

