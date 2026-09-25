"""
Chitti Laptop Agent Command Parser.
Parses natural-language commands (English, Hindi, Hinglish) into deterministic StructuredAction objects.
"""

import re
from typing import Optional

from src.agent.actions import ActionType, RiskLevel, StructuredAction
from src.agent.registry import DEFAULT_URL_MAP
from src.utils.logging import log_debug


class ActionParser:
    """Parses natural language requests into structured laptop action commands."""

    @classmethod
    def parse_command(cls, user_text: str) -> Optional[StructuredAction]:
        raw = user_text.strip()
        lower = raw.lower()

        # Clean Chitti invocation prefix if present
        clean = re.sub(r"^(?:chitti,?\s*|hey chitti,?\s*|bhai,?\s*|please\s+)", "", raw, flags=re.IGNORECASE).strip()
        clean_lower = clean.lower()

        # 1. SCREENSHOT COMMANDS
        screenshot_patterns = [
            r"(?i)\b(?:take\s+(?:a\s+)?screenshot|capture\s+(?:the\s+)?screen|screen\s+capture)\b",
            r"(?i)\b(?:screenshot\s+(?:le\s+lo|lo|kheecho|khincho|lelo)|screen\s+(?:capture\s+karo|ka\s+screenshot))\b",
            r"(?:स्क्रीनशॉट\s+लो|स्क्रीनशॉट\s+ले\s+लो|स्क्रीन\s+कैप्चर\s+करो)",
        ]
        if any(re.search(pat, clean) for pat in screenshot_patterns):
            return StructuredAction(
                action=ActionType.TAKE_SCREENSHOT,
                parameters={},
                risk_level=RiskLevel.LOW,
                requires_confirmation=False,
                raw_input=raw,
            )

        # 2. SYSTEM INFORMATION COMMANDS
        system_info_patterns = [
            r"(?i)\b(?:how\s+much\s+(?:ram|memory|disk\s+space|storage)|what\s+(?:is\s+my\s+)?(?:disk\s+space|ram|gpu|cpu|operating\s+system|os))\b",
            r"(?i)\b(?:system\s+(?:info|information|specs)|specs\s+batao|ram\s+kitni\s+hai|disk\s+space\s+kitna\s+hai)\b",
            r"(?:रैम\s+कितनी\s+है|डिस्क\s+स्पेस\s+कितना\s+है|सिस्टम\s+इन्फो)",
        ]
        if any(re.search(pat, clean) for pat in system_info_patterns):
            q_type = "general"
            if "ram" in clean_lower or "memory" in clean_lower:
                q_type = "ram"
            elif "disk" in clean_lower or "storage" in clean_lower:
                q_type = "disk"
            elif "gpu" in clean_lower or "graphics" in clean_lower:
                q_type = "gpu"
            elif "cpu" in clean_lower or "processor" in clean_lower:
                q_type = "cpu"
            elif "os" in clean_lower or "operating system" in clean_lower:
                q_type = "os"

            return StructuredAction(
                action=ActionType.GET_SYSTEM_INFO,
                parameters={"query_type": q_type},
                risk_level=RiskLevel.LOW,
                requires_confirmation=False,
                raw_input=raw,
            )

        # 3. VOLUME CONTROL COMMANDS
        volume_patterns = [
            r"(?i)\b(?:(?:increase|raise|turn\s+up|boost)\s+volume|volume\s+(?:badhao|badha\s+do|tez\s+karo))\b",
            r"(?i)\b(?:(?:decrease|lower|turn\s+down|reduce)\s+volume|volume\s+(?:kam\s+karo|kam\s+kar\s+do|dheemi\s+karo))\b",
            r"(?i)\b(?:mute(?:\s+audio|\s+sound|\s+volume)?|mute\s+kar\s+do|awaaz\s+band\s+karo)\b",
            r"(?i)\b(?:unmute(?:\s+audio|\s+sound|\s+volume)?|unmute\s+kar\s+do)\b",
            r"(?:वॉल्यूम\s+बढ़ाओ|वॉल्यूम\s+कम\s+करो|म्यूट\s+करो|अनम्यूट\s+करो)",
        ]
        if any(re.search(pat, clean) for pat in volume_patterns):
            op = "increase"
            if any(k in clean_lower for k in ["decrease", "lower", "kam", "reduce", "turn down", "dheemi", "कम"]):
                op = "decrease"
            elif any(k in clean_lower for k in ["unmute", "अनम्यूट"]):
                op = "unmute"
            elif any(k in clean_lower for k in ["mute", "band karo", "म्यूट"]):
                op = "mute"

            return StructuredAction(
                action=ActionType.SET_VOLUME,
                parameters={"operation": op},
                risk_level=RiskLevel.LOW,
                requires_confirmation=False,
                raw_input=raw,
            )

        # 4. DESTRUCTIVE ACTIONS (Delete File / Folder)
        m_del_folder = re.search(r"(?i)\b(?:delete|remove)\s+(?:the\s+)?folder\s+([A-Za-z0-9_\-\.\s]+)", clean) or \
                       re.search(r"(?i)\b([A-Za-z0-9_\-\.]+)\s+folder\s+(?:delete|hatao|remove)\s+karo\b", clean)
        if m_del_folder:
            target = m_del_folder.group(1).strip().rstrip(".!? \t\n")
            return StructuredAction(
                action=ActionType.DELETE_FOLDER,
                parameters={"target": target},
                risk_level=RiskLevel.HIGH,
                requires_confirmation=True,
                raw_input=raw,
            )

        m_del_file = re.search(r"(?i)\b(?:delete|remove)\s+(?:the\s+)?file\s+([A-Za-z0-9_\-\.\s]+)", clean) or \
                     re.search(r"(?i)\b([A-Za-z0-9_\-\.]+)\s+file\s+(?:delete|hatao|remove)\s+karo\b", clean)
        if m_del_file:
            target = m_del_file.group(1).strip().rstrip(".!? \t\n")
            return StructuredAction(
                action=ActionType.DELETE_FILE,
                parameters={"target": target},
                risk_level=RiskLevel.HIGH,
                requires_confirmation=True,
                raw_input=raw,
            )

        # 5. CREATE TEXT FILE COMMANDS
        m_create_file = re.search(r"(?i)\b(?:create\s+(?:a\s+)?(?:text\s+)?file|make\s+(?:a\s+)?file)\s+([A-Za-z0-9_\-\.]+)(?:\s+with\s+(?:content|text)\s+(.*))?", clean) or \
                        re.search(r"(?i)\b([A-Za-z0-9_\-\.]+)\s+(?:text\s+)?file\s+(?:banao|create\s+karo)(?:\s+(?:with|me|mein)\s+(.*))?", clean)
        if m_create_file:
            filename = m_create_file.group(1).strip()
            content = m_create_file.group(2).strip() if m_create_file.group(2) else ""
            return StructuredAction(
                action=ActionType.CREATE_TEXT_FILE,
                parameters={"name": filename, "content": content},
                risk_level=RiskLevel.LOW,
                requires_confirmation=False,
                raw_input=raw,
            )

        # 6. CREATE FOLDER COMMANDS
        m_create_folder = re.search(r"(?i)\b(?:create\s+(?:a\s+)?(?:new\s+)?folder(?:\s+named|\s+called)?)\s+([A-Za-z0-9_\-\s]+)", clean) or \
                          re.search(r"(?i)\b([A-Za-z0-9_\-\s]+)\s+naam\s+ka\s+folder\s+banao\b", clean) or \
                          re.search(r"(?i)\b([A-Za-z0-9_\-\s]+)\s+folder\s+(?:create\s+karo|banao)\b", clean)
        if m_create_folder:
            foldername = m_create_folder.group(1).strip().rstrip(".!? \t\n")
            if foldername.lower() not in {"a", "new", "this"}:
                return StructuredAction(
                    action=ActionType.CREATE_FOLDER,
                    parameters={"name": foldername},
                    risk_level=RiskLevel.LOW,
                    requires_confirmation=False,
                    raw_input=raw,
                )

        # 7. OPEN URL COMMANDS
        # Check standard URL names (GitHub, Google, YouTube) or explicit http/https
        m_url = re.search(r"(?i)^(?:open|launch|visit|go\s+to)\s+(https?://\S+|www\.\S+)", clean)
        if m_url:
            url = m_url.group(1).strip()
            if not url.startswith("http"):
                url = f"https://{url}"
            return StructuredAction(
                action=ActionType.OPEN_URL,
                parameters={"url": url},
                risk_level=RiskLevel.LOW,
                requires_confirmation=False,
                raw_input=raw,
            )

        for site_name, url in DEFAULT_URL_MAP.items():
            if re.search(rf"(?i)(?:open\s+{site_name}|^{site_name}\s+(?:kholo|chalao|open\s+karo|खोलो))(?:\s+|$|[.,!?])", clean):
                return StructuredAction(
                    action=ActionType.OPEN_URL,
                    parameters={"url": url, "site_name": site_name.title()},
                    risk_level=RiskLevel.LOW,
                    requires_confirmation=False,
                    raw_input=raw,
                )

        # 8. OPEN FOLDER COMMANDS
        m_folder = re.search(r"(?i)^(?:open\s+(?:the\s+)?([A-Za-z0-9_\-\s]+?)\s+folder|([A-Za-z0-9_\-\s]+?)\s+folder\s+(?:kholo|open\s+karo|खोलो))(?:\s+|$|[.,!?])", clean)
        if m_folder:
            target_folder = (m_folder.group(1) or m_folder.group(2)).strip()
            return StructuredAction(
                action=ActionType.OPEN_FOLDER,
                parameters={"target": target_folder},
                risk_level=RiskLevel.LOW,
                requires_confirmation=False,
                raw_input=raw,
            )

        # Direct folder alias checks (e.g., "Open Downloads", "Downloads kholo")
        folder_aliases = ["downloads", "documents", "desktop", "pictures", "videos", "music"]
        for fa in folder_aliases:
            if re.search(rf"(?i)(?:^open\s+{fa}|^{fa}\s+(?:kholo|open\s+karo|खोलो))(?:\s+|$|[.,!?])", clean):
                return StructuredAction(
                    action=ActionType.OPEN_FOLDER,
                    parameters={"target": fa.title()},
                    risk_level=RiskLevel.LOW,
                    requires_confirmation=False,
                    raw_input=raw,
                )

        # 9. CLOSE APPLICATION COMMANDS
        m_close_app = re.search(r"(?i)^(?:close|exit|terminate|kill|shut\s+down)\s+([A-Za-z0-9_\-\s]+)", clean) or \
                      re.search(r"(?i)^([A-Za-z0-9_\-\s]+?)\s+(?:band\s+karo|band\s+kar\s+do|close\s+karo|बंद\s+करो)(?:\s+|$|[.,!?])", clean)
        if m_close_app:
            app_target = m_close_app.group(1).strip().rstrip(".!? \t\n")
            if app_target.lower() not in {"this", "window", "folder", "chitti"}:
                return StructuredAction(
                    action=ActionType.CLOSE_APPLICATION,
                    parameters={"target": app_target},
                    risk_level=RiskLevel.MEDIUM,
                    requires_confirmation=False,
                    raw_input=raw,
                )

        # 10. OPEN APPLICATION COMMANDS (Hindi / Hinglish target-first patterns)
        # e.g. "Chrome kholo", "Chrome ko open karo", "VS Code chala do", "Chrome खोलो"
        m_app_hi = re.search(
            r"(?i)^([A-Za-z0-9_\-\s]+?)(?:\s+ko|\s+app)?\s+(?:kholo|chalao|open\s+karo|chala\s+do|खोलो|चलाओ)(?:\s+|$|[.,!?])",
            clean
        )
        if m_app_hi:
            app_target = m_app_hi.group(1).strip().rstrip(".!? \t\n")
            if app_target.lower() not in {"the", "a", "my", "this", "file", "folder", "url"}:
                app_target = re.sub(r"^(?:the\s+|app\s+)", "", app_target, flags=re.IGNORECASE).strip()
                return StructuredAction(
                    action=ActionType.OPEN_APPLICATION,
                    parameters={"target": app_target},
                    risk_level=RiskLevel.LOW,
                    requires_confirmation=False,
                    raw_input=raw,
                )

        # 11. OPEN APPLICATION COMMANDS (English verb-first patterns)
        # e.g. "Open Chrome", "Launch VS Code", "Start Calculator"
        m_app_en = re.search(r"(?i)^(?:open|launch|start|run)\s+(?:the\s+|app\s+)?([A-Za-z0-9_\-\s]+)$", clean)
        if m_app_en:
            app_target = m_app_en.group(1).strip().rstrip(".!? \t\n")
            # Exclude non-app words (e.g. general questions or verbs)
            if app_target.lower() not in {
                "the", "a", "my", "this", "deep learning", "cnn", "dbms", "leetcode",
                "file", "folder", "url", "question", "problem", "solution"
            }:
                return StructuredAction(
                    action=ActionType.OPEN_APPLICATION,
                    parameters={"target": app_target},
                    risk_level=RiskLevel.LOW,
                    requires_confirmation=False,
                    raw_input=raw,
                )

        return None
