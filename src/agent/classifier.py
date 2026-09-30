"""
Chitti Task & Project Intent Classifier (Phase 6).
Classifies user requests into distinct actionable vs conversational categories,
identifies project creation/modification/execution intents, extracts structured
requirements, and prevents memory router interference with computer operations.
"""

import enum
import re
from dataclasses import dataclass, field
from typing import List, Optional


class TaskIntent(str, enum.Enum):
    # Conversational & Informational Categories
    GENERAL_CONVERSATION = "GENERAL_CONVERSATION"
    GENERAL_KNOWLEDGE = "GENERAL_KNOWLEDGE"
    TECHNICAL_QUERY = "TECHNICAL_QUERY"

    # Project-Building & Software Development Categories (Phase 6 Agent)
    CREATE_PROJECT = "CREATE_PROJECT"
    MODIFY_PROJECT = "MODIFY_PROJECT"
    RUN_PROJECT = "RUN_PROJECT"
    DEBUG_PROJECT = "DEBUG_PROJECT"
    BUILD_PROJECT = "BUILD_PROJECT"
    TEST_PROJECT = "TEST_PROJECT"

    # Computer Control & Operations Categories
    COMPUTER_TASK = "COMPUTER_TASK"
    FILE_OPERATION = "FILE_OPERATION"
    BROWSER_TASK = "BROWSER_TASK"

    # Personal & Memory Categories
    PERSONAL_MEMORY = "PERSONAL_MEMORY"
    RELATIONSHIP_MEMORY = "RELATIONSHIP_MEMORY"
    PROJECT_MEMORY = "PROJECT_MEMORY"
    PREFERENCE_MEMORY = "PREFERENCE_MEMORY"
    EXPLICIT_MEMORY = "EXPLICIT_MEMORY"

    # Backward compatibility aliases
    CODE_GENERATION = "CREATE_PROJECT"
    PERSONAL_QUERY = "PERSONAL_MEMORY"


@dataclass
class ClassificationResult:
    intent: TaskIntent
    is_actionable_task: bool
    confidence: float
    reason: str
    is_computer_task: bool = False  # Backward compatibility field
    project_type: Optional[str] = None
    ui_required: bool = False
    execution_requested: bool = False
    extracted_requirements: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.is_actionable_task and not self.is_computer_task:
            self.is_computer_task = True


from src.utils.text import normalize_typos


class TaskClassifier:
    """
    Dedicated Intent Classification Layer for actionable computer/project tasks vs conversational queries.
    Decoupled from long-term memory retrieval.
    """

    # 1. Personal & Identity Query Patterns
    PERSONAL_PATTERNS = [
        r"(?i)^(?:what\s+is\s+my\s+name|who\s+am\s+i|who\s+created\s+you|who\s+made\s+you|tell\s+me\s+about\s+my\s+college|mera\s+naam\s+kya\s+hai|tumhe\s+kisne\s+banaya|who\s+create\s+u)\b",
        r"(?i)^(?:do\s+you\s+remember\s+me|kya\s+tum\s+mujhe\s+jaante\s+ho)\b",
    ]

    # 2. Pure Informational / Conceptual Questions ("what is", "explain", "why")
    KNOWLEDGE_PATTERNS = [
        r"(?i)^(?:what\s+is|what\s+are|define|explain|meaning\s+of|difference\s+between|why\s+is|why\s+do|how\s+does)\s+(?:a\s+|an\s+|the\s+)?(?:[a-zA-Z0-9_\-\s]+)[\?\.]?$",
        r"(?i)\b(?:kya\s+hota\s+hai|kya\s+hai|samjhao|explain\s+karo)\b",
        r"(?i)^(?:tell\s+me\s+about|what\s+do\s+you\s+know\s+about)\s+",
    ]

    # 3. Technical How-To / Code Snippet Questions ("how do I", "give me code for")
    TECHNICAL_QUERY_PATTERNS = [
        r"(?i)^(?:how\s+(?:do\s+i|can\s+i|to)|how\s+would\s+i)\s+(?:write|create|make|build|code|implement|use|setup)\b",
        r"(?i)^(?:give\s+me|show\s+me|provide|share)\s+(?:an?\s+)?(?:example|code|snippet|sample|tutorial|syntax)\s+(?:of|for|to)\b",
        r"(?i)^(?:how\s+does\s+[a-zA-Z0-9_]+\s+work)\b",
    ]

    # 4. Project Modification & Open Patterns
    PROJECT_MODIFY_PATTERNS = [
        r"(?i)\b(?:open\s+(?:my\s+)?(?:existing\s+)?([a-zA-Z0-9_\-]+)\s+project\s+(?:and|aur)\s+(?:add|modify|update|change|implement))\b",
        r"(?i)\b(?:open\s+(?:my\s+)?([a-zA-Z0-9_\-]+)\s+project(?:\s+in\s+.*)?)\b",
        r"(?i)\b(?:add\s+.*(?:to|in)\s+(?:my\s+)?(?:existing\s+)?project)\b",
        r"(?i)\b(?:modify|update|refactor|enhance)\s+(?:my\s+)?(?:existing\s+)?(?:project|code|app)\b",
    ]

    # 5. Project Run / Test / Debug Patterns
    PROJECT_RUN_PATTERNS = [
        r"(?i)\b(?:run\s+(?:the\s+)?(?:project|tests?|pytest|app|application)|test\s+(?:the\s+)?project|is\s+program\s+ko\s+run\s+karo|test\s+chalao)\b",
        r"(?i)\b(?:debug|fix\s+the\s+error|find\s+bug)\s+(?:in|of)\s+.*",
    ]

    # 6. Computer Control & Application Patterns
    COMPUTER_CONTROL_PATTERNS = [
        r"(?i)\b(?:open\s+.*(?:whatsapp|telegram|slack|discord|gmail|email|messages?)|whatsapp\s+(?:web\s+)?(?:kholo|open|chalao)|(?:message|msg|send(?:\s+a)?\s+message|send)\s+.*to\s+.*|.*ko\s+.*(?:message|bhejo|msg)|send\s+(?:an?\s+)?(?:email|mail))\b",
        r"(?i)\b(?:send|bhejo|dispatch)\s+.*(?:on|via|pe|par)\s+(?:whatsapp|telegram|slack|discord|gmail|email)\b",
        r"(?i)\b(?:open|launch|visit|navigate|go\s+to)\s+(?:https?://\S+|www\.\S+|[a-zA-Z0-9_\-\.]+\.[a-zA-Z]{2,}(?:/\S*)?)(?:\s+in\s+browser)?\b",
        r"(?i)\b(?:take|capture)\s+(?:a\s+)?screenshot\b",
        r"(?i)\b(?:play\s+.*(?:song|music|track)|play\s+.*on\s+youtube|youtube\s+pe.*chalao|gaana\s+chalao|search\s+youtube\s+for)\b",
        r"(?i)\b(?:search\s+(?:for\s+)?.*on\s+(?:google|chrome|browser|bing))\b",
        r"(?i)\b(?:type\s+.*into\s+[a-zA-Z0-9_\-]+|open\s+[a-zA-Z0-9_\-]+\s+and\s+type)\b",
        r"(?i)\b(?:create|make|delete|remove|rename|move)\s+(?:folder|directory|file)\b",
        r"(?i)\b(?:open|launch|kholo|chalao|show)\s+(?:the\s+)?(?:[a-zA-Z0-9_\-]+\s+)?(?:folder|directory)\b",
        r"(?i)\b(?:open|launch|kholo|chalao)\s+(?:downloads|documents|desktop|pictures|music|videos)\b",
        r"(?i)^(?:open|launch|start|run|visit|go\s+to)\s+(?:the\s+|app\s+|my\s+)?([a-zA-Z0-9_\-\.\:\/\s]+)$",
        r"(?i)^([a-zA-Z0-9_\-\.\:\/\s]+?)(?:\s+ko|\s+app)?\s+(?:kholo|chalao|open\s+karo|chala\s+do|खोलो|चलाओ)(?:\s+|$|[.,!?])",
        r"(?i)^(?:close|exit|terminate|kill|shut\s+down)\s+(?:the\s+|app\s+)?([a-zA-Z0-9_\-\s]+)",
        r"(?i)^([a-zA-Z0-9_\-\s]+?)\s+(?:band\s+karo|band\s+kar\s+do|close\s+karo|बंद\s+करो)(?:\s+|$|[.,!?])",
    ]

    @classmethod
    def _normalize_text(cls, text: str) -> str:
        """Corrects common typographical errors in action verbs and application names."""
        return normalize_typos(text)

    @classmethod
    def classify(cls, user_text: str) -> ClassificationResult:
        """
        Classifies the user input into clear semantic intent categories and extracts project metadata.
        """
        raw = cls._normalize_text(user_text.strip())
        lower = raw.lower()

        # 1. Check for Personal & Identity Queries
        if any(re.search(pat, raw) for pat in cls.PERSONAL_PATTERNS):
            return ClassificationResult(
                intent=TaskIntent.PERSONAL_MEMORY,
                is_actionable_task=False,
                confidence=0.95,
                reason="User asking personal identity, name, or creator relationship question",
            )

        # 2. Check for Project Modification Queries
        if any(re.search(pat, raw) for pat in cls.PROJECT_MODIFY_PATTERNS):
            return ClassificationResult(
                intent=TaskIntent.MODIFY_PROJECT,
                is_actionable_task=True,
                confidence=0.95,
                reason="User requested modifying or enhancing an existing software project",
                execution_requested=bool(re.search(r"(?i)\b(?:run|test|execute)\b", raw)),
            )

        # 3. Check for Project Run / Test / Debug Queries
        if any(re.search(pat, raw) for pat in cls.PROJECT_RUN_PATTERNS):
            intent = TaskIntent.TEST_PROJECT if "test" in lower or "pytest" in lower else (
                TaskIntent.DEBUG_PROJECT if "debug" in lower or "fix" in lower else TaskIntent.RUN_PROJECT
            )
            return ClassificationResult(
                intent=intent,
                is_actionable_task=True,
                confidence=0.95,
                reason=f"User requested project operation: {intent.value}",
                execution_requested=True,
            )

        # 4. Check for Technical "How-to" / Informational Code Snippet Questions
        # If user explicitly asks "how do I create...", "give me code for...", "explain..." WITHOUT asking Chitti to do it
        is_informational_howto = any(re.search(pat, raw) for pat in cls.TECHNICAL_QUERY_PATTERNS)
        is_pure_conceptual = any(re.search(pat, raw) for pat in cls.KNOWLEDGE_PATTERNS)

        # Note: If user says "Create a todo list for me", it is NOT a how-to question.
        # But "How do I create a todo list in Python?" is a technical query.
        if is_informational_howto and not re.search(r"(?i)\b(?:for\s+me|in\s+vs\s*code|and\s+run\s+it|aur\s+run\s+karo|on\s+my\s+computer|bana\s+do|banao)\b", raw):
            return ClassificationResult(
                intent=TaskIntent.TECHNICAL_QUERY,
                is_actionable_task=False,
                confidence=0.90,
                reason="User asking conversational technical query or how-to explanation",
            )

        if is_pure_conceptual and not re.search(r"(?i)\b(?:banao|likho|create|build|make|develop|implement|generate|run\s+it|for\s+me)\b", raw):
            return ClassificationResult(
                intent=TaskIntent.GENERAL_KNOWLEDGE,
                is_actionable_task=False,
                confidence=0.92,
                reason="Conceptual or general knowledge question",
            )

        # 5. Check for Software Project Creation / Code Synthesis (Semantic Detection)
        # Signals: Action verbs (create, build, make, develop, implement, generate, write, design, banao, likho)
        # OR Desires (I want an application that..., I need a tool that..., can you build me...)
        is_creation_request = (
            # Optional VS Code prefix + Action verb + target / intent
            bool(re.search(r"(?i)\b(?:open\s+(?:vs\s*code|vscode|ide)\s*(?:and|aur|,)?\s*)?(?:create|build|make|develop|implement|generate|design|write)\s+(?:me\s+|us\s+|for\s+me\s+)?(?:an?|the|some|ek)?\s*(?:fully\s+functional\s+|interactive\s+|simple\s+|modern\s+|clean\s+|complete\s+)?(?:[a-zA-Z0-9_\-\s]+)", raw)) or
            # Desire structure: "I want an app...", "I need a system...", "can you make..."
            bool(re.search(r"(?i)\b(?:i\s+want|i\s+need|can\s+you\s+(?:build|create|make|develop)|please\s+(?:build|create|make|develop))\s+(?:an?|the|some)?\s*(?:application|app|project|website|system|tool|dashboard|tracker|script|solution|software|service|game|interface|api|program)\b", raw)) or
            # Hinglish: "X banao", "X likho", "X create karo"
            bool(re.search(r"(?i)\b(?:banao|bana\s+do|likho|create\s+karo|develop\s+karo|build\s+karo|bana\s+kro)\b", raw)) or
            # Direct programming language + task
            bool(re.search(r"(?i)\b(?:python|cpp|c\+\+|java|javascript|typescript|rust|golang|go|c\#|csharp|react|html|css|flask|fastapi|express|django|node)\b.*(?:banao|likho|create|write|implement|build|develop|app|project|code|program|script|system|tracker|calculator|dashboard|api|website)", raw))
        )

        if is_creation_request:
            # Extract metadata
            ui_required = bool(
                re.search(r"(?i)\b(?:ui|interactive\s+ui|good\s+ui|modern\s+ui|responsive\s+ui|gui|frontend|interface|user\s+interface|webpage|dashboard|website|portal|browser|web)\b", raw)
            )
            execution_requested = bool(
                re.search(r"(?i)\b(?:run|execute|chalao|run\s+karo|execute\s+karo|chala\s+do|and\s+run|aur\s+run|test\s+it|test\s+karo|in\s+browser|browser\s+me)\b", raw)
            )

            # Determine clean project type semantic description
            # E.g. "todo list", "calculator", "expense tracker", "attendance system"
            clean_proj = raw
            clean_proj = re.sub(r"(?i)^(?:chitti,?\s*|hey\s+chitti,?\s*|please\s+|can\s+you\s+|i\s+want\s+(?:an?|the)?\s*|i\s+need\s+(?:an?|the)?\s*)", "", clean_proj).strip()
            clean_proj = re.sub(r"(?i)^(?:open\s+(?:vs\s*code|vscode|ide)\s*(?:and|aur|,)?\s*)", "", clean_proj).strip()
            clean_proj = re.sub(r"(?i)^(?:create|build|make|develop|implement|generate|design|write)\s+(?:me\s+|us\s+|for\s+me\s+)?(?:an?|the|some|ek)?\s*(?:fully\s+functional\s+|interactive\s+|simple\s+|modern\s+|clean\s+|complete\s+)?", "", clean_proj).strip()
            clean_proj = re.sub(r"(?i)\s*(?:with\s+(?:an?\s+)?(?:interactive|good|modern|responsive|clean)?\s*ui|for\s+me|and\s+run\s+it|aur\s+run\s+karo|in\s+vs\s*code|in\s+python|in\s+react|in\s+c\+\+|in\s+java).*$", "", clean_proj).strip()
            clean_proj = re.sub(r"(?i)\s+(?:banao|likho|create\s+karo|develop\s+karo|kro|karo|do)$", "", clean_proj).strip()
            slug_type = re.sub(r"[^a-zA-Z0-9_\s]", "", clean_proj).strip().replace(" ", "_").lower() or "application"

            requirements = [
                f"Core functionality for {clean_proj or 'requested application'}",
            ]
            if "fully functional" in lower:
                requirements.append("Fully functional implementation with complete business logic (zero placeholders)")
            if ui_required:
                requirements.append("Interactive user interface with dynamic styling and event controls")

            return ClassificationResult(
                intent=TaskIntent.CREATE_PROJECT,
                is_actionable_task=True,
                confidence=0.96,
                reason=f"User requested software project creation: '{clean_proj or slug_type}'",
                project_type=slug_type,
                ui_required=ui_required,
                execution_requested=execution_requested,
                extracted_requirements=requirements,
            )

        # 6. Check for Dedicated Computer Control / App Launch / Browser / Messaging
        if any(re.search(pat, raw) for pat in cls.COMPUTER_CONTROL_PATTERNS):
            if any(w in lower for w in ["whatsapp", "youtube", "song", "gaana", "browser", "chrome", "web", "gmail", "message", "telegram", "slack"]):
                intent = TaskIntent.BROWSER_TASK
            elif "folder" in lower or "file" in lower or "desktop" in lower or "delete" in lower:
                intent = TaskIntent.FILE_OPERATION
            else:
                intent = TaskIntent.COMPUTER_TASK

            return ClassificationResult(
                intent=intent,
                is_actionable_task=True,
                confidence=0.95,
                reason=f"Direct computer control trigger: {intent.value}",
            )

        # 7. Default Fallback
        return ClassificationResult(
            intent=TaskIntent.GENERAL_CONVERSATION,
            is_actionable_task=False,
            confidence=0.70,
            reason="General conversation routing",
        )
