"""
Chitti Unified Master Router.
Acts as the central decision-making brain layer for incoming requests:
Determines whether a user prompt is a Chat/Identity query, General/Technical Knowledge query,
Personal/Relationship/Project Memory retrieval, Phase 5 Computer/OS Task, or Phase 6 Project-Building Agent.
"""

import re
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from src.utils.logging import log_chitti, log_debug, log_info


class MasterRoute(str, Enum):
    # 1. Conversational & Self Identity
    CHAT = "CHAT"
    SELF_IDENTITY = "SELF_IDENTITY"

    # 2. Informational & Technical Knowledge
    GENERAL_KNOWLEDGE = "GENERAL_KNOWLEDGE"
    TECHNICAL_KNOWLEDGE = "TECHNICAL_KNOWLEDGE"

    # 3. Memory Subsystems
    PERSONAL_MEMORY = "PERSONAL_MEMORY"
    RELATIONSHIP_MEMORY = "RELATIONSHIP_MEMORY"
    PROJECT_MEMORY = "PROJECT_MEMORY"
    PREFERENCE_MEMORY = "PREFERENCE_MEMORY"
    EXPLICIT_MEMORY = "EXPLICIT_MEMORY"

    # 4. Phase 6 Project-Building Agent
    PROJECT_CREATION = "PROJECT_CREATION"
    PROJECT_MODIFICATION = "PROJECT_MODIFICATION"
    PROJECT_EXECUTION = "PROJECT_EXECUTION"
    PROJECT_DEBUGGING = "PROJECT_DEBUGGING"

    # 5. Phase 5 Real Computer Control & Multi-Step OS Actions
    COMPUTER_TASK = "COMPUTER_TASK"
    FILE_OPERATION = "FILE_OPERATION"
    BROWSER_TASK = "BROWSER_TASK"
    SYSTEM_TASK = "SYSTEM_TASK"

    # 6. Specialized Multimodal / Service Tasks
    VISION_TASK = "VISION_TASK"
    TRANSLATION_TASK = "TRANSLATION_TASK"

    # Fallback
    UNKNOWN = "UNKNOWN"


@dataclass
class MasterRouteDecision:
    route: MasterRoute
    confidence: float
    requires_memory: bool
    requires_computer: bool
    requires_phase5: bool
    requires_phase6: bool
    explanation: str = ""
    extracted_requirements: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MasterRouter:
    """
    Central Master Router for Chitti.
    Examines semantic meaning and structure of user requests to determine the appropriate subsystem.
    """

    # 1. Self-Identity Patterns
    SELF_IDENTITY_PATTERNS = [
        r"(?i)^(?:tell\s+me\s+about\s+yourself|who\s+are\s+you|what\s+are\s+you|introduce\s+yourself|your\s+introduction)\b",
        r"(?i)^(?:tum\s+kaun\s+ho|apne\s+baare\s+me(?:in)?\s+batao|apna\s+intro\s+do|tum\s+kya\s+ho)\b",
        r"(?:तुम\s+कौन\s+हो|अपने\s+बारे\s+में\s+बताओ)",
    ]

    # 2. Personal & Long-term Memory Query Patterns
    PERSONAL_MEMORY_PATTERNS = [
        r"(?i)\b(?:what\s+do\s+(?:you|u)\s+know\s+about\s+me|tell\s+me\s+about\s+me|what\s+is\s+my\s+name|who\s+(?:am\s+i|i\s+am))\b",
        r"(?i)\b(?:do\s+(?:you|u)\s+know\s+(?:who\s+(?:i\s+am|am\s+i)|me)|do\s+(?:you|u)\s+remember\s+me|what\s+do\s+(?:you|u)\s+remember\s+about\s+me|list\s+my\s+memories|my\s+memories)\b",
        r"(?i)\b(?:who\s+(?:created|made|built|developed)\s+(?:you|u)|who\s+is\s+your\s+creator|who\s+developed\s+you)\b",
        r"(?i)\b(?:what\s+is\s+my\s+(?:college|university|job|profession|work|branch|degree|goal|hobby|favorite\s+\w+|preferred\s+\w+))\b",
        r"(?i)\b(?:tell\s+me\s+about\s+my\s+(?:college|university|friends|family|projects|work|sister|brother|best\s+friend|hobbies))\b",
        r"(?i)\b(?:who\s+is\s+(?:my\s+)?(?:best\s+friend|friend|sister|brother|father|mother|teammate|coworker|partner|girlfriend|boyfriend|wife|husband))\b",
        r"(?i)\b(?:mera\s+naam\s+kya\s+hai|mujhe\s+jaante\s+ho|mere\s+baare\s+me(?:in)?\s+kya\s+jaante\s+ho|tumhe\s+kisne\s+banaya|main\s+kaun\s+hoon)\b",
        r"(?i)\b(?:mera|meri|mere)\s+(?:preferred|favourite|favorite|college|naam|dost|project|goal|degree|language|programming\s+language)\b.*(?:kya\s+hai|kya\s+tha|batao|yaad\s+hai)",
        r"(?:तुम्हें\s+मेरे\s+बारे\s+में\s+क्या\s+याद\s+है|मेरा\s+नाम\s+क्या\s+है|तुम्हें\s+किसने\s+बनाया)",
    ]

    # 3. Explicit Memory Modification / Store Patterns
    EXPLICIT_MEMORY_PATTERNS = [
        r"(?i)\b(?:remember\s+that|don'?t\s+forget\s+that|save\s+this\s+fact|keep\s+in\s+mind\s+that|note\s+down)\b",
        r"(?i)\b(?:yaad\s+rakhna\s+ki|yaad\s+rakho\s+ki|ye\s+yaad\s+rakhna|bhoolna\s+mat)\b",
        r"(?i)\b(?:forget\s+that|delete\s+my\s+memory|clear\s+my\s+memories)\b",
        r"(?:याद\s+रखना\s+कि|याद\s+रखो)",
    ]

    # 4. Project Creation Patterns (Phase 6 Agent)
    PROJECT_CREATION_PATTERNS = [
        # Explicit creation directives (including with VS Code prefix)
        r"(?i)\b(?:open\s+(?:vs\s*code|vscode|ide)\s*(?:and|aur|,)?\s*)?(?:create|build|make|develop|implement|generate|design|write)\s+(?:a\s+|an\s+|the\s+)?(?:fully\s+functional\s+|complete\s+|simple\s+|interactive\s+|modern\s+|clean\s+ui\s+|good\s+ui\s+)?([a-zA-Z0-9_\-\s]+?)(?:\s+for\s+me|\s+with\s+.*|\s+in\s+[a-zA-Z0-9_\+\#]+|\s+using\s+.*|\s+and\s+run\s+it)?$",
        # Desire-based project requests ("I want an app that...", "I need a project for...")
        r"(?i)\b(?:i\s+want|i\s+need|can\s+you\s+(?:build|make|create)|please\s+(?:build|make|create))\s+(?:a\s+|an\s+|the\s+)?([a-zA-Z0-9_\-\s]+?(?:application|app|project|website|dashboard|tracker|tool|system))\b",
        # Multilingual Hindi / Hinglish project building
        r"(?i)\b(?:mere\s+liye\s+)?(?:ek\s+)?([a-zA-Z0-9_\-\s]+?)\s+(?:app|project|system|tracker|dashboard|tool|website)?\s*(?:banao|bana\s+do|bana\s+ke\s+run\s+karo|bana\s+dijiye|create\s+karo|develop\s+karo)\b",
    ]

    # 5. Project Modification / Debugging / Execution (Phase 5/6)
    PROJECT_ACTION_PATTERNS = [
        (r"(?i)\b(?:open\s+(?:vs\s*code|vscode|ide)\s*,\s*.*(?:find\s+(?:the\s+)?error|fix\s+(?:the\s+)?error|debug))\b", MasterRoute.PROJECT_DEBUGGING),
        (r"(?i)\b(?:open\s+(?:my\s+)?(?:existing\s+)?([a-zA-Z0-9_\-]+)\s+project\s+(?:and|aur)\s+(?:add|modify|update|change|implement))\b", MasterRoute.PROJECT_MODIFICATION),
        (r"(?i)\b(?:modify|update|refactor|enhance)\s+(?:my\s+)?(?:existing\s+)?([a-zA-Z0-9_\-]+)\s*(?:project|app|code)?\b", MasterRoute.PROJECT_MODIFICATION),
        (r"(?i)\b(?:debug|fix\s+(?:the\s+)?errors?|fix\s+the\s+bug|find\s+bugs?)\s+(?:in|of)\s+.*(?:project|program|code|file|vs\s*code)", MasterRoute.PROJECT_DEBUGGING),
        (r"(?i)\b(?:run\s+(?:my\s+)?(?:existing\s+)?([a-zA-Z0-9_\-]+)\s+project|run\s+my\s+project|is\s+project\s+ko\s+run\s+karo)\b", MasterRoute.PROJECT_EXECUTION),
    ]

    # 6. Phase 5 Direct OS / Application / Browser & Communication Tasks
    COMPUTER_TASK_PATTERNS = [
        # Messaging / Communication / Email / Web App Tasks
        (r"(?i)\b(?:send\s+(?:an?\s+)?(?:email|mail)|email\s+.*to\s+.*|mail\s+.*to\s+.*|[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b", MasterRoute.BROWSER_TASK),
        (r"(?i)\b(?:open\s+.*(?:whatsapp|telegram|slack|discord|gmail|email|messages?)|whatsapp\s+(?:web\s+)?(?:kholo|open|chalao)|(?:message|msg|send\s+message)\s+.*to\s+.*|.*ko\s+.*(?:message|bhejo|msg))\b", MasterRoute.BROWSER_TASK),
        (r"(?i)\b(?:whatsapp|telegram|slack|discord|gmail|reddit|twitter|x\.com|github|wikipedia|amazon|flipkart|netflix|spotify|chatgpt|youtube|yt)\b.*(?:kholo|open|chalao|visit|message|send|search|browse|play)", MasterRoute.BROWSER_TASK),
        (r"(?i)\b(?:open|launch|visit|navigate|go\s+to)\s+.*(?:whatsapp|telegram|slack|discord|gmail|reddit|twitter|github|wikipedia|amazon|netflix|spotify|chatgpt|youtube|yt|browser|web)\b", MasterRoute.BROWSER_TASK),
        (r"(?i)\b(?:search\s+(?:for\s+)?.*on\s+(?:google|chrome|browser|bing|youtube|web)|open\s+(?:chrome|browser|edge)\s+(?:and|aur)\s+search(?:\s+for)?\s+.*)\b", MasterRoute.BROWSER_TASK),
        (r"(?i)\b(?:open|launch|visit|navigate|go\s+to)\s+(?:https?://\S+|www\.\S+|[a-zA-Z0-9_\-\.]+\.[a-zA-Z]{2,}(?:/\S*)?)(?:\s+in\s+browser)?\b", MasterRoute.BROWSER_TASK),
        (r"(?i)\b(?:https?://\S+|www\.\S+|[a-zA-Z0-9_\-\.]+\.[a-zA-Z]{2,}(?:/\S*)?)\s+(?:kholo|open|chalao|visit|open\s+karo)\b", MasterRoute.BROWSER_TASK),
        (r"(?i)\b(?:play\s+.*(?:song|music|track)|play\s+.*on\s+youtube|go\s+to\s+youtube\s+and\s+play|youtube\s+(?:pe|par).*chalao|gaana\s+chalao|search\s+youtube\s+for)\b", MasterRoute.BROWSER_TASK),
        (r"(?i)\b(?:take|capture)\s+(?:a\s+)?screenshot|screenshot\s+(?:le\s+lo|kheecho)\b", MasterRoute.SYSTEM_TASK),
        (r"(?i)\b(?:increase|decrease|mute|unmute)\s+volume|volume\s+(?:kam|badhao|mute)\b", MasterRoute.SYSTEM_TASK),
        (r"(?i)\b(?:create|make|delete|remove|rename|move)\s+(?:folder|directory|file)\b", MasterRoute.FILE_OPERATION),
        (r"(?i)\b(?:open|launch|kholo|chalao|show)\s+(?:the\s+)?(?:[a-zA-Z0-9_\-]+\s+)?(?:folder|directory)\b", MasterRoute.FILE_OPERATION),
        (r"(?i)\b(?:open|launch|kholo|chalao)\s+(?:downloads|documents|desktop|pictures|music|videos)\b", MasterRoute.FILE_OPERATION),
        # Dynamic Application / Resource Launch & Control Commands (English & Hindi)
        (r"(?i)\b(?:type\s+.*into\s+[a-zA-Z0-9_\-]+|open\s+[a-zA-Z0-9_\-]+\s+and\s+type)\b", MasterRoute.COMPUTER_TASK),
        (r"(?i)^(?:open|launch|start|run|visit|go\s+to)\s+(?:the\s+|app\s+|my\s+)?([a-zA-Z0-9_\-\.\:\/\s]+)$", MasterRoute.COMPUTER_TASK),
        (r"(?i)^([a-zA-Z0-9_\-\.\:\/\s]+?)(?:\s+ko|\s+app)?\s+(?:kholo|chalao|open\s+karo|chala\s+do|खोलो|चलाओ)(?:\s+|$|[.,!?])", MasterRoute.COMPUTER_TASK),
        (r"(?i)^(?:close|exit|terminate|kill|shut\s+down)\s+(?:the\s+|app\s+)?([a-zA-Z0-9_\-\s]+)", MasterRoute.COMPUTER_TASK),
        (r"(?i)^([a-zA-Z0-9_\-\s]+?)\s+(?:band\s+karo|band\s+kar\s+do|close\s+karo|बंद\s+करो)(?:\s+|$|[.,!?])", MasterRoute.COMPUTER_TASK),
        # Agent confirmation responses (e.g. Yes, No, Proceed, Cancel, Haan)
        (r"(?i)^\s*(?:yes|proceed|confirm|sure|do\s+it|yep|yeah|haan|sahi|ha|ha\s+kar\s+do|no|cancel|stop|abort|don'?t|nope|nahi|nahin|mat\s+karo)\s*$", MasterRoute.COMPUTER_TASK),
    ]

    # 7. Conversational Chat & Greetings
    CHAT_GREETING_PATTERNS = [
        r"(?i)^(?:hi|hello|hey|greetings|namaste|kem\s+cho|wassup|yo|good\s+(?:morning|afternoon|evening))\b",
        r"(?i)^(?:how\s+are\s+you|how's\s+it\s+going|kaise\s+ho|kya\s+haal\s+hai|sab\s+badhiya)\b",
        r"(?i)^(?:thank\s+you|thanks|shukriya|dhanyawad|bye|goodbye|alvida|tata)\b",
    ]

    # 8. Technical Knowledge Inquiries (How-to, explanation, concept definition)
    TECHNICAL_KNOWLEDGE_PATTERNS = [
        r"(?i)^(?:what\s+is|what\s+are|explain|define|meaning\s+of|difference\s+between|why\s+is|how\s+does)\s+(?:a\s+|an\s+|the\s+)?(?:[a-zA-Z0-9_\+\#\-\s]+)\??$",
        r"(?i)^(?:how\s+(?:do\s+i|can\s+i|to)|how\s+would\s+i)\s+(?:write|create|make|build|code|implement|use|setup|install)\b",
        r"(?i)^(?:give\s+me|show\s+me|provide|share)\s+(?:an?\s+)?(?:example|code|snippet|sample|tutorial|syntax)\s+(?:of|for|to)\b",
        r"(?i)\b(?:kya\s+hota\s+hai|kya\s+hai|samjhao|explain\s+karo|kaise\s+kaam\s+karta\s+hai)\b",
        r"(?i)\b(?:why\s+is\s+my\s+.*(?:giving|throwing)\s+(?:this\s+)?error|what\s+causes\s+.*error)\b",
    ]

    @classmethod
    def _normalize_text(cls, text: str) -> str:
        """Corrects common typographical errors in action verbs and application names."""
        from src.utils.text import normalize_typos
        return normalize_typos(text)

    @classmethod
    def classify_request(cls, user_text: str) -> MasterRouteDecision:
        """
        Classifies the incoming user text into a structured MasterRouteDecision.
        """
        raw = user_text.strip()
        normalized_raw = cls._normalize_text(raw)
        lower = normalized_raw.lower()

        # Clean invocation prefix & trailing filler
        clean = re.sub(r"^(?:chitti,?\s*|hey chitti,?\s*|bhai,?\s*|please\s+)", "", normalized_raw, flags=re.IGNORECASE).strip()
        clean = re.sub(r"(?i)\s+(?:for\s+me|in\s+(?:any\s+|my\s+)?browser|browser\s+me(?:in)?|on\s+(?:my\s+)?computer)$", "", clean).strip()
        clean_lower = clean.lower()

        # 1. Check for Explicit Memory Storage / Update Commands
        if any(re.search(pat, clean) for pat in cls.EXPLICIT_MEMORY_PATTERNS):
            return cls._make_decision(
                route=MasterRoute.EXPLICIT_MEMORY,
                confidence=0.98,
                requires_memory=True,
                requires_computer=False,
                requires_phase5=False,
                requires_phase6=False,
                explanation="User explicitly commanding to store, update, or clear long-term memory",
            )

        # 2. Check for Self-Identity / Introduction Query
        if any(re.search(pat, clean) for pat in cls.SELF_IDENTITY_PATTERNS):
            return cls._make_decision(
                route=MasterRoute.SELF_IDENTITY,
                confidence=0.95,
                requires_memory=False,
                requires_computer=False,
                requires_phase5=False,
                requires_phase6=False,
                explanation="User inquiring about Chitti's identity, background, or purpose",
            )

        # 3. Check for Personal / Creator / Relationship Memory Query
        if any(re.search(pat, clean) for pat in cls.PERSONAL_MEMORY_PATTERNS):
            return cls._make_decision(
                route=MasterRoute.PERSONAL_MEMORY,
                confidence=0.96,
                requires_memory=True,
                requires_computer=False,
                requires_phase5=False,
                requires_phase6=False,
                explanation="User querying personal identity, memories, creator, or relationships",
            )

        # 4. Check for Dedicated Project Modification / Debugging / Execution
        for pat, action_route in cls.PROJECT_ACTION_PATTERNS:
            if re.search(pat, clean):
                return cls._make_decision(
                    route=action_route,
                    confidence=0.94,
                    requires_memory=False,
                    requires_computer=True,
                    requires_phase5=True,
                    requires_phase6=True,
                    explanation=f"User requested project lifecycle action: {action_route.value}",
                    metadata={"execution_requested": bool(re.search(r"(?i)\b(?:run|test|execute)\b", clean))},
                )

        # 4B. Check for Email Drafting / Generative Assistance vs Real Email Sending
        is_email_draft = bool(re.search(r"(?i)\b(?:write|draft|compose|suggest)\s+(?:an?\s+)?email\b", clean)) and not bool(re.search(r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})", clean)) and not bool(re.search(r"(?i)\b(?:send|shoot|dispatch|mail\s+this|email\s+this)\b", clean))
        if is_email_draft:
            return cls._make_decision(
                route=MasterRoute.CHAT,
                confidence=0.95,
                requires_memory=False,
                requires_computer=False,
                requires_phase5=False,
                requires_phase6=False,
                explanation="User requested drafting/writing email content (conversational assistance)",
            )

        # 5. Check for Technical / Conversational "How-to" vs Project Creation
        # Distinguish: "How do I create a todo list?" (Technical Knowledge) vs "Create a todo list for me" (Project Creation)
        is_informational = any(re.search(pat, clean) for pat in cls.TECHNICAL_KNOWLEDGE_PATTERNS)
        is_actionable_directive = bool(
            re.search(r"(?i)\b(?:for\s+me|in\s+vs\s*code|and\s+run\s+it|aur\s+run\s+karo|on\s+my\s+computer|mere\s+liye|bana\s+do|banao)\b", clean)
            or re.search(r"(?i)\b(?:interactive\s+ui|good\s+ui|fully\s+functional|run\s+it|open\s+in\s+browser)\b", clean)
        )

        # 6. Check for Project Creation (Phase 6 Agent)
        is_project_signal = (
            any(re.search(pat, clean) for pat in cls.PROJECT_CREATION_PATTERNS)
            or bool(re.search(r"(?i)\b(?:create|build|make|develop|implement)\s+(?:a\s+|an\s+)?(?:[a-zA-Z0-9_\-\s]+?\s+)?(?:app|project|application|tool|system|dashboard|tracker|website|api|game|interface|quiz|calculator|timer)\b", clean))
            or bool(re.search(r"(?i)\b(?:python|cpp|c\+\+|java|javascript|typescript|rust|go|react|html|css|flask|fastapi|express|django|node)\b.*(?:banao|likho|create|write|implement|build|develop|app|project|code|program|script|system|tracker|calculator|dashboard|api|website)", clean))
        )

        if is_project_signal and not (is_informational and not is_actionable_directive):
            # Extract metadata and requirements
            exec_req = bool(re.search(r"(?i)\b(?:and\s+run(?:\s+it)?|aur\s+run\s+karo|run\s+it|run\s+the\s+app|execute)\b", clean))
            ui_req = bool(re.search(r"(?i)\b(?:ui|gui|interface|frontend|browser|web|interactive|visual|dashboard|page|html|react)\b", clean))
            
            # Derive structured requirements
            reqs = []
            if "fully functional" in clean_lower or "complete" in clean_lower:
                reqs.append("fully functional application logic (no placeholders)")
            if ui_req:
                reqs.append("interactive and styled user interface")
            if "database" in clean_lower or "storage" in clean_lower:
                reqs.append("persistent data storage")

            return cls._make_decision(
                route=MasterRoute.PROJECT_CREATION,
                confidence=0.96,
                requires_memory=False,
                requires_computer=True,
                requires_phase5=True,
                requires_phase6=True,
                explanation="User requested creating a new software project (Phase 6 Agent activated)",
                extracted_requirements=reqs,
                metadata={
                    "execution_requested": exec_req,
                    "ui_required": ui_req,
                },
            )

        # 7. Check for Dedicated Computer Control / OS / Browser / Messaging Tasks (Phase 5)
        for pat, comp_route in cls.COMPUTER_TASK_PATTERNS:
            if re.search(pat, clean):
                return cls._make_decision(
                    route=comp_route,
                    confidence=0.95,
                    requires_memory=False,
                    requires_computer=True,
                    requires_phase5=True,
                    requires_phase6=False,
                    explanation=f"Direct computer/OS action detected: {comp_route.value}",
                )

        # 8. Check for Casual Greeting / Small Talk / Name Call
        if not clean or any(re.search(pat, raw) for pat in cls.CHAT_GREETING_PATTERNS) or clean_lower in ("hello", "hi", "hey", "namaste", "sup", "yo", "chitti", "hey chitti"):
            return cls._make_decision(
                route=MasterRoute.CHAT,
                confidence=0.95,
                requires_memory=False,
                requires_computer=False,
                requires_phase5=False,
                requires_phase6=False,
                explanation="Casual greeting or conversational chat",
            )

        # 9. Check for General Knowledge Question ("what is the capital of France", "where is Mount Everest", "how many...")
        if (
            re.search(r"(?i)^(?:what\s+is\s+the\s+capital|where\s+is|when\s+was|how\s+many|which\s+is|why\s+is\s+the\s+sky)\b", clean)
            or re.search(r"(?i)\b(?:capital\s+of|highest\s+mountain|continents|president\s+of|prime\s+minister\s+of)\b", clean)
        ):
            return cls._make_decision(
                route=MasterRoute.GENERAL_KNOWLEDGE,
                confidence=0.92,
                requires_memory=False,
                requires_computer=False,
                requires_phase5=False,
                requires_phase6=False,
                explanation="General world knowledge or factual inquiry",
            )

        # 10. Check for Pure Technical Inquiries (No computer action, pure brain LLM answer)
        if is_informational:
            return cls._make_decision(
                route=MasterRoute.TECHNICAL_KNOWLEDGE,
                confidence=0.92,
                requires_memory=False,
                requires_computer=False,
                requires_phase5=False,
                requires_phase6=False,
                explanation="User asking technical explanation or programming concept",
            )

        # 11. General Knowledge Question Fallback ("what is X", "who is X")
        if re.search(r"(?i)^(?:what\s+is|who\s+is|where\s+is|when\s+was|how\s+many|which\s+is|why\s+is)\b", clean):
            return cls._make_decision(
                route=MasterRoute.GENERAL_KNOWLEDGE,
                confidence=0.90,
                requires_memory=False,
                requires_computer=False,
                requires_phase5=False,
                requires_phase6=False,
                explanation="General world knowledge or factual inquiry",
            )

        # 12. Action Intent Safety Net: If the request contains explicit action verbs/directives
        # targeting applications, web, messaging, or files, ensure it is NOT dropped to UNKNOWN chat!
        action_verb_match = re.search(r"(?i)\b(?:open|launch|kholo|chalao|bhejo|send|message|search|play|create|build|delete|remove|close|run|type|write|likho|banao)\b", clean)
        if action_verb_match and not re.search(r"(?i)^(?:what|who|where|when|why|how)\b", clean):
            if any(w in clean_lower for w in ["web", "browser", "chrome", "edge", "youtube", "whatsapp", "message", "search", "google", "site", "url"]):
                return cls._make_decision(
                    route=MasterRoute.BROWSER_TASK,
                    confidence=0.88,
                    requires_memory=False,
                    requires_computer=True,
                    requires_phase5=True,
                    requires_phase6=False,
                    explanation="Action intent detected targeting browser or communication subsystem",
                )
            return cls._make_decision(
                route=MasterRoute.COMPUTER_TASK,
                confidence=0.85,
                requires_memory=False,
                requires_computer=True,
                requires_phase5=True,
                requires_phase6=False,
                explanation="Action intent detected targeting computer control subsystem",
            )

        # 13. Dynamic Target / Web / App Resolution for single-token or shorthand requests (e.g. "gemmini", "whatsapp", "yt", "calc")
        try:
            from src.agent.registry import ResourceDiscovery, AppDiscovery
            if ResourceDiscovery.is_url(clean) or ResourceDiscovery.is_url(clean_lower):
                return cls._make_decision(
                    route=MasterRoute.BROWSER_TASK,
                    confidence=0.90,
                    requires_memory=False,
                    requires_computer=True,
                    requires_phase5=True,
                    requires_phase6=False,
                    explanation=f"Direct web resource target detected: {clean}",
                )
            if not re.search(r"\b(?:what|who|where|why|how|explain|define|tell|mean|is|are)\b", clean_lower):
                resolved_app = AppDiscovery.resolve_application(clean)
                if resolved_app:
                    return cls._make_decision(
                        route=MasterRoute.COMPUTER_TASK,
                        confidence=0.88,
                        requires_memory=False,
                        requires_computer=True,
                        requires_phase5=True,
                        requires_phase6=False,
                        explanation=f"Direct application target detected: {clean}",
                    )
        except Exception:
            pass

        # Default fallback to Conversational Brain
        return cls._make_decision(
            route=MasterRoute.UNKNOWN,
            confidence=0.70,
            requires_memory=False,
            requires_computer=False,
            requires_phase5=False,
            requires_phase6=False,
            explanation="General request routed to conversational LLM brain",
        )

    @classmethod
    def _make_decision(
        cls,
        route: MasterRoute,
        confidence: float,
        requires_memory: bool,
        requires_computer: bool,
        requires_phase5: bool,
        requires_phase6: bool,
        explanation: str = "",
        extracted_requirements: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MasterRouteDecision:
        decision = MasterRouteDecision(
            route=route,
            confidence=confidence,
            requires_memory=requires_memory,
            requires_computer=requires_computer,
            requires_phase5=requires_phase5,
            requires_phase6=requires_phase6,
            explanation=explanation,
            extracted_requirements=extracted_requirements or [],
            metadata=metadata or {},
        )
        cls.log_decision(decision)
        return decision

    @classmethod
    def log_decision(cls, decision: MasterRouteDecision) -> None:
        """Emits structured logging according to the required specification."""
        log_chitti(f"[CHITTI] [MASTER ROUTER] Route: {decision.route.value}")
        log_chitti(f"[CHITTI] [MASTER ROUTER] Confidence: {decision.confidence:.2f}")
        
        mem_status = "ENABLED" if decision.requires_memory else "SKIPPED"
        log_chitti(f"[CHITTI] [MASTER ROUTER] Memory retrieval: {mem_status}")

        if decision.requires_phase6:
            log_chitti("[CHITTI] [MASTER ROUTER] Phase 6: ACTIVATED")
        elif decision.requires_phase5 or decision.requires_computer:
            log_chitti("[CHITTI] [MASTER ROUTER] Phase 5: ACTIVATED")
        else:
            log_chitti("[CHITTI] [MASTER ROUTER] Computer agent: DISABLED")
