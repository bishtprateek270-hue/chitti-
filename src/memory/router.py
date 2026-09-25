"""
Chitti Memory Decision Router Module.
Provides intent-first classification for memory storage and retrieval routing.
Ensures personal, relationship, project, preference, and explicit memories are persisted
and retrieved while general-knowledge and technical queries skip unnecessary memory operations.
"""

import re
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional

from src.utils.logging import log_debug, log_chitti


class MemoryIntent(str, Enum):
    PERSONAL_MEMORY = "PERSONAL_MEMORY"
    RELATIONSHIP_MEMORY = "RELATIONSHIP_MEMORY"
    PROJECT_MEMORY = "PROJECT_MEMORY"
    PREFERENCE_MEMORY = "PREFERENCE_MEMORY"
    EDUCATION_MEMORY = "EDUCATION_MEMORY"
    EXPLICIT_MEMORY = "EXPLICIT_MEMORY"
    GENERAL_KNOWLEDGE = "GENERAL_KNOWLEDGE"
    TECHNICAL_QUERY = "TECHNICAL_QUERY"
    TASK_REQUEST = "TASK_REQUEST"
    CASUAL_CONVERSATION = "CASUAL_CONVERSATION"
    UNKNOWN = "UNKNOWN"


@dataclass
class MemoryRouteDecision:
    intent: MemoryIntent
    should_retrieve_memory: bool
    should_store_memory: bool
    is_explicit_memory: bool = False
    target_categories: List[str] = field(default_factory=list)
    explanation: str = ""


# Explicit memory command patterns
EXPLICIT_REMEMBER_PATTERNS = [
    r"(?i)\b(?:r[e]?m[e]?mb[e]?r|rmbr|yaad\s+r[a]?khna|yaad\s+r[a]?kho|yaad\s+r[a]?kh|isse\s+yaad\s+rakhna|ye\s+yaad\s+rakhna)\b",
    r"(?i)\b(?:don'?t\s+forget|keep\s+(?:this\s+)?in\s+memory|save\s+this\s+(?:in\s+memory)?|note\s+down|bhoolna\s+mat)\b",
    r"(?:याद\s+रखना|याद\s+रखो|भूलना\s+मत)",
]

# Explicit memory inspection patterns
EXPLICIT_RECALL_PATTERNS = [
    r"(?i)\b(?:what\s+do\s+you\s+remember|show\s+me\s+what\s+you\s+remember|list\s+(?:my\s+)?memories|tell\s+me\s+what\s+you\s+remember)\b",
    r"(?i)\b(?:what\s+do\s+you\s+know\s+about\s+me|tell\s+me\s+about\s+me|meri\s+memories|kya\s+yaad\s+hai)\b",
    r"(?:तुम्हें\s+मेरे\s+बारे\s+में\s+क्या\s+याद\s+है|मेरी\s+यादें\s+बताओ)",
]

# Technical terms and general technical query indicators
TECHNICAL_KEYWORDS = [
    "deep learning", "machine learning", "neural network", "cnn", "rnn", "lstm", "transformer", "bert", "gpt",
    "backpropagation", "gradient descent", "loss function", "optimizer", "svm", "random forest", "xgboost",
    "dbms", "sql", "normalization", "nosql", "mongodb", "postgresql", "index", "acid properties",
    "operating system", "process", "thread", "deadlock", "paging", "virtual memory", "semaphores",
    "kubernetes", "docker", "container", "devops", "ci/cd", "microservices", "api gateway",
    "c++", "python", "java", "rust", "golang", "javascript", "typescript", "html", "css",
    "leetcode", "data structure", "binary tree", "graph", "dynamic programming", "sorting", "dijkstra",
    "algorithm", "pointer", "recursion", "array", "linked list", "hash table",
]

# Pure technical command verbs
TECHNICAL_VERBS = [
    r"^(?:explain|describe|what\s+is|what\s+are|how\s+does|how\s+do|how\s+to|write\s+a|solve|implement|code|generate|debug)\b",
    r"^(?:samjhao|batao|kya\s+hota\s+hai|kaise\s+kaam\s+karta\s+hai)\b",
]

# Personal / Identity / Relationship / Life query patterns (triggers memory retrieval)
PERSONAL_QUERY_PATTERNS = [
    r"(?i)\b(?:who\s+am\s+i|what\s+is\s+my\s+name|what'?s\s+my\s+name|tell\s+me\s+my\s+name|do\s+you\s+know\s+my\s+name)\b",
    r"(?i)\b(?:who\s+(?:created|made|built|developed)\s+(?:you|u)|who\s+is\s+your\s+creator|who\s+developed\s+you)\b",
    r"(?i)\b(?:what\s+is\s+my\s+(?:job|profession|work|occupation|college|university|branch|degree|goal|hobby|favorite\s+\w+))\b",
    r"(?i)\b(?:where\s+do\s+i\s+(?:study|work|live)|which\s+college\s+(?:do\s+i|did\s+i))\b",
    r"(?i)\b(?:who\s+is\s+(?:my\s+)?(?:best\s+friend|friend|sister|brother|father|mother|teammate|coworker|girlfriend|boyfriend|wife|husband))\b",
    r"(?i)\b(?:who\s+is\s+[A-Z][a-z]+)\b",  # e.g., "Who is Rahul?", "Who is Ananya?"
    r"(?i)\b(?:what\s+(?:project|projects|app)\s+(?:am\s+i|do\s+i|are\s+we)\s+(?:building|working\s+on|making))\b",
    r"(?i)\b(?:which\s+(?:\w+\s+)?project\s+am\s+i\s+(?:building|working\s+on|making))\b",
    r"(?i)\b(?:tell\s+me\s+about\s+my\s+(?:college|university|friends|family|projects|work|sister|brother|best\s+friend|hobbies))\b",
    r"(?i)\b(?:mera\s+naam\s+kya\s+hai|tujhe\s+kisne\s+banaya|main\s+kaun\s+hoon|mera\s+college\s+kaunsa\s+hai|mera\s+best\s+friend\s+kaun\s+hai)\b",
    r"(?:मेरा\s+नाम\s+क्या\s+है|तुम्हें\s+किसने\s+बनाया|मेरा\s+कॉलेज\s+कौन\s+सा\s+है|मेरा\s+दोस्त\s+कौन\s+है)",
]

# Personal disclosure patterns (triggers memory storage)
PERSONAL_DISCLOSURE_PATTERNS = [
    # Name & Identity
    r"(?i)\b(?:my\s+name\s+is|i\s+am|i'm)\s+([A-Za-z\s]+?)(?:\s+(?:and|who|i|also)|$)",
    r"(?i)\b(?:mera\s+naam|main\s+[A-Za-z\s]+?\s+hoon)\b",
    # Creator
    r"(?i)\b(?:i\s+(?:have\s+)?(?:created|create|made|built|developed)\s+you|i\s+am\s+your\s+creator|maine\s+tumhe\s+banaya)\b",
    # College / Education
    r"(?i)\b(?:my\s+college\s+is|i\s+study\s+at|i\s+am\s+studying\s+at|i\s+go\s+to\s+college\s+at|my\s+university\s+is|i'm\s+in\s+(?:first|second|third|fourth|\d(?:st|nd|rd|th))\s+year|my\s+branch\s+is|i'm\s+studying\s+[A-Za-z0-9_\-\s]+)\b",
    r"(?i)\b(?:mera\s+college|main\s+padhta\s+hoon|meri\s+university)\b",
    # Relationships
    r"(?i)\b(?:my\s+(?:best\s+friend|friend|sister|brother|father|mother|mom|dad|teammate|coworker|partner|wife|husband)(?:'s|\s+name)*\s+(?:is|=))\b",
    r"(?i)\b([A-Z][a-z]+)\s+is\s+my\s+(?:best\s+friend|friend|sister|brother|father|mother|teammate|coworker)\b",
    r"(?i)\b([A-Z][a-z]+)\s+works\s+with\s+me\b",
    r"(?i)\b(?:mera\s+best\s+friend|meri\s+behen|mera\s+bhai|mera\s+dost)\b",
    # Projects
    r"(?i)\b(?:i'm\s+building|i\s+am\s+building|i'm\s+working\s+on|my\s+project\s+is|i\s+am\s+developing)\s+([A-Za-z0-9_\-\s]+)\b",
    # Profession / Occupation
    r"(?i)\b(?:i\s+am\s+an?|i'm\s+an?|i\s+work\s+as\s+an?)\s+([A-Za-z0-9_\-\s]+?(?:engineer|developer|scientist|student|designer|doctor|researcher))\b",
    # Goals / Preferences
    r"(?i)\b(?:my\s+goal\s+is|my\s+favorite\s+\w+\s+is|my\s+favourite\s+\w+\s+is|i\s+(?:really\s+)?(?:love|like|prefer|enjoy|hate)|i\s+live\s+(?:with|in))\b",
]


class MemoryRouter:
    """Classifies user inputs to make intelligent memory routing decisions."""

    @classmethod
    def classify_intent(cls, user_text: str) -> MemoryRouteDecision:
        raw = user_text.strip()
        lower = raw.lower()

        # 1. Check for Explicit Memory Commands (Highest Precedence)
        if any(re.search(pat, raw) for pat in EXPLICIT_REMEMBER_PATTERNS):
            log_debug(f"[MEMORY ROUTER] Explicit memory directive detected: '{raw}'")
            return MemoryRouteDecision(
                intent=MemoryIntent.EXPLICIT_MEMORY,
                should_retrieve_memory=False,
                should_store_memory=True,
                is_explicit_memory=True,
                target_categories=["explicit", "fact", "preference", "relationship", "project"],
                explanation="Explicit memory storage command detected."
            )

        # 2. Check for Explicit Memory Recall Inspection
        if any(re.search(pat, raw) for pat in EXPLICIT_RECALL_PATTERNS):
            return MemoryRouteDecision(
                intent=MemoryIntent.PERSONAL_MEMORY,
                should_retrieve_memory=True,
                should_store_memory=False,
                target_categories=["identity", "relationship", "education", "project", "preference", "fact"],
                explanation="User asked for memory inspection / overview."
            )

        # 3. Check for Personal / Identity / Relationship / Education / Project Queries
        is_personal_query = any(re.search(pat, raw) for pat in PERSONAL_QUERY_PATTERNS)
        if is_personal_query:
            # Determine specific sub-intent
            if any(k in lower for k in ["creator", "create you", "built you", "made you", "kisne banaya"]):
                return MemoryRouteDecision(
                    intent=MemoryIntent.RELATIONSHIP_MEMORY,
                    should_retrieve_memory=True,
                    should_store_memory=False,
                    target_categories=["relationship", "identity"],
                    explanation="Query about creator identity."
                )
            if any(k in lower for k in ["my name", "who am i", "mera naam", "mera nam"]):
                return MemoryRouteDecision(
                    intent=MemoryIntent.PERSONAL_MEMORY,
                    should_retrieve_memory=True,
                    should_store_memory=False,
                    target_categories=["identity"],
                    explanation="Query about user name."
                )
            if any(k in lower for k in ["college", "university", "study", "branch", "degree", "padhta"]):
                return MemoryRouteDecision(
                    intent=MemoryIntent.EDUCATION_MEMORY,
                    should_retrieve_memory=True,
                    should_store_memory=False,
                    target_categories=["education", "personal"],
                    explanation="Query about user college/education."
                )
            if any(k in lower for k in ["project", "building", "app", "chitti"]):
                return MemoryRouteDecision(
                    intent=MemoryIntent.PROJECT_MEMORY,
                    should_retrieve_memory=True,
                    should_store_memory=False,
                    target_categories=["project"],
                    explanation="Query about user project."
                )
            if any(k in lower for k in ["friend", "sister", "brother", "father", "mother", "teammate", "rahul", "ananya", "harsh", "rishabh"]):
                return MemoryRouteDecision(
                    intent=MemoryIntent.RELATIONSHIP_MEMORY,
                    should_retrieve_memory=True,
                    should_store_memory=False,
                    target_categories=["relationship"],
                    explanation="Query about user relationships or acquaintances."
                )

            return MemoryRouteDecision(
                intent=MemoryIntent.PERSONAL_MEMORY,
                should_retrieve_memory=True,
                should_store_memory=False,
                target_categories=["identity", "relationship", "education", "project", "preference", "fact"],
                explanation="General personal query."
            )

        # 4. Check for Personal Disclosures (Statements to store into long-term memory)
        is_personal_disclosure = False
        disclosure_category = "personal"

        for pat in PERSONAL_DISCLOSURE_PATTERNS:
            if re.search(pat, raw):
                is_personal_disclosure = True
                break

        if is_personal_disclosure:
            if any(k in lower for k in ["college", "university", "studying", "study at", "branch", "year"]):
                return MemoryRouteDecision(
                    intent=MemoryIntent.EDUCATION_MEMORY,
                    should_retrieve_memory=False,
                    should_store_memory=True,
                    target_categories=["education"],
                    explanation="User disclosed education / college details."
                )
            if any(k in lower for k in ["friend", "sister", "brother", "father", "mother", "mom", "dad", "teammate", "coworker", "created you", "create you", "banaya"]):
                return MemoryRouteDecision(
                    intent=MemoryIntent.RELATIONSHIP_MEMORY,
                    should_retrieve_memory=False,
                    should_store_memory=True,
                    target_categories=["relationship"],
                    explanation="User disclosed relationship / family / friend details."
                )
            if any(k in lower for k in ["building", "working on", "developing", "project"]):
                return MemoryRouteDecision(
                    intent=MemoryIntent.PROJECT_MEMORY,
                    should_retrieve_memory=False,
                    should_store_memory=True,
                    target_categories=["project"],
                    explanation="User disclosed project information."
                )
            if any(k in lower for k in ["favorite", "favourite", "prefer", "love to", "goal"]):
                return MemoryRouteDecision(
                    intent=MemoryIntent.PREFERENCE_MEMORY,
                    should_retrieve_memory=False,
                    should_store_memory=True,
                    target_categories=["preference"],
                    explanation="User disclosed personal preference or goal."
                )
            return MemoryRouteDecision(
                intent=MemoryIntent.PERSONAL_MEMORY,
                should_retrieve_memory=False,
                should_store_memory=True,
                target_categories=["identity", "personal"],
                explanation="User disclosed personal identity or background."
            )

        # 5. Check for General Knowledge / Technical Queries (Explicitly Gate Retrieval)
        # Check if query contains technical verbs or technical topics
        has_tech_keyword = any(k in lower for k in TECHNICAL_KEYWORDS)
        has_tech_verb = any(re.search(pat, lower) for pat in TECHNICAL_VERBS)

        # Notice: Check if the user explicitly tied the technical query to their personal self:
        # e.g., "What deep-learning project am I working on?" or "Which CNN project am I building?"
        is_self_referential_tech = bool(re.search(r"(?i)\b(?:am\s+i|my|mine|i'm\s+working|i\s+work|my\s+project)\b", raw))

        if has_tech_keyword or has_tech_verb or any(k in lower for k in ["solve", "program", "code", "leetcode", "write a", "explain"]):
            if is_self_referential_tech:
                return MemoryRouteDecision(
                    intent=MemoryIntent.PROJECT_MEMORY,
                    should_retrieve_memory=True,
                    should_store_memory=False,
                    target_categories=["project", "preference"],
                    explanation="Technical query explicitly connected to personal user context."
                )
            else:
                return MemoryRouteDecision(
                    intent=MemoryIntent.TECHNICAL_QUERY,
                    should_retrieve_memory=False,
                    should_store_memory=False,
                    explanation="Pure technical query or general knowledge. Long-term memory retrieval skipped."
                )

        # 6. Check for Task Requests
        if any(re.search(pat, lower) for pat in [r"^(?:summarize|translate|calculate|convert|format|generate)\b"]):
            return MemoryRouteDecision(
                intent=MemoryIntent.TASK_REQUEST,
                should_retrieve_memory=False,
                should_store_memory=False,
                explanation="Task execution request. Memory retrieval skipped."
            )

        # 7. Check for Casual Chit-Chat
        casual_greetings = ["hello", "hi", "hey", "namaste", "good morning", "good afternoon", "good evening", "how are you", "who are you", "bye", "good night", "thanks", "thank you"]
        if lower.rstrip(".!? \t\n") in casual_greetings or len(raw.split()) <= 2:
            return MemoryRouteDecision(
                intent=MemoryIntent.CASUAL_CONVERSATION,
                should_retrieve_memory=False,
                should_store_memory=False,
                explanation="Casual conversation. Memory retrieval skipped."
            )

        # Default fallback
        return MemoryRouteDecision(
            intent=MemoryIntent.UNKNOWN,
            should_retrieve_memory=False,
            should_store_memory=False,
            explanation="Unclassified query. Long-term memory retrieval not required."
        )
