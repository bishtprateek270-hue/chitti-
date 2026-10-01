"""
Chitti Memory & Context Synthesizer (Phase 3).
Combines long-term factual memories, Knowledge Graph entity relationships,
and proactive upcoming scheduled reminders into a unified prompt context.
"""

from typing import Any, Dict, List, Optional, Set
from src.brain.knowledge_graph import KnowledgeGraph
from src.brain.proactive_scheduler import ProactiveScheduler
from src.memory.database import MemoryDatabase, MemoryRecord


class ContextSynthesizer:
    """
    Synthesizes multi-dimensional memory context:
    1. Direct semantic / keyword factual memories
    2. Knowledge Graph relational entities & connections
    3. Active upcoming proactive reminders
    """

    def __init__(
        self,
        memory_db: Optional[MemoryDatabase] = None,
        knowledge_graph: Optional[KnowledgeGraph] = None,
        scheduler: Optional[ProactiveScheduler] = None,
    ):
        self.memory_db = memory_db or MemoryDatabase()
        self.kg = knowledge_graph or KnowledgeGraph()
        self.scheduler = scheduler or ProactiveScheduler()

    def extract_entity_names_from_text(self, text: str) -> List[str]:
        """Finds known entities mentioned in user input."""
        import re
        entities = self.kg.list_entities()
        mentioned = []
        clean_text = text.lower()
        for e in entities:
            # Check for word boundary match
            if re.search(rf"\b{re.escape(e.name.lower())}\b", clean_text):
                mentioned.append(e.name)
        return mentioned

    def synthesize_context(
        self,
        user_query: str,
        memories: Optional[List[MemoryRecord]] = None,
        include_reminders: bool = True,
    ) -> str:
        """
        Builds the complete memory + knowledge graph + scheduler context block.
        """
        blocks = []

        # 1. Stored Factual Memories
        mem_records = memories if memories is not None else []
        if mem_records:
            lines = ["\n[STORED LONG-TERM MEMORIES - FACTUAL GROUND TRUTH]:"]
            for m in mem_records:
                content = m.content if hasattr(m, "content") else str(m)
                lines.append(f"- {content}")
            blocks.append("\n".join(lines))

        # 2. Knowledge Graph Context
        mentioned_entities = self.extract_entity_names_from_text(user_query)
        kg_context = self.kg.format_graph_context(mentioned_entities)
        if kg_context:
            blocks.append(kg_context)

        # 3. Active Upcoming Reminders
        if include_reminders:
            active_tasks = self.scheduler.list_active_tasks()
            if active_tasks:
                lines = ["\n[ACTIVE PENDING REMINDERS]:"]
                for t in active_tasks[:5]:
                    due_in = int(t.time_until_due_sec)
                    lines.append(f"- #{t.id} '{t.title}': {t.message} (due in {due_in}s)")
                blocks.append("\n".join(lines))

        return "\n\n".join(blocks)
