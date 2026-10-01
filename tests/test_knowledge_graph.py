"""
Comprehensive Test Suite for Phase 3: Dynamic Knowledge Graph & Proactive Memory.
Tests:
- KnowledgeGraph (Entity CRUD, Directed Relationships, Traversal, Subgraphs, Extraction)
- ProactiveScheduler (Scheduled Tasks, Recurring Triggers, Natural Language Parsing, Listeners)
- ContextSynthesizer (Multi-dimensional prompt injection)
- Integration with MasterRouter and Personal Memory
"""

import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.brain.knowledge_graph import Entity, KnowledgeGraph, Relationship
from src.brain.proactive_scheduler import ProactiveScheduler, ScheduledTask, TriggerType
from src.brain.context_synthesizer import ContextSynthesizer
from src.memory.database import MemoryDatabase, MemoryRecord
from src.router.master_router import MasterRouter, MasterRoute


class TestKnowledgeGraphPhase3:
    """Test suite for Phase 3 Knowledge Graph and Proactive Memory components."""

    # =========================================================================
    # 1. KNOWLEDGE GRAPH ENTITY & RELATIONSHIP TESTS
    # =========================================================================

    def test_entity_creation_and_retrieval(self, tmp_path):
        """Verify adding, updating, and retrieving graph entities."""
        db_file = tmp_path / "kg_test.db"
        kg = KnowledgeGraph(db_path=str(db_file))

        # Add entity
        e1 = kg.add_or_update_entity("Rohit", entity_type="PERSON", attributes={"role": "Project Lead"})
        assert e1.id is not None
        assert e1.name == "Rohit"
        assert e1.entity_type == "PERSON"
        assert e1.attributes["role"] == "Project Lead"

        # Update entity
        e1_updated = kg.add_or_update_entity("rohit", entity_type="PERSON", attributes={"email": "rohit@company.com"})
        assert e1_updated.id == e1.id
        assert e1_updated.attributes["role"] == "Project Lead"
        assert e1_updated.attributes["email"] == "rohit@company.com"

        # Retrieve
        fetched = kg.get_entity("Rohit")
        assert fetched is not None
        assert fetched.name == "Rohit"
        assert fetched.attributes["email"] == "rohit@company.com"

    def test_relationship_creation_and_traversal(self, tmp_path):
        """Verify creating relationships and traversing connected nodes."""
        db_file = tmp_path / "kg_rel_test.db"
        kg = KnowledgeGraph(db_path=str(db_file))

        # Create entities & edges
        kg.add_or_update_entity("Rohit", entity_type="PERSON")
        kg.add_or_update_entity("Chitti", entity_type="PROJECT")
        kg.add_or_update_entity("Python", entity_type="TOOL")

        r1 = kg.add_relationship("Rohit", "Chitti", relation_type="LEADS")
        r2 = kg.add_relationship("Chitti", "Python", relation_type="USES")

        assert r1.source_name == "Rohit"
        assert r1.target_name == "Chitti"
        assert r1.relation_type == "LEADS"

        # Query outgoing from Rohit
        rohit_rels = kg.get_relationships("Rohit", direction="outgoing")
        assert len(rohit_rels) == 1
        assert rohit_rels[0].target_name == "Chitti"

        # Query incoming to Python
        py_rels = kg.get_relationships("Python", direction="incoming")
        assert len(py_rels) == 1
        assert py_rels[0].source_name == "Chitti"

        # Query related entities
        related_to_chitti = kg.get_related_entities("Chitti")
        assert len(related_to_chitti) == 2
        names = [ent.name for ent, _ in related_to_chitti]
        assert "Rohit" in names
        assert "Python" in names

    def test_subgraph_extraction(self, tmp_path):
        """Verify extracting subgraphs up to depth N."""
        db_file = tmp_path / "kg_sub_test.db"
        kg = KnowledgeGraph(db_path=str(db_file))

        kg.add_relationship("Alice", "ProjectA", relation_type="WORKS_ON")
        kg.add_relationship("ProjectA", "DatabaseX", relation_type="DEPENDS_ON")

        subgraph = kg.find_subgraph("Alice", max_depth=2)
        assert len(subgraph["nodes"]) >= 3
        assert len(subgraph["edges"]) >= 2
        node_names = [n["name"] for n in subgraph["nodes"]]
        assert "Alice" in node_names
        assert "ProjectA" in node_names
        assert "DatabaseX" in node_names

    def test_dialogue_knowledge_extraction(self, tmp_path):
        """Verify parsing natural language statements into graph entities and relationships."""
        db_file = tmp_path / "kg_extract_test.db"
        kg = KnowledgeGraph(db_path=str(db_file))

        statement = "Rohit is my project lead and his email is rohit@company.com"
        rels = kg.extract_and_store_from_text(statement)
        assert len(rels) >= 2

        rohit = kg.get_entity("Rohit")
        assert rohit is not None
        assert rohit.attributes.get("email") == "rohit@company.com"
        assert rohit.attributes.get("role") == "project lead"

    def test_format_graph_context_for_llm(self, tmp_path):
        """Verify generating structured graph context prompt block."""
        db_file = tmp_path / "kg_prompt_test.db"
        kg = KnowledgeGraph(db_path=str(db_file))

        kg.add_or_update_entity("Ayush", entity_type="PERSON", attributes={"role": "Teammate"})
        kg.add_relationship("Ayush", "Frontend", relation_type="WORKS_ON")

        prompt_block = kg.format_graph_context(["Ayush"])
        assert "KNOWLEDGE GRAPH RELATIONAL GROUND TRUTH" in prompt_block
        assert "Entity: Ayush" in prompt_block
        assert "role: Teammate" in prompt_block
        assert "WORKS_ON -> Frontend" in prompt_block

    def test_entity_deletion_cascades(self, tmp_path):
        """Verify deleting an entity removes connected edges."""
        db_file = tmp_path / "kg_del_test.db"
        kg = KnowledgeGraph(db_path=str(db_file))

        kg.add_relationship("X", "Y", relation_type="CONNECTS")
        assert len(kg.get_relationships("X")) == 1

        deleted = kg.delete_entity("X")
        assert deleted is True
        assert kg.get_entity("X") is None
        assert len(kg.get_relationships("Y")) == 0

    # =========================================================================
    # 2. PROACTIVE SCHEDULER & AMBIENT TRIGGER TESTS
    # =========================================================================

    def test_proactive_scheduler_reminder_crud(self, tmp_path):
        """Verify scheduling, listing, and canceling reminders."""
        db_file = tmp_path / "sched_test.db"
        scheduler = ProactiveScheduler(db_path=str(db_file), poll_interval_sec=0.1)

        task = scheduler.schedule_reminder(
            title="Git Commit",
            message="Commit working changes to origin/main",
            due_in_seconds=600.0,
        )
        assert task.id is not None
        assert task.title == "Git Commit"
        assert task.is_active is True

        active = scheduler.list_active_tasks()
        assert len(active) == 1
        assert active[0].id == task.id

        canceled = scheduler.cancel_task(task.id)
        assert canceled is True
        assert len(scheduler.list_active_tasks()) == 0

    def test_proactive_scheduler_natural_language_parser(self):
        """Verify parsing natural language reminder expressions."""
        p1 = ProactiveScheduler.parse_natural_language_reminder("remind me in 10 minutes to drink water")
        assert p1 is not None
        assert p1["due_in_seconds"] == 600.0
        assert "drink water" in p1["message"].lower()

        p2 = ProactiveScheduler.parse_natural_language_reminder("remind me in 2 hours to call Rohit")
        assert p2 is not None
        assert p2["due_in_seconds"] == 7200.0

        p3 = ProactiveScheduler.parse_natural_language_reminder("remind me to check the server")
        assert p3 is not None
        assert "check the server" in p3["message"].lower()

    def test_proactive_scheduler_trigger_execution(self, tmp_path):
        """Verify background worker loop triggers due tasks and notifies listeners."""
        db_file = tmp_path / "sched_trigger_test.db"
        scheduler = ProactiveScheduler(db_path=str(db_file), poll_interval_sec=0.05)

        triggered_tasks = []
        scheduler.register_listener(lambda t: triggered_tasks.append(t))

        # Schedule reminder due in 0.1 seconds
        scheduler.schedule_reminder(
            title="Quick Alert",
            message="Test prompt alert",
            due_in_seconds=0.1,
        )

        scheduler.start()
        time.sleep(0.3)
        scheduler.stop()

        assert len(triggered_tasks) >= 1
        assert triggered_tasks[0].title == "Quick Alert"

    # =========================================================================
    # 3. UNIFIED CONTEXT SYNTHESIZER TESTS
    # =========================================================================

    def test_context_synthesizer_unified_context(self, tmp_path):
        """Verify merging factual memories, graph relations, and upcoming reminders."""
        db_mem = tmp_path / "mem.db"
        db_kg = tmp_path / "kg.db"
        db_sched = tmp_path / "sched.db"

        mem_db = MemoryDatabase(db_path=str(db_mem))
        kg = KnowledgeGraph(db_path=str(db_kg))
        sched = ProactiveScheduler(db_path=str(db_sched))

        # Populate
        rec = MemoryRecord(content="User prefers Python over C++", memory_type="preference")
        mem_db.insert_memory(rec)

        kg.add_or_update_entity("Rohit", entity_type="PERSON", attributes={"email": "rohit@company.com"})
        kg.add_relationship("Rohit", "Chitti", relation_type="LEADS")

        sched.schedule_reminder("Sync Meeting", "Weekly team sync", due_in_seconds=300)

        synthesizer = ContextSynthesizer(memory_db=mem_db, knowledge_graph=kg, scheduler=sched)
        full_context = synthesizer.synthesize_context(
            user_query="Tell me about Rohit and my project",
            memories=[rec],
            include_reminders=True,
        )

        assert "STORED LONG-TERM MEMORIES" in full_context
        assert "User prefers Python over C++" in full_context
        assert "KNOWLEDGE GRAPH RELATIONAL GROUND TRUTH" in full_context
        assert "Entity: Rohit" in full_context
        assert "LEADS -> Chitti" in full_context
        assert "ACTIVE PENDING REMINDERS" in full_context
        assert "Sync Meeting" in full_context

    # =========================================================================
    # 4. MASTER ROUTER PHASE 3 INTEGRATION
    # =========================================================================

    def test_master_router_reminders_and_graph_queries(self):
        """Verify proactive reminders and entity memory routing in MasterRouter."""
        prompts = [
            "remind me in 10 minutes to take a break",
            "remind me at 6 PM to commit my code",
            "list my reminders",
            "cancel reminder 2",
            "remember that Rohit is my project lead",
        ]
        for p in prompts:
            res = MasterRouter.classify_request(p)
            assert res.requires_memory is True
            assert res.route in (MasterRoute.EXPLICIT_MEMORY, MasterRoute.PERSONAL_MEMORY)
