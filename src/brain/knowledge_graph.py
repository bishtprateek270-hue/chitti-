"""
Chitti Dynamic Knowledge Graph Engine (Phase 3).
Provides persistent entity-relationship network storage, semantic traversal,
relationship inference, and automated knowledge extraction from dialogue.
"""

import json
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from src.brain.llm import BaseLLM
from src.utils.logging import log_chitti, log_debug, log_info, log_warn, log_error


def get_utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Entity:
    """Represents a node in the Knowledge Graph."""
    id: Optional[int] = None
    name: str = ""
    entity_type: str = "CONCEPT"  # PERSON, PROJECT, WORKSPACE, PREFERENCE, ORGANIZATION, TOOL, EVENT, CONCEPT
    attributes: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    created_at: str = field(default_factory=get_utc_now_iso)
    updated_at: str = field(default_factory=get_utc_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "entity_type": self.entity_type,
            "attributes": self.attributes,
            "confidence": self.confidence,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class Relationship:
    """Represents a directed edge between two entities in the Knowledge Graph."""
    id: Optional[int] = None
    source_name: str = ""
    target_name: str = ""
    relation_type: str = "RELATED_TO"  # CREATED_BY, LEADS, WORKS_ON, EMAIL_IS, PREFERS, LOCATED_IN, FRIEND_OF, USES, DEPENDS_ON, HAS_ROLE
    weight: float = 1.0
    attributes: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=get_utc_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source_name,
            "target": self.target_name,
            "relation": self.relation_type,
            "weight": self.weight,
            "attributes": self.attributes,
            "created_at": self.created_at,
        }


class KnowledgeGraph:
    """
    Persistent SQLite-backed Knowledge Graph for entity-relationship intelligence.
    """

    def __init__(self, db_path: str = "data/memory/knowledge_graph.db"):
        self.db_path = Path(db_path)
        if not self.db_path.is_absolute():
            root_dir = Path(__file__).resolve().parent.parent.parent
            self.db_path = root_dir / self.db_path

        self._lock = threading.Lock()
        self._ensure_dir()
        self._init_tables()

    def _ensure_dir(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_tables(self):
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS kg_entities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL COLLATE NOCASE,
                    entity_type TEXT NOT NULL,
                    attributes TEXT DEFAULT '{}',
                    confidence REAL DEFAULT 1.0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS kg_relationships (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_name TEXT NOT NULL COLLATE NOCASE,
                    target_name TEXT NOT NULL COLLATE NOCASE,
                    relation_type TEXT NOT NULL COLLATE NOCASE,
                    weight REAL DEFAULT 1.0,
                    attributes TEXT DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    UNIQUE(source_name, target_name, relation_type)
                );
            """)

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_kg_entities_name ON kg_entities(name);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_kg_entities_type ON kg_entities(entity_type);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_kg_rel_source ON kg_relationships(source_name);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_kg_rel_target ON kg_relationships(target_name);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_kg_rel_type ON kg_relationships(relation_type);")
            conn.commit()

    # -------------------------------------------------------------------------
    # ENTITY CRUD OPERATIONS
    # -------------------------------------------------------------------------

    def add_or_update_entity(
        self,
        name: str,
        entity_type: str = "CONCEPT",
        attributes: Optional[Dict[str, Any]] = None,
        confidence: float = 1.0,
    ) -> Entity:
        """Adds a new entity or updates attributes if entity already exists."""
        clean_name = name.strip()
        now = get_utc_now_iso()
        attrs = attributes or {}

        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, attributes, confidence FROM kg_entities WHERE name = ?", (clean_name,))
            row = cursor.fetchone()

            if row:
                entity_id = row["id"]
                existing_attrs = json.loads(row["attributes"]) if row["attributes"] else {}
                existing_attrs.update(attrs)
                cursor.execute("""
                    UPDATE kg_entities
                    SET entity_type = ?, attributes = ?, confidence = ?, updated_at = ?
                    WHERE id = ?
                """, (entity_type.upper(), json.dumps(existing_attrs), confidence, now, entity_id))
                conn.commit()
                return Entity(
                    id=entity_id,
                    name=clean_name,
                    entity_type=entity_type.upper(),
                    attributes=existing_attrs,
                    confidence=confidence,
                    updated_at=now,
                )
            else:
                cursor.execute("""
                    INSERT INTO kg_entities (name, entity_type, attributes, confidence, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (clean_name, entity_type.upper(), json.dumps(attrs), confidence, now, now))
                conn.commit()
                entity_id = cursor.lastrowid
                return Entity(
                    id=entity_id,
                    name=clean_name,
                    entity_type=entity_type.upper(),
                    attributes=attrs,
                    confidence=confidence,
                    created_at=now,
                    updated_at=now,
                )

    def get_entity(self, name: str) -> Optional[Entity]:
        """Retrieves an entity by its name (case-insensitive)."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM kg_entities WHERE name = ?", (name.strip(),))
            row = cursor.fetchone()
            if not row:
                return None
            return Entity(
                id=row["id"],
                name=row["name"],
                entity_type=row["entity_type"],
                attributes=json.loads(row["attributes"]) if row["attributes"] else {},
                confidence=row["confidence"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )

    def list_entities(self, entity_type: Optional[str] = None) -> List[Entity]:
        """Returns all entities, optionally filtered by type."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            if entity_type:
                cursor.execute("SELECT * FROM kg_entities WHERE entity_type = ? ORDER BY name ASC", (entity_type.upper(),))
            else:
                cursor.execute("SELECT * FROM kg_entities ORDER BY name ASC")
            rows = cursor.fetchall()
            return [
                Entity(
                    id=r["id"],
                    name=r["name"],
                    entity_type=r["entity_type"],
                    attributes=json.loads(r["attributes"]) if r["attributes"] else {},
                    confidence=r["confidence"],
                    created_at=r["created_at"],
                    updated_at=r["updated_at"],
                )
                for r in rows
            ]

    def delete_entity(self, name: str) -> bool:
        """Deletes an entity and all associated incoming and outgoing relationships."""
        clean_name = name.strip()
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM kg_relationships WHERE source_name = ? OR target_name = ?", (clean_name, clean_name))
            cursor.execute("DELETE FROM kg_entities WHERE name = ?", (clean_name,))
            conn.commit()
            return cursor.rowcount > 0

    # -------------------------------------------------------------------------
    # RELATIONSHIP CRUD & TRAVERSAL
    # -------------------------------------------------------------------------

    def add_relationship(
        self,
        source_name: str,
        target_name: str,
        relation_type: str = "RELATED_TO",
        attributes: Optional[Dict[str, Any]] = None,
        weight: float = 1.0,
        auto_create_entities: bool = True,
    ) -> Relationship:
        """
        Creates or updates a directed relationship between two entities.
        Automatically registers unknown entities if auto_create_entities is True.
        """
        src = source_name.strip()
        tgt = target_name.strip()
        rel = relation_type.strip().upper()
        now = get_utc_now_iso()
        attrs = attributes or {}

        if auto_create_entities:
            if not self.get_entity(src):
                self.add_or_update_entity(src, entity_type="CONCEPT")
            if not self.get_entity(tgt):
                self.add_or_update_entity(tgt, entity_type="CONCEPT")

        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO kg_relationships (source_name, target_name, relation_type, weight, attributes, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_name, target_name, relation_type)
                DO UPDATE SET weight = excluded.weight, attributes = excluded.attributes
            """, (src, tgt, rel, weight, json.dumps(attrs), now))
            conn.commit()
            rel_id = cursor.lastrowid
            log_debug(f"[KNOWLEDGE GRAPH] Added relationship: ({src}) -[{rel}]-> ({tgt})")
            return Relationship(
                id=rel_id,
                source_name=src,
                target_name=tgt,
                relation_type=rel,
                weight=weight,
                attributes=attrs,
                created_at=now,
            )

    def get_relationships(
        self,
        entity_name: str,
        relation_type: Optional[str] = None,
        direction: str = "both",  # "outgoing", "incoming", or "both"
    ) -> List[Relationship]:
        """Returns relationships attached to an entity."""
        name = entity_name.strip()
        results: List[Relationship] = []

        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            queries = []
            params = []

            if direction in ("outgoing", "both"):
                q = "SELECT * FROM kg_relationships WHERE source_name = ?"
                p = [name]
                if relation_type:
                    q += " AND relation_type = ?"
                    p.append(relation_type.upper())
                queries.append((q, p))

            if direction in ("incoming", "both"):
                q = "SELECT * FROM kg_relationships WHERE target_name = ?"
                p = [name]
                if relation_type:
                    q += " AND relation_type = ?"
                    p.append(relation_type.upper())
                queries.append((q, p))

            for q, p in queries:
                cursor.execute(q, tuple(p))
                for r in cursor.fetchall():
                    results.append(Relationship(
                        id=r["id"],
                        source_name=r["source_name"],
                        target_name=r["target_name"],
                        relation_type=r["relation_type"],
                        weight=r["weight"],
                        attributes=json.loads(r["attributes"]) if r["attributes"] else {},
                        created_at=r["created_at"],
                    ))

        return results

    def get_related_entities(
        self,
        entity_name: str,
        relation_type: Optional[str] = None,
    ) -> List[Tuple[Entity, str]]:
        """Returns pairs of (Entity, relation_type) connected directly to entity_name."""
        rels = self.get_relationships(entity_name, relation_type=relation_type, direction="both")
        related: List[Tuple[Entity, str]] = []
        seen_names: Set[str] = set()

        for r in rels:
            neighbor = r.target_name if r.source_name.lower() == entity_name.lower() else r.source_name
            if neighbor.lower() not in seen_names:
                seen_names.add(neighbor.lower())
                ent = self.get_entity(neighbor)
                if ent:
                    related.append((ent, r.relation_type))

        return related

    def find_subgraph(self, root_entity_name: str, max_depth: int = 2) -> Dict[str, Any]:
        """Traverses the graph up to max_depth and returns structured nodes and edges."""
        visited_nodes: Set[str] = set()
        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []

        def traverse(current: str, depth: int):
            if depth > max_depth or current.lower() in visited_nodes:
                return
            visited_nodes.add(current.lower())
            ent = self.get_entity(current)
            if ent:
                nodes.append(ent.to_dict())

            rels = self.get_relationships(current, direction="outgoing")
            for r in rels:
                edges.append(r.to_dict())
                traverse(r.target_name, depth + 1)

        traverse(root_entity_name, 0)
        return {"nodes": nodes, "edges": edges}

    # -------------------------------------------------------------------------
    # CONTEXT SYNTHESIS FOR PROMPT INJECTION
    # -------------------------------------------------------------------------

    def format_graph_context(self, entity_names: List[str]) -> str:
        """
        Formats relational knowledge about specific entities into a clean context block
        for injecting into Chitti's LLM system prompt.
        """
        if not entity_names:
            return ""

        lines = ["\n[KNOWLEDGE GRAPH RELATIONAL GROUND TRUTH]:"]
        found_any = False

        for name in entity_names:
            ent = self.get_entity(name)
            if not ent:
                continue

            found_any = True
            lines.append(f"\nEntity: {ent.name} (Type: {ent.entity_type})")
            if ent.attributes:
                for k, v in ent.attributes.items():
                    lines.append(f"  - {k}: {v}")

            # Outgoing relations
            out_rels = self.get_relationships(ent.name, direction="outgoing")
            for r in out_rels:
                lines.append(f"  - {r.relation_type} -> {r.target_name}")

            # Incoming relations
            in_rels = self.get_relationships(ent.name, direction="incoming")
            for r in in_rels:
                lines.append(f"  - (is {r.relation_type} of) <- {r.source_name}")

        if not found_any:
            return ""

        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # DIALOGUE KNOWLEDGE EXTRACTION
    # -------------------------------------------------------------------------

    def extract_and_store_from_text(self, text: str, llm: Optional[BaseLLM] = None) -> List[Relationship]:
        """
        Extracts entities and relationships from conversational text and stores them in the graph.
        """
        import re
        extracted: List[Relationship] = []

        # 1. Regex Heuristic Extraction for Common Patterns
        # Pattern 1: "X is my [role/lead/friend/boss]" -> (X, Role, HAS_ROLE), (User, X, FRIEND_OF/WORKS_WITH)
        m_role = re.search(r"(?i)\b([A-Za-z0-9_\-\s]+?)\s+is\s+my\s+([A-Za-z0-9_\-\s]+?)(?:\s+and\s+|\s*$|[.,])", text)
        if m_role:
            person = m_role.group(1).strip().title()
            role = m_role.group(2).strip()
            self.add_or_update_entity(person, entity_type="PERSON", attributes={"role": role})
            rel = self.add_relationship(person, role, relation_type="HAS_ROLE")
            extracted.append(rel)

        # Pattern 2: "email is X@Y.com" / "X's email is Y"
        m_email = re.search(r"(?i)\b(?:his|her|their)?\s*email\s+(?:is\s+)?([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})", text)
        if m_email:
            email_addr = m_email.group(1).strip()
            # If person identified
            if m_role:
                person = m_role.group(1).strip().title()
                self.add_or_update_entity(person, attributes={"email": email_addr})
                rel = self.add_relationship(person, email_addr, relation_type="EMAIL_IS")
                extracted.append(rel)

        # Pattern 3: "working on X project" / "project is X"
        m_proj = re.search(r"(?i)\b(?:working\s+on|leading|developing|building)\s+(?:the\s+|my\s+)?([A-Za-z0-9_\-]+)\s+(?:project|app|code)", text)
        if m_proj:
            proj_name = m_proj.group(1).strip().title()
            self.add_or_update_entity(proj_name, entity_type="PROJECT")
            if m_role:
                person = m_role.group(1).strip().title()
                rel = self.add_relationship(person, proj_name, relation_type="WORKS_ON")
                extracted.append(rel)

        # 2. LLM Enhanced Extraction if available
        if llm:
            prompt = f"""Extract entities and factual relationships from this statement into a structured JSON array.
Statement: "{text}"

Output Format:
[
  {{"source": "EntityName", "source_type": "PERSON|PROJECT|PREFERENCE", "relation": "LEADS|WORKS_ON|EMAIL_IS|FRIEND_OF|PREFERS", "target": "TargetName", "target_type": "CONCEPT|PROJECT"}}
]

Respond ONLY with valid JSON.
"""
            try:
                res = llm.generate_response(prompt).strip()
                match = re.search(r"\[.*\]", res, re.DOTALL)
                if match:
                    items = json.loads(match.group(0))
                    for it in items:
                        src = it.get("source", "").strip()
                        tgt = it.get("target", "").strip()
                        rel = it.get("relation", "RELATED_TO").strip()
                        src_type = it.get("source_type", "CONCEPT")
                        tgt_type = it.get("target_type", "CONCEPT")
                        if src and tgt:
                            self.add_or_update_entity(src, entity_type=src_type)
                            self.add_or_update_entity(tgt, entity_type=tgt_type)
                            r = self.add_relationship(src, tgt, relation_type=rel)
                            extracted.append(r)
            except Exception as e:
                log_debug(f"[KNOWLEDGE GRAPH] LLM extraction notice: {e}")

        return extracted
