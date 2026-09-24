"""
Chitti Memory Manager Module.
Coordinates memory extraction, persistent storage, semantic retrieval, deduplication, and conversational memory commands.
"""

from typing import List, Dict, Any, Optional, Tuple

from src.memory.database import MemoryDatabase, MemoryRecord
from src.memory.retriever import MemoryRetriever, get_embedding_engine
from src.memory.extractor import MemoryExtractor, ExtractedMemoryCommand
from src.config import get_config
from src.utils.logging import log_debug, log_warning, log_chitti, log_state


class MemoryManager:
    """Central manager for Chitti's long-term memory system."""

    def __init__(
        self,
        db_path: Optional[str] = None,
        similarity_threshold: Optional[float] = None,
        importance_threshold: Optional[int] = None,
        embedding_model: Optional[str] = None,
    ):
        cfg = get_config()
        path = db_path or cfg.memory.db_path
        sim_thresh = similarity_threshold if similarity_threshold is not None else cfg.memory.similarity_threshold
        imp_thresh = importance_threshold if importance_threshold is not None else cfg.memory.importance_threshold
        emb_model = embedding_model or cfg.memory.embedding_model

        self.db = MemoryDatabase(db_path=path)
        self.embedding_engine = get_embedding_engine(model_name=emb_model)
        self.retriever = MemoryRetriever(
            db=self.db,
            embedding_engine=self.embedding_engine,
            similarity_threshold=sim_thresh,
            importance_threshold=imp_thresh,
        )
        self.extractor = MemoryExtractor()
        self.pending_confirmation: Optional[str] = None

    def remember(
        self,
        content: str,
        memory_type: str = "fact",
        importance: int = 4,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str, Optional[MemoryRecord]]:
        """
        Stores or updates a memory. Checks for duplicates before saving.
        """
        if not content or not content.strip():
            return False, "Nothing to remember.", None

        if self.extractor.is_sensitive(content):
            return (
                False,
                "For security reasons, I cannot store passwords, API keys, or secret tokens in memory.",
                None,
            )

        clean_content = self.extractor.clean_remembered_content(content)

        # Check for semantically similar existing memory (deduplication)
        dup = self.retriever.find_duplicate(clean_content, threshold=0.90)
        if dup:
            existing_rec, score = dup
            new_emb = self.embedding_engine.embed_text(clean_content)
            self.db.update_memory(
                existing_rec.id,
                content=clean_content,
                memory_type=memory_type,
                importance=max(existing_rec.importance, importance),
                embedding=new_emb,
            )
            log_chitti(f"[MEMORY] Updated existing memory (ID: {existing_rec.id}): '{clean_content}'")
            return True, "I've updated that in my memory.", existing_rec

        # Insert new record
        emb = self.embedding_engine.embed_text(clean_content)
        record = MemoryRecord(
            content=clean_content,
            memory_type=memory_type,
            importance=importance,
            embedding=emb,
            metadata=metadata or {},
        )
        mem_id = self.db.insert_memory(record)
        record.id = mem_id
        log_chitti(f"[MEMORY] New memory created (ID: {mem_id}): '{clean_content}'")
        return True, "I'll remember that.", record

    def recall(self, query: str, top_k: int = 5) -> List[MemoryRecord]:
        """Retrieves top_k relevant memories for a given query."""
        results = self.retriever.retrieve(query, top_k=top_k)
        return [item[0] for item in results]

    def forget_by_query(self, query: str) -> Tuple[bool, str]:
        """Finds and removes the memory most relevant to the query."""
        if not query or not query.strip():
            return False, "Please specify what you would like me to forget."

        # Search for matching memories with lower threshold for deletion match
        matches = self.retriever.retrieve(query, top_k=1, min_similarity=0.40)
        if not matches:
            # Fallback to substring matching
            all_mems = self.db.list_all_memories()
            for m in all_mems:
                if query.lower() in m.content.lower():
                    self.db.delete_memory(m.id)
                    log_chitti(f"[MEMORY] Memory deleted (ID: {m.id}): '{m.content}'")
                    return True, f"I have forgotten: '{m.content}'."
            return False, "I couldn't find a matching memory to forget."

        target_rec = matches[0][0]
        self.db.delete_memory(target_rec.id)
        log_chitti(f"[MEMORY] Memory deleted (ID: {target_rec.id}): '{target_rec.content}'")
        return True, f"I have forgotten: '{target_rec.content}'."

    def forget(self, memory_id: int) -> bool:
        """Deletes a memory by its unique ID."""
        success = self.db.delete_memory(memory_id)
        if success:
            log_chitti(f"[MEMORY] Memory deleted (ID: {memory_id})")
        return success

    def list_memories(self, memory_type: Optional[str] = None) -> List[MemoryRecord]:
        """Lists all stored memories, optionally by category."""
        return self.db.list_all_memories(memory_type=memory_type)

    def clear_all_memories(self) -> int:
        """Wipes all stored memories permanently."""
        count = self.db.clear_all_memories()
        log_chitti(f"[MEMORY] All {count} memories cleared.")
        return count

    def handle_interaction(self, user_text: str) -> Optional[Tuple[str, str]]:
        """
        Evaluates user input for explicit memory actions or pending confirmations.
        Returns (action_tag, response_message) if handled as a memory command, or None for standard LLM flow.
        """
        text = user_text.strip()
        lower = text.lower()

        # 1. Handle Pending Confirmation
        if self.pending_confirmation == "forget_all":
            if any(w in lower for w in ["yes", "proceed", "confirm", "sure", "do it", "yep", "yeah"]):
                self.pending_confirmation = None
                count = self.clear_all_memories()
                return ("forget_all_confirmed", f"All stored memories have been deleted ({count} cleared).")
            elif any(w in lower for w in ["no", "cancel", "stop", "abort", "don't", "dont", "nope"]):
                self.pending_confirmation = None
                return ("cancelled", "Memory deletion cancelled. All your memories are safe.")
            else:
                return ("confirm_required", "Please confirm with 'Yes' to delete all memories, or 'No' to cancel.")

        # 2. Extract Memory Command
        cmd = self.extractor.extract_command(text)

        if cmd.action == "sensitive_rejected":
            log_warning("[MEMORY] Sensitive credential storage rejected.")
            return ("rejected", cmd.rejection_reason or "I cannot store sensitive secrets.")

        if cmd.action == "remember":
            success, msg, _ = self.remember(cmd.content, cmd.memory_type, cmd.importance)
            return ("remembered", msg)

        if cmd.action == "forget":
            success, msg = self.forget_by_query(cmd.content)
            return ("forgotten", msg)

        if cmd.action == "forget_all":
            self.pending_confirmation = "forget_all"
            return (
                "confirm_required",
                "You asked me to delete all stored memories. Should I proceed? Please say Yes or No."
            )

        if cmd.action == "recall":
            cat = cmd.target_category
            memories = self.list_memories(memory_type=cat) if cat else self.list_memories()
            if not memories:
                resp = "I don't have any stored memories yet." if not cat else f"I don't have any memories stored under '{cat}'."
                return ("recalled", resp)

            summary_lines = ["Here is what I remember:"]
            for m in memories[:8]:  # Limit spoken summary to 8 items
                summary_lines.append(f"- {m.content}")
            if len(memories) > 8:
                summary_lines.append(f"...and {len(memories) - 8} more.")
            return ("recalled", "\n".join(summary_lines))

        # Not an explicit memory command -> return None so LLM handles it with memory context
        return None
