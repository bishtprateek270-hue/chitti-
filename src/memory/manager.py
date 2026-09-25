"""
Chitti Memory Manager Module.
Coordinates memory extraction, persistent storage, multi-fact decomposition,
semantic retrieval, deduplication, identity resolution, and conversational memory commands.
"""

import re
from typing import List, Dict, Any, Optional, Tuple

from src.memory.database import MemoryDatabase, MemoryRecord
from src.memory.retriever import MemoryRetriever, get_embedding_engine
from src.memory.extractor import MemoryExtractor, ExtractedMemoryCommand, ExtractedFact
from src.config import get_config
from src.utils.logging import log_debug, log_warning, log_chitti, log_state


class MemoryManager:
    """Central manager for Chitti's long-term memory and identity knowledge system."""

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
        Stores or updates a single memory. Checks for duplicates before saving.
        """
        if not content or not content.strip():
            return False, "Nothing to remember.", None

        if self.extractor.is_sensitive(content):
            return (
                False,
                "For security reasons, I cannot store passwords, API keys, or secret tokens in memory.",
                None,
            )

        clean_content = content.strip().rstrip(".!? \t\n") + "."

        # Check for semantically similar existing memory (deduplication)
        dup = self.retriever.find_duplicate(clean_content, threshold=0.88)
        if dup:
            existing_rec, score = dup
            new_emb = self.embedding_engine.embed_text(clean_content)
            merged_meta = existing_rec.metadata.copy() if existing_rec.metadata else {}
            if metadata:
                merged_meta.update(metadata)

            self.db.update_memory(
                existing_rec.id,
                content=clean_content,
                memory_type=memory_type,
                importance=max(existing_rec.importance, importance),
                embedding=new_emb,
                metadata=merged_meta,
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

    def remember_facts(self, facts: List[ExtractedFact]) -> Tuple[bool, str, List[MemoryRecord]]:
        """Stores a list of extracted discrete facts."""
        if not facts:
            return False, "Nothing to remember.", []

        stored_records: List[MemoryRecord] = []
        for fact in facts:
            meta = {}
            if fact.key:
                meta["key"] = fact.key
            if fact.value:
                meta["value"] = fact.value

            success, msg, rec = self.remember(
                content=fact.content,
                memory_type=fact.memory_type,
                importance=fact.importance,
                metadata=meta,
            )
            if success and rec:
                stored_records.append(rec)

        if len(stored_records) == 1:
            return True, "I'll remember that.", stored_records
        elif len(stored_records) > 1:
            return True, f"I've saved {len(stored_records)} details to memory.", stored_records

        return False, "Failed to store memories.", []

    def recall(self, query: str, top_k: int = 5) -> List[MemoryRecord]:
        """Retrieves top_k relevant memories for a given query."""
        results = self.retriever.retrieve(query, top_k=top_k)
        return [item[0] for item in results]

    def get_user_name(self) -> Optional[str]:
        """Retrieves stored user name if present in memory database."""
        for rec in self.db.list_all_memories():
            if rec.metadata and rec.metadata.get("key") == "user_name" and rec.metadata.get("value"):
                return rec.metadata.get("value")
            # Fallback to regex in content
            m = re.search(r"(?i)\buser'?s\s+name\s+is\s+([A-Za-z\s]+?)(?:\.|$)", rec.content)
            if m:
                return m.group(1).strip()
        return None

    def get_creator_name(self) -> Optional[str]:
        """Retrieves creator identity from memory database."""
        user_name = self.get_user_name()
        for rec in self.db.list_all_memories():
            if rec.metadata and rec.metadata.get("key") == "creator" and rec.metadata.get("value"):
                val = rec.metadata.get("value")
                if val.lower() == "user" and user_name:
                    return user_name
                return val
            # Fallback regex in content
            m = re.search(r"(?i)\b([A-Za-z\s]+?)\s+is\s+my\s+creator", rec.content)
            if m:
                c_name = m.group(1).strip()
                if c_name.lower() == "user" and user_name:
                    return user_name
                return c_name
        return user_name

    def get_user_occupation(self) -> Optional[str]:
        """Retrieves stored occupation / profession from memory database."""
        for rec in self.db.list_all_memories():
            if rec.metadata and rec.metadata.get("key") == "occupation" and rec.metadata.get("value"):
                return rec.metadata.get("value")
            m = re.search(r"(?i)\buser\s+is\s+an?\s+([A-Za-z0-9_\-\s]+?)(?:\.|$)", rec.content)
            if m:
                return m.group(1).strip()
        return None

    def resolve_identity_query(self, raw_query: str, lang: str = "en") -> Optional[str]:
        """
        Directly and reliably resolves identity, creator, and occupation questions
        without hallucinations or placeholder leakage.
        """
        lower = raw_query.lower().strip()

        # 1. Creator Queries
        creator_patterns = [
            r"(?i)\b(?:who\s+(?:created|create|makes|made|make|built|build)\s+(?:you|u)|who\s+is\s+your\s+creator|who\s+developed\s+you)\b",
            r"(?i)\b(?:tujhe\s+kisne\s+(?:banaya|bnaya)|tumhe\s+kisne\s+banaya|tumhara\s+creator\s+kaun\s+hai|tera\s+creator\s+kaun\s+hai|aapko\s+kisne\s+banaya)\b",
            r"(?:तुम्हें\s+किसने\s+बनाया|तुम्हारा\s+क्रिएटर\s+कौन\s+है|आपको\s+किसने\s+बनाया)",
        ]
        if any(re.search(pat, lower) for pat in creator_patterns):
            creator = self.get_creator_name()
            if creator:
                if lang == "hi":
                    return f"मुझे {creator} ने बनाया है।"
                elif lang in ("hinglish", "mixed"):
                    return f"Mujhe {creator} ne banaya hai."
                else:
                    return f"You created me, {creator}." if creator == self.get_user_name() else f"{creator} created me."
            else:
                if lang == "hi":
                    return "मुझे अभी अपनी मेमोरी में मेरे क्रिएटर की जानकारी नहीं मिली।"
                elif lang in ("hinglish", "mixed"):
                    return "Mujhe abhi memory mein mere creator ki information nahi mili."
                else:
                    return "I don't have information about who created me stored in my memory yet."

        # 2. User Name Queries
        name_patterns = [
            r"(?i)\b(?:what\s+is\s+my\s+name|who\s+am\s+i|do\s+you\s+know\s+my\s+name|what'?s\s+my\s+name|tell\s+me\s+my\s+name)\b",
            r"(?i)\b(?:mera\s+na+m\s+kya\s+(?:hai|h)|main\s+kaun\s+(?:hoon|hu)|mera\s+na+m\s+yaad\s+hai|(?:kya\s+)?(?:tujhe|tumhe|aapko)\s+mera\s+naam\s+(?:pata|yaad)\s+hai)\b",
            r"(?:मेरा\s+नाम\s+क्या\s+है|मैं\s+कौन\s+हूँ|मेरा\s+नाम\s+याद\s+है|क्या\s+(?:तुम|तुम्हें|आप|आपको)\s+मेरा\s+नाम\s+(?:जानते|पता)\s+हो)",
        ]
        if any(re.search(pat, lower) for pat in name_patterns):
            user_name = self.get_user_name()
            if user_name:
                if lang == "hi":
                    return f"तुम्हारा नाम {user_name} है।"
                elif lang in ("hinglish", "mixed"):
                    return f"Tumhara naam {user_name} hai."
                else:
                    return f"Your name is {user_name}."
            else:
                if lang == "hi":
                    return "मुझे अभी मेमोरी में आपका नाम नहीं मिला।"
                elif lang in ("hinglish", "mixed"):
                    return "Mujhe abhi tumhara naam memory mein nahi mila."
                else:
                    return "I don't have your name stored in my memory yet."

        # 3. User Occupation Queries
        occ_patterns = [
            r"(?i)\b(?:what\s+do\s+i\s+do|what\s+is\s+my\s+(?:job|profession|work|occupation)|what'?s\s+my\s+(?:job|profession|work|occupation))\b",
            r"(?i)\b(?:main\s+kya\s+karta\s+(?:hoon|hu)|mera\s+profession\s+kya\s+hai|mera\s+kaam\s+kya\s+hai)\b",
            r"(?:मैं\s+क्या\s+करता\s+हूँ|मेरा\s+प्रोफेशन\s+क्या\s+है|मेरा\s+काम\s+क्या\s+है)",
        ]
        if any(re.search(pat, lower) for pat in occ_patterns):
            occ = self.get_user_occupation()
            if occ:
                if lang == "hi":
                    return f"आप एक {occ} हैं।"
                elif lang in ("hinglish", "mixed"):
                    return f"Tum {occ} ho."
                else:
                    return f"You are an {occ}."
            else:
                if lang == "hi":
                    return "मुझे अभी मेमोरी में आपके प्रोफेशन की जानकारी नहीं मिली।"
                elif lang in ("hinglish", "mixed"):
                    return "Mujhe abhi tumhare profession ki information memory mein nahi mili."
                else:
                    return "I don't have your profession stored in my memory yet."

        return None

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
            if re.search(r"\b(?:no|cancel|stop|abort|don'?t|nope|nahi|nahin|mat karo)\b", lower):
                self.pending_confirmation = None
                return ("cancelled", "Memory deletion cancelled. All your memories are safe.")
            elif re.search(r"\b(?:yes|proceed|confirm|sure|do it|yep|yeah|haan|sahi|ha)\b", lower):
                self.pending_confirmation = None
                count = self.clear_all_memories()
                return ("forget_all_confirmed", f"All stored memories have been deleted ({count} cleared).")
            else:
                return ("confirm_required", "Please confirm with 'Yes' to delete all memories, or 'No' to cancel.")

        # 2. Extract Memory Command
        cmd = self.extractor.extract_command(text)

        if cmd.action == "sensitive_rejected":
            log_warning("[MEMORY] Sensitive credential storage rejected.")
            return ("rejected", cmd.rejection_reason or "I cannot store sensitive secrets.")

        if cmd.action == "remember":
            if cmd.facts:
                success, msg, _ = self.remember_facts(cmd.facts)
                return ("remembered", msg)
            elif cmd.content:
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
            for m in memories[:8]:
                summary_lines.append(f"- {m.content}")
            if len(memories) > 8:
                summary_lines.append(f"...and {len(memories) - 8} more.")
            return ("recalled", "\n".join(summary_lines))

        # Not an explicit memory command
        return None
