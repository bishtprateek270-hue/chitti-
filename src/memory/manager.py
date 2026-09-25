"""
Chitti Memory Manager Module.
Coordinates memory extraction, persistent storage, multi-fact decomposition,
semantic retrieval, deduplication, identity & relationship resolution, and intelligent memory routing.
"""

import re
from typing import List, Dict, Any, Optional, Tuple

from src.memory.database import MemoryDatabase, MemoryRecord
from src.memory.retriever import MemoryRetriever, get_embedding_engine
from src.memory.extractor import MemoryExtractor, ExtractedMemoryCommand, ExtractedFact
from src.memory.router import MemoryRouter, MemoryIntent, MemoryRouteDecision
from src.config import get_config
from src.utils.logging import log_debug, log_warning, log_chitti, log_error, log_state


class MemoryManager:
    """Central manager for Chitti's long-term memory, personal identity, and knowledge system."""

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
        self.router = MemoryRouter()
        self.pending_confirmation: Optional[str] = None

    def remember(
        self,
        content: str,
        memory_type: str = "fact",
        importance: int = 4,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str, Optional[MemoryRecord]]:
        """
        Stores or updates a single memory. Checks for key collisions and duplicate entries.
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
        target_key = metadata.get("key") if metadata else None

        log_chitti("[MEMORY ROUTER] Personal information detected")
        if target_key:
            log_chitti(f"[MEMORY] Persisting structured memory ({target_key})...")
        else:
            log_chitti(f"[MEMORY] Persisting memory ({memory_type})...")

        # 1. Check for structured key collision (e.g. key="user_name", key="creator", key="college", key="best_friend")
        if target_key:
            for existing_rec in self.db.list_all_memories():
                if existing_rec.metadata and existing_rec.metadata.get("key") == target_key:
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
                    verified = self.db.get_memory(existing_rec.id)
                    log_chitti(f"[MEMORY] Verified successfully (ID: {existing_rec.id}): '{clean_content}' (Updated key='{target_key}')")
                    return True, "I've updated that in my memory.", verified

        # 2. Check for semantically similar existing memory (deduplication)
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
            verified = self.db.get_memory(existing_rec.id)
            log_chitti(f"[MEMORY] Verified successfully (ID: {existing_rec.id}): '{clean_content}' (Updated existing)")
            return True, "I've updated that in my memory.", verified

        # 3. Insert new record with immediate write verification
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

        # Verification check
        verified = self.db.get_memory(mem_id)
        if not verified:
            log_error(f"[MEMORY ERROR] Failed to verify persistent storage of memory (ID: {mem_id})")
            return False, "Failed to persist memory in database.", None

        log_chitti(f"[MEMORY] Verified successfully (ID: {mem_id}): '{clean_content}'")
        return True, "I'll remember that.", verified

    def remember_facts(self, facts: List[ExtractedFact], lang: str = "en") -> Tuple[bool, str, List[MemoryRecord]]:
        """Stores a list of extracted discrete facts and returns a natural confirmation."""
        if not facts:
            return False, "Nothing to remember.", []

        stored_records: List[MemoryRecord] = []
        user_name_stored: Optional[str] = None
        creator_stored = False
        occ_stored: Optional[str] = None
        college_stored: Optional[str] = None
        best_friend_stored: Optional[str] = None

        for fact in facts:
            meta = {}
            if fact.key:
                meta["key"] = fact.key
            if fact.value:
                meta["value"] = fact.value

            if fact.key == "user_name":
                user_name_stored = fact.value
            elif fact.key == "creator":
                creator_stored = True
            elif fact.key == "occupation":
                occ_stored = fact.value
            elif fact.key == "college":
                college_stored = fact.value
            elif fact.key == "best_friend":
                best_friend_stored = fact.value

            success, msg, rec = self.remember(
                content=fact.content,
                memory_type=fact.memory_type,
                importance=fact.importance,
                metadata=meta,
            )
            if success and rec:
                stored_records.append(rec)

        if not stored_records:
            return False, "Failed to store memories.", []

        # If user_name was stored and creator was also stored, update creator memory value to user_name
        if user_name_stored:
            for rec in stored_records:
                if rec.metadata and rec.metadata.get("key") == "creator" and (rec.metadata.get("value") == "User" or not rec.metadata.get("value")):
                    rec.metadata["value"] = user_name_stored
                    rec.content = f"{user_name_stored} is my creator (User created Chitti)."
                    self.db.update_memory(rec.id, content=rec.content, metadata=rec.metadata)

        # Build rich, natural confirmation
        if user_name_stored and creator_stored:
            if lang == "hi":
                return True, f"समझ गया। मुझे याद रहेगा कि आप {user_name_stored} हैं और आपने मुझे बनाया है।", stored_records
            elif lang in ("hinglish", "mixed"):
                return True, f"Got it. Yaad rahega — tum {user_name_stored} ho aur tumne mujhe banaya hai.", stored_records
            else:
                return True, f"Got it. I'll remember that you're {user_name_stored} and that you created me.", stored_records
        elif user_name_stored:
            if lang == "hi":
                return True, f"समझ गया। मुझे याद रहेगा कि आपका नाम {user_name_stored} है।", stored_records
            elif lang in ("hinglish", "mixed"):
                return True, f"Got it. Mujhe yaad rahega ki tumhara naam {user_name_stored} hai.", stored_records
            else:
                return True, f"Got it. I'll remember that your name is {user_name_stored}.", stored_records
        elif creator_stored:
            if lang == "hi":
                return True, "समझ गया। मुझे याद रहेगा कि आपने मुझे बनाया है।", stored_records
            elif lang in ("hinglish", "mixed"):
                return True, "Got it. Mujhe yaad rahega ki tumne mujhe banaya hai.", stored_records
            else:
                return True, "Got it. I'll remember that you created me.", stored_records
        elif college_stored:
            if lang == "hi":
                return True, f"समझ गया। मुझे याद रहेगा कि आपका कॉलेज {college_stored} है।", stored_records
            elif lang in ("hinglish", "mixed"):
                return True, f"Got it. Yaad rahega ki tumhara college {college_stored} hai.", stored_records
            else:
                return True, f"Got it. I'll remember that your college is {college_stored}.", stored_records
        elif best_friend_stored:
            if lang == "hi":
                return True, f"समझ गया। मुझे याद रहेगा कि {best_friend_stored} आपका बेस्ट फ्रेंड है।", stored_records
            elif lang in ("hinglish", "mixed"):
                return True, f"Got it. Yaad rahega ki {best_friend_stored} tumhara best friend hai.", stored_records
            else:
                return True, f"Got it. I'll remember that your best friend is {best_friend_stored}.", stored_records
        elif occ_stored:
            if lang == "hi":
                return True, f"समझ गया। मुझे याद रहेगा कि आप एक {occ_stored} हैं।", stored_records
            elif lang in ("hinglish", "mixed"):
                return True, f"Got it. Yaad rahega ki tum {occ_stored} ho.", stored_records
            else:
                return True, f"Got it. I'll remember that you are an {occ_stored}.", stored_records

        if len(stored_records) == 1:
            return True, "I'll remember that.", stored_records
        return True, f"I've saved {len(stored_records)} details to memory.", stored_records

    def recall(self, query: str, top_k: int = 5, memory_type: Optional[str] = None) -> List[MemoryRecord]:
        """Retrieves top_k relevant memories for a given query."""
        results = self.retriever.retrieve(query, top_k=top_k, memory_type=memory_type)
        recs = [item[0] for item in results]
        log_chitti(f"[MEMORY] Retrieved {len(recs)} relevant memories")
        return recs

    def get_user_name(self) -> Optional[str]:
        """Retrieves stored user name if present in memory database."""
        for rec in self.db.list_all_memories():
            if rec.metadata and rec.metadata.get("key") == "user_name" and rec.metadata.get("value"):
                return rec.metadata.get("value")
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
            m = re.search(r"(?i)\b([A-Za-z\s]+?)\s+is\s+my\s+creator", rec.content)
            if m:
                c_name = m.group(1).strip()
                if c_name.lower() == "user" and user_name:
                    return user_name
                return c_name
            if "user created chitti" in rec.content.lower() and user_name:
                return user_name
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

    def get_college_name(self) -> Optional[str]:
        """Retrieves user's college or university from memory database."""
        for rec in self.db.list_all_memories():
            if rec.metadata and rec.metadata.get("key") in ("college", "university") and rec.metadata.get("value"):
                return rec.metadata.get("value")
            m = re.search(r"(?i)\buser'?s\s+(?:college|university)\s+is\s+([A-Za-z0-9_\-\s]+?)(?:\.|$)", rec.content)
            if m:
                return m.group(1).strip()
        return None

    def get_relationship(self, relation_key: str) -> Optional[str]:
        """Retrieves a relationship by key (e.g. 'best_friend', 'sister', 'brother')."""
        for rec in self.db.list_all_memories():
            if rec.metadata and rec.metadata.get("key") == relation_key and rec.metadata.get("value"):
                return rec.metadata.get("value")
            m = re.search(rf"(?i)\buser'?s\s+{relation_key.replace('_', ' ')}\s+is\s+([A-Za-z\s]+?)(?:\.|$)", rec.content)
            if m:
                return m.group(1).strip()
        return None

    def get_entity_relationship(self, entity_name: str) -> Optional[str]:
        """Looks up the relation of a specific person's name (e.g., 'Rahul' -> 'best friend')."""
        clean_name = entity_name.strip().lower()
        for rec in self.db.list_all_memories():
            if rec.metadata and rec.metadata.get("value", "").lower() == clean_name:
                key = rec.metadata.get("key", "")
                if key == "best_friend":
                    return f"{entity_name} is your best friend."
                if key == "sister":
                    return f"{entity_name} is your sister."
                if key == "brother":
                    return f"{entity_name} is your brother."
                if "teammate" in key:
                    return f"{entity_name} is your teammate."
                if "friend" in key:
                    return f"{entity_name} is your friend."
            if clean_name in rec.content.lower():
                return rec.content
        return None

    def get_project_name(self) -> Optional[str]:
        """Retrieves active user project from memory database."""
        for rec in self.db.list_all_memories():
            if rec.metadata and rec.metadata.get("key") == "project" and rec.metadata.get("value"):
                return rec.metadata.get("value")
            if rec.memory_type == "project":
                return rec.content
        return None

    def resolve_identity_query(self, raw_query: str, lang: str = "en") -> Optional[str]:
        """
        Directly and reliably resolves identity, creator, college, relationship, and occupation questions
        without hallucinations or placeholder leakage.
        """
        lower = raw_query.lower().strip()

        # 1. Creator Queries
        creator_patterns = [
            r"(?i)\b(?:who\s+(?:created|create|makes|made|make|built|build|developed|develop)\s+(?:you|u)|who\s+is\s+your\s+creator|who\s+developed\s+you|who\s+built\s+you)\b",
            r"(?i)\b(?:tujhe\s+kisne\s+(?:banaya|bnaya)|tumhe\s+kisne\s+banaya|tumhara\s+creator\s+kaun\s+hai|tera\s+creator\s+kaun\s+hai|aapko\s+kisne\s+banaya|kisne\s+banaya\s+(?:tujhe|tumhe|aapko))\b",
            r"(?:तुम्हें\s+किसने\s+बनाया|तुम्हारा\s+क्रिएटर\s+कौन\s+है|आपको\s+किसने\s+बनाया|किसने\s+बनाया\s+तुम्हें)",
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
            r"(?i)\b(?:what\s+is\s+my\s+name|who\s+am\s+i|tell\s+me\s+who\s+i\s+am|do\s+you\s+know\s+my\s+name|what'?s\s+my\s+name|tell\s+me\s+my\s+name)\b",
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

        # 3. College / University Queries
        college_patterns = [
            r"(?i)\b(?:what\s+is\s+my\s+college|where\s+do\s+i\s+study|tell\s+me\s+about\s+my\s+college|which\s+college\s+(?:do\s+i|am\s+i))\b",
            r"(?i)\b(?:mera\s+college\s+kaunsa\s+hai|main\s+kaha\s+padhta\s+hoon|mera\s+college)\b",
            r"(?:मेरा\s+कॉलेज\s+कौन\s+सा\s+है|मैं\s+कहाँ\s+पढ़ता\s+हूँ)",
        ]
        if any(re.search(pat, lower) for pat in college_patterns):
            college = self.get_college_name()
            if college:
                if lang == "hi":
                    return f"आपका कॉलेज {college} है।"
                elif lang in ("hinglish", "mixed"):
                    return f"Tumhara college {college} hai."
                else:
                    return f"Your college is {college}."
            else:
                if lang == "hi":
                    return "मुझे अभी मेमोरी में आपके कॉलेज की जानकारी नहीं मिली।"
                elif lang in ("hinglish", "mixed"):
                    return "Mujhe abhi memory mein tumhare college ki information nahi mili."
                else:
                    return "I don't have information about your college stored in memory yet."

        # 4. Best Friend / Specific Relationship Queries
        if re.search(r"(?i)\b(?:who\s+is\s+my\s+best\s+friend|mera\s+best\s+friend\s+kaun\s+hai)\b", lower):
            bf = self.get_relationship("best_friend")
            if bf:
                if lang == "hi":
                    return f"आपका बेस्ट फ्रेंड {bf} है।"
                elif lang in ("hinglish", "mixed"):
                    return f"Tumhara best friend {bf} hai."
                else:
                    return f"Your best friend is {bf}."
            else:
                return "I don't have information about your best friend in memory yet."

        if re.search(r"(?i)\b(?:who\s+is\s+my\s+sister|what\s+is\s+my\s+sister'?s\s+name|meri\s+behen\s+kaun\s+hai)\b", lower):
            sis = self.get_relationship("sister")
            if sis:
                if lang == "hi":
                    return f"आपकी बहन {sis} है।"
                elif lang in ("hinglish", "mixed"):
                    return f"Tumhari sister {sis} hai."
                else:
                    return f"Your sister's name is {sis}."
            else:
                return "I don't have information about your sister in memory yet."

        # 5. Specific Person Entity Query (e.g. "Who is Rahul?", "Who is Ananya?")
        m_entity = re.search(r"(?i)\bwho\s+is\s+([A-Z][a-z]+)\b", raw_query)
        if m_entity:
            target_name = m_entity.group(1).strip()
            if target_name.lower() not in {"chitti", "user", "creator"}:
                rel_info = self.get_entity_relationship(target_name)
                if rel_info:
                    return rel_info

        # 6. User Occupation Queries
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

        # 7. Project Query (e.g. "What project am I building?", "What am I building?")
        if re.search(r"(?i)\b(?:what\s+(?:project\s+)?am\s+i\s+(?:building|working\s+on)|what\s+am\s+i\s+making)\b", lower):
            proj = self.get_project_name()
            if proj:
                return f"You are building {proj}." if not proj.lower().startswith("user") else proj
            else:
                return "I don't have information about your project in memory yet."

        return None

    def forget_by_query(self, query: str) -> Tuple[bool, str]:
        """Finds and removes the memory most relevant to the query."""
        if not query or not query.strip():
            return False, "Please specify what you would like me to forget."

        matches = self.retriever.retrieve(query, top_k=1, min_similarity=0.40)
        if not matches:
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

    def handle_interaction(self, user_text: str, lang: str = "en") -> Optional[Tuple[str, str]]:
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
                success, msg, _ = self.remember_facts(cmd.facts, lang=lang)
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

    def auto_capture_interaction(
        self,
        user_text: str,
        assistant_response: str,
        lang: str = "en",
    ) -> Tuple[List[MemoryRecord], Optional[MemoryRecord]]:
        """
        Selectively captures and persists personal, relationship, education, project, and preference facts.
        General technical questions and knowledge queries are strictly excluded from long-term memory.
        """
        if not user_text or not user_text.strip() or not assistant_response or not assistant_response.strip():
            return [], None

        # Check for sensitive data
        if self.extractor.is_sensitive(user_text) or self.extractor.is_sensitive(assistant_response):
            log_warning("[MEMORY] Sensitive credential detected. Skipped auto-capture.")
            return [], None

        user_clean = user_text.strip()
        decision = self.router.classify_intent(user_clean)

        # Gate storage intent
        if not decision.should_store_memory:
            log_debug(f"[MEMORY ROUTER] Intent: {decision.intent.value}. Persistent storage skipped.")
            return [], None

        captured_facts: List[MemoryRecord] = []
        implicit_facts = self.extractor.extract_implicit_facts(user_clean)

        for fact in implicit_facts:
            meta = {"auto_captured": True}
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
                captured_facts.append(rec)

        return captured_facts, None
