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

        # Preload embedding model asynchronously so first query experiences zero lag
        if hasattr(self.embedding_engine, "_load"):
            import threading
            threading.Thread(target=self.embedding_engine._load, daemon=True, name="ChittiEmbeddingPreload").start()

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
        all_recs = self.db.list_all_memories()
        for rec in all_recs:
            if rec.metadata and rec.metadata.get("key") == "occupation" and rec.metadata.get("value"):
                return rec.metadata.get("value")
        for rec in all_recs:
            m = re.search(r"(?i)\buser\s+is\s+an?\s+([A-Za-z0-9_\-\.\s]+?)(?:\.|$)", rec.content)
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

        # 7. User Projects Queries
        project_patterns = [
            r"(?i)\b(?:what\s+projects?\s+(?:have\s+i|did\s+i|i\s+have)\s+(?:made|built|developed|created|worked\s+on)|tell\s+me\s+about\s+my\s+projects?|what\s+are\s+my\s+projects?|list\s+(?:all\s+)?my\s+projects?|my\s+projects?|what\s+is\s+in\s+my\s+portfolio|projects\s+on\s+my\s+github|projects\s+in\s+my\s+portfolio|projects\s+i\s+have\s+made|what\s+did\s+i\s+make)\b",
            r"(?i)\b(?:mere\s+projects?\s+(?:kaunse\s+hai|batao|kya\s+hai|ke\s+baare\s+me(?:in)?\s+batao)|maine\s+kaunse\s+projects?\s+banaye\s+hai|mere\s+projects?)\b",
            r"(?:मेरे\s+प्रोजेक्ट्स?\s+कौन\s+से\s+हैं|मेरे\s+प्रोजेक्ट्स?\s+के\s+बारे\s+में\s+बताओ)",
        ]
        if any(re.search(pat, lower) for pat in project_patterns):
            project_mems = [m for m in self.db.list_all_memories() if m.memory_type == "project" and "Project " in m.content]
            if project_mems:
                lines = []
                for pm in sorted(project_mems, key=lambda x: x.id):
                    text = pm.content.strip().rstrip(".")
                    if ". Stack:" in text:
                        text = text.split(". Stack:")[0]
                    if ". Live:" in text:
                        text = text.split(". Live:")[0]
                    if " | GitHub:" in text:
                        text = text.split(" | GitHub:")[0]
                    lines.append(text)

                if lang == "hi":
                    return "आपने ये प्रमुख प्रोजेक्ट्स बनाए हैं जो आपके पोर्टफोलियो और गिटहब पर उपलब्ध हैं:\n• " + "\n• ".join(lines)
                elif lang in ("hinglish", "mixed"):
                    return "Aapne yeh sabhi flagship projects banaye hain jo aapke Portfolio aur GitHub par hain:\n• " + "\n• ".join(lines)
                else:
                    return "Here are your major projects available across your GitHub and portfolio:\n• " + "\n• ".join(lines)

        # 7.1. Portfolio & GitHub Links Queries
        links_patterns = [
            r"(?i)\b(?:what\s+is\s+my\s+portfolio(?:\s+link|\s+url|\s+website)?|where\s+is\s+my\s+portfolio|my\s+portfolio\s+link|what\s+is\s+my\s+github(?:\s+link|\s+url|\s+profile)?|where\s+is\s+my\s+github|my\s+github\s+link|show\s+my\s+portfolio|show\s+my\s+github|my\s+portfolio|my\s+github)\b",
            r"(?i)\b(?:mera\s+portfolio\s+kya\s+hai|mera\s+github\s+link|mera\s+portfolio\s+link|mera\s+portfolio|mera\s+github)\b",
        ]
        if any(re.search(pat, lower) for pat in links_patterns):
            portfolio_url = "https://bishtprateek270-hue.github.io/Portfolio/"
            github_url = "https://github.com/bishtprateek270-hue"
            linkedin_url = "https://linkedin.com/in/prateek-singh-bisht"
            if "github" in lower:
                return f"Your GitHub profile is: {github_url}"
            if "portfolio" in lower:
                return f"Your live portfolio website is: {portfolio_url}"
            return f"Here are your profile links:\n• Portfolio: {portfolio_url}\n• GitHub: {github_url}\n• LinkedIn: {linkedin_url}"

        # 7.2. Tech Stack & Skills Queries
        skills_patterns = [
            r"(?i)\b(?:what\s+is\s+my\s+(?:tech\s+stack|technical\s+stack|skills?)|what\s+are\s+my\s+skills?|what\s+technologies\s+do\s+i\s+(?:know|use)|my\s+tech\s+stack|my\s+skills?)\b",
            r"(?i)\b(?:mera\s+tech\s+stack\s+kya\s+hai|meri\s+skills?\s+kya\s+hai|mujhe\s+kaunsi\s+technologies\s+aati\s+hai)\b",
        ]
        if any(re.search(pat, lower) for pat in skills_patterns):
            skill_mems = [m for m in self.db.list_all_memories() if m.memory_type == "skill"]
            if skill_mems:
                lines = [m.content.strip().rstrip(".") for m in skill_mems]
                if lang in ("hi", "hinglish", "mixed"):
                    return "Yeh aapka technical stack hai:\n• " + "\n• ".join(lines)
                else:
                    return "Here is your technical stack across AI, Backend, Frontend, and 3D/DevOps:\n• " + "\n• ".join(lines)

        # 7.3. Hackathons & Achievements Queries
        achieve_patterns = [
            r"(?i)\b(?:what\s+(?:hackathons?|achievements?|awards?|certifications?)\s+(?:have\s+i|did\s+i)\s+(?:won|done|participated)|tell\s+me\s+about\s+my\s+(?:hackathons?|achievements?|certifications?)|my\s+hackathons?|my\s+achievements?)\b",
            r"(?i)\b(?:meri\s+achievements?\s+kya\s+hai|maine\s+kaunse\s+hackathons?\s+kiye\s+hai|mere\s+awards?)\b",
        ]
        if any(re.search(pat, lower) for pat in achieve_patterns):
            ach_mems = [m for m in self.db.list_all_memories() if m.memory_type == "achievement"]
            if ach_mems:
                lines = [m.content.strip().rstrip(".") for m in ach_mems]
                if lang in ("hi", "hinglish", "mixed"):
                    return "Aapke milestones, hackathons aur certifications:\n• " + "\n• ".join(lines)
                else:
                    return "Here are your key milestones, hackathons, and certifications:\n• " + "\n• ".join(lines)

        # 7.4. Work Experience & Internship Queries
        exp_patterns = [
            r"(?i)\b(?:where\s+did\s+i\s+(?:intern|work)|what\s+is\s+my\s+work\s+experience|tell\s+me\s+about\s+my\s+internship|my\s+internship|my\s+work\s+experience)\b",
            r"(?i)\b(?:meri\s+internship\s+kaha(?:n)?\s+(?:thi|hui)|mera\s+work\s+experience)\b",
        ]
        if any(re.search(pat, lower) for pat in exp_patterns):
            exp_mems = [m for m in self.db.list_all_memories() if m.memory_type == "experience"]
            if exp_mems:
                return exp_mems[0].content

        # 7.5. Academic Record & Grades Queries
        academic_patterns = [
            r"(?i)\b(?:what\s+is\s+my\s+(?:cgpa|sgpa|academic\s+record|grades?|marks)|how\s+much\s+(?:cgpa|sgpa)\s+did\s+i\s+get|my\s+academic\s+record|my\s+cgpa)\b",
            r"(?i)\b(?:mera\s+cgpa\s+kya\s+hai|mere\s+marks\s+kitne\s+hai|mera\s+academic\s+record)\b",
        ]
        if any(re.search(pat, lower) for pat in academic_patterns):
            for rec in self.db.list_all_memories():
                if rec.metadata and rec.metadata.get("key") == "academic_record":
                    return rec.content

        # 8. General Identity & Memory Summary ("What do you know about me?", "Tell me about me", etc.)
        know_me_patterns = [
            r"(?i)\b(?:what\s+do\s+(?:you|u)\s+(?:know|remember)\s+about\s+me|tell\s+me\s+(?:everything\s+)?about\s+me|what\s+do\s+(?:you|u)\s+have\s+on\s+me|list\s+(?:all\s+)?my\s+memories|my\s+memories)\b",
            r"(?i)\b(?:mere\s+baare\s+me(?:in)?\s+(?:kya\s+jaante\s+ho|batao|kya\s+yaad\s+hai)|mujhe\s+kitna\s+jaante\s+ho)\b",
            r"(?:तुम्हें\s+मेरे\s+बारे\s+में\s+क्या\s+(?:याद|पता)\s+है|मेरे\s+बारे\s+में\s+बताओ)",
        ]
        if any(re.search(pat, lower) for pat in know_me_patterns):
            facts = []
            name = self.get_user_name()
            if name:
                facts.append(f"Name: {name}")
            creator = self.get_creator_name()
            if creator:
                facts.append(f"Creator: {creator}")
            college = self.get_college_name()
            if college:
                facts.append(f"College: {college}")
            occ = self.get_user_occupation()
            if occ:
                facts.append(f"Profession: {occ}")
            facts.append("Portfolio: https://bishtprateek270-hue.github.io/Portfolio/")
            facts.append("GitHub: https://github.com/bishtprateek270-hue (14 projects including Chitti, 3D-ULPIN, KarigarAI, DocForensics AI, DoseBand, ChurnOps)")

            # Also include high-importance memories
            all_mems = self.db.list_all_memories()
            sorted_mems = sorted(all_mems, key=lambda m: m.importance, reverse=True)
            for m in sorted_mems:
                c = m.content.strip().rstrip(".")
                if c and not any(c.lower() in f.lower() for f in facts) and len(facts) < 10:
                    facts.append(c)

            if facts:
                if lang == "hi":
                    return "मुझे आपके बारे में यह जानकारी याद है:\n• " + "\n• ".join(facts)
                elif lang in ("hinglish", "mixed"):
                    return "Mujhe aapke baare mein yeh sab yaad hai:\n• " + "\n• ".join(facts)
                else:
                    return "Here is what I know about you:\n• " + "\n• ".join(facts)
            else:
                if lang == "hi":
                    return "मुझे अभी मेमोरी में आपके बारे में कोई विशेष जानकारी नहीं मिली।"
                elif lang in ("hinglish", "mixed"):
                    return "Mujhe abhi memory mein aapke baare mein koi information nahi mili."
                else:
                    return "I don't have any personal details stored about you in my memory yet."

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
