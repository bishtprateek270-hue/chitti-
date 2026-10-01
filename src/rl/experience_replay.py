"""
Chitti Persistent Experience Replay & Episodic Memory Buffer.
Stores full execution trajectories, failure reflexions, and DPO training pairs in SQLite.
Provides semantic and keyword lookup to retrieve past lessons for future planning.
"""

import json
import os
from pathlib import Path
import sqlite3
import threading
from typing import Any, Dict, List, Optional, Tuple

from src.rl.models import Episode, ReflexionLesson, TrajectoryStep, DPOPreferencePair, TaskOutcome
from src.utils.logging import log_chitti, log_debug, log_info, log_warn


class ExperienceReplayBuffer:
    """
    Persistent episodic experience store for Reinforcement Learning & Self-Correction.
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            base_dir = Path(__file__).resolve().parent.parent.parent
            self.db_path = base_dir / "data" / "memory" / "experience_replay.db"
        else:
            self.db_path = Path(db_path)

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            # Episodes
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS episodes (
                    episode_id TEXT PRIMARY KEY,
                    task_description TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    total_reward REAL NOT NULL,
                    outcome TEXT NOT NULL,
                    user_feedback_score REAL,
                    start_time TEXT NOT NULL,
                    end_time TEXT
                )
            """)
            # Trajectory steps
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trajectory_steps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    episode_id TEXT NOT NULL,
                    step_id INTEGER NOT NULL,
                    action_type TEXT NOT NULL,
                    action_payload TEXT NOT NULL,
                    observation_before TEXT,
                    observation_after TEXT,
                    reward REAL NOT NULL,
                    verified INTEGER NOT NULL,
                    error_message TEXT,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (episode_id) REFERENCES episodes (episode_id)
                )
            """)
            # Reflexion Lessons
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS reflexion_lessons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_pattern TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    failed_action TEXT NOT NULL,
                    root_cause TEXT NOT NULL,
                    correction_rule TEXT NOT NULL,
                    reward_delta REAL NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            # DPO Pairs
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS dpo_pairs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prompt TEXT NOT NULL,
                    chosen_response TEXT NOT NULL,
                    rejected_response TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                )
            """)
            conn.commit()

    def record_episode(self, episode: Episode):
        """Records a full execution episode and its state-action steps into persistent storage."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO episodes 
                (episode_id, task_description, domain, total_reward, outcome, user_feedback_score, start_time, end_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                episode.episode_id,
                episode.task_description,
                episode.domain,
                episode.total_reward,
                episode.outcome.value if isinstance(episode.outcome, TaskOutcome) else str(episode.outcome),
                episode.user_feedback_score,
                episode.start_time,
                episode.end_time,
            ))

            for step in episode.steps:
                cursor.execute("""
                    INSERT INTO trajectory_steps
                    (episode_id, step_id, action_type, action_payload, observation_before, observation_after, reward, verified, error_message, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    episode.episode_id,
                    step.step_id,
                    step.action_type,
                    json.dumps(step.action_payload),
                    step.observation_before,
                    step.observation_after,
                    step.reward,
                    1 if step.verified else 0,
                    step.error_message,
                    step.timestamp,
                ))

            # Record any distilled reflexion lessons
            for lesson in episode.lessons_learned:
                cursor.execute("""
                    INSERT INTO reflexion_lessons
                    (task_pattern, domain, failed_action, root_cause, correction_rule, reward_delta, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    lesson.task_pattern,
                    lesson.domain,
                    lesson.failed_action,
                    lesson.root_cause,
                    lesson.correction_rule,
                    lesson.reward_delta,
                    lesson.created_at,
                ))

            conn.commit()
            log_debug(f"[RL BUFFER] Saved episode {episode.episode_id} with reward {episode.total_reward:.2f}")

    def store_lesson(self, lesson: ReflexionLesson) -> int:
        """Stores a standalone reflexion rule learned from an error."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO reflexion_lessons
                (task_pattern, domain, failed_action, root_cause, correction_rule, reward_delta, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                lesson.task_pattern,
                lesson.domain,
                lesson.failed_action,
                lesson.root_cause,
                lesson.correction_rule,
                lesson.reward_delta,
                lesson.created_at,
            ))
            conn.commit()
            return cursor.lastrowid

    def get_relevant_lessons(
        self,
        task_query: str,
        domain: Optional[str] = None,
        limit: int = 4,
    ) -> List[ReflexionLesson]:
        """
        Retrieves the most relevant past failure lessons and self-correction rules
        for a given task description or application keyword.
        """
        if not task_query:
            return []

        words = [w.lower() for w in task_query.split() if len(w) > 3]
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            if domain:
                cursor.execute("SELECT * FROM reflexion_lessons WHERE domain = ? ORDER BY id DESC LIMIT 50", (domain,))
            else:
                cursor.execute("SELECT * FROM reflexion_lessons ORDER BY id DESC LIMIT 50")

            rows = cursor.fetchall()

        matched_lessons = []
        for r in rows:
            lesson = ReflexionLesson(
                id=r["id"],
                task_pattern=r["task_pattern"],
                domain=r["domain"],
                failed_action=r["failed_action"],
                root_cause=r["root_cause"],
                correction_rule=r["correction_rule"],
                reward_delta=r["reward_delta"],
                created_at=r["created_at"],
            )

            # Score relevance
            text_corpus = f"{lesson.task_pattern} {lesson.failed_action} {lesson.root_cause}".lower()
            score = sum(1 for w in words if w in text_corpus)
            if score > 0 or not words:
                matched_lessons.append((score, lesson))

        matched_lessons.sort(key=lambda x: x[0], reverse=True)
        return [l for score, l in matched_lessons[:limit]]

    def store_dpo_pair(self, pair: DPOPreferencePair) -> int:
        """Stores a pairwise preference sample for offline model fine-tuning."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO dpo_pairs (prompt, chosen_response, rejected_response, domain, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (
                pair.prompt,
                pair.chosen_response,
                pair.rejected_response,
                pair.domain,
                pair.timestamp,
            ))
            conn.commit()
            return cursor.lastrowid

    def export_dpo_dataset(self, output_path: str) -> int:
        """Exports all DPO preference pairs to a standard JSONL dataset."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT prompt, chosen_response, rejected_response, domain FROM dpo_pairs")
            rows = cursor.fetchall()

        count = 0
        with open(out, "w", encoding="utf-8") as f:
            for r in rows:
                entry = {
                    "prompt": r["prompt"],
                    "chosen": r["chosen_response"],
                    "rejected": r["rejected_response"],
                    "domain": r["domain"],
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                count += 1

        log_info(f"[RL DPO] Exported {count} preference pairs to {output_path}")
        return count

    def get_stats(self) -> Dict[str, Any]:
        """Returns aggregate metrics of episodes, lessons, and rewards."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*), AVG(total_reward) FROM episodes")
            ep_count, avg_reward = cursor.fetchone()
            cursor.execute("SELECT COUNT(*) FROM reflexion_lessons")
            lesson_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM dpo_pairs")
            dpo_count = cursor.fetchone()[0]

        return {
            "total_episodes": ep_count or 0,
            "average_reward": round(avg_reward or 0.0, 3),
            "total_lessons_learned": lesson_count or 0,
            "dpo_training_pairs": dpo_count or 0,
        }
