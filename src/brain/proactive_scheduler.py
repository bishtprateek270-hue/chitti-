"""
Chitti Proactive Job Scheduler & Ambient Trigger Engine (Phase 3).
Manages persistent time-based reminders, recurring background health checks,
morning briefings, Git branch commit reminders, and spoken/visual notifications.
"""

import json
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from src.utils.logging import log_chitti, log_debug, log_info, log_warn, log_error


class TriggerType(str, Enum):
    ONE_TIME_REMINDER = "ONE_TIME_REMINDER"
    RECURRING_DAILY = "RECURRING_DAILY"
    RECURRING_INTERVAL = "RECURRING_INTERVAL"
    GIT_STATUS_CHECK = "GIT_STATUS_CHECK"
    PROACTIVE_BRIEFING = "PROACTIVE_BRIEFING"


def get_utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ScheduledTask:
    """Represents a scheduled background task or reminder."""
    id: Optional[int] = None
    title: str = ""
    message: str = ""
    trigger_type: TriggerType = TriggerType.ONE_TIME_REMINDER
    due_timestamp: float = 0.0  # Unix Epoch seconds
    recurrence_interval_sec: Optional[float] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    last_executed: Optional[str] = None
    created_at: str = field(default_factory=get_utc_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "message": self.message,
            "trigger_type": self.trigger_type.value if hasattr(self.trigger_type, "value") else str(self.trigger_type),
            "due_timestamp": self.due_timestamp,
            "recurrence_interval_sec": self.recurrence_interval_sec,
            "payload": self.payload,
            "is_active": self.is_active,
            "last_executed": self.last_executed,
            "created_at": self.created_at,
        }

    @property
    def time_until_due_sec(self) -> float:
        return max(0.0, self.due_timestamp - time.time())


class ProactiveScheduler:
    """
    Persistent SQLite-backed task scheduler running on a background worker thread.
    """

    def __init__(self, db_path: str = "data/memory/scheduled_tasks.db", poll_interval_sec: float = 1.0):
        self.db_path = Path(db_path)
        if not self.db_path.is_absolute():
            root_dir = Path(__file__).resolve().parent.parent.parent
            self.db_path = root_dir / self.db_path

        self.poll_interval = poll_interval_sec
        self._lock = threading.Lock()
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None
        self._listeners: List[Callable[[ScheduledTask], None]] = []

        self._ensure_dir()
        self._init_tables()

    def _ensure_dir(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_tables(self):
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS scheduled_tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    message TEXT NOT NULL,
                    trigger_type TEXT NOT NULL,
                    due_timestamp REAL NOT NULL,
                    recurrence_interval_sec REAL,
                    payload TEXT DEFAULT '{}',
                    is_active INTEGER DEFAULT 1,
                    last_executed TEXT,
                    created_at TEXT NOT NULL
                );
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_tasks_due ON scheduled_tasks(due_timestamp, is_active);")
            conn.commit()

    # -------------------------------------------------------------------------
    # LISTENER REGISTRATION
    # -------------------------------------------------------------------------

    def register_listener(self, callback: Callable[[ScheduledTask], None]):
        """Registers a callback function to be called when a task triggers."""
        self._listeners.append(callback)

    def _dispatch_trigger(self, task: ScheduledTask):
        """Notifies all registered listeners and logs the trigger."""
        log_chitti(f"[CHITTI] [PROACTIVE ALERT] Reminder triggered: '{task.title}' - {task.message}")
        for listener in self._listeners:
            try:
                listener(task)
            except Exception as e:
                log_warn(f"[SCHEDULER] Listener error on task {task.id}: {e}")

    # -------------------------------------------------------------------------
    # SCHEDULING CRUD
    # -------------------------------------------------------------------------

    def schedule_reminder(
        self,
        title: str,
        message: str,
        due_in_seconds: Optional[float] = None,
        due_datetime: Optional[datetime] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> ScheduledTask:
        """Schedules a one-time reminder."""
        now_ts = time.time()
        if due_in_seconds is not None:
            due_ts = now_ts + due_in_seconds
        elif due_datetime is not None:
            due_ts = due_datetime.timestamp()
        else:
            due_ts = now_ts + 60.0  # Default 1 minute

        now_iso = get_utc_now_iso()
        task_payload = payload or {}

        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO scheduled_tasks (
                    title, message, trigger_type, due_timestamp, recurrence_interval_sec, payload, is_active, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                title.strip(),
                message.strip(),
                TriggerType.ONE_TIME_REMINDER.value,
                due_ts,
                None,
                json.dumps(task_payload),
                1,
                now_iso,
            ))
            conn.commit()
            task_id = cursor.lastrowid
            log_info(f"[SCHEDULER] Scheduled reminder #{task_id}: '{title}' in {due_ts - now_ts:.1f}s")
            return ScheduledTask(
                id=task_id,
                title=title.strip(),
                message=message.strip(),
                trigger_type=TriggerType.ONE_TIME_REMINDER,
                due_timestamp=due_ts,
                payload=task_payload,
                is_active=True,
                created_at=now_iso,
            )

    def schedule_interval(
        self,
        title: str,
        message: str,
        interval_seconds: float,
        payload: Optional[Dict[str, Any]] = None,
        trigger_type: TriggerType = TriggerType.RECURRING_INTERVAL,
    ) -> ScheduledTask:
        """Schedules a recurring interval task."""
        now_ts = time.time()
        due_ts = now_ts + interval_seconds
        now_iso = get_utc_now_iso()
        task_payload = payload or {}

        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO scheduled_tasks (
                    title, message, trigger_type, due_timestamp, recurrence_interval_sec, payload, is_active, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                title.strip(),
                message.strip(),
                trigger_type.value,
                due_ts,
                interval_seconds,
                json.dumps(task_payload),
                1,
                now_iso,
            ))
            conn.commit()
            task_id = cursor.lastrowid
            return ScheduledTask(
                id=task_id,
                title=title.strip(),
                message=message.strip(),
                trigger_type=trigger_type,
                due_timestamp=due_ts,
                recurrence_interval_sec=interval_seconds,
                payload=task_payload,
                is_active=True,
                created_at=now_iso,
            )

    def list_active_tasks(self) -> List[ScheduledTask]:
        """Returns all currently active tasks sorted by due timestamp."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM scheduled_tasks WHERE is_active = 1 ORDER BY due_timestamp ASC")
            rows = cursor.fetchall()
            return [
                ScheduledTask(
                    id=r["id"],
                    title=r["title"],
                    message=r["message"],
                    trigger_type=TriggerType(r["trigger_type"]),
                    due_timestamp=r["due_timestamp"],
                    recurrence_interval_sec=r["recurrence_interval_sec"],
                    payload=json.loads(r["payload"]) if r["payload"] else {},
                    is_active=bool(r["is_active"]),
                    last_executed=r["last_executed"],
                    created_at=r["created_at"],
                )
                for r in rows
            ]

    def cancel_task(self, task_id: int) -> bool:
        """Deactivates a scheduled task."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE scheduled_tasks SET is_active = 0 WHERE id = ?", (task_id,))
            conn.commit()
            return cursor.rowcount > 0

    # -------------------------------------------------------------------------
    # NATURAL LANGUAGE REMINDER PARSER
    # -------------------------------------------------------------------------

    @classmethod
    def parse_natural_language_reminder(cls, text: str) -> Optional[Dict[str, Any]]:
        """
        Parses expressions like:
        - "remind me in 10 minutes to drink water"
        - "remind me at 6 PM to commit my git changes"
        - "remind me in 2 hours to call Rohit"
        - "mujhe 5 minute me yaad dilana ki pani pina hai"
        """
        raw = text.strip()
        
        # Pattern 1: in X minutes/hours/seconds
        m_in = re.search(r"(?i)\b(?:remind\s+me|yaad\s+dilana)\s+(?:in|ke\s+baad)\s+(\d+)\s+(seconds?|minutes?|mins?|hours?|hrs?|sec)\s+(?:to\s+|ki\s+|that\s+)?(.*)", raw)
        if m_in:
            amount = int(m_in.group(1))
            unit = m_in.group(2).lower()
            task_msg = m_in.group(3).strip() or "Reminder"
            
            multipliers = {
                "s": 1, "sec": 1, "second": 1, "seconds": 1,
                "m": 60, "min": 60, "mins": 60, "minute": 60, "minutes": 60,
                "h": 3600, "hr": 3600, "hrs": 3600, "hour": 3600, "hours": 3600,
            }
            seconds = amount * multipliers.get(unit, 60)
            return {
                "title": task_msg.title()[:40],
                "message": task_msg,
                "due_in_seconds": seconds,
            }

        # Pattern 2: at X PM/AM
        m_at = re.search(r"(?i)\b(?:remind\s+me|yaad\s+dilana)\s+at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s+(?:to\s+|ki\s+|that\s+)?(.*)", raw)
        if m_at:
            hour = int(m_at.group(1))
            minute = int(m_at.group(2)) if m_at.group(2) else 0
            meridiem = (m_at.group(3) or "").lower()
            task_msg = m_at.group(4).strip() or "Reminder"

            if meridiem == "pm" and hour < 12:
                hour += 12
            elif meridiem == "am" and hour == 12:
                hour = 0

            now = datetime.now()
            target_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if target_time <= now:
                target_time += timedelta(days=1)

            diff_sec = (target_time - now).total_seconds()
            return {
                "title": task_msg.title()[:40],
                "message": task_msg,
                "due_in_seconds": diff_sec,
            }

        # Pattern 3: Generic reminder directive ("remind me to X")
        m_gen = re.search(r"(?i)\b(?:remind\s+me|yaad\s+dilana)\s+(?:to\s+|ki\s+|that\s+)?(.*)", raw)
        if m_gen:
            task_msg = m_gen.group(1).strip()
            if task_msg:
                return {
                    "title": task_msg.title()[:40],
                    "message": task_msg,
                    "due_in_seconds": 300.0,  # Default 5 minutes
                }

        return None

    # -------------------------------------------------------------------------
    # BACKGROUND DAEMON LOOP
    # -------------------------------------------------------------------------

    def start(self):
        """Starts the background worker thread."""
        if self._running:
            return
        self._running = True
        self._worker_thread = threading.Thread(target=self._run_loop, daemon=True, name="ProactiveSchedulerThread")
        self._worker_thread.start()
        log_info("[SCHEDULER] Proactive Scheduler background daemon started.")

    def stop(self):
        """Stops the background worker thread."""
        self._running = False
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
        log_info("[SCHEDULER] Proactive Scheduler background daemon stopped.")

    def _run_loop(self):
        """Main daemon loop checking for due tasks."""
        while self._running:
            try:
                self._check_and_execute_due_tasks()
            except Exception as e:
                log_error(f"[SCHEDULER] Error in scheduler worker loop: {e}")
            time.sleep(self.poll_interval)

    def _check_and_execute_due_tasks(self):
        """Finds due tasks and executes them."""
        now_ts = time.time()
        now_iso = get_utc_now_iso()
        due_tasks: List[ScheduledTask] = []

        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM scheduled_tasks WHERE is_active = 1 AND due_timestamp <= ?", (now_ts,))
            rows = cursor.fetchall()
            for r in rows:
                due_tasks.append(ScheduledTask(
                    id=r["id"],
                    title=r["title"],
                    message=r["message"],
                    trigger_type=TriggerType(r["trigger_type"]),
                    due_timestamp=r["due_timestamp"],
                    recurrence_interval_sec=r["recurrence_interval_sec"],
                    payload=json.loads(r["payload"]) if r["payload"] else {},
                    is_active=bool(r["is_active"]),
                    last_executed=r["last_executed"],
                    created_at=r["created_at"],
                ))

            # Update database status
            for t in due_tasks:
                if t.recurrence_interval_sec and t.recurrence_interval_sec > 0:
                    next_due = now_ts + t.recurrence_interval_sec
                    cursor.execute("""
                        UPDATE scheduled_tasks
                        SET due_timestamp = ?, last_executed = ?
                        WHERE id = ?
                    """, (next_due, now_iso, t.id))
                else:
                    cursor.execute("""
                        UPDATE scheduled_tasks
                        SET is_active = 0, last_executed = ?
                        WHERE id = ?
                    """, (now_iso, t.id))
            conn.commit()

        # Dispatch triggers outside database lock
        for t in due_tasks:
            self._dispatch_trigger(t)
