"""
Chitti Centralized State Manager & Global Interruption Architecture.
Acts as the single authoritative source of truth for Chitti's lifecycle and operational states,
coordinating high-priority interrupts, request generation invalidation, cancellation tokens,
background task cleanup, and thread synchronization.
"""

import enum
import logging
import subprocess
import threading
import time
from typing import Callable, Dict, List, Optional, Set

from src.utils.logging import log_chitti, log_debug, log_info, log_warning


class SystemState(str, enum.Enum):
    """Authoritative lifecycle states of Chitti."""
    STARTING = "STARTING"
    SLEEPING = "SLEEPING"
    AWAKE = "AWAKE"
    LISTENING = "LISTENING"
    PROCESSING = "PROCESSING"
    EXECUTING = "EXECUTING"
    SPEAKING = "SPEAKING"
    INTERRUPTING = "INTERRUPTING"
    STOPPING = "STOPPING"


class CommandPriority(int, enum.Enum):
    """Global Command Priority Hierarchy."""
    PRIORITY_0_SHUTDOWN = 0
    PRIORITY_1_INTERRUPT = 1
    PRIORITY_2_SLEEP = 2
    PRIORITY_3_WAKE = 3
    PRIORITY_4_USER_COMMAND = 4
    PRIORITY_5_BACKGROUND_TASKS = 5


class SystemStateManager:
    """
    Thread-safe, authoritative state manager for Chitti.
    Enforces valid state transitions and guarantees high-priority immediate cancellation.
    """

    LEGAL_TRANSITIONS: Dict[SystemState, Set[SystemState]] = {
        SystemState.STARTING: {SystemState.SLEEPING, SystemState.AWAKE, SystemState.STOPPING},
        SystemState.SLEEPING: {SystemState.AWAKE, SystemState.INTERRUPTING, SystemState.STOPPING},
        SystemState.AWAKE: {
            SystemState.LISTENING,
            SystemState.PROCESSING,
            SystemState.EXECUTING,
            SystemState.SPEAKING,
            SystemState.SLEEPING,
            SystemState.INTERRUPTING,
            SystemState.STOPPING,
        },
        SystemState.LISTENING: {
            SystemState.PROCESSING,
            SystemState.AWAKE,
            SystemState.SLEEPING,
            SystemState.INTERRUPTING,
            SystemState.STOPPING,
        },
        SystemState.PROCESSING: {
            SystemState.EXECUTING,
            SystemState.SPEAKING,
            SystemState.AWAKE,
            SystemState.SLEEPING,
            SystemState.INTERRUPTING,
            SystemState.STOPPING,
        },
        SystemState.EXECUTING: {
            SystemState.SPEAKING,
            SystemState.AWAKE,
            SystemState.SLEEPING,
            SystemState.INTERRUPTING,
            SystemState.STOPPING,
        },
        SystemState.SPEAKING: {
            SystemState.LISTENING,
            SystemState.AWAKE,
            SystemState.SLEEPING,
            SystemState.INTERRUPTING,
            SystemState.STOPPING,
        },
        SystemState.INTERRUPTING: {SystemState.AWAKE, SystemState.SLEEPING, SystemState.STOPPING},
        SystemState.STOPPING: set(),
    }

    def __init__(self, initial_state: SystemState = SystemState.AWAKE):
        self._state = initial_state
        self._lock = threading.RLock()

        # Cancellation synchronization primitives
        self._interrupt_event = threading.Event()
        self._task_cancel_event = threading.Event()
        self._tts_cancel_event = threading.Event()

        # Request generation counter for invalidating stale tasks / TTS
        self._current_generation_id = 0

        # Registered active subprocesses
        self._active_processes: Set[subprocess.Popen] = set()
        self._process_lock = threading.Lock()

        # Callbacks
        self._on_interrupt_callbacks: List[Callable[[], None]] = []
        self._on_sleep_callbacks: List[Callable[[], None]] = []
        self._on_wake_callbacks: List[Callable[[], None]] = []
        self._on_state_change_callbacks: List[Callable[[SystemState, SystemState], None]] = []

        # Operating mode: "wake_word" (Mode A) vs "continuous" (Mode B)
        self._operating_mode = "wake_word"

    @property
    def operating_mode(self) -> str:
        with self._lock:
            return self._operating_mode

    @operating_mode.setter
    def operating_mode(self, mode: str):
        with self._lock:
            self._operating_mode = "continuous" if mode.lower().startswith("cont") else "wake_word"
            log_info(f"[CHITTI][CONFIG] Operating mode set to: {self._operating_mode.upper()}")

    # ---------------- Request Generation Protocol ----------------

    def new_request(self) -> int:
        """
        Starts a new user request session.
        Increments request generation ID and resets interrupt signals.
        """
        with self._lock:
            self._current_generation_id += 1
            gen_id = self._current_generation_id
            self.reset_interrupt()
            log_info(f"[CHITTI][REQUEST] Started Request #{gen_id}")
            return gen_id

    def get_current_generation(self) -> int:
        """Returns the current request generation ID."""
        with self._lock:
            return self._current_generation_id

    def invalidate_generation(self, gen_id: Optional[int] = None) -> int:
        """
        Atomically invalidates the active request generation.
        Any worker or TTS with an earlier generation ID will be discarded.
        """
        with self._lock:
            old_id = self._current_generation_id
            self._current_generation_id += 1
            new_id = self._current_generation_id
            log_info(f"[CHITTI][CONTROL] Request #{old_id} invalidated -> #{new_id}")
            return new_id

    def is_generation_valid(self, gen_id: Optional[int] = None) -> bool:
        """
        Validates whether a generation is still active and valid.
        Returns False if the system is sleeping, interrupted, or gen_id is superseded.
        """
        with self._lock:
            if self._state == SystemState.SLEEPING:
                return False
            if self._interrupt_event.is_set() or self._task_cancel_event.is_set():
                return False
            if gen_id is not None and gen_id != self._current_generation_id:
                return False
            return True

    def can_speak(self, gen_id: Optional[int] = None) -> bool:
        """Returns True if speech playback is permitted for the given generation."""
        with self._lock:
            if self._state == SystemState.SLEEPING:
                return False
            if self._tts_cancel_event.is_set() or self._interrupt_event.is_set():
                return False
            if gen_id is not None and gen_id != self._current_generation_id:
                return False
            return True

    # ---------------- Authoritative State Accessors ----------------

    def get_state(self) -> SystemState:
        """Returns the current authoritative state."""
        with self._lock:
            return self._state

    def is_sleeping(self) -> bool:
        """Returns True if Chitti is currently sleeping."""
        with self._lock:
            return self._state == SystemState.SLEEPING

    def is_awake(self) -> bool:
        """Returns True if Chitti is awake and able to interact."""
        with self._lock:
            return self._state not in (SystemState.SLEEPING, SystemState.STARTING, SystemState.STOPPING)

    def is_interrupted(self) -> bool:
        """Returns True if a high-priority cancellation or interruption is active."""
        return self._interrupt_event.is_set() or self._task_cancel_event.is_set()

    def get_cancellation_event(self) -> threading.Event:
        """Returns the shared threading.Event used as a cancellation token."""
        return self._task_cancel_event

    def get_tts_cancellation_event(self) -> threading.Event:
        """Returns the shared event used to immediately abort ongoing TTS."""
        return self._tts_cancel_event

    # ---------------- State Transition Callbacks ----------------

    def register_on_interrupt(self, callback: Callable[[], None]):
        """Registers a callback executed immediately upon global interrupt."""
        with self._lock:
            if callback not in self._on_interrupt_callbacks:
                self._on_interrupt_callbacks.append(callback)

    def register_on_sleep(self, callback: Callable[[], None]):
        """Registers a callback executed when entering sleep state."""
        with self._lock:
            if callback not in self._on_sleep_callbacks:
                self._on_sleep_callbacks.append(callback)

    def register_on_wake(self, callback: Callable[[], None]):
        """Registers a callback executed when waking up."""
        with self._lock:
            if callback not in self._on_wake_callbacks:
                self._on_wake_callbacks.append(callback)

    def register_on_state_change(self, callback: Callable[[SystemState, SystemState], None]):
        """Registers a callback for state transition events."""
        with self._lock:
            if callback not in self._on_state_change_callbacks:
                self._on_state_change_callbacks.append(callback)

    def set_state(self, new_state: SystemState) -> bool:
        """
        Thread-safely transitions to a new state if valid.
        Logs state transitions in standard format [CHITTI][STATE] OLD -> NEW.
        """
        with self._lock:
            old_state = self._state
            if old_state == new_state:
                return True

            # Reject changing SLEEPING to active states without waking
            if old_state == SystemState.SLEEPING and new_state in (
                SystemState.EXECUTING,
                SystemState.PROCESSING,
                SystemState.SPEAKING,
                SystemState.LISTENING,
            ):
                log_warning(
                    f"[CHITTI][STATE] Rejected invalid transition: {old_state.value} -> {new_state.value} "
                    f"(System is SLEEPING; must wake first)"
                )
                return False

            self._state = new_state
            log_info(f"[CHITTI][STATE] {old_state.value} -> {new_state.value}")

            # Notify listeners outside the lock
            callbacks = list(self._on_state_change_callbacks)

        for cb in callbacks:
            try:
                cb(old_state, new_state)
            except Exception as e:
                log_debug(f"[CHITTI][STATE] Callback error: {e}")

        return True

    # ---------------- Explicit State Lifecycle Methods ----------------

    def wake(self) -> bool:
        """
        Transitions Chitti from SLEEPING to AWAKE state.
        Resets cancellation signals and notifies wake listeners.
        """
        with self._lock:
            if self._state == SystemState.AWAKE:
                return True

            self.reset_interrupt()
            old_state = self._state
            self._state = SystemState.AWAKE
            log_chitti(f"[CHITTI][STATE] {old_state.value} -> AWAKE")
            callbacks = list(self._on_wake_callbacks)

        for cb in callbacks:
            try:
                cb()
            except Exception as e:
                log_debug(f"[CHITTI][WAKE] Wake callback notice: {e}")

        return True

    def sleep(self) -> bool:
        """
        Enters SLEEPING state:
        1. Immediately cancels current tasks and invalidates generation ID.
        2. Halts TTS speech playback.
        3. Clears pending speech and execution.
        4. Transitions state to SLEEPING.
        """
        log_chitti("[CHITTI][STATE] Entering SLEEP state...")
        self.interrupt(reason="sleep")

        with self._lock:
            old_state = self._state
            self._state = SystemState.SLEEPING
            log_info(f"[CHITTI][STATE] {old_state.value} -> SLEEPING")
            callbacks = list(self._on_sleep_callbacks)

        for cb in callbacks:
            try:
                cb()
            except Exception as e:
                log_debug(f"[CHITTI][SLEEP] Sleep callback notice: {e}")

        return True

    def begin_listening(self) -> bool:
        """Transitions state to LISTENING. Rejects if SLEEPING."""
        with self._lock:
            if self._state == SystemState.SLEEPING:
                log_warning("[CHITTI][STATE] Rejected begin_listening: System is SLEEPING.")
                return False
            return self.set_state(SystemState.LISTENING)

    def begin_processing(self, gen_id: Optional[int] = None) -> bool:
        """Transitions state to PROCESSING. Verifies generation ID and sleeping state."""
        with self._lock:
            if not self.is_generation_valid(gen_id):
                log_warning(f"[CHITTI][STATE] Rejected begin_processing: Request {gen_id} is invalid or sleeping.")
                return False
            return self.set_state(SystemState.PROCESSING)

    def begin_execution(self, gen_id: Optional[int] = None) -> bool:
        """Transitions state to EXECUTING. Verifies generation ID and sleeping state."""
        with self._lock:
            if not self.is_generation_valid(gen_id):
                log_warning(f"[CHITTI][STATE] Rejected begin_execution: Request {gen_id} is invalid or sleeping.")
                return False
            return self.set_state(SystemState.EXECUTING)

    def begin_speaking(self, gen_id: Optional[int] = None) -> bool:
        """Transitions state to SPEAKING. Strictly rejects if sleeping or interrupted."""
        with self._lock:
            if not self.can_speak(gen_id):
                log_warning(f"[CHITTI][STATE] Rejected begin_speaking: Speech disallowed for gen #{gen_id} (state={self._state.value})")
                return False
            return self.set_state(SystemState.SPEAKING)

    def finish_task(self, gen_id: Optional[int] = None) -> bool:
        """Concludes active processing or speaking, returning to AWAKE or configured state."""
        with self._lock:
            if self._state == SystemState.SLEEPING:
                return True
            if self._state in (SystemState.SPEAKING, SystemState.EXECUTING, SystemState.PROCESSING, SystemState.LISTENING):
                next_state = SystemState.LISTENING if self._operating_mode == "continuous" else SystemState.AWAKE
                return self.set_state(next_state)
            return True

    def shutdown(self) -> bool:
        """Transitions state to STOPPING and releases resources."""
        log_info("[CHITTI][STATE] Shutting down...")
        self.interrupt(reason="shutdown")
        with self._lock:
            self._state = SystemState.STOPPING
        return True

    def interrupt(self, reason: str = "user_stop") -> bool:
        """
        HIGH-PRIORITY GLOBAL INTERRUPT (Priority 1).
        Instantly halts:
        1. Ongoing TTS playback.
        2. Cancellable active tasks, computer actions, LLM requests.
        3. Terminating registered child subprocesses.
        4. Invalidates the active request generation ID.
        """
        log_chitti(f"[CHITTI][INTERRUPT] 🛑 Global cancellation signal issued (Reason: {reason})")

        # 1. Invalidate request generation ID atomically
        self.invalidate_generation()

        # 2. Set interruption flags atomically
        self._interrupt_event.set()
        self._task_cancel_event.set()
        self._tts_cancel_event.set()

        # 3. Terminate any active child subprocesses
        self.terminate_active_processes()

        # 4. Transition to INTERRUPTING
        with self._lock:
            old_state = self._state
            self._state = SystemState.INTERRUPTING
            log_info(f"[CHITTI][STATE] {old_state.value} -> INTERRUPTING")
            callbacks = list(self._on_interrupt_callbacks)

        # 5. Trigger registered interrupt handlers immediately
        for cb in callbacks:
            try:
                cb()
            except Exception as e:
                log_debug(f"[CHITTI][INTERRUPT] Interrupt callback notice: {e}")

        # 6. Transition to resting state (SLEEPING if we were sleeping or sleep requested, AWAKE otherwise)
        with self._lock:
            next_state = SystemState.SLEEPING if reason == "sleep" else SystemState.AWAKE
            log_info(f"[CHITTI][STATE] INTERRUPTING -> {next_state.value}")
            self._state = next_state

        return True

    def reset_interrupt(self):
        """Clears interruption and cancellation flags so new tasks can run."""
        self._interrupt_event.clear()
        self._task_cancel_event.clear()
        self._tts_cancel_event.clear()
        log_debug("[CHITTI][INTERRUPT] Cancellation flags reset.")

    def cancel_current_task(self) -> bool:
        """Explicitly cancels the active background or agent task."""
        log_info("[CHITTI][TASK] Current task cancelled.")
        self.invalidate_generation()
        self._task_cancel_event.set()
        self.terminate_active_processes()
        return True

    def stop_speaking(self) -> bool:
        """Halts TTS playback immediately."""
        log_info("[CHITTI][TTS] Playback stopped.")
        self._tts_cancel_event.set()
        return True

    # ---------------- Subprocess Tracking & Cleanup ----------------

    def register_process(self, proc: subprocess.Popen):
        """Registers a child subprocess so it can be terminated on interrupt."""
        with self._process_lock:
            self._active_processes.add(proc)

    def unregister_process(self, proc: subprocess.Popen):
        """Unregisters a completed subprocess."""
        with self._process_lock:
            self._active_processes.discard(proc)

    def terminate_active_processes(self):
        """Gracefully terminates registered subprocesses, escalating to kill if needed."""
        with self._process_lock:
            procs = list(self._active_processes)
            self._active_processes.clear()

        for proc in procs:
            if proc.poll() is None:  # Still running
                try:
                    log_info(f"[CHITTI][PROCESS] Terminating child process PID {proc.pid}...")
                    proc.terminate()
                    try:
                        proc.wait(timeout=0.5)
                    except subprocess.TimeoutExpired:
                        log_warning(f"[CHITTI][PROCESS] Process PID {proc.pid} did not exit in 0.5s. Killing...")
                        proc.kill()
                        proc.wait(timeout=0.5)
                except Exception as e:
                    log_debug(f"[CHITTI][PROCESS] Process termination notice: {e}")


# Global Singleton Instance
_STATE_MANAGER: Optional[SystemStateManager] = None
_STATE_LOCK = threading.Lock()


def get_state_manager() -> SystemStateManager:
    """Returns the application-wide authoritative State Manager instance."""
    global _STATE_MANAGER
    if _STATE_MANAGER is None:
        with _STATE_LOCK:
            if _STATE_MANAGER is None:
                _STATE_MANAGER = SystemStateManager()
    return _STATE_MANAGER
