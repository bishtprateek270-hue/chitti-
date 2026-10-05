"""
Chitti Floating Desktop Dynamic Island HUD Overlay (Phase 4 & 5).
Provides a frameless, glassmorphism-styled floating pill widget on top of all windows
with full keyboard focus, instant text typing, voice capture, and ambient status.
"""

import os
import queue
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from src.ui.visualizer import HUDMode, HUDState, HUDStepInfo, HUD_THEME
from src.utils.logging import log_chitti, log_debug, log_info, log_warn


class FloatingHUD:
    """
    Lightweight, always-on-top floating dynamic island widget with 100% reliable Windows keyboard focus.
    """

    PLACEHOLDER_TEXT = "Type a command here (or say 'Hey Chitti')..."

    def __init__(self, on_command_submit: Optional[Callable[[str], None]] = None):
        self.on_command_submit = on_command_submit
        self.state = HUDState()
        self._queue: queue.Queue = queue.Queue()
        self._root = None
        self._is_running = False
        self._thread: Optional[threading.Thread] = None
        self._has_placeholder = True

        # GUI elements
        self._status_lbl = None
        self._mode_dot = None
        self._entry = None
        self._progress_lbl = None

    @property
    def is_visible(self) -> bool:
        return self.state.is_visible

    # -------------------------------------------------------------------------
    # THREAD-SAFE STATE UPDATE APIS
    # -------------------------------------------------------------------------

    def set_mode(self, mode: HUDMode, status_text: Optional[str] = None):
        """Updates the visual mode and status text."""
        self._queue.put(("set_mode", mode, status_text))

    def set_progress(self, current_step_or_desc: Any = "", total_steps: int = 0, step_desc: str = ""):
        """Updates the live execution progress bar and step description."""
        if isinstance(current_step_or_desc, str) and total_steps == 0:
            desc = current_step_or_desc
            cur, tot = 0, 0
        else:
            cur = int(current_step_or_desc) if isinstance(current_step_or_desc, (int, float)) else 0
            tot = total_steps
            desc = step_desc
        self._queue.put(("set_progress", cur, tot, desc))

    def set_response(self, text: str):
        """Displays assistant response text in the HUD."""
        self._queue.put(("set_response", text))

    def show(self):
        """Makes the HUD visible on top of all windows."""
        self._queue.put(("show",))

    def hide(self):
        """Hides the HUD from view."""
        self._queue.put(("hide",))

    def toggle(self):
        """Toggles visibility between shown and hidden."""
        self._queue.put(("toggle",))

    # -------------------------------------------------------------------------
    # LIFECYCLE & THREAD MANAGEMENT
    # -------------------------------------------------------------------------

    def start(self, non_blocking: bool = True):
        """Starts the HUD window."""
        if self._is_running:
            return

        self._is_running = True
        self._ready_event = threading.Event()
        if non_blocking:
            self._thread = threading.Thread(target=self._run_gui, daemon=True, name="ChittiHUDThread")
            self._thread.start()
            self._ready_event.wait(timeout=3.0)
            log_info("[HUD] Floating HUD background thread started.")
        else:
            self._run_gui()

    def stop(self):
        """Terminates the HUD window."""
        self._is_running = False
        self._queue.put(("destroy",))
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        log_info("[HUD] Floating HUD stopped.")

    # -------------------------------------------------------------------------
    # TKINTER GUI IMPLEMENTATION
    # -------------------------------------------------------------------------

    def _run_gui(self):
        """Initializes and runs the Tkinter mainloop with dark glassmorphism styling."""
        try:
            import tkinter as tk
            from tkinter import ttk
        except ImportError:
            log_warn("[HUD] Tkinter is not available. HUD running in headless mock mode.")
            if hasattr(self, "_ready_event"):
                self._ready_event.set()
            return

        try:
            self._root = tk.Tk()
            self._root.title("Chitti Dynamic Island")
            self._root.attributes("-topmost", True)

            # Center top positioning: 640x165 pill at top center of screen
            screen_w = self._root.winfo_screenwidth()
            hud_w, hud_h = 640, 165
            hud_x = (screen_w - hud_w) // 2
            hud_y = 25
            self._root.geometry(f"{hud_w}x{hud_h}+{hud_x}+{hud_y}")
            self._root.configure(bg=HUD_THEME["bg_dark"])
            if sys.platform == "win32":
                try:
                    self._root.attributes("-toolwindow", True)
                except Exception:
                    pass

            # Outer border container
            border_frame = tk.Frame(
                self._root,
                bg=HUD_THEME["border"],
                padx=1,
                pady=1,
            )
            border_frame.pack(fill="both", expand=True, padx=2, pady=2)

            # Main Pill Background Card
            card = tk.Frame(border_frame, bg=HUD_THEME["bg_card"], padx=14, pady=10)
            card.pack(fill="both", expand=True)

            # TOP ROW: Status Indicator & Mode
            top_row = tk.Frame(card, bg=HUD_THEME["bg_card"])
            top_row.pack(fill="x", pady=(0, 6))

            self._mode_dot = tk.Label(
                top_row,
                text="🤖",
                font=("Segoe UI Emoji", 12),
                bg=HUD_THEME["bg_card"],
                fg=HUD_THEME["accent_blue"],
            )
            self._mode_dot.pack(side="left", padx=(0, 8))

            self._status_lbl = tk.Label(
                top_row,
                text="Chitti Desktop AI Companion (Alt+Space)",
                font=("Segoe UI", 10, "bold"),
                bg=HUD_THEME["bg_card"],
                fg=HUD_THEME["text_primary"],
                anchor="w",
            )
            self._status_lbl.pack(side="left", fill="x", expand=True)

            # Close / Dismiss Button
            close_btn = tk.Label(
                top_row,
                text="✕",
                font=("Segoe UI", 10, "bold"),
                bg=HUD_THEME["bg_card"],
                fg=HUD_THEME["text_secondary"],
                cursor="hand2",
            )
            close_btn.pack(side="right")
            close_btn.bind("<Button-1>", lambda e: self.hide())

            # MIDDLE ROW: Quick Input Entry with Active Focus
            entry_frame = tk.Frame(card, bg=HUD_THEME["bg_input"], padx=10, pady=5)
            entry_frame.pack(fill="x", pady=(0, 6))

            self._entry = tk.Entry(
                entry_frame,
                font=("Segoe UI", 11),
                bg=HUD_THEME["bg_input"],
                fg=HUD_THEME["text_secondary"],
                insertbackground=HUD_THEME["accent_blue"],
                insertwidth=2,
                relief="flat",
                highlightthickness=0,
            )
            self._entry.insert(0, self.PLACEHOLDER_TEXT)
            self._has_placeholder = True
            self._entry.pack(fill="x", side="left", expand=True)

            def on_entry_focus_in(event=None):
                if self._has_placeholder or self._entry.get() == self.PLACEHOLDER_TEXT:
                    self._entry.delete(0, "end")
                    self._entry.config(fg=HUD_THEME["text_primary"])
                    self._has_placeholder = False
                self._entry.focus_set()

            def on_entry_focus_out(event=None):
                if not self._entry.get().strip():
                    self._entry.delete(0, "end")
                    self._entry.insert(0, self.PLACEHOLDER_TEXT)
                    self._entry.config(fg=HUD_THEME["text_secondary"])
                    self._has_placeholder = True

            def on_entry_key_down(event=None):
                if self._has_placeholder or self._entry.get() == self.PLACEHOLDER_TEXT:
                    self._entry.delete(0, "end")
                    self._entry.config(fg=HUD_THEME["text_primary"])
                    self._has_placeholder = False

            self._entry.bind("<FocusIn>", on_entry_focus_in)
            self._entry.bind("<FocusOut>", on_entry_focus_out)
            self._entry.bind("<Button-1>", on_entry_focus_in)
            self._entry.bind("<KeyPress>", on_entry_key_down)
            self._entry.bind("<Return>", self._on_enter_pressed)
            entry_frame.bind("<Button-1>", on_entry_focus_in)
            self._root.bind("<Escape>", lambda e: self.hide())

            # Send Button
            send_btn = tk.Label(
                entry_frame,
                text="➔",
                font=("Segoe UI", 11, "bold"),
                bg=HUD_THEME["bg_input"],
                fg=HUD_THEME["accent_blue"],
                cursor="hand2",
                padx=4,
            )
            send_btn.pack(side="right")
            send_btn.bind("<Button-1>", lambda e: self._on_enter_pressed(None))

            # BOTTOM ROW: Quick Action Chips
            chips_row = tk.Frame(card, bg=HUD_THEME["bg_card"])
            chips_row.pack(fill="x", pady=(0, 4))

            def add_chip(label: str, cmd: str):
                chip = tk.Label(
                    chips_row,
                    text=label,
                    font=("Segoe UI", 8),
                    bg=HUD_THEME["bg_input"],
                    fg=HUD_THEME["accent_cyan"],
                    padx=6,
                    pady=2,
                    cursor="hand2",
                )
                chip.pack(side="left", padx=(0, 6))
                chip.bind("<Button-1>", lambda e: self._submit_text(cmd))

            add_chip("👁️ Screen Info", "look at my screen")
            add_chip("📄 Doc Summary", "summarize the document on screen")
            add_chip("🔁 Self Heal", "run tests and fix whatever is broken")
            add_chip("🧠 Memories", "list my memories")
            add_chip("🗺️ Roadmap", "roadmap k bare m batao")

            # Progress Bar Frame (initially empty)
            self._progress_lbl = tk.Label(
                card,
                text="",
                font=("Segoe UI", 8),
                bg=HUD_THEME["bg_card"],
                fg=HUD_THEME["text_secondary"],
                anchor="w",
            )
            self._progress_lbl.pack(fill="x", pady=(2, 0))

            # Start message queue polling
            self._root.after(50, self._process_queue)
            self.state.is_visible = True
            if hasattr(self, "_ready_event"):
                self._ready_event.set()
            self._root.mainloop()

        except Exception as e:
            log_warn(f"[HUD] Tkinter execution notice: {e}")

    def _process_queue(self):
        """Processes thread-safe message updates on the Tkinter mainloop."""
        if not self._root:
            return

        try:
            while not self._queue.empty():
                item = self._queue.get_nowait()
                msg_type = item[0]

                if msg_type == "set_mode":
                    mode, text = item[1], item[2]
                    self.state.mode = mode
                    if text:
                        self.state.status_text = text
                    if self._mode_dot:
                        self._mode_dot.config(text=self.state.mode_icon, fg=self.state.mode_color)
                    if self._status_lbl and text:
                        self._status_lbl.config(text=text)

                elif msg_type == "set_progress":
                    cur, tot, desc = item[1], item[2], item[3]
                    self.state.current_step = cur
                    self.state.total_steps = tot
                    if self._progress_lbl:
                        txt = f"[Step {cur}/{tot}] {desc}" if tot > 0 else desc
                        self._progress_lbl.config(text=txt, fg=HUD_THEME["accent_orange"])

                elif msg_type == "set_response":
                    resp_text = item[1]
                    self.state.last_response = resp_text
                    if self._status_lbl:
                        self._status_lbl.config(text=resp_text[:60] + "..." if len(resp_text) > 60 else resp_text)

                elif msg_type == "show":
                    self.state.is_visible = True
                    if self._root:
                        self._root.deiconify()
                        self._root.lift()
                        self._root.attributes("-topmost", True)
                        if sys.platform == "win32":
                            try:
                                import ctypes
                                top_hwnd = ctypes.windll.user32.GetParent(self._root.winfo_id()) or self._root.winfo_id()
                                fg_hwnd = ctypes.windll.user32.GetForegroundWindow()
                                fg_thread = ctypes.windll.user32.GetWindowThreadProcessId(fg_hwnd, None)
                                cur_thread = ctypes.windll.kernel32.GetCurrentThreadId()
                                if fg_thread != cur_thread:
                                    ctypes.windll.user32.AttachThreadInput(cur_thread, fg_thread, True)
                                    ctypes.windll.user32.SetForegroundWindow(top_hwnd)
                                    ctypes.windll.user32.SetFocus(top_hwnd)
                                    ctypes.windll.user32.AttachThreadInput(cur_thread, fg_thread, False)
                                else:
                                    ctypes.windll.user32.SetForegroundWindow(top_hwnd)
                                    ctypes.windll.user32.SetFocus(top_hwnd)
                            except Exception:
                                pass
                        if self._entry:
                            self._entry.focus_set()
                            self._entry.focus_force()

                elif msg_type == "hide":
                    self.state.is_visible = False
                    if self._root:
                        self._root.withdraw()

                elif msg_type == "toggle":
                    is_currently_shown = bool(self._root.winfo_viewable()) if self._root else self.state.is_visible
                    if is_currently_shown:
                        self.state.is_visible = False
                        if self._root:
                            self._root.withdraw()
                    else:
                        self.state.is_visible = True
                        if self._root:
                            self._root.deiconify()
                            self._root.lift()
                            self._root.attributes("-topmost", True)
                            if sys.platform == "win32":
                                try:
                                    import ctypes
                                    top_hwnd = ctypes.windll.user32.GetParent(self._root.winfo_id()) or self._root.winfo_id()
                                    fg_hwnd = ctypes.windll.user32.GetForegroundWindow()
                                    fg_thread = ctypes.windll.user32.GetWindowThreadProcessId(fg_hwnd, None)
                                    cur_thread = ctypes.windll.kernel32.GetCurrentThreadId()
                                    if fg_thread != cur_thread:
                                        ctypes.windll.user32.AttachThreadInput(cur_thread, fg_thread, True)
                                        ctypes.windll.user32.SetForegroundWindow(top_hwnd)
                                        ctypes.windll.user32.SetFocus(top_hwnd)
                                        ctypes.windll.user32.AttachThreadInput(cur_thread, fg_thread, False)
                                    else:
                                        ctypes.windll.user32.SetForegroundWindow(top_hwnd)
                                        ctypes.windll.user32.SetFocus(top_hwnd)
                                except Exception:
                                    pass
                            if self._entry:
                                self._entry.focus_set()
                                self._entry.focus_force()

                elif msg_type == "destroy":
                    if self._root:
                        self._root.quit()
                        self._root.destroy()
                        self._root = None
                    return

        except Exception as e:
            log_debug(f"[HUD] Queue processing notice: {e}")

        if self._root:
            self._root.after(50, self._process_queue)

    def _on_enter_pressed(self, event=None):
        """Handles Enter key or Send button in the quick input box."""
        if self._entry:
            text = self._entry.get().strip()
            if text == self.PLACEHOLDER_TEXT:
                text = ""
            self._entry.delete(0, "end")
            self._entry.config(fg=HUD_THEME["text_primary"])
            self._has_placeholder = False
            if text:
                self._submit_text(text)

    def _submit_text(self, text: str):
        """Dispatches submitted command to the callback listener."""
        log_chitti(f"[CHITTI] [HUD] User submitted command: '{text}'")
        self.set_mode(HUDMode.THINKING, f"Processing: '{text}'")
        if self.on_command_submit:
            threading.Thread(target=self.on_command_submit, args=(text,), daemon=True).start()
