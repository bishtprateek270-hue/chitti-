"""
Chitti Structured Computer-Use Actions.
Defines atomic computer-use action commands, parameters, and results.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from src.computer_use.state import RiskLevel, VerificationStatus


class ComputerActionType(str, Enum):
    OPEN_APPLICATION = "OPEN_APPLICATION"
    CLOSE_APPLICATION = "CLOSE_APPLICATION"
    OPEN_URL = "OPEN_URL"
    SEARCH_WEB = "SEARCH_WEB"
    SEARCH_YOUTUBE = "SEARCH_YOUTUBE"
    PLAY_SONG = "PLAY_SONG"
    INSPECT_SCREEN = "INSPECT_SCREEN"
    READ_SCREEN = "READ_SCREEN"
    FIND_UI_ELEMENT = "FIND_UI_ELEMENT"
    CLICK = "CLICK"
    DOUBLE_CLICK = "DOUBLE_CLICK"
    RIGHT_CLICK = "RIGHT_CLICK"
    MOVE_MOUSE = "MOVE_MOUSE"
    SCROLL = "SCROLL"
    DRAG = "DRAG"
    WAIT = "WAIT"
    WAIT_FOR_UI = "WAIT_FOR_UI"
    TYPE_TEXT = "TYPE_TEXT"
    PRESS_KEY = "PRESS_KEY"
    HOTKEY = "HOTKEY"
    COPY = "COPY"
    PASTE = "PASTE"
    CLIPBOARD_READ = "CLIPBOARD_READ"
    CLIPBOARD_WRITE = "CLIPBOARD_WRITE"
    FOCUS_WINDOW = "FOCUS_WINDOW"
    GET_ACTIVE_WINDOW = "GET_ACTIVE_WINDOW"
    VERIFY_UI_STATE = "VERIFY_UI_STATE"
    VERIFY_TEXT = "VERIFY_TEXT"
    VERIFY_ELEMENT = "VERIFY_ELEMENT"
    VERIFY_APPLICATION_STATE = "VERIFY_APPLICATION_STATE"
    CHECK_AUTHENTICATION = "CHECK_AUTHENTICATION"
    SEARCH_CONTACT = "SEARCH_CONTACT"
    SELECT_CONVERSATION = "SELECT_CONVERSATION"
    SEND_MESSAGE = "SEND_MESSAGE"
    VERIFY_MESSAGE_SENT = "VERIFY_MESSAGE_SENT"
    COMPOSE_EMAIL = "COMPOSE_EMAIL"
    CONFIRM_SEND = "CONFIRM_SEND"
    SEND_EMAIL = "SEND_EMAIL"
    VERIFY_EMAIL_SENT = "VERIFY_EMAIL_SENT"
    CREATE_FILE = "CREATE_FILE"
    WRITE_FILE = "WRITE_FILE"
    CREATE_DIRECTORY = "CREATE_DIRECTORY"
    CREATE_FOLDER = "CREATE_FOLDER"
    DELETE_FILE = "DELETE_FILE"
    DELETE_DIRECTORY = "DELETE_DIRECTORY"
    RUN_TERMINAL = "RUN_TERMINAL"
    CUSTOM = "CUSTOM"


@dataclass
class ComputerActionResult:
    """Outcome of an executed computer action."""
    action_type: ComputerActionType
    success: bool
    message: str = ""
    evidence: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    verification_status: VerificationStatus = VerificationStatus.UNKNOWN


@dataclass
class ComputerActionStep:
    """Atomic step within a multi-step computer task plan."""
    step_id: int
    description: str
    action_type: ComputerActionType
    parameters: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[int] = field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.LOW_RISK
    requires_confirmation: bool = False
    result: Optional[ComputerActionResult] = None
    retry_count: int = 0
    max_retries: int = 3
