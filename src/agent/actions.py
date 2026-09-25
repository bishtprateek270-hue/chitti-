"""
Chitti Laptop Agent Actions Module.
Defines supported laptop actions, risk levels, parameter definitions, and action results.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List


class ActionType(str, Enum):
    OPEN_APPLICATION = "OPEN_APPLICATION"
    CLOSE_APPLICATION = "CLOSE_APPLICATION"
    OPEN_FOLDER = "OPEN_FOLDER"
    OPEN_FILE = "OPEN_FILE"
    OPEN_URL = "OPEN_URL"
    CREATE_FOLDER = "CREATE_FOLDER"
    CREATE_TEXT_FILE = "CREATE_TEXT_FILE"
    TAKE_SCREENSHOT = "TAKE_SCREENSHOT"
    GET_SYSTEM_INFO = "GET_SYSTEM_INFO"
    SET_VOLUME = "SET_VOLUME"
    DELETE_FILE = "DELETE_FILE"
    DELETE_FOLDER = "DELETE_FOLDER"
    UNKNOWN = "UNKNOWN"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass
class ActionDefinition:
    action_type: ActionType
    description: str
    risk_level: RiskLevel = RiskLevel.LOW
    requires_confirmation: bool = False
    required_parameters: List[str] = field(default_factory=list)
    optional_parameters: List[str] = field(default_factory=list)


@dataclass
class StructuredAction:
    action: ActionType
    parameters: Dict[str, Any] = field(default_factory=dict)
    risk_level: RiskLevel = RiskLevel.LOW
    requires_confirmation: bool = False
    raw_input: str = ""

    @property
    def target(self) -> Optional[str]:
        return self.parameters.get("target") or self.parameters.get("name") or self.parameters.get("url")


@dataclass
class ActionResult:
    success: bool
    action: ActionType
    message: str
    target: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class ActionRegistry:
    """Central registry of all supported laptop actions and their safety constraints."""

    _DEFINITIONS: Dict[ActionType, ActionDefinition] = {
        ActionType.OPEN_APPLICATION: ActionDefinition(
            action_type=ActionType.OPEN_APPLICATION,
            description="Opens or launches a desktop application on Windows.",
            risk_level=RiskLevel.LOW,
            requires_confirmation=False,
            required_parameters=["target"],
        ),
        ActionType.CLOSE_APPLICATION: ActionDefinition(
            action_type=ActionType.CLOSE_APPLICATION,
            description="Closes or terminates a running application.",
            risk_level=RiskLevel.MEDIUM,
            requires_confirmation=False,
            required_parameters=["target"],
        ),
        ActionType.OPEN_FOLDER: ActionDefinition(
            action_type=ActionType.OPEN_FOLDER,
            description="Opens a folder directory in Windows File Explorer.",
            risk_level=RiskLevel.LOW,
            requires_confirmation=False,
            required_parameters=["target"],
        ),
        ActionType.OPEN_FILE: ActionDefinition(
            action_type=ActionType.OPEN_FILE,
            description="Opens a specific file using its default system application.",
            risk_level=RiskLevel.LOW,
            requires_confirmation=False,
            required_parameters=["target"],
        ),
        ActionType.OPEN_URL: ActionDefinition(
            action_type=ActionType.OPEN_URL,
            description="Opens a web URL in the default browser.",
            risk_level=RiskLevel.LOW,
            requires_confirmation=False,
            required_parameters=["url"],
        ),
        ActionType.CREATE_FOLDER: ActionDefinition(
            action_type=ActionType.CREATE_FOLDER,
            description="Creates a new folder directory in a safe workspace.",
            risk_level=RiskLevel.LOW,
            requires_confirmation=False,
            required_parameters=["name"],
            optional_parameters=["parent_dir"],
        ),
        ActionType.CREATE_TEXT_FILE: ActionDefinition(
            action_type=ActionType.CREATE_TEXT_FILE,
            description="Creates a text file with specified content in a safe workspace.",
            risk_level=RiskLevel.LOW,
            requires_confirmation=False,
            required_parameters=["name"],
            optional_parameters=["content", "parent_dir"],
        ),
        ActionType.TAKE_SCREENSHOT: ActionDefinition(
            action_type=ActionType.TAKE_SCREENSHOT,
            description="Captures a screenshot of the primary display.",
            risk_level=RiskLevel.LOW,
            requires_confirmation=False,
            optional_parameters=["output_path"],
        ),
        ActionType.GET_SYSTEM_INFO: ActionDefinition(
            action_type=ActionType.GET_SYSTEM_INFO,
            description="Retrieves read-only system metrics (RAM, disk, CPU, GPU, OS).",
            risk_level=RiskLevel.LOW,
            requires_confirmation=False,
            optional_parameters=["query_type"],
        ),
        ActionType.SET_VOLUME: ActionDefinition(
            action_type=ActionType.SET_VOLUME,
            description="Adjusts or mutes the system audio volume.",
            risk_level=RiskLevel.LOW,
            requires_confirmation=False,
            required_parameters=["operation"],
            optional_parameters=["level"],
        ),
        ActionType.DELETE_FILE: ActionDefinition(
            action_type=ActionType.DELETE_FILE,
            description="Deletes a file (Destructive action requiring confirmation).",
            risk_level=RiskLevel.HIGH,
            requires_confirmation=True,
            required_parameters=["target"],
        ),
        ActionType.DELETE_FOLDER: ActionDefinition(
            action_type=ActionType.DELETE_FOLDER,
            description="Deletes a folder directory (Destructive action requiring confirmation).",
            risk_level=RiskLevel.HIGH,
            requires_confirmation=True,
            required_parameters=["target"],
        ),
    }

    @classmethod
    def get_definition(cls, action_type: ActionType) -> Optional[ActionDefinition]:
        return cls._DEFINITIONS.get(action_type)

    @classmethod
    def is_supported(cls, action_type: ActionType) -> bool:
        return action_type in cls._DEFINITIONS
