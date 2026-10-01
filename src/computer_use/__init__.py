"""
Chitti Computer-Use Automation Framework.
Provides closed-loop UI perception, OS device control, verification, and safety.
"""

from src.computer_use.actions import (
    ComputerActionType,
    ComputerActionResult,
    ComputerActionStep,
)
from src.computer_use.controller import (
    ComputerController,
    WindowInfo,
)
from src.computer_use.planner import ComputerUseEngine
from src.computer_use.recovery import ComputerActionRecovery
from src.computer_use.safety import ComputerSafetyPolicy
from src.computer_use.screen import (
    ScreenObservation,
    ScreenObserver,
)
from src.computer_use.state import (
    ActionHistoryEntry,
    ComputerTaskState,
    RiskLevel,
    VerificationStatus,
)
from src.computer_use.ui_detector import (
    UIElement,
    UIElementDetector,
)
from src.computer_use.verifier import (
    ComputerActionVerifier,
    VerificationResult,
)

__all__ = [
    "ComputerActionType",
    "ComputerActionResult",
    "ComputerActionStep",
    "ComputerController",
    "WindowInfo",
    "ComputerUseEngine",
    "ComputerActionRecovery",
    "ComputerSafetyPolicy",
    "ScreenObservation",
    "ScreenObserver",
    "ActionHistoryEntry",
    "ComputerTaskState",
    "RiskLevel",
    "VerificationStatus",
    "UIElement",
    "UIElementDetector",
    "ComputerActionVerifier",
    "VerificationResult",
]
