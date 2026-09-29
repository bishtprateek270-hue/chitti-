"""
Chitti Computer-Use Module.
Full device control & closed-loop computer-use agent for Windows.
"""

from src.computer_use.actions import (
    ComputerActionResult,
    ComputerActionStep,
    ComputerActionType,
)
from src.computer_use.controller import ComputerController, WindowInfo
from src.computer_use.planner import ComputerUseEngine
from src.computer_use.recovery import ComputerActionRecovery
from src.computer_use.safety import ComputerSafetyPolicy
from src.computer_use.screen import ScreenObservation, ScreenObserver
from src.computer_use.state import (
    ActionHistoryEntry,
    ComputerTaskState,
    RiskLevel,
    VerificationStatus,
)
from src.computer_use.ui_detector import UIElement, UIElementDetector
from src.computer_use.verifier import (
    ComputerActionVerifier,
    VerificationResult,
)

__all__ = [
    "ComputerUseEngine",
    "ComputerController",
    "WindowInfo",
    "ScreenObserver",
    "ScreenObservation",
    "UIElementDetector",
    "UIElement",
    "ComputerActionVerifier",
    "VerificationResult",
    "ComputerActionRecovery",
    "ComputerTaskState",
    "ActionHistoryEntry",
    "VerificationStatus",
    "RiskLevel",
    "ComputerSafetyPolicy",
    "ComputerActionType",
    "ComputerActionStep",
    "ComputerActionResult",
]
