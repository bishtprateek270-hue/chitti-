"""
Chitti Controlled Laptop Agent Package (Phase 5).
Provides safe, deterministic, multilingual Windows desktop tools and action execution.
"""

from src.agent.actions import ActionType, RiskLevel, ActionDefinition, StructuredAction, ActionResult, ActionRegistry
from src.agent.registry import AppDiscovery, FolderDiscovery, DEFAULT_APP_MAP, DEFAULT_URL_MAP
from src.agent.parser import ActionParser
from src.agent.validator import ActionValidator
from src.agent.executor import ActionExecutor
from src.agent.manager import LaptopAgentManager

__all__ = [
    "ActionType",
    "RiskLevel",
    "ActionDefinition",
    "StructuredAction",
    "ActionResult",
    "ActionRegistry",
    "AppDiscovery",
    "FolderDiscovery",
    "DEFAULT_APP_MAP",
    "DEFAULT_URL_MAP",
    "ActionParser",
    "ActionValidator",
    "ActionExecutor",
    "LaptopAgentManager",
]
