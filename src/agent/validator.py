"""
Chitti Laptop Agent Action Validator.
Validates requested actions, parameter constraints, application availability,
path safety, and risk/confirmation requirements prior to execution.
"""

from pathlib import Path
from typing import Tuple, Optional

from src.agent.actions import ActionType, RiskLevel, StructuredAction, ActionRegistry
from src.agent.registry import AppDiscovery, FolderDiscovery
from src.utils.logging import log_chitti, log_warning, log_debug


class ActionValidator:
    """Deterministic validation layer ensuring safety before laptop tool execution."""

    @classmethod
    def validate(cls, action: StructuredAction) -> Tuple[bool, str, Optional[StructuredAction]]:
        """
        Validates an action against safety rules and parameter requirements.
        Returns (is_valid, rejection_reason, validated_action).
        """
        # 1. Action definition check
        definition = ActionRegistry.get_definition(action.action)
        if not definition:
            log_warning(f"[AGENT] Validation: FAILED | Reason: Unknown action {action.action}")
            return False, f"Action '{action.action}' is not supported.", None

        # 2. Check required parameters
        for req_param in definition.required_parameters:
            if req_param not in action.parameters or not action.parameters[req_param]:
                log_warning(f"[AGENT] Validation: FAILED | Reason: Missing required parameter '{req_param}'")
                return False, f"Missing required parameter: '{req_param}'.", None

        # 3. Action-Specific Validations
        if action.action == ActionType.OPEN_APPLICATION:
            app_target = action.parameters["target"]
            # Strip redundant words like "application", "app", "the"
            import re
            clean_target = re.sub(r"(?i)^(?:application|app|the|my|a|an)\s+", "", app_target).strip()
            action.parameters["target"] = clean_target

            resolved = AppDiscovery.resolve_application(clean_target)
            if not resolved:
                # Check if target is a web URL, website, or mapped web service (YouTube, Google, WhatsApp, etc.)
                from src.agent.registry import ResourceDiscovery, DEFAULT_URL_MAP
                if ResourceDiscovery.is_url(clean_target) or clean_target.lower() in DEFAULT_URL_MAP:
                    target_url = ResourceDiscovery.to_url(clean_target)
                    log_chitti(f"[AGENT] Re-routing web target '{clean_target}' from OPEN_APPLICATION to OPEN_URL: {target_url}")
                    action.action = ActionType.OPEN_URL
                    action.parameters["url"] = target_url
                    action.parameters["site_name"] = clean_target.title()
                    return True, "Validation successful.", action

                log_warning(f"[AGENT] Validation: FAILED | Reason: Application '{clean_target}' could not be resolved on Windows.")
                return False, f"I couldn't find '{clean_target}' on this computer.", None
            action.parameters["resolved_app"] = resolved

        elif action.action == ActionType.CLOSE_APPLICATION:
            import re
            app_target = action.parameters["target"]
            app_target = re.sub(r"(?i)^(?:application|app|the|my|a|an)\s+", "", app_target).strip()
            action.parameters["target"] = app_target
            # Basic validation of target string
            if not app_target or len(app_target) < 2:
                return False, "Please specify a valid application name to close.", None

        elif action.action == ActionType.OPEN_FOLDER:
            folder_target = action.parameters.get("target", "Desktop")
            resolved_path = FolderDiscovery.resolve_folder_path(folder_target)
            if not resolved_path or not resolved_path.exists():
                resolved_path = Path.home()
            action.parameters["resolved_path"] = resolved_path

        elif action.action == ActionType.OPEN_URL:
            url = action.parameters["url"]
            if not (url.startswith("http://") or url.startswith("https://")):
                return False, "Invalid web URL. Must start with http:// or https://.", None

        elif action.action in (ActionType.CREATE_FOLDER, ActionType.CREATE_TEXT_FILE):
            name = action.parameters["name"]
            # Path safety checks: no forbidden characters or path traversal
            forbidden_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
            if any(char in name for char in forbidden_chars if char not in ('/', '\\')):
                return False, "File or folder name contains invalid characters.", None
            if ".." in name:
                return False, "Path traversal ('..') is not permitted for file operations.", None

        elif action.action in (ActionType.DELETE_FILE, ActionType.DELETE_FOLDER):
            target = action.parameters["target"]
            if ".." in target or target.strip() in ("/", "\\", "C:", "C:\\"):
                log_warning(f"[AGENT] Validation: FAILED | Reason: Dangerous path '{target}' blocked.")
                return False, "This path is protected and cannot be deleted.", None

        log_debug(f"[AGENT] Validation: PASSED for action {action.action.value}")
        return True, "Validation successful.", action
