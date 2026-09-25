"""
Chitti Project & Environment Registry.
Manages user projects and environment paths persistently without polluting personal identity memory.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from src.utils.logging import log_info, log_warn, log_debug


class ProjectRegistry:
    """Manages configured development projects and their absolute paths on the user's laptop."""

    def __init__(self, registry_file: str = "data/projects.json"):
        self.registry_file = Path(registry_file)
        self.projects: Dict[str, str] = {}
        self._load_registry()

    def _load_registry(self) -> None:
        """Loads projects from the JSON configuration file."""
        if not self.registry_file.exists():
            # Seed default project if in workspace
            current_dir = str(Path.cwd().resolve())
            self.projects = {
                "chitti": current_dir,
            }
            self._save_registry()
            return

        try:
            with open(self.registry_file, "r", encoding="utf-8") as f:
                self.projects = json.load(f)
        except Exception as e:
            log_warn(f"Failed to load project registry from {self.registry_file}: {e}")
            self.projects = {}

    def _save_registry(self) -> None:
        """Saves projects to the JSON configuration file."""
        try:
            self.registry_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.registry_file, "w", encoding="utf-8") as f:
                json.dump(self.projects, f, indent=2, ensure_ascii=False)
        except Exception as e:
            log_warn(f"Failed to save project registry to {self.registry_file}: {e}")

    def register_project(self, name: str, path: str) -> bool:
        """Registers or updates a project path."""
        clean_name = name.strip().lower()
        resolved_path = str(Path(path).resolve())
        self.projects[clean_name] = resolved_path
        self._save_registry()
        log_info(f"Registered project '{name}' -> '{resolved_path}'")
        return True

    def get_project_path(self, name: str) -> Optional[str]:
        """Resolves a project name to its path."""
        clean_name = name.strip().lower()
        # Direct lookup
        if clean_name in self.projects:
            return self.projects[clean_name]
        
        # Substring / fuzzy match
        for key, p in self.projects.items():
            if clean_name in key or key in clean_name:
                return p

        # Check if project folder exists in user's home or standard development directories
        user_home = Path.home()
        candidates = [
            user_home / "OneDrive" / "Desktop" / name,
            user_home / "Desktop" / name,
            user_home / "Projects" / name,
            user_home / "source" / "repos" / name,
            user_home / "Documents" / name,
            Path("C:/Projects") / name,
        ]
        for cand in candidates:
            if cand.exists() and cand.is_dir():
                resolved = str(cand.resolve())
                self.register_project(name, resolved)
                return resolved

        return None

    def list_projects(self) -> Dict[str, str]:
        """Returns all registered projects."""
        return dict(self.projects)

    def remove_project(self, name: str) -> bool:
        """Removes a project from the registry."""
        clean_name = name.strip().lower()
        if clean_name in self.projects:
            del self.projects[clean_name]
            self._save_registry()
            return True
        return False
