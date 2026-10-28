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
        """Resolves a project name, repo name, or speech transcription to its absolute path."""
        import re
        import difflib

        clean_name = name.strip().lower()

        # Handle generic 'project repo', 'my repo', 'the project', 'current project'
        if clean_name in ("project repo", "my repo", "the repo", "repo", "project", "the project", "current project", "chitti repo", "chitti"):
            curr = str(Path.cwd().resolve())
            return curr

        # Direct lookup in registry
        if clean_name in self.projects:
            return self.projects[clean_name]

        # Strip generic stopwords: project, repo, repository, code, app, folder
        stripped = re.sub(r"(?i)\b(?:the\s+|my\s+|a\s+|an\s+)?(?:project|repo|repository|code|app|folder|workspace|program|application)\b", "", clean_name).strip()
        stripped = re.sub(r"\s+", " ", stripped).strip()

        # Speech recognition typo normalizations
        STT_PROJECT_ALIASES = {
            "3d wallpaper": "3d unplin",
            "wallpaper": "3d unplin",
            "3d ulpin": "3d unplin",
            "3d-ulpin": "3d unplin",
            "ulpin": "3d unplin",
            "unplin": "3d unplin",
            "karigar": "karigarai",
            "doc forensic": "DocForencic AI",
            "doc forensics": "DocForencic AI",
            "document tampering": "document tempring",
        }
        for alias_key, mapped in STT_PROJECT_ALIASES.items():
            if alias_key in clean_name or alias_key in stripped:
                stripped = mapped.lower()
                break

        # Never match pronouns or meaningless words as projects
        if clean_name in ("it", "this", "that", "them", "me", "him", "her", "us", "file", "folder", "app", "window"):
            return None

        # Check stripped against registry
        if stripped and stripped in self.projects:
            return self.projects[stripped]

        # Substring match in registry (only for queries >= 4 chars to avoid false positives like "it" matching "chitti")
        for key, p in self.projects.items():
            if stripped and len(stripped) >= 4 and (stripped in key or key in stripped):
                return p
            if len(clean_name) >= 4 and (clean_name in key or key in clean_name):
                return p

        # Difflib close match in registered projects
        if len(stripped or clean_name) >= 4:
            matches = difflib.get_close_matches(stripped or clean_name, list(self.projects.keys()), n=1, cutoff=0.6)
            if matches:
                return self.projects[matches[0]]

        # Search bases
        user_home = Path.home()
        search_bases = [
            user_home / "OneDrive" / "Desktop",
            user_home / "Desktop",
            user_home / "Projects",
            user_home / "source" / "repos",
            user_home / "Documents",
        ]

        # Direct existence checks
        for query in (stripped, clean_name):
            if not query:
                continue
            for base in search_bases:
                cand = base / query
                if cand.exists() and cand.is_dir():
                    resolved = str(cand.resolve())
                    self.register_project(query, resolved)
                    return resolved

        # Scan subdirectories for token overlap or fuzzy match
        target_tokens = set(re.findall(r"[a-zA-Z0-9]+", (stripped or clean_name).lower()))
        for base in search_bases:
            if not base.exists():
                continue
            try:
                for child in base.iterdir():
                    if child.is_dir():
                        c_lower = child.name.lower()
                        child_tokens = set(re.findall(r"[a-zA-Z0-9]+", c_lower))

                        # If tokens overlap strongly
                        if target_tokens and target_tokens.issubset(child_tokens):
                            resolved = str(child.resolve())
                            self.register_project(clean_name, resolved)
                            return resolved

                        # Substring or fuzzy sequence matcher
                        if (stripped and stripped in c_lower) or c_lower in clean_name:
                            resolved = str(child.resolve())
                            self.register_project(clean_name, resolved)
                            return resolved

                        if difflib.SequenceMatcher(None, stripped or clean_name, c_lower).ratio() > 0.65:
                            resolved = str(child.resolve())
                            self.register_project(clean_name, resolved)
                            return resolved
            except Exception:
                continue

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
