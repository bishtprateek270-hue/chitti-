"""
Chitti Application & Folder Discovery Registry.
Maps human-friendly application names and folder aliases to safe Windows paths and commands.
"""

import os
import shutil
from pathlib import Path
from typing import Optional, Dict, List


# Standard application executables and command aliases
DEFAULT_APP_MAP: Dict[str, List[str]] = {
    "chrome": ["chrome.exe", "google chrome", "chrome"],
    "google chrome": ["chrome.exe", "google chrome"],
    "vs code": ["code.cmd", "code.exe", "code"],
    "vscode": ["code.cmd", "code.exe", "code"],
    "code": ["code.cmd", "code.exe", "code"],
    "visual studio code": ["code.cmd", "code.exe", "code"],
    "notepad": ["notepad.exe", "notepad"],
    "calculator": ["calc.exe", "calculator"],
    "calc": ["calc.exe", "calculator"],
    "terminal": ["wt.exe", "cmd.exe", "powershell.exe"],
    "windows terminal": ["wt.exe"],
    "cmd": ["cmd.exe"],
    "command prompt": ["cmd.exe"],
    "powershell": ["powershell.exe"],
    "file explorer": ["explorer.exe"],
    "explorer": ["explorer.exe"],
    "paint": ["mspaint.exe", "paint"],
    "mspaint": ["mspaint.exe"],
    "task manager": ["taskmgr.exe"],
    "taskmgr": ["taskmgr.exe"],
    "spotify": ["spotify.exe", "spotify"],
    "word": ["winword.exe"],
    "excel": ["excel.exe"],
}

# Standard URL mappings
DEFAULT_URL_MAP: Dict[str, str] = {
    "google": "https://www.google.com",
    "github": "https://www.github.com",
    "youtube": "https://www.youtube.com",
    "chatgpt": "https://chat.openai.com",
    "gmail": "https://mail.google.com",
    "stackoverflow": "https://stackoverflow.com",
    "wikipedia": "https://www.wikipedia.org",
    "reddit": "https://www.reddit.com",
    "twitter": "https://www.x.com",
    "x": "https://www.x.com",
    "linkedin": "https://www.linkedin.com",
}


class AppDiscovery:
    """Discovers and resolves application executable paths on Windows."""

    @classmethod
    def resolve_application(cls, app_name: str) -> Optional[str]:
        clean = app_name.strip().lower()

        # 1. Direct match in default map
        candidates = DEFAULT_APP_MAP.get(clean, [clean, f"{clean}.exe"])

        # 2. Check via PATH (shutil.which)
        for candidate in candidates:
            resolved = shutil.which(candidate)
            if resolved:
                return resolved

        # 3. Check common Windows Program Files & LocalAppData locations
        program_files = os.environ.get("ProgramFiles", "C:\\Program Files")
        program_files_x86 = os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)")
        local_app_data = os.environ.get("LOCALAPPDATA", "")
        user_profile = os.environ.get("USERPROFILE", "")

        search_roots = [program_files, program_files_x86, local_app_data, user_profile]

        # Specific known path checks
        if clean in ("chrome", "google chrome"):
            chrome_paths = [
                Path(program_files) / "Google" / "Chrome" / "Application" / "chrome.exe",
                Path(program_files_x86) / "Google" / "Chrome" / "Application" / "chrome.exe",
                Path(local_app_data) / "Google" / "Chrome" / "Application" / "chrome.exe",
            ]
            for p in chrome_paths:
                if p.exists():
                    return str(p)

        if clean in ("vs code", "vscode", "code", "visual studio code"):
            code_paths = [
                Path(local_app_data) / "Programs" / "Microsoft VS Code" / "Code.exe",
                Path(program_files) / "Microsoft VS Code" / "Code.exe",
            ]
            for p in code_paths:
                if p.exists():
                    return str(p)

        # Fallback for Windows built-in apps that can launch by name
        if clean in ("notepad", "calc", "calculator", "explorer", "cmd", "powershell", "taskmgr", "mspaint"):
            return clean

        return None

    # Alias for convenience
    resolve_app = resolve_application


class FolderDiscovery:
    """Resolves standard user folder directories and aliases safely."""

    @classmethod
    def resolve_folder_path(cls, folder_alias: str) -> Optional[Path]:
        clean = folder_alias.strip().lower()
        home = Path.home()

        alias_map = {
            "downloads": home / "Downloads",
            "download": home / "Downloads",
            "documents": home / "Documents",
            "docs": home / "Documents",
            "document": home / "Documents",
            "desktop": home / "Desktop",
            "pictures": home / "Pictures",
            "photos": home / "Pictures",
            "pics": home / "Pictures",
            "videos": home / "Videos",
            "video": home / "Videos",
            "music": home / "Music",
            "home": home,
            "user": home,
        }

        # Check standard alias
        if clean in alias_map:
            p = alias_map[clean]
            if p.exists():
                return p

        # Check direct path
        direct_path = Path(folder_alias)
        if direct_path.is_absolute() and direct_path.exists() and direct_path.is_dir():
            return direct_path

        # Check relative to home
        rel_path = home / folder_alias
        if rel_path.exists() and rel_path.is_dir():
            return rel_path

        return None
