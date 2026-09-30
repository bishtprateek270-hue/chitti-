"""
Chitti Application, Resource & Folder Dynamic Discovery Registry.
Dynamically resolves installed applications, user folders, and web resources
without rigid hardcoded command lists.
"""

import os
import sys
import re
import shutil
import platform
import difflib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List, Any

try:
    import winreg
except ImportError:
    winreg = None

from src.utils.logging import log_debug, log_info, log_warning


@dataclass
class DiscoveredApp:
    name: str
    path: Optional[str]
    executable: str
    is_running: bool = False


# Generic descriptive semantic aliases
SEMANTIC_APP_ALIASES: Dict[str, str] = {
    "editor": "code",
    "my editor": "code",
    "the editor": "code",
    "code editor": "code",
    "my code editor": "code",
    "browser": "chrome",
    "the browser": "chrome",
    "my browser": "chrome",
    "the web browser": "chrome",
    "web browser": "chrome",
    "that browser": "chrome",
    "terminal": "wt",
    "the terminal": "wt",
    "my terminal": "wt",
    "command line": "cmd",
    "the command line": "cmd",
    "music player": "spotify",
    "the music player": "spotify",
    "music site": "youtube",
    "the music site": "youtube",
    "music app": "spotify",
    "the music app": "spotify",
    "messaging app": "whatsapp",
    "the messaging app": "whatsapp",
    "chat app": "whatsapp",
    "the chat app": "whatsapp",
    "mail": "gmail",
    "the mail": "gmail",
    "my mail": "gmail",
    "email": "gmail",
    "the email": "gmail",
    "email app": "gmail",
    "calculator": "calc",
    "the calculator": "calc",
    "task manager": "taskmgr",
    "the task manager": "taskmgr",
    "file manager": "explorer",
    "the file manager": "explorer",
    "explorer": "explorer",
    "the explorer": "explorer",
    "paint": "mspaint",
    "paint app": "mspaint",
    "the paint": "mspaint",
}

# Backward compatibility alias map
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

# Standard URL mappings for popular services
DEFAULT_URL_MAP: Dict[str, str] = {
    "google": "https://www.google.com",
    "gemini": "https://gemini.google.com",
    "github": "https://www.github.com",
    "youtube": "https://www.youtube.com",
    "yt": "https://www.youtube.com",
    "whatsapp": "https://web.whatsapp.com",
    "whatsapp web": "https://web.whatsapp.com",
    "chatgpt": "https://chatgpt.com",
    "claude": "https://claude.ai",
    "telegram": "https://web.telegram.org",
    "discord": "https://discord.com/app",
    "slack": "https://app.slack.com",
    "teams": "https://teams.microsoft.com",
    "gmail": "https://mail.google.com",
    "stackoverflow": "https://stackoverflow.com",
    "wikipedia": "https://www.wikipedia.org",
    "reddit": "https://www.reddit.com",
    "twitter": "https://www.x.com",
    "x": "https://www.x.com",
    "linkedin": "https://www.linkedin.com",
    "spotify": "https://open.spotify.com",
    "netflix": "https://www.netflix.com",
    "amazon": "https://www.amazon.com",
}


class AppDiscovery:
    """Dynamically discovers and resolves application executable paths and shortcuts on the host OS."""

    _cached_app_paths: Optional[Dict[str, str]] = None

    @classmethod
    def _normalize_name(cls, app_name: str) -> str:
        clean = app_name.strip().lower()
        if clean in SEMANTIC_APP_ALIASES:
            return SEMANTIC_APP_ALIASES[clean]
        
        # Check without determiners
        stripped = re.sub(r"^(the|my|that|this|a|an)\s+", "", clean).strip()
        if stripped in SEMANTIC_APP_ALIASES:
            return SEMANTIC_APP_ALIASES[stripped]
            
        return clean

    @classmethod
    def get_registered_app_paths(cls) -> Dict[str, str]:
        """Queries the Windows Registry App Paths for all registered machine and user applications."""
        if cls._cached_app_paths is not None:
            return cls._cached_app_paths

        app_paths: Dict[str, str] = {}
        if platform.system() != "Windows" or winreg is None:
            cls._cached_app_paths = app_paths
            return app_paths

        reg_roots = [
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths"),
            (winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\App Paths"),
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\App Paths"),
        ]

        for root_key, sub_key in reg_roots:
            try:
                with winreg.OpenKey(root_key, sub_key) as key:
                    num_subkeys, _, _ = winreg.QueryInfoKey(key)
                    for i in range(num_subkeys):
                        try:
                            app_sub = winreg.EnumKey(key, i)
                            with winreg.OpenKey(key, app_sub) as app_key:
                                exec_path, _ = winreg.QueryValueEx(app_key, "")
                                if exec_path:
                                    exec_path_clean = exec_path.strip('"')
                                    base_name = app_sub.lower().replace(".exe", "")
                                    app_paths[base_name] = exec_path_clean
                                    app_paths[app_sub.lower()] = exec_path_clean
                        except Exception:
                            continue
            except Exception:
                continue

        cls._cached_app_paths = app_paths
        return app_paths

    @classmethod
    def find_start_menu_shortcuts(cls, query: str) -> Optional[str]:
        """Searches Start Menu program shortcuts for matching applications."""
        if platform.system() != "Windows":
            return None

        clean_query = query.lower().replace(" ", "")
        program_data = os.environ.get("ProgramData", "C:\\ProgramData")
        app_data = os.environ.get("APPDATA", "")

        search_dirs = [
            Path(program_data) / "Microsoft" / "Windows" / "Start Menu" / "Programs",
            Path(app_data) / "Microsoft" / "Windows" / "Start Menu" / "Programs" if app_data else None,
        ]

        for s_dir in search_dirs:
            if not s_dir or not s_dir.exists():
                continue
            try:
                for lnk in s_dir.rglob("*.lnk"):
                    stem_clean = lnk.stem.lower().replace(" ", "")
                    if clean_query == stem_clean or clean_query in stem_clean or stem_clean in clean_query:
                        return str(lnk.resolve())
            except Exception:
                continue

        return None

    @classmethod
    def resolve_application(cls, app_name: str) -> Optional[str]:
        """
        Dynamically resolves an application target to an executable path, shortcut, or shell command.
        Uses PATH lookup, Windows Registry App Paths, Start Menu shortcut discovery, and standard Program Files.
        """
        raw_clean = app_name.strip()
        clean = cls._normalize_name(raw_clean)
        clean_no_ext = clean.replace(".exe", "").replace(".cmd", "").replace(".bat", "")

        # 0. Check Windows built-in utility commands
        builtins = {
            "notepad": "notepad.exe",
            "calc": "calc.exe",
            "calculator": "calc.exe",
            "explorer": "explorer.exe",
            "file explorer": "explorer.exe",
            "cmd": "cmd.exe",
            "command prompt": "cmd.exe",
            "powershell": "powershell.exe",
            "terminal": "wt.exe",
            "windows terminal": "wt.exe",
            "wt": "wt.exe",
            "taskmgr": "taskmgr.exe",
            "task manager": "taskmgr.exe",
            "mspaint": "mspaint.exe",
            "paint": "mspaint.exe",
            "control": "control.exe",
            "settings": "ms-settings:",
        }
        if clean in builtins or clean_no_ext in builtins:
            b_target = builtins.get(clean, builtins.get(clean_no_ext))
            if shutil.which(b_target):
                return shutil.which(b_target)
            return b_target

        # 1. Check system PATH via shutil.which
        candidate_names = [clean, f"{clean}.exe", f"{clean}.cmd", f"{clean}.bat", clean_no_ext, f"{clean_no_ext}.exe"]
        if clean in ("vs code", "vscode", "visual studio code"):
            candidate_names = ["code.cmd", "code.exe", "code"] + candidate_names
        elif clean in ("chrome", "google chrome"):
            candidate_names = ["chrome.exe", "chrome"] + candidate_names

        for c in candidate_names:
            resolved = shutil.which(c)
            if resolved:
                return resolved

        # 2. Check Windows Registry App Paths
        reg_apps = cls.get_registered_app_paths()
        if clean in reg_apps:
            path_val = reg_apps[clean]
            if os.path.exists(path_val):
                return path_val
        if clean_no_ext in reg_apps:
            path_val = reg_apps[clean_no_ext]
            if os.path.exists(path_val):
                return path_val

        # Substring search in registry apps
        for reg_k, reg_v in reg_apps.items():
            if clean_no_ext == reg_k or clean_no_ext in reg_k or reg_k in clean_no_ext:
                if os.path.exists(reg_v):
                    return reg_v

        # 3. Check Start Menu Shortcuts (.lnk)
        start_menu_lnk = cls.find_start_menu_shortcuts(clean_no_ext)
        if start_menu_lnk:
            return start_menu_lnk

        # 4. Search Common Program Directories
        program_files = os.environ.get("ProgramFiles", "C:\\Program Files")
        program_files_x86 = os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)")
        local_app_data = os.environ.get("LOCALAPPDATA", "")
        user_profile = os.environ.get("USERPROFILE", "")

        search_roots = [
            Path(local_app_data) / "Programs" if local_app_data else None,
            Path(program_files),
            Path(program_files_x86),
            Path(local_app_data) if local_app_data else None,
            Path(user_profile) if user_profile else None,
        ]

        for root in search_roots:
            if not root or not root.exists():
                continue
            try:
                # Direct folder check
                app_dir = root / clean_no_ext.title()
                if app_dir.exists() and app_dir.is_dir():
                    for exe_candidate in [f"{clean_no_ext}.exe", f"{clean_no_ext.title()}.exe"]:
                        target_p = app_dir / exe_candidate
                        if target_p.exists():
                            return str(target_p.resolve())
            except Exception:
                continue

        # 5. Check if it's an executable file directly provided
        direct_p = Path(raw_clean)
        if direct_p.exists() and direct_p.is_file():
            return str(direct_p.resolve())

        # 6. Fuzzy Match against known apps, builtins, and registered apps
        all_known = list(builtins.keys()) + list(DEFAULT_APP_MAP.keys()) + list(reg_apps.keys())
        fuzzy_app = difflib.get_close_matches(clean_no_ext, all_known, n=1, cutoff=0.7)
        if fuzzy_app and fuzzy_app[0] != clean_no_ext:
            matched_name = fuzzy_app[0]
            log_debug(f"[APP_DISCOVERY] Fuzzy matched '{clean_no_ext}' -> '{matched_name}'")
            return cls.resolve_application(matched_name)

        # Fallback: if it's a simple alphanumeric name, allow standard Windows shell execution
        if re.match(r"^[a-zA-Z0-9_\-]+$", clean_no_ext):
            return clean_no_ext

        return None

    # Alias for convenience
    resolve_app = resolve_application


class FolderDiscovery:
    """Resolves standard user folder directories and aliases safely."""

    @classmethod
    def resolve_folder_path(cls, folder_alias: str) -> Optional[Path]:
        raw = folder_alias.strip() if folder_alias else ""
        clean = raw.lower().strip(".!? \t\n\"'")
        # Strip common leading verbs and determiners
        clean = re.sub(r"^(?:open\s+|show\s+|launch\s+|the\s+|my\s+|a\s+|an\s+|this\s+|that\s+|some\s+)+", "", clean).strip()
        # Strip trailing keywords
        clean = re.sub(r"(?:\s+(?:folder|directory|dir|kholo|dikhao|open\s+karo|karo))+$", "", clean).strip()

        home = Path.home()
        desktop = (home / "OneDrive" / "Desktop") if (home / "OneDrive" / "Desktop").exists() else (home / "Desktop")
        documents = (home / "OneDrive" / "Documents") if (home / "OneDrive" / "Documents").exists() else (home / "Documents")
        pictures = (home / "OneDrive" / "Pictures") if (home / "OneDrive" / "Pictures").exists() else (home / "Pictures")

        # Generic / empty folder request -> default to Desktop or Home
        if not clean or clean in (
            "folder", "directory", "dir", "folders", "directories", "files",
            "my files", "file explorer", "explorer", "my pc", "this pc", "computer",
            "a", "the", "my", "some"
        ):
            return desktop if desktop.exists() else home

        alias_map = {
            "downloads": home / "Downloads",
            "download": home / "Downloads",
            "documents": documents,
            "docs": documents,
            "doc": documents,
            "document": documents,
            "desktop": desktop,
            "pictures": pictures,
            "photos": pictures,
            "pics": pictures,
            "images": pictures,
            "picture": pictures,
            "videos": home / "Videos",
            "video": home / "Videos",
            "movies": home / "Videos",
            "music": home / "Music",
            "songs": home / "Music",
            "audio": home / "Music",
            "home": home,
            "user": home,
            "userprofile": home,
            "profile": home,
            "workspace": Path.cwd(),
            "work": Path.cwd(),
            "current": Path.cwd(),
            "cwd": Path.cwd(),
            "project": Path.cwd(),
            "chitti": Path.cwd(),
            "c drive": Path("C:\\"),
            "c:": Path("C:\\"),
            "c": Path("C:\\"),
        }

        # Check standard alias
        if clean in alias_map:
            p = alias_map[clean]
            if p.exists():
                return p

        # Check direct path
        try:
            for cand_str in (raw, clean):
                direct_path = Path(cand_str)
                if direct_path.is_absolute() and direct_path.exists() and direct_path.is_dir():
                    return direct_path

            # Check relative to standard locations
            search_bases = [Path.cwd(), desktop, home / "Downloads", documents, home]
            for base in search_bases:
                if not base or not base.exists():
                    continue
                cand_rel = base / clean
                if cand_rel.exists() and cand_rel.is_dir():
                    return cand_rel

            # Search immediate subdirectories (case-insensitive)
            for base in search_bases:
                if not base or not base.exists():
                    continue
                try:
                    for child in base.iterdir():
                        if child.is_dir() and child.name.lower() == clean:
                            return child
                except Exception:
                    continue
        except Exception:
            pass

        # Fallback to Desktop or Home rather than returning None if a folder was requested
        return desktop if desktop.exists() else home


class ResourceDiscovery:
    """Categorizes arbitrary user targets into URLs, Folders, Applications, or Files."""

    @classmethod
    def is_url(cls, target: str) -> bool:
        clean = target.strip().lower()
        if clean.startswith("http://") or clean.startswith("https://") or clean.startswith("www."):
            return True
        # Check standard web domains / TLDs
        if re.search(r"\b[a-zA-Z0-9_\-]+\.(?:com|org|net|io|in|co|ai|app|dev|edu|gov|xyz|tech|tv)(?:/\S*)?$", clean):
            return True
        if clean in DEFAULT_URL_MAP:
            return True
        # Check fuzzy match in DEFAULT_URL_MAP
        matches = difflib.get_close_matches(clean, list(DEFAULT_URL_MAP.keys()), n=1, cutoff=0.7)
        if matches:
            return True
        return False

    @classmethod
    def to_url(cls, target: str) -> str:
        clean = target.strip().lower()
        if clean in DEFAULT_URL_MAP:
            return DEFAULT_URL_MAP[clean]
        matches = difflib.get_close_matches(clean, list(DEFAULT_URL_MAP.keys()), n=1, cutoff=0.7)
        if matches:
            return DEFAULT_URL_MAP[matches[0]]
        if clean.startswith("http://") or clean.startswith("https://"):
            return target.strip()
        if clean.startswith("www."):
            return f"https://{target.strip()}"
        return f"https://{target.strip()}"
