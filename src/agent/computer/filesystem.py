"""
Chitti Filesystem Controller.
Provides structured, safe file & directory operations across Windows storage.
"""

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.logging import log_debug, log_info, log_warn


@dataclass
class FileInfo:
    name: str
    path: str
    is_dir: bool
    size_bytes: int
    modified_time: float


class FilesystemController:
    """Manages file and folder creation, editing, deletion, searching, and metadata inspection."""

    def __init__(self, default_workspace: str = "data/workspace"):
        self.default_workspace = Path(default_workspace)
        self.default_workspace.mkdir(parents=True, exist_ok=True)

    def resolve_path(self, path_str: str) -> Path:
        """Resolves path string relative to workspace or standard PC locations."""
        raw = path_str.strip().strip('"\'')
        if not raw:
            return self.default_workspace

        p = Path(raw)
        if p.is_absolute():
            return p.resolve()

        if raw.startswith("~"):
            return Path(raw).expanduser().resolve()

        # If an explicit custom workspace is set (e.g. test tmp_path or project workspace)
        if self.default_workspace != Path("data/workspace") and not str(self.default_workspace).endswith("data/workspace"):
            return (self.default_workspace / p).resolve()

        # 1. Check if relative to default_workspace
        ws_path = (self.default_workspace / p).resolve()
        if ws_path.exists():
            return ws_path

        home = Path.home()
        desktop = (home / "OneDrive" / "Desktop") if (home / "OneDrive" / "Desktop").exists() else (home / "Desktop")
        documents = (home / "OneDrive" / "Documents") if (home / "OneDrive" / "Documents").exists() else (home / "Documents")
        downloads = home / "Downloads"

        # Check candidate locations across PC in order of priority
        candidates = [
            Path.cwd() / p,
            desktop / p,
            downloads / p,
            documents / p,
            home / p,
        ]

        for cand in candidates:
            if cand.exists():
                return cand.resolve()

        # If creating a new file/folder with relative path, default to default_workspace
        return ws_path

    def list_directory(self, dir_path: Optional[str] = None, recursive: bool = False) -> List[FileInfo]:
        """Lists files and subdirectories."""
        target = self.resolve_path(dir_path) if dir_path else Path.cwd()
        if not target.exists() or not target.is_dir():
            raise FileNotFoundError(f"Directory not found: {target}")

        results: List[FileInfo] = []
        if recursive:
            for item in target.rglob("*"):
                try:
                    stat = item.stat()
                    results.append(FileInfo(
                        name=item.name,
                        path=str(item.resolve()),
                        is_dir=item.is_dir(),
                        size_bytes=stat.st_size if not item.is_dir() else 0,
                        modified_time=stat.st_mtime,
                    ))
                except Exception:
                    pass
        else:
            for item in target.iterdir():
                try:
                    stat = item.stat()
                    results.append(FileInfo(
                        name=item.name,
                        path=str(item.resolve()),
                        is_dir=item.is_dir(),
                        size_bytes=stat.st_size if not item.is_dir() else 0,
                        modified_time=stat.st_mtime,
                    ))
                except Exception:
                    pass

        return sorted(results, key=lambda x: (not x.is_dir, x.name.lower()))

    def create_file(self, file_path: str, content: str = "") -> str:
        """Creates a new text file."""
        target = self.resolve_path(file_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            f.write(content)
        log_info(f"Created file: {target}")
        return str(target)

    def read_file(self, file_path: str, max_chars: int = 10000) -> str:
        """Reads content from a text file."""
        target = self.resolve_path(file_path)
        if not target.exists() or not target.is_file():
            raise FileNotFoundError(f"File not found: {target}")
        with open(target, "r", encoding="utf-8", errors="replace") as f:
            content = f.read(max_chars)
        return content

    def write_file(self, file_path: str, content: str) -> str:
        """Overwrites content in a file."""
        return self.create_file(file_path, content)

    def modify_file(self, file_path: str, content: str) -> str:
        """Modifies content in a file."""
        return self.create_file(file_path, content)

    def append_file(self, file_path: str, content: str) -> str:
        """Appends content to an existing file."""
        target = self.resolve_path(file_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "a", encoding="utf-8") as f:
            f.write(content)
        log_info(f"Appended to file: {target}")
        return str(target)

    def copy_file(self, src: str, dst: str) -> str:
        """Copies a file from src to dst."""
        src_path = self.resolve_path(src)
        dst_path = self.resolve_path(dst)
        if not src_path.exists():
            raise FileNotFoundError(f"Source file not found: {src_path}")
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)
        log_info(f"Copied {src_path} -> {dst_path}")
        return str(dst_path)

    def move_file(self, src: str, dst: str) -> str:
        """Moves a file or directory from src to dst."""
        src_path = self.resolve_path(src)
        dst_path = self.resolve_path(dst)
        if not src_path.exists():
            raise FileNotFoundError(f"Source not found: {src_path}")
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src_path), str(dst_path))
        log_info(f"Moved {src_path} -> {dst_path}")
        return str(dst_path)

    def rename_file(self, src: str, new_name: str) -> str:
        """Renames a file or directory within its current parent directory."""
        src_path = self.resolve_path(src)
        if not src_path.exists():
            raise FileNotFoundError(f"Source not found: {src_path}")
        dst_path = src_path.parent / new_name
        src_path.rename(dst_path)
        log_info(f"Renamed {src_path} -> {dst_path}")
        return str(dst_path)

    def delete_file(self, file_path: str) -> bool:
        """Deletes a single file."""
        target = self.resolve_path(file_path)
        if not target.exists():
            raise FileNotFoundError(f"File not found: {target}")
        if target.is_dir():
            raise IsADirectoryError(f"Target is a directory, not a file: {target}")
        target.unlink()
        log_info(f"Deleted file: {target}")
        return True

    def create_directory(self, dir_path: str) -> str:
        """Creates a new directory."""
        target = self.resolve_path(dir_path)
        target.mkdir(parents=True, exist_ok=True)
        log_info(f"Created directory: {target}")
        return str(target)

    def delete_directory(self, dir_path: str, recursive: bool = True) -> bool:
        """Deletes a directory."""
        target = self.resolve_path(dir_path)
        if not target.exists():
            raise FileNotFoundError(f"Directory not found: {target}")
        if not target.is_dir():
            raise NotADirectoryError(f"Target is not a directory: {target}")
        if recursive:
            shutil.rmtree(target)
        else:
            target.rmdir()
        log_info(f"Deleted directory: {target}")
        return True

    def search_files(self, pattern: str, root_path: Optional[str] = None) -> List[str]:
        """Searches for files matching a glob pattern (e.g. '*.pdf', '*.py', 'report.docx')."""
        if root_path:
            root = self.resolve_path(root_path)
            if not root.exists():
                return []
            return [str(p.resolve()) for p in root.rglob(pattern)]

        # Search across primary PC locations
        home = Path.home()
        desktop = (home / "OneDrive" / "Desktop") if (home / "OneDrive" / "Desktop").exists() else (home / "Desktop")
        documents = (home / "OneDrive" / "Documents") if (home / "OneDrive" / "Documents").exists() else (home / "Documents")
        downloads = home / "Downloads"

        search_roots = [Path.cwd(), desktop, downloads, documents, self.default_workspace]
        matches: List[str] = []
        seen = set()

        for r in search_roots:
            if not r or not r.exists():
                continue
            try:
                for p in r.rglob(pattern):
                    resolved_str = str(p.resolve())
                    if resolved_str not in seen:
                        seen.add(resolved_str)
                        matches.append(resolved_str)
            except Exception:
                continue

        return matches

    def get_file_info(self, file_path: str) -> FileInfo:
        """Retrieves metadata for a file or directory."""
        target = self.resolve_path(file_path)
        if not target.exists():
            raise FileNotFoundError(f"Path not found: {target}")
        stat = target.stat()
        return FileInfo(
            name=target.name,
            path=str(target.resolve()),
            is_dir=target.is_dir(),
            size_bytes=stat.st_size if not target.is_dir() else 0,
            modified_time=stat.st_mtime,
        )
