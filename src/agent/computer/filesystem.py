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
        """Resolves path string relative to workspace or as absolute path."""
        p = Path(path_str.strip())
        if p.is_absolute():
            return p.resolve()
        return (self.default_workspace / p).resolve()

    def list_directory(self, dir_path: Optional[str] = None, recursive: bool = False) -> List[FileInfo]:
        """Lists files and subdirectories."""
        target = self.resolve_path(dir_path) if dir_path else self.default_workspace
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
        """Searches for files matching a glob pattern (e.g. '*.pdf', '*.py')."""
        root = self.resolve_path(root_path) if root_path else self.default_workspace
        if not root.exists():
            return []
        matches = [str(p.resolve()) for p in root.rglob(pattern)]
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
