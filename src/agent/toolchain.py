"""
Chitti Programming Language Toolchain & Compiler Support.
Detects installed compilers/interpreters and constructs build/run commands.
"""

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.utils.logging import log_debug, log_info, log_warn


@dataclass
class ToolchainInfo:
    language: str
    binaries: List[str]
    is_compiled: bool
    default_extension: str
    run_template: str
    compile_template: Optional[str] = None


class ToolchainManager:
    """Detects installed compilers and constructs build/run commands for arbitrary languages."""

    TOOLCHAINS: Dict[str, ToolchainInfo] = {
        "python": ToolchainInfo(
            language="python",
            binaries=["python", "py", "python3"],
            is_compiled=False,
            default_extension=".py",
            run_template='python "{file}"',
        ),
        "cpp": ToolchainInfo(
            language="cpp",
            binaries=["g++", "clang++", "cl"],
            is_compiled=True,
            default_extension=".cpp",
            compile_template='g++ -std=c++17 "{file}" -o "{out}"',
            run_template='"{out}"',
        ),
        "c": ToolchainInfo(
            language="c",
            binaries=["gcc", "clang", "cl"],
            is_compiled=True,
            default_extension=".c",
            compile_template='gcc "{file}" -o "{out}"',
            run_template='"{out}"',
        ),
        "java": ToolchainInfo(
            language="java",
            binaries=["javac", "java"],
            is_compiled=True,
            default_extension=".java",
            compile_template='javac "{file}"',
            run_template='java -cp "{dir}" {classname}',
        ),
        "javascript": ToolchainInfo(
            language="javascript",
            binaries=["node"],
            is_compiled=False,
            default_extension=".js",
            run_template='node "{file}"',
        ),
        "typescript": ToolchainInfo(
            language="typescript",
            binaries=["ts-node", "tsx", "npx", "deno", "bun"],
            is_compiled=False,
            default_extension=".ts",
            run_template='npx ts-node "{file}"',
        ),
        "rust": ToolchainInfo(
            language="rust",
            binaries=["rustc", "cargo"],
            is_compiled=True,
            default_extension=".rs",
            compile_template='rustc "{file}" -o "{out}"',
            run_template='"{out}"',
        ),
        "go": ToolchainInfo(
            language="go",
            binaries=["go"],
            is_compiled=False,  # Can run directly via `go run`
            default_extension=".go",
            run_template='go run "{file}"',
        ),
        "csharp": ToolchainInfo(
            language="csharp",
            binaries=["dotnet", "csc"],
            is_compiled=True,
            default_extension=".cs",
            compile_template='csc "{file}" /out:"{out}"',
            run_template='"{out}"',
        ),
    }

    # Language name aliases mapping
    LANGUAGE_ALIASES: Dict[str, str] = {
        "python": "python",
        "py": "python",
        "c++": "cpp",
        "cpp": "cpp",
        "c": "c",
        "java": "java",
        "javascript": "javascript",
        "js": "javascript",
        "node": "javascript",
        "nodejs": "javascript",
        "typescript": "typescript",
        "ts": "typescript",
        "rust": "rust",
        "rs": "rust",
        "golang": "go",
        "go": "go",
        "c#": "csharp",
        "csharp": "csharp",
        "dotnet": "csharp",
        "react": "javascript",
        "html": "html",
        "css": "css",
        "bash": "bash",
        "shell": "bash",
        "powershell": "powershell",
    }

    @classmethod
    def normalize_language(cls, lang_str: str) -> str:
        """Normalizes language string to standard canonical identifier."""
        clean = lang_str.lower().strip()
        return cls.LANGUAGE_ALIASES.get(clean, clean)

    @classmethod
    def find_binary(cls, binaries: List[str]) -> Optional[str]:
        """Finds the first available executable in PATH."""
        for b in binaries:
            found = shutil.which(b)
            if found:
                return found
        return None

    @classmethod
    def is_toolchain_available(cls, language: str) -> Tuple[bool, Optional[str]]:
        """Checks if a toolchain/compiler is installed for the requested language."""
        lang = cls.normalize_language(language)
        info = cls.TOOLCHAINS.get(lang)
        if not info:
            # For web / markup languages, no compiler needed
            if lang in ("html", "css", "markdown", "json"):
                return True, "browser"
            return False, None

        bin_path = cls.find_binary(info.binaries)
        if bin_path:
            return True, bin_path
        return False, None

    @classmethod
    def get_extension_for_language(cls, language: str) -> str:
        """Returns standard file extension for language."""
        lang = cls.normalize_language(language)
        if lang in cls.TOOLCHAINS:
            return cls.TOOLCHAINS[lang].default_extension
        if lang == "react":
            return ".jsx"
        if lang == "html":
            return ".html"
        if lang == "css":
            return ".css"
        return ".txt"

    @classmethod
    def build_execution_commands(cls, language: str, file_path: str) -> Tuple[Optional[str], str]:
        """
        Builds (compile_command, run_command) for target language and file.
        """
        lang = cls.normalize_language(language)
        p = Path(file_path).resolve()
        info = cls.TOOLCHAINS.get(lang)
        if not info:
            return None, f'python "{p}"'

        out_bin = str(p.with_suffix(".exe"))

        if lang == "cpp":
            compile_cmd = f'g++ -std=c++17 "{p}" -o "{out_bin}"'
            return compile_cmd, f'"{out_bin}"'

        elif lang == "c":
            compile_cmd = f'gcc "{p}" -o "{out_bin}"'
            return compile_cmd, f'"{out_bin}"'

        elif lang == "java":
            compile_cmd = f'javac "{p}"'
            classname = p.stem
            return compile_cmd, f'java -cp "{p.parent}" {classname}'

        elif lang == "rust":
            compile_cmd = f'rustc "{p}" -o "{out_bin}"'
            return compile_cmd, f'"{out_bin}"'

        elif lang == "csharp":
            compile_cmd = f'csc "{p}" /out:"{out_bin}"'
            return compile_cmd, f'"{out_bin}"'

        elif lang == "go":
            return None, f'go run "{p}"'

        elif lang == "javascript":
            return None, f'node "{p}"'

        elif lang == "typescript":
            return None, f'npx ts-node "{p}"'

        # Default Python
        return None, f'python "{p}"'
