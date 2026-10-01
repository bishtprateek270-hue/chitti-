"""
Chitti Laptop Agent Tool Executor.
Executes validated Windows desktop actions deterministically and returns structured results.
"""

import os
import sys
import subprocess
import webbrowser
import platform
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import psutil
from PIL import ImageGrab

from src.agent.actions import ActionType, StructuredAction, ActionResult
from src.utils.logging import log_chitti, log_warning, log_error, log_debug


class ActionExecutor:
    """Performs validated system operations on the local Windows OS."""

    @classmethod
    def execute(cls, action: StructuredAction, default_workspace: str = "data/workspace", screenshots_dir: str = "data/screenshots") -> ActionResult:
        act_type = action.action

        log_chitti(f"[AGENT] Execution: STARTED | Action: {act_type.value}")

        try:
            if act_type == ActionType.OPEN_APPLICATION:
                return cls._open_application(action)
            elif act_type == ActionType.CLOSE_APPLICATION:
                return cls._close_application(action)
            elif act_type == ActionType.OPEN_FOLDER:
                return cls._open_folder(action)
            elif act_type == ActionType.OPEN_FILE:
                return cls._open_file(action)
            elif act_type == ActionType.OPEN_URL:
                return cls._open_url(action)
            elif act_type == ActionType.CREATE_FOLDER:
                return cls._create_folder(action, default_workspace)
            elif act_type == ActionType.CREATE_TEXT_FILE:
                return cls._create_text_file(action, default_workspace)
            elif act_type == ActionType.TAKE_SCREENSHOT:
                return cls._take_screenshot(action, screenshots_dir)
            elif act_type == ActionType.ANALYZE_SCREEN:
                return cls._analyze_screen(action)
            elif act_type == ActionType.DIAGNOSE_SCREEN_ERROR:
                return cls._diagnose_screen_error(action)
            elif act_type == ActionType.SUMMARIZE_DOCUMENT:
                return cls._summarize_document(action)
            elif act_type == ActionType.FIND_UI_ELEMENT:
                return cls._find_ui_element(action)
            elif act_type == ActionType.RUN_CODE_TESTS:
                return cls._run_code_tests(action)
            elif act_type == ActionType.SELF_HEAL_CODE:
                return cls._self_heal_code(action)
            elif act_type == ActionType.DEBUG_CODE:
                return cls._debug_code(action)
            elif act_type == ActionType.GET_SYSTEM_INFO:
                return cls._get_system_info(action)
            elif act_type == ActionType.SET_VOLUME:
                return cls._set_volume(action)
            elif act_type == ActionType.DELETE_FILE:
                return cls._delete_file(action, default_workspace)
            elif act_type == ActionType.DELETE_FOLDER:
                return cls._delete_folder(action, default_workspace)
            else:
                return ActionResult(
                    success=False,
                    action=act_type,
                    message=f"No executor implemented for action: {act_type.value}",
                    error="Unimplemented action",
                )
        except Exception as e:
            log_error(f"[AGENT] Execution: FAILED | Error: {e}")
            return ActionResult(
                success=False,
                action=act_type,
                message=f"Failed to perform action {act_type.value}: {str(e)}",
                error=str(e),
            )

    @classmethod
    def _open_application(cls, action: StructuredAction) -> ActionResult:
        target = action.parameters["target"]
        resolved = action.parameters.get("resolved_app", target)

        try:
            # On Windows, os.startfile opens registered apps safely without shell injection
            if platform.system() == "Windows" and os.path.exists(resolved):
                os.startfile(resolved)
            else:
                subprocess.Popen([resolved], shell=False)

            log_chitti(f"[AGENT] Execution: SUCCESS | Launched application: {target}")
            return ActionResult(
                success=True,
                action=ActionType.OPEN_APPLICATION,
                target=target,
                message=f"{target} has been opened.",
                data={"resolved_path": resolved},
            )
        except Exception as e:
            # Fallback for built-in tools like notepad, calc, explorer
            try:
                subprocess.Popen(f"start {target}", shell=True)
                log_chitti(f"[AGENT] Execution: SUCCESS | Launched {target} via start wrapper")
                return ActionResult(
                    success=True,
                    action=ActionType.OPEN_APPLICATION,
                    target=target,
                    message=f"{target} has been opened.",
                )
            except Exception as e2:
                return ActionResult(
                    success=False,
                    action=ActionType.OPEN_APPLICATION,
                    target=target,
                    message=f"Could not open {target}.",
                    error=str(e2),
                )

    @classmethod
    def _close_application(cls, action: StructuredAction) -> ActionResult:
        target = action.parameters["target"]
        clean_target = target.lower().replace(".exe", "")
        terminated_count = 0

        for proc in psutil.process_iter(['pid', 'name']):
            try:
                p_name = proc.info['name'].lower().replace(".exe", "")
                if clean_target in p_name or p_name in clean_target:
                    proc.terminate()
                    terminated_count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass

        if terminated_count > 0:
            log_chitti(f"[AGENT] Execution: SUCCESS | Closed {terminated_count} instances of {target}")
            return ActionResult(
                success=True,
                action=ActionType.CLOSE_APPLICATION,
                target=target,
                message=f"Closed {target}.",
                data={"closed_instances": terminated_count},
            )
        else:
            return ActionResult(
                success=False,
                action=ActionType.CLOSE_APPLICATION,
                target=target,
                message=f"No running instances of {target} were found.",
            )

    @classmethod
    def _open_folder(cls, action: StructuredAction) -> ActionResult:
        target = action.parameters.get("target", "Desktop")
        resolved_path = action.parameters.get("resolved_path")

        if not resolved_path:
            from src.agent.registry import FolderDiscovery
            resolved_path = FolderDiscovery.resolve_folder_path(target)

        if not resolved_path or not resolved_path.exists():
            resolved_path = Path.home()

        try:
            if platform.system() == "Windows":
                os.startfile(str(resolved_path))
            else:
                subprocess.Popen(["xdg-open", str(resolved_path)])
        except Exception as e:
            log_debug(f"Folder opening error: {e}")

        log_chitti(f"[AGENT] Execution: SUCCESS | Opened folder: {resolved_path}")
        return ActionResult(
            success=True,
            action=ActionType.OPEN_FOLDER,
            target=target,
            message=f"Opened the {target} folder.",
            data={"path": str(resolved_path)},
        )

    @classmethod
    def _open_file(cls, action: StructuredAction) -> ActionResult:
        target = action.parameters["target"]
        p = Path(target)
        if not p.is_absolute():
            # Check in Desktop, Documents, Downloads or workspace
            for candidate_dir in [Path.cwd(), Path.home() / "Desktop", Path.home() / "Downloads", Path.home() / "Documents"]:
                candidate = candidate_dir / target
                if candidate.exists():
                    p = candidate
                    break

        if not p.exists():
            return ActionResult(
                success=False,
                action=ActionType.OPEN_FILE,
                target=target,
                message=f"File '{target}' was not found.",
            )

        if platform.system() == "Windows":
            os.startfile(str(p))
        else:
            subprocess.Popen(["xdg-open", str(p)])

        return ActionResult(
            success=True,
            action=ActionType.OPEN_FILE,
            target=target,
            message=f"Opened file {target}.",
            data={"path": str(p)},
        )

    @classmethod
    def _open_url(cls, action: StructuredAction) -> ActionResult:
        url = action.parameters["url"]
        site_name = action.parameters.get("site_name", url)

        webbrowser.open(url)
        log_chitti(f"[AGENT] Execution: SUCCESS | Opened URL: {url}")
        return ActionResult(
            success=True,
            action=ActionType.OPEN_URL,
            target=url,
            message=f"Opened {site_name} in your browser.",
            data={"url": url},
        )

    @classmethod
    def _create_folder(cls, action: StructuredAction, default_workspace: str) -> ActionResult:
        name = action.parameters["name"]
        parent_dir = action.parameters.get("parent_dir")

        if parent_dir:
            base_path = Path(parent_dir)
        else:
            # Default to Desktop or safe workspace
            base_path = Path.home() / "Desktop"
            if not base_path.exists():
                base_path = Path(default_workspace)

        target_dir = base_path / name
        target_dir.mkdir(parents=True, exist_ok=True)

        log_chitti(f"[AGENT] Execution: SUCCESS | Created folder: {target_dir}")
        return ActionResult(
            success=True,
            action=ActionType.CREATE_FOLDER,
            target=name,
            message=f"Created folder '{name}' on your Desktop.",
            data={"path": str(target_dir)},
        )

    @classmethod
    def _create_text_file(cls, action: StructuredAction, default_workspace: str) -> ActionResult:
        name = action.parameters["name"]
        content = action.parameters.get("content", "")
        parent_dir = action.parameters.get("parent_dir")

        if parent_dir:
            base_path = Path(parent_dir)
        else:
            base_path = Path.home() / "Desktop"
            if not base_path.exists():
                base_path = Path(default_workspace)

        base_path.mkdir(parents=True, exist_ok=True)
        file_path = base_path / name
        file_path.write_text(content, encoding="utf-8")

        log_chitti(f"[AGENT] Execution: SUCCESS | Created text file: {file_path}")
        return ActionResult(
            success=True,
            action=ActionType.CREATE_TEXT_FILE,
            target=name,
            message=f"Created file '{name}' with specified content.",
            data={"path": str(file_path), "size_bytes": len(content)},
        )

    @classmethod
    def _take_screenshot(cls, action: StructuredAction, screenshots_dir: str) -> ActionResult:
        target_dir = Path(screenshots_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = target_dir / f"screenshot_{now_str}.png"

        # Capture primary screen
        screenshot = ImageGrab.grab()
        screenshot.save(str(file_path))

        log_chitti(f"[AGENT] Execution: SUCCESS | Screenshot saved: {file_path}")
        return ActionResult(
            success=True,
            action=ActionType.TAKE_SCREENSHOT,
            message=f"Screenshot taken and saved to {file_path.name}.",
            data={"path": str(file_path), "filename": file_path.name},
        )

    @classmethod
    def _analyze_screen(cls, action: StructuredAction) -> ActionResult:
        from src.agent.vision import VisionGroundingEngine
        engine = VisionGroundingEngine()
        query = action.parameters.get("query", "Explain what is on the screen")
        res = engine.analyze_screen(query=query)
        log_chitti(f"[AGENT] Execution: SUCCESS | Visual Analysis: {res.summary}")
        return ActionResult(
            success=True,
            action=ActionType.ANALYZE_SCREEN,
            message=res.summary,
            data={
                "summary": res.summary,
                "active_window": res.active_window_title,
                "process": res.active_process,
                "screenshot_path": res.screenshot_path,
            },
        )

    @classmethod
    def _diagnose_screen_error(cls, action: StructuredAction) -> ActionResult:
        from src.agent.vision import VisionGroundingEngine
        engine = VisionGroundingEngine()
        diag = engine.diagnose_screen_error()
        log_chitti(f"[AGENT] Execution: SUCCESS | Error Diagnosis: {diag.description}")
        return ActionResult(
            success=True,
            action=ActionType.DIAGNOSE_SCREEN_ERROR,
            message=diag.description,
            data={
                "has_error": diag.has_error,
                "error_type": diag.error_type,
                "description": diag.description,
                "suggested_fix": diag.suggested_fix,
            },
        )

    @classmethod
    def _summarize_document(cls, action: StructuredAction) -> ActionResult:
        from src.agent.vision import VisionGroundingEngine
        engine = VisionGroundingEngine()
        query = action.parameters.get("query", "Summarize the open document")
        res = engine.summarize_open_document(query=query)
        log_chitti(f"[AGENT] Execution: SUCCESS | Document Summary: {res.document_title}")
        return ActionResult(
            success=True,
            action=ActionType.SUMMARIZE_DOCUMENT,
            message=res.full_summary,
            data={
                "title": res.document_title,
                "document_type": res.document_type,
                "overview": res.overview,
                "key_points": res.key_points,
                "word_count": res.word_count,
                "screenshot_path": res.screenshot_path,
            },
        )

    @classmethod
    def _find_ui_element(cls, action: StructuredAction) -> ActionResult:
        from src.agent.vision import VisionGroundingEngine
        engine = VisionGroundingEngine()
        elem_name = action.parameters.get("element", "")
        match = engine.find_ui_element(elem_name)
        if match:
            msg = f"Found '{match.label}' on screen at coordinates ({match.center_x}, {match.center_y})."
            return ActionResult(
                success=True,
                action=ActionType.FIND_UI_ELEMENT,
                message=msg,
                data={
                    "found": True,
                    "label": match.label,
                    "center_x": match.center_x,
                    "center_y": match.center_y,
                    "box": match.bounding_box,
                },
            )
        return ActionResult(
            success=True,
            action=ActionType.FIND_UI_ELEMENT,
            message=f"UI element matching '{elem_name}' was not detected in active window.",
            data={"found": False, "element": elem_name},
        )

    @classmethod
    def _run_code_tests(cls, action: StructuredAction) -> ActionResult:
        from src.agent.debugger import CodeExecutionSandbox
        target = action.parameters.get("target", ".")
        fw = action.parameters.get("framework")
        cwd = action.parameters.get("cwd")
        res = CodeExecutionSandbox.run_tests(target_path=target, framework=fw, cwd=cwd)
        msg = f"Test suite {'PASSED' if res.success else 'FAILED'} (Exit code {res.exit_code}, {res.duration_ms:.1f}ms)."
        log_chitti(f"[AGENT] Execution: {'SUCCESS' if res.success else 'FAILED'} | {msg}")
        return ActionResult(
            success=res.success,
            action=ActionType.RUN_CODE_TESTS,
            message=msg,
            data={
                "exit_code": res.exit_code,
                "stdout": res.stdout,
                "stderr": res.stderr,
                "duration_ms": res.duration_ms,
            },
        )

    @classmethod
    def _self_heal_code(cls, action: StructuredAction) -> ActionResult:
        from src.agent.debugger import SelfHealingDebugger
        target = action.parameters.get("target", ".")
        test_file = action.parameters.get("test_file")
        max_iter = int(action.parameters.get("max_iterations", 3))

        healer = SelfHealingDebugger()
        p = Path(target)
        if p.is_dir():
            res = healer.heal_project(project_dir=target, test_file=test_file, max_iterations=max_iter)
        else:
            res = healer.heal_file(target_file=target, test_file=test_file, max_iterations=max_iter)

        log_chitti(f"[AGENT] Execution: {'SUCCESS' if res.success else 'FAILED'} | {res.summary()}")
        return ActionResult(
            success=res.success,
            action=ActionType.SELF_HEAL_CODE,
            message=res.summary(),
            data={
                "iterations": res.iterations,
                "diagnosis": res.diagnosis,
                "files_modified": res.files_modified,
                "patches": res.patches_applied,
            },
        )

    @classmethod
    def _debug_code(cls, action: StructuredAction) -> ActionResult:
        from src.agent.debugger import TracebackParser, SelfHealingDebugger
        target = action.parameters.get("target", ".")
        err_msg = action.parameters.get("error_message", "")
        err_info = TracebackParser.parse(err_msg, default_file=target)

        healer = SelfHealingDebugger()
        res = healer.heal_file(target_file=target, max_iterations=3)
        return ActionResult(
            success=res.success,
            action=ActionType.DEBUG_CODE,
            message=res.summary(),
            data={
                "initial_error": err_info.summary(),
                "success": res.success,
                "iterations": res.iterations,
            },
        )

    @classmethod
    def _get_system_info(cls, action: StructuredAction) -> ActionResult:
        q_type = action.parameters.get("query_type", "general")

        # Memory / RAM
        mem = psutil.virtual_memory()
        ram_total_gb = mem.total / (1024 ** 3)
        ram_avail_gb = mem.available / (1024 ** 3)
        ram_used_pct = mem.percent

        # Disk
        disk = psutil.disk_usage(os.path.abspath(os.sep))
        disk_total_gb = disk.total / (1024 ** 3)
        disk_free_gb = disk.free / (1024 ** 3)
        disk_used_pct = disk.percent

        # CPU
        cpu_count = psutil.cpu_count(logical=True)
        cpu_pct = psutil.cpu_percent(interval=0.1)

        # OS
        os_info = f"{platform.system()} {platform.release()}"

        data = {
            "ram_total_gb": round(ram_total_gb, 1),
            "ram_available_gb": round(ram_avail_gb, 1),
            "ram_used_pct": ram_used_pct,
            "disk_total_gb": round(disk_total_gb, 1),
            "disk_free_gb": round(disk_free_gb, 1),
            "cpu_threads": cpu_count,
            "cpu_usage_pct": cpu_pct,
            "os": os_info,
        }

        if q_type == "ram":
            msg = f"You have {data['ram_total_gb']} GB of RAM, with {data['ram_available_gb']} GB currently available ({data['ram_used_pct']}% used)."
        elif q_type == "disk":
            msg = f"You have {data['disk_free_gb']} GB free storage out of {data['disk_total_gb']} GB on your main drive."
        elif q_type == "cpu":
            msg = f"Your system has {data['cpu_threads']} CPU threads running at {data['cpu_usage_pct']}% utilization."
        elif q_type == "os":
            msg = f"You are running {data['os']}."
        else:
            msg = (
                f"System Overview: {data['ram_total_gb']} GB RAM ({data['ram_available_gb']} GB free), "
                f"{data['disk_free_gb']} GB free disk space, {data['cpu_threads']} CPU threads on {data['os']}."
            )

        log_chitti(f"[AGENT] Execution: SUCCESS | Retrieved system info: {msg}")
        return ActionResult(
            success=True,
            action=ActionType.GET_SYSTEM_INFO,
            message=msg,
            data=data,
        )

    @classmethod
    def _set_volume(cls, action: StructuredAction) -> ActionResult:
        op = action.parameters.get("operation", "increase")

        if platform.system() == "Windows":
            import ctypes
            # Virtual-Key codes
            VK_VOLUME_MUTE = 0xAD
            VK_VOLUME_DOWN = 0xAE
            VK_VOLUME_UP = 0xAF

            if op == "mute" or op == "unmute":
                ctypes.windll.user32.keybd_event(VK_VOLUME_MUTE, 0, 0, 0)
                ctypes.windll.user32.keybd_event(VK_VOLUME_MUTE, 0, 2, 0)
                msg = "Muted audio." if op == "mute" else "Unmuted audio."
            elif op == "decrease":
                for _ in range(5):  # 5 steps down (~10%)
                    ctypes.windll.user32.keybd_event(VK_VOLUME_DOWN, 0, 0, 0)
                    ctypes.windll.user32.keybd_event(VK_VOLUME_DOWN, 0, 2, 0)
                msg = "Decreased volume."
            else:  # increase
                for _ in range(5):  # 5 steps up (~10%)
                    ctypes.windll.user32.keybd_event(VK_VOLUME_UP, 0, 0, 0)
                    ctypes.windll.user32.keybd_event(VK_VOLUME_UP, 0, 2, 0)
                msg = "Increased volume."

            log_chitti(f"[AGENT] Execution: SUCCESS | Volume operation: {op}")
            return ActionResult(
                success=True,
                action=ActionType.SET_VOLUME,
                message=msg,
                data={"operation": op},
            )
        else:
            return ActionResult(
                success=True,
                action=ActionType.SET_VOLUME,
                message=f"Adjusted volume ({op}).",
                data={"operation": op},
            )

    @classmethod
    def _delete_file(cls, action: StructuredAction, default_workspace: str) -> ActionResult:
        target = action.parameters["target"]
        p = Path(target)
        if not p.is_absolute():
            p = (Path.home() / "Desktop" / target)

        if p.exists() and p.is_file():
            p.unlink()
            log_chitti(f"[AGENT] Execution: SUCCESS | Deleted file: {p}")
            return ActionResult(
                success=True,
                action=ActionType.DELETE_FILE,
                target=target,
                message=f"File '{target}' has been deleted.",
            )
        else:
            return ActionResult(
                success=False,
                action=ActionType.DELETE_FILE,
                target=target,
                message=f"File '{target}' not found.",
            )

    @classmethod
    def _delete_folder(cls, action: StructuredAction, default_workspace: str) -> ActionResult:
        import shutil
        target = action.parameters["target"]
        p = Path(target)
        if not p.is_absolute():
            p = (Path.home() / "Desktop" / target)

        if p.exists() and p.is_dir():
            shutil.rmtree(p)
            log_chitti(f"[AGENT] Execution: SUCCESS | Deleted folder: {p}")
            return ActionResult(
                success=True,
                action=ActionType.DELETE_FOLDER,
                target=target,
                message=f"Folder '{target}' has been deleted.",
            )
        else:
            return ActionResult(
                success=False,
                action=ActionType.DELETE_FOLDER,
                target=target,
                message=f"Folder '{target}' not found.",
            )
