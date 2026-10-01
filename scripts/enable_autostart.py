"""
Chitti Windows Autostart Setup Script.
Registers Chitti in the user's Windows Startup folder so it automatically
launches in the background every time the laptop boots or logs in.
"""

import os
from pathlib import Path
import sys

def enable_autostart(silent: bool = True):
    """Adds Chitti to the Windows Startup folder."""
    if sys.platform != "win32":
        print("[ERROR] Autostart configuration is only supported on Windows.")
        return False

    appdata = os.getenv("APPDATA")
    if not appdata:
        print("[ERROR] Could not resolve APPDATA environment variable.")
        return False

    startup_dir = Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"
    if not startup_dir.exists():
        startup_dir.mkdir(parents=True, exist_ok=True)

    project_root = Path(__file__).resolve().parent.parent
    scripts_dir = project_root / "scripts"

    launcher_bat = scripts_dir / "chitti_launcher.bat"

    # Clean up any legacy or corrupt files first
    for old_file in ["ChittiAI.bat", "ChittiAI.vbs", "Chitti.lnk"]:
        old_path = startup_dir / old_file
        if old_path.exists():
            try:
                old_path.unlink()
            except Exception:
                pass

    if silent:
        destination_file = startup_dir / "ChittiAI.vbs"
        bat_escaped = str(launcher_bat).replace("\\", "\\\\")
        root_escaped = str(project_root).replace("\\", "\\\\")
        content = (
            'Set WshShell = CreateObject("WScript.Shell")\r\n'
            f'WshShell.CurrentDirectory = "{project_root}"\r\n'
            f'WshShell.Run Chr(34) & "{launcher_bat}" & Chr(34), 0, False\r\n'
            'Set WshShell = Nothing\r\n'
        )
        with open(destination_file, "w", encoding="utf-8") as f:
            f.write(content)
    else:
        destination_file = startup_dir / "ChittiAI.bat"
        content = (
            '@echo off\r\n'
            f'cd /d "{project_root}"\r\n'
            f'call "{launcher_bat}"\r\n'
        )
        with open(destination_file, "w", encoding="utf-8") as f:
            f.write(content)

    print("\n==================================================================")
    print("  [SUCCESS] Chitti Autostart Successfully Enabled on Windows!     ")
    print("==================================================================")
    print(f"Startup Entry Created at:")
    print(f"  -> {destination_file}")
    print("\nHow it works:")
    print("  1. When you turn on/login to your laptop, Windows executes this script.")
    print("  2. Ollama AI server is started automatically if not already running.")
    print("  3. Chitti begins ambient listening ('Hey Chitti') & HUD overlay (Alt+Space).")
    print("\nTo disable autostart at any time, run:")
    print("  python scripts/disable_autostart.py\n")
    return True


if __name__ == "__main__":
    # Default to silent background mode unless --visible is explicitly passed
    is_visible = "--visible" in sys.argv or "-v" in sys.argv
    enable_autostart(silent=not is_visible)
