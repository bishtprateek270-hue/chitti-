"""
Chitti Windows Autostart Setup Script.
Registers Chitti in the user's Windows Startup folder so it automatically
launches in the background every time the laptop boots or logs in.
"""

import os
from pathlib import Path
import sys

def enable_autostart(silent: bool = False):
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
    launcher_vbs = scripts_dir / "chitti_silent_launcher.vbs"

    target_script = launcher_vbs if silent else launcher_bat
    target_link_name = "ChittiAI.vbs" if silent else "ChittiAI.bat"
    destination_file = startup_dir / target_link_name

    # Create startup launcher in Startup folder
    if silent:
        content = f'''Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = "{project_root}"
WshShell.Run "cmd /c """ & "{launcher_bat}"""", 0, False
Set WshShell = Nothing
'''
        with open(destination_file, "w", encoding="utf-8") as f:
            f.write(content)
    else:
        content = f'''@echo off
cd /d "{project_root}"
call "{launcher_bat}"
'''
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
    # If passed --silent, runs without visible console window
    silent_mode = "--silent" in sys.argv or "-s" in sys.argv
    enable_autostart(silent=silent_mode)
