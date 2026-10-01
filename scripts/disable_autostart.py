"""
Chitti Windows Autostart Teardown Script.
Removes Chitti from the user's Windows Startup folder.
"""

import os
from pathlib import Path
import sys

def disable_autostart():
    """Removes Chitti entries from Windows Startup."""
    if sys.platform != "win32":
        print("[ERROR] Autostart configuration is only supported on Windows.")
        return False

    appdata = os.getenv("APPDATA")
    if not appdata:
        print("[ERROR] Could not resolve APPDATA environment variable.")
        return False

    startup_dir = Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"
    
    files_to_remove = ["ChittiAI.bat", "ChittiAI.vbs", "Chitti.lnk"]
    removed_any = False

    for filename in files_to_remove:
        file_path = startup_dir / filename
        if file_path.exists():
            try:
                file_path.unlink()
                print(f"[OK] Removed autostart entry: {file_path}")
                removed_any = True
            except Exception as e:
                print(f"[ERROR] Could not remove {file_path}: {e}")

    if removed_any:
        print("\n==================================================================")
        print("  [SUCCESS] Chitti Autostart Successfully Disabled.               ")
        print("==================================================================")
    else:
        print("\n[INFO] No active Chitti autostart entries found in Windows Startup.")

    return True

if __name__ == "__main__":
    disable_autostart()
