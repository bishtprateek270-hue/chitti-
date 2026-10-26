' Chitti AI 24/7 Watchdog Supervisor Launcher
' Runs the watchdog monitor silently in background to auto-restart Chitti if it ever stops.
Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
projectDir = fso.GetParentFolderName(WScript.ScriptFullName)

WshShell.CurrentDirectory = projectDir
WshShell.Run "pythonw scripts\chitti_watchdog.py", 0, False

Set WshShell = Nothing
Set fso = Nothing
