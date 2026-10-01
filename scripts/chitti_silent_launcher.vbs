' Chitti AI Silent Background Launcher
' Starts Chitti and Ollama silently without persistent command prompt window
Set WshShell = CreateObject("WScript.Shell")
strPath = WshShell.CurrentDirectory

' Get directory of this script
Set fso = CreateObject("Scripting.FileSystemObject")
scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
projectDir = fso.GetParentFolderName(scriptDir)

' Run launcher script hidden (0 = hidden window)
WshShell.CurrentDirectory = projectDir
WshShell.Run "cmd /c """ & scriptDir & "\chitti_launcher.bat""", 0, False
Set WshShell = Nothing
Set fso = Nothing
