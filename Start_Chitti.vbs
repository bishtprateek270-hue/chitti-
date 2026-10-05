' Chitti AI 1-Click Silent Background Launcher
' Starts Chitti and Ollama silently without leaving a CMD window open
Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
projectDir = fso.GetParentFolderName(WScript.ScriptFullName)

' 1. Check if Ollama is running, launch in background if not
WshShell.Run "cmd /c tasklist /FI ""IMAGENAME eq ollama.exe"" 2>NUL | find /I /N ""ollama.exe"" >NUL || start /B ollama serve", 0, True

' 2. Launch Chitti Background Daemon
WshShell.CurrentDirectory = projectDir
WshShell.Run "python src\main.py --daemon", 0, False

Set WshShell = Nothing
Set fso = Nothing
