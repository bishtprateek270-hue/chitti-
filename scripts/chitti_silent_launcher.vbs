' Chitti AI Silent Background Launcher
' Starts Chitti and Ollama silently without persistent command prompt window
Set WshShell = CreateObject("WScript.Shell")

' Get directory of this script
Set fso = CreateObject("Scripting.FileSystemObject")
scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
projectDir = fso.GetParentFolderName(scriptDir)

' Ensure Ollama is running
WshShell.Run "cmd /c tasklist /FI ""IMAGENAME eq ollama.exe"" 2>NUL | find /I /N ""ollama.exe"" >NUL || start /B ollama serve", 0, True

' Run Chitti in 24/7 background mode
WshShell.CurrentDirectory = projectDir
WshShell.Run "python src\main.py --daemon", 0, False

Set WshShell = Nothing
Set fso = Nothing
