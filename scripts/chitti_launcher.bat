@echo off
REM Chitti AI Desktop Companion Background Launcher
cd /d "%~dp0\.."

REM Ensure Ollama server is running
powershell -NoProfile -Command "$p = Get-Process ollama -ErrorAction SilentlyContinue; if (-not $p) { Start-Process 'ollama' -ArgumentList 'serve' -WindowStyle Hidden; Start-Sleep -Seconds 3 }"

REM Start Chitti Main Process in 24/7 background mode (Windowless Daemon)
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
if exist "%LOCALAPPDATA%\Programs\Python\Python314\pythonw.exe" (
    start "" "%LOCALAPPDATA%\Programs\Python\Python314\pythonw.exe" src\main.py --daemon
) else (
    start "" pythonw src\main.py --daemon
)
exit
