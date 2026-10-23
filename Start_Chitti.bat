@echo off
title Chitti AI Launcher
cd /d "%~dp0"

REM 1. Ensure Ollama server is running in background
powershell -NoProfile -Command "$p = Get-Process ollama -ErrorAction SilentlyContinue; if (-not $p) { Start-Process 'ollama' -ArgumentList 'serve' -WindowStyle Hidden; Start-Sleep -Seconds 3 }"

REM 2. Start Chitti in background
if exist "%LOCALAPPDATA%\Programs\Python\Python314\pythonw.exe" (
    start "" "%LOCALAPPDATA%\Programs\Python\Python314\pythonw.exe" src\main.py --daemon
) else (
    start "" pythonw src\main.py --daemon
)
exit
