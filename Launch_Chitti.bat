@echo off
title Chitti AI Companion Launcher
cd /d "%~dp0"

REM 1. Ensure Ollama server is running in background
powershell -NoProfile -Command "$p = Get-Process ollama -ErrorAction SilentlyContinue; if (-not $p) { Start-Process 'ollama' -ArgumentList 'serve' -WindowStyle Hidden; Start-Sleep -Seconds 3 }"

REM 2. Start Chitti silently in 24/7 background mode (No persistent terminal window)
start "" pythonw src\main.py --daemon
exit
