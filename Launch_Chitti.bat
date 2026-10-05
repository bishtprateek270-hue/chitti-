@echo off
title Chitti AI Companion Launcher
cd /d "%~dp0"

echo [CHITTI] Checking Ollama AI server...
powershell -NoProfile -Command "$p = Get-Process ollama -ErrorAction SilentlyContinue; if (-not $p) { Start-Process 'ollama' -ArgumentList 'serve' -WindowStyle Hidden; Start-Sleep -Seconds 2 }"

echo [CHITTI] Starting Chitti 24/7 background AI Companion...
set PYTHON_EXE=C:\Users\Bisht\AppData\Local\Programs\Python\Python314\pythonw.exe
if exist "%PYTHON_EXE%" (
    start "" "%PYTHON_EXE%" src\main.py --daemon
) else (
    start "" pythonw src\main.py --daemon
)

echo [CHITTI] Online! Say 'Hey Chitti' or press Alt+Space.
timeout /t 2 /nobreak >nul
exit
