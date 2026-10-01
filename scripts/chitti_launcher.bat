@echo off
REM Chitti AI Desktop Companion Launcher
cd /d "%~dp0\.."

REM Check if Ollama is running, start if needed
tasklist /FI "IMAGENAME eq ollama.exe" 2>NUL | find /I /N "ollama.exe">NUL
if "%ERRORLEVEL%"=="1" (
    start /B ollama serve >NUL 2>&1
    ping 127.0.0.1 -n 3 >nul 2>&1
)

REM Start Chitti Main Process
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
python src\main.py
