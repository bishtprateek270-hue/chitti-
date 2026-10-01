@echo off
REM Chitti AI Desktop Companion Launcher
cd /d "%~dp0\.."

REM Check if Ollama is running, start if needed
tasklist /FI "IMAGENAME eq ollama.exe" 2>NUL | find /I /N "ollama.exe">NUL
if "%ERRORLEVEL%"=="1" (
    start /B ollama serve >NUL 2>&1
    timeout /t 2 >NUL
)

REM Start Chitti Main Process
python src\main.py
