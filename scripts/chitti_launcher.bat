@echo off
REM ==============================================================================
REM Chitti AI Desktop Companion - Safe Background Launcher
REM Engineered with boot grace periods, CPU safety guards, and below-normal priority
REM to prevent any laptop freezes, driver conflicts, or system sluggishness.
REM ==============================================================================

cd /d "%~dp0\.."

REM Optional boot delay if called with --delayed
if "%1"=="--delayed" (
    timeout /t 15 /nobreak >nul 2>&1
)

REM Check if Ollama is running, launch quietly if needed
powershell -NoProfile -Command "$p = Get-Process ollama -ErrorAction SilentlyContinue; if (-not $p) { Start-Process 'ollama' -ArgumentList 'serve' -WindowStyle Hidden; Start-Sleep -Seconds 2 }"

REM Ensure CPU safety mode for background audio to prevent GPU/CUDA driver contention
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set STT_DEVICE=cpu

REM Launch Chitti as a windowless daemon with BELOW_NORMAL priority class
if exist "%LOCALAPPDATA%\Programs\Python\Python314\pythonw.exe" (
    start "" /BELOWNORMAL "%LOCALAPPDATA%\Programs\Python\Python314\pythonw.exe" src\main.py --daemon
) else (
    start "" /BELOWNORMAL pythonw src\main.py --daemon
)

exit
