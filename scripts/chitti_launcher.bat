@echo off
REM ==============================================================================
REM Chitti AI Desktop Companion - Safe Background Launcher
REM Engineered with boot grace periods, CPU safety guards, and passive OpenMP
REM to prevent any laptop freezes, driver conflicts, or system sluggishness.
REM ==============================================================================

cd /d "%~dp0\.."

REM Optional boot delay if called with --delayed
if "%1"=="--delayed" (
    timeout /t 25 /nobreak >nul 2>&1
)

REM Check if Ollama is running, launch quietly if needed
powershell -NoProfile -Command "$p = Get-Process ollama -ErrorAction SilentlyContinue; if (-not $p) { Start-Process 'ollama' -ArgumentList 'serve' -WindowStyle Hidden; Start-Sleep -Seconds 2 }"

REM Environment guards for low-latency voice and passive OpenMP thread management
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set OMP_NUM_THREADS=2
set MKL_NUM_THREADS=2
set KMP_BLOCKTIME=1
set OMP_WAIT_POLICY=PASSIVE

REM Launch Chitti as a windowless background daemon at standard priority
if exist "%LOCALAPPDATA%\Programs\Python\Python314\pythonw.exe" (
    start "" "%LOCALAPPDATA%\Programs\Python\Python314\pythonw.exe" src\main.py --daemon
) else (
    start "" pythonw src\main.py --daemon
)

exit
