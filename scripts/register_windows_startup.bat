@echo off
title Register Chitti & n8n 24/7 Windows Startup
cd /d "%~dp0\.."

echo ====================================================================
echo         REGISTER CHITTI & N8N 24/7 WINDOWS AUTOSTART
echo ====================================================================
echo.
echo This script configures Windows to launch Chitti and n8n silently
echo in the background whenever your laptop boots up / you log in.
echo.

set "STARTUP_DIR=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup"
set "VBS_FILE=%STARTUP_DIR%\Chitti_24x7_Autostart.vbs"
set "BAT_PATH=%~dp0..\Start_Chitti_Daemon.bat"

echo Creating silent background runner at:
echo "%VBS_FILE%"
echo.

(
echo Set WshShell = CreateObject^("WScript.Shell"^)
echo WshShell.Run chr^(34^) ^& "%BAT_PATH%" ^& chr^(34^), 0, False
) > "%VBS_FILE%"

if exist "%VBS_FILE%" (
    echo [✓] SUCCESS: Chitti and n8n are now registered for 24/7 Windows autostart!
    echo [i] They will start automatically every time you log in, silently.
    echo.
) else (
    echo [!] Failed to create startup shortcut. Check permissions.
)

pause
