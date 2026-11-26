@echo off
title Unregister Chitti 24/7 Autostart
set "STARTUP_DIR=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup"
set "VBS_FILE=%STARTUP_DIR%\Chitti_24x7_Autostart.vbs"

if exist "%VBS_FILE%" (
    del "%VBS_FILE%"
    echo [✓] Removed Chitti autostart from Windows Startup folder.
) else (
    echo [i] Chitti was not registered in the Windows Startup folder.
)
pause
