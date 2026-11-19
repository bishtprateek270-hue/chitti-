@echo off
title Chitti 24/7 Background Desktop Assistant
cd /d "%~dp0"

echo =================================================================
echo             Launching Chitti 24/7 Desktop Companion
echo =================================================================
echo [i] Starting background daemon...
echo [i] Wake phrase: "Hello Chitti" / "Hey Chitti"
echo [i] Hotkeys: Alt+Space, Ctrl+Space, Alt+C
echo =================================================================

start "" "C:\Users\Bisht\AppData\Local\Programs\Python\Python314\pythonw.exe" src\main.py --daemon

echo [✓] Chitti is now running silently in the background!
echo [i] Press Alt+Space anytime or say "Hello Chitti".
timeout /t 3 >nul
exit
