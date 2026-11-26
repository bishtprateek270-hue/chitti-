@echo off
title n8n Workflow Automation Server for Chitti
cd /d "%~dp0\.."

echo ====================================================================
echo                 N8N WORKFLOW AUTOMATION FOR CHITTI
echo ====================================================================
echo.
echo n8n empowers Chitti with 400+ native app integrations (WhatsApp,
echo Telegram, Gmail, Notion, GitHub, Google Drive, Slack, etc.)!
echo.
echo Checking for local runtime (Node.js / npx or Docker)...
echo.

where npx >nul 2>nul
if %errorlevel% equ 0 (
    echo [+] Node.js / npx detected!
    echo [+] Starting local n8n server on http://localhost:5678 ...
    echo.
    echo Once loaded, open your browser at:
    echo     http://localhost:5678
    echo.
    echo Chitti Inbound Notification Webhook:
    echo     POST http://localhost:8765/chitti/notify
    echo.
    echo ====================================================================
    npx n8n
    goto :done
)

where docker >nul 2>nul
if %errorlevel% equ 0 (
    echo [+] Docker detected! Starting n8n container...
    echo [+] Web UI: http://localhost:5678
    echo.
    docker run -it --rm --name n8n -p 5678:5678 -v "%USERPROFILE%\.n8n:/home/node/.n8n" docker.n8n.io/n8nio/n8n
    goto :done
)

echo [!] Neither 'npx' (Node.js) nor 'docker' was found on your PATH.
echo.
echo To run n8n on Windows:
echo 1. Install Node.js from https://nodejs.org/ (LTS version)
echo 2. Run this script again, or run 'npx n8n' in your terminal.
echo.
pause
exit /b 1

:done
pause
