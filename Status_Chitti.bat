@echo off
title Chitti 24/7 Brain and Service Status
cd /d "%~dp0"
python scripts\check_status.py
pause
