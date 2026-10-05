@echo off
echo Stopping Chitti background processes...
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*src\main.py*' -or $_.CommandLine -like '*src/main.py*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
echo [OK] Chitti stopped cleanly.
ping 127.0.0.1 -n 2 >nul
