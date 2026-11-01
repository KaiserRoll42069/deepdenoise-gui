@echo off
setlocal
cd /d "%~dp0"
echo ==========================================================
echo   Deep-Rep Windows Setup Launcher
echo ==========================================================
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_windows.ps1" %*
pause
