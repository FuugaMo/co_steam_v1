@echo off
chcp 65001
echo ================================================
echo   Stopping all pipeline services...
echo ================================================

:: Kill all Python processes (most reliable)
echo Killing all Python processes...
taskkill /f /im python.exe >nul 2>&1
if %errorlevel%==0 (echo   Done.) else (echo   No Python processes found.)

:: Wait for cleanup
ping 127.0.0.1 -n 4 >nul

echo ================================================
echo   All services stopped.
echo ================================================
pause
