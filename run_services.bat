@echo off
chcp 65001 >nul
title Pipeline Launcher
set ROOT=%~dp0
pushd "%ROOT%"

echo ================================================
echo   Co-Steam Pipeline Services
echo ================================================
echo.

:: First, clean up any existing Python processes
echo Cleaning up old processes...
taskkill /f /im python.exe >nul 2>&1
ping 127.0.0.1 -n 4 >nul
echo.

echo [1/7] Starting ComfyUI...
start "" "%ROOT%start_comfyui.bat"
timeout /t 8 /nobreak >nul

echo [2/7] Starting Bridge...
start "" "%ROOT%start_bridge.bat"
timeout /t 3 /nobreak >nul

echo [3/7] Starting SLM...
start "" "%ROOT%start_slm.bat"
timeout /t 3 /nobreak >nul

echo [4/7] Starting ASR...
start "" "%ROOT%start_asr.bat"
timeout /t 3 /nobreak >nul

echo [5/7] Starting T2I...
start "" "%ROOT%start_t2i.bat"
timeout /t 3 /nobreak >nul

echo [6/7] Starting Image Viewer...
start "" "%ROOT%start_imageviewer.bat"
timeout /t 2 /nobreak >nul

echo [7/7] Starting Control Pad...
start "" "%ROOT%start_controlpad.bat"
timeout /t 2 /nobreak >nul

echo.
echo ================================================
echo   All services started!
echo ================================================
echo.
echo   ASR:          ws://localhost:5551
echo   SLM:          ws://localhost:5552
echo   T2I:          ws://localhost:5554
echo   Bridge:       ws://localhost:5555
echo   Control Pad:  http://localhost:5560
echo   Image Viewer: http://localhost:5565
echo.
echo   ComfyUI:      http://localhost:8188
echo.
echo ================================================

start http://localhost:5560
start http://localhost:5565
pause
