@echo off
title ComfyUI :8188
set ROOT=%~dp0
if not defined COMFYUI_DIR set COMFYUI_DIR=%ROOT%ComfyUI_cu126\ComfyUI_windows_portable
if not exist "%COMFYUI_DIR%" (
    echo [ERROR] COMFYUI_DIR not found: "%COMFYUI_DIR%"
    echo Please set COMFYUI_DIR to your ComfyUI folder, e.g.
    echo    set COMFYUI_DIR=D:\path\to\ComfyUI
    exit /b 1
)
pushd "%COMFYUI_DIR%"
python_embeded\python.exe -s ComfyUI\main.py --windows-standalone-build
pause
