@echo off
title T2I :5554
set ROOT=%~dp0
pushd "%ROOT%"
set PYTHONUNBUFFERED=1
if not defined CONDA_BAT set CONDA_BAT=D:\Miniconda3\condabin\conda.bat
if not defined PYTHON_EXE set PYTHON_EXE=D:\Miniconda3\envs\asr\python.exe

echo ========================================
echo T2I Text-to-Image Service
echo ========================================
echo Port: 5554
echo ComfyUI: http://127.0.0.1:8188
echo ========================================
echo.

call %CONDA_BAT% activate asr
%PYTHON_EXE% -u t2i\service.py ^
    --port 5554 ^
    --slm-host localhost ^
    --comfyui-url http://127.0.0.1:8188 ^
    --workflow sd15_fast ^
    --output-dir "%ROOT%data\generated_images" ^
    --style "" ^
    --vram-mode 8gb

pause
