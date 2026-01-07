@echo off
title Image Viewer :5565
set ROOT=%~dp0
pushd "%ROOT%"
set PYTHONUNBUFFERED=1
if not defined CONDA_BAT set CONDA_BAT=D:\Miniconda3\condabin\conda.bat
if not defined PYTHON_EXE set PYTHON_EXE=D:\Miniconda3\envs\asr\python.exe

echo ========================================
echo T2I Image Viewer
echo ========================================
echo Web UI: http://localhost:5565
echo ========================================
echo.

call %CONDA_BAT% activate asr
%PYTHON_EXE% -u t2i\image_viewer.py

pause
