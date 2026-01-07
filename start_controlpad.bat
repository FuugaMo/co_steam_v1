@echo off
title ControlPad :5560
set ROOT=%~dp0
pushd "%ROOT%"
if not defined CONDA_BAT set CONDA_BAT=D:\Miniconda3\condabin\conda.bat
echo Starting Control Pad...
%CONDA_BAT% activate asr && python -u bridge\control_pad.py
pause
