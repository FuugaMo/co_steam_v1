@echo off
title SLM :5552
set ROOT=%~dp0
pushd "%ROOT%"
if not defined CONDA_BAT set CONDA_BAT=D:\Miniconda3\condabin\conda.bat
echo Starting SLM service...
%CONDA_BAT% activate asr && python -u slm\service.py --workers 2 --num-predict 150
pause
