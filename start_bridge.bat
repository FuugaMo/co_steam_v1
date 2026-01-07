@echo off
title Bridge :5555
set ROOT=%~dp0
pushd "%ROOT%"
if not defined CONDA_BAT set CONDA_BAT=D:\Miniconda3\condabin\conda.bat
echo Starting Bridge service...
echo Connecting to ASR :5551 and SLM :5552...
%CONDA_BAT% activate asr && python -u bridge\service.py
pause
