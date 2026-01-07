@echo off
title ASR :5551
set ROOT=%~dp0
pushd "%ROOT%"
if not defined CONDA_BAT set CONDA_BAT=D:\Miniconda3\condabin\conda.bat
echo Starting ASR service...
%CONDA_BAT% activate asr && python -u asr\service.py
pause
