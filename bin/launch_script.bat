@echo off
setlocal enabledelayedexpansion

set CONDA_ENV=envNTT
set SCRIPT_PATH=C:\Users\EMALAB\Desktop\TW_DAQ\fast_tagger_gui\main.py

:: Activate the Conda environment and run the script
call conda activate %CONDA_ENV%
if %errorlevel% neq 0 (
    echo Failed to activate Conda environment.
    pause
    exit /b %errorlevel%
)

python %SCRIPT_PATH%
if %errorlevel% neq 0 (
    echo Script execution failed.
    pause
    exit /b %errorlevel%
)

:: Deactivate the Conda environment
call conda deactivate

echo Script executed successfully.
pause