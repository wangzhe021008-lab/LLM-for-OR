@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Missing location/.venv. Please see README.md.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" --version >nul 2>&1
if errorlevel 1 (
    echo The Python environment belongs to another computer or is unavailable.
    echo See README.md to recreate location/.venv on this computer.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -X utf8 start_web.py
if errorlevel 1 pause
exit /b %errorlevel%
