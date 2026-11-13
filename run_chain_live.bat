@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Please install the location/.venv environment first. See README.md.
    exit /b 1
)
".venv\Scripts\python.exe" -X utf8 scripts/run_chain_batch.py --pipeline live %*
set "result=%errorlevel%"
if "%~1"=="" pause
exit /b %result%
