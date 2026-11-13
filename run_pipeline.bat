@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Please install the location/.venv environment first. See README.md.
    exit /b 1
)
if "%~1"=="" (
    ".venv\Scripts\python.exe" -X utf8 main.py run --description-file examples/problem.txt --solve copt --time-limit 300 --mip-gap 0.08
) else (
    ".venv\Scripts\python.exe" -X utf8 main.py run --description-file examples/problem.txt --solve copt --time-limit 300 --mip-gap 0.08 %*
)
exit /b %errorlevel%
