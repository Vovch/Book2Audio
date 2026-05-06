@echo off
setlocal EnableExtensions
cd /d "%~dp0"

if /i "%~1"=="amd" (set "IS_AMD=1") else (set "IS_AMD=0")

where python >nul 2>&1
if errorlevel 1 (
    echo ERROR: python is not on PATH. Install Python 3.10+ from python.org or use the "py" launcher.
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 exit /b 1
)

echo Upgrading pip...
call ".venv\Scripts\python.exe" -m pip install -U pip
if errorlevel 1 exit /b 1

if "%IS_AMD%"=="1" (
    echo Installing Book2Audio with DirectML profile ^(.[amd]^)...
    call ".venv\Scripts\pip.exe" install ".[amd]"
) else (
    echo Installing Book2Audio ^(default^)...
    call ".venv\Scripts\pip.exe" install .
)
if errorlevel 1 exit /b 1

echo.
echo Done. Activate:  .venv\Scripts\activate
echo Then run:      book2audio-tts
endlocal
exit /b 0
