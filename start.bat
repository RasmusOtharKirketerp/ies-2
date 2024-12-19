@echo off
echo Setting up virtual environment...

REM Check if Python is installed
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Create and activate virtual environment
if not exist .venv (
    echo Creating new virtual environment...
    python -m venv .venv
)

call .venv\Scripts\activate.bat
if %ERRORLEVEL% NEQ 0 (
    echo Failed to activate virtual environment
    pause
    exit /b 1
)

REM Install dependencies
echo Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo Failed to install dependencies
    pause
    exit /b 1
)

echo Starting application...
python app.py

pause