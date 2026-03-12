@echo off
cd /d "%~dp0"

if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Python not found. Install Python 3.11+ from python.org
        pause
        exit /b 1
    )
)

venv\Scripts\pip install -r requirements.txt --quiet 2>nul

rem Show splash immediately (PyQt6 only, no torch — instant startup)
start "" venv\Scripts\pythonw.exe presplash.pyw

rem Start main app (imports torch ~1.5s, loads model ~3.5s)
start "" venv\Scripts\pythonw.exe main.py
