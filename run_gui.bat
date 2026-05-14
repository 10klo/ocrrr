@echo off
setlocal
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo Virtual environment not found. Run: python -m venv venv ^&^& venv\Scripts\pip install -r requirements.txt
    pause >nul
    exit /b 1
)

call venv\Scripts\python gui.py
if errorlevel 1 (
    echo.
    echo GUI exited with an error. Press any key to close.
    pause >nul
)
