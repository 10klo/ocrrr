@echo off
setlocal
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate
    pip install -r requirements.txt
    pip install pyinstaller
) else (
    call venv\Scripts\activate
)

echo Building ocrrr.exe...
pyinstaller ocrrr.spec

if errorlevel 1 (
    echo.
    echo Build failed. Check the output above for errors.
    pause >nul
    exit /b 1
)

echo.
echo Build complete: dist\ocrrr\ocrrr.exe
pause >nul
