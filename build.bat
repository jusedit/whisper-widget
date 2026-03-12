@echo off
cd /d "%~dp0"

echo ========================================
echo   Building WhisperWidget.exe
echo ========================================
echo.

REM --- Download embedded Python if not present ---
if not exist "python-embed.zip" (
    echo Downloading Python 3.13 embeddable package...
    powershell -Command "Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.13.2/python-3.13.2-embed-amd64.zip' -OutFile 'python-embed.zip'"
    if errorlevel 1 (
        echo Failed to download Python embeddable package.
        pause
        exit /b 1
    )
    echo   Done.
) else (
    echo Python embeddable package already downloaded.
)

REM --- Download get-pip.py if not present ---
if not exist "get-pip.py" (
    echo Downloading get-pip.py...
    powershell -Command "Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile 'get-pip.py'"
    if errorlevel 1 (
        echo Failed to download get-pip.py.
        pause
        exit /b 1
    )
    echo   Done.
) else (
    echo get-pip.py already downloaded.
)

echo.

REM --- Install PyInstaller ---
echo Installing PyInstaller...
venv\Scripts\pip install pyinstaller --quiet 2>nul

echo Building WhisperWidget.exe...
venv\Scripts\pyinstaller launcher.py ^
    --onefile ^
    --noconsole ^
    --name WhisperWidget ^
    --icon favicon.ico ^
    --add-data "main.py;." ^
    --add-data "presplash.pyw;." ^
    --add-data "overlay.py;." ^
    --add-data "config.py;." ^
    --add-data "recorder.py;." ^
    --add-data "transcriber.py;." ^
    --add-data "settings_dialog.py;." ^
    --add-data "model_downloader.py;." ^
    --add-data "assets.py;." ^
    --add-data "perf_logger.py;." ^
    --add-data "requirements.txt;." ^
    --add-data "favicon.ico;." ^
    --add-data "python-embed.zip;." ^
    --add-data "get-pip.py;." ^
    --clean

echo.
if exist dist\WhisperWidget.exe (
    echo ========================================
    echo   dist\WhisperWidget.exe ready!
    echo.
    echo   Drop it anywhere - Desktop, Downloads,
    echo   pin to Start Menu or Taskbar.
    echo.
    echo   No Python installation required!
    echo   App installs to: %%LOCALAPPDATA%%\WhisperWidget
    echo ========================================
) else (
    echo Build failed.
)
pause
