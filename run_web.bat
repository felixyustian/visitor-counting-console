@echo off
REM Browser console launcher (Windows). Opens http://localhost:8780
REM   run_web.bat                 video mode (files listed in config\sources.json)
REM   run_web.bat --mode live     live mode (RTSP URLs listed in config\sources.json)
REM   run_web.bat --sources a.mp4 rtsp://user:pass@host/stream
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
    echo Virtual environment not found - see README.md quick start.
    exit /b 1
)
.venv\Scripts\python.exe webapp.py %*
