@echo off
REM Visitor Counting Console - three sample videos in a 2x2 window grid.
REM Extra arguments are passed through, e.g.  run_demo.bat --setup   or   run_demo.bat --sources rtsp://... a.mp4
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
    echo Virtual environment not found. See README.md "Quick start".
    exit /b 1
)
.venv\Scripts\python.exe app.py %*
