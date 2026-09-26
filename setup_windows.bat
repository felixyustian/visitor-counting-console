@echo off
REM Visitor Counting Console - one-time environment set-up for Windows 10/11 with an NVIDIA GPU.
REM Creates .venv (Python 3.12), installs CUDA-enabled torch, the requirements and pre-downloads the model weights.
REM Re-running is safe.
REM
REM   setup_windows.bat                 CUDA 12.8 wheels (driver >= 570) - default
REM   set CUDA_INDEX=cu126 ^&^& setup_windows.bat    older driver (>= 525)
REM   set CUDA_INDEX=cpu   ^&^& setup_windows.bat    no NVIDIA GPU
setlocal
cd /d "%~dp0"
if "%CUDA_INDEX%"=="" set CUDA_INDEX=cu128
set PY=.venv\Scripts\python.exe

where uv >nul 2>nul
if errorlevel 1 (
    echo uv not found. Install it with:  winget install --id=astral-sh.uv -e   ^(or: pip install uv^)  then re-open the terminal.
    exit /b 1
)
where ffmpeg >nul 2>nul
if errorlevel 1 echo NOTE: ffmpeg not on PATH - needed only for download_demo_videos.py ^(winget install Gyan.FFmpeg^)

if not exist "%PY%" (
    echo [1/4] Creating .venv ^(Python 3.12^)
    uv venv --python 3.12 .venv || exit /b 1
) else (
    echo [1/4] .venv already exists, reusing it
)

echo [2/4] Installing torch / torchvision ^(%CUDA_INDEX%^)
uv pip install --python "%PY%" torch torchvision --index-url https://download.pytorch.org/whl/%CUDA_INDEX% || exit /b 1

echo [3/4] Installing requirements.txt
uv pip install --python "%PY%" -r requirements.txt || exit /b 1

echo [4/4] Pre-downloading model weights ^(detector + classifier^) and checking the install
"%PY%" -c "import torch,cv2,ultralytics,transformers; c=torch.cuda.is_available(); print(f'  torch {torch.__version__}  CUDA available: {c}' + (f'  ({torch.cuda.get_device_name(0)})' if c else '  -> will run on CPU')); from ultralytics import YOLO; YOLO('yolo11s.pt'); from humanmonitor.classifier import DEFAULT_MODEL; from transformers import AutoModel, AutoTokenizer; AutoModel.from_pretrained(DEFAULT_MODEL); AutoTokenizer.from_pretrained(DEFAULT_MODEL); import humanmonitor.engine; print('  models downloaded, project modules import OK')" || exit /b 1

if not exist sample_data\videos mkdir sample_data\videos
if not exist output mkdir output

rem config\*.json are site-specific and gitignored: seed them from the committed
rem templates so a pull never collides with this machine's own calibration.
if not exist config\sources.json copy /y config\sources.json.example config\sources.json >nul
if not exist config\lines.json copy /y config\lines.json.example config\lines.json >nul

echo.

echo Done. Next steps:
echo   1. Demo footage:   %PY% download_demo_videos.py      (or put your own files in sample_data\videos\)
echo   2. Browser demo:   run_web.bat --mode video          then open http://localhost:8780
echo      Live cameras:   edit config\sources.json, then run_web.bat --mode live
echo   3. OpenCV windows: run_demo.bat --sources sample_data\videos\cam1_store_entrance.mp4 ...
echo   Optional TensorRT: %PY% export_tensorrt.py   then add  --model yolo11s.engine
endlocal
