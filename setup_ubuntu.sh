#!/usr/bin/env bash
# Visitor Counting Console - one-time environment set-up for Ubuntu 22.04/24.04 with an NVIDIA GPU.
# Creates .venv (Python 3.12), installs CUDA-enabled torch, the requirements and pre-downloads the model weights.
# Re-running is safe: existing packages are kept and only missing pieces are added.
#
#   ./setup_ubuntu.sh                # CUDA 12.8 wheels (driver >= 570) - default
#   CUDA_INDEX=cu126 ./setup_ubuntu.sh   # older driver (>= 525): CUDA 12.6 wheels
#   CUDA_INDEX=cpu   ./setup_ubuntu.sh   # no NVIDIA GPU
set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python
CUDA_INDEX="${CUDA_INDEX:-cu128}"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found. Install it with:  curl -LsSf https://astral.sh/uv/install.sh | sh   then re-open the shell."
    exit 1
fi

# OpenCV (video decode / GUI) and ffmpeg (RTSP, demo video download) system libraries
if command -v apt-get >/dev/null 2>&1; then
    NEED=()
    for pkg in libgl1 libglib2.0-0 ffmpeg; do
        dpkg -s "$pkg" >/dev/null 2>&1 || NEED+=("$pkg")
    done
    if [ ${#NEED[@]} -gt 0 ]; then
        echo "[0/4] Installing system packages: ${NEED[*]}  (sudo)"
        sudo apt-get update -qq && sudo apt-get install -y -qq "${NEED[@]}"
    fi
fi

if [ ! -x "$PY" ]; then
    echo "[1/4] Creating .venv (Python 3.12)"
    uv venv --python 3.12 .venv
else
    echo "[1/4] .venv already exists, reusing it"
fi

echo "[2/4] Installing torch / torchvision ($CUDA_INDEX)"
uv pip install --python "$PY" torch torchvision --index-url "https://download.pytorch.org/whl/$CUDA_INDEX"

echo "[3/4] Installing requirements.txt"
uv pip install --python "$PY" -r requirements.txt

echo "[4/4] Pre-downloading model weights (detector + classifier) and checking the install"
"$PY" - <<'PYEOF'
import torch, cv2, ultralytics, transformers
cuda = torch.cuda.is_available()
print(f"  torch {torch.__version__}  CUDA available: {cuda}" + (f"  ({torch.cuda.get_device_name(0)})" if cuda else "  -> will run on CPU"))
print(f"  opencv {cv2.__version__}  ultralytics {ultralytics.__version__}  transformers {transformers.__version__}")
from ultralytics import YOLO
YOLO("yolo11s.pt")
from humanmonitor.classifier import DEFAULT_MODEL
from transformers import AutoModel, AutoTokenizer
AutoModel.from_pretrained(DEFAULT_MODEL)
AutoTokenizer.from_pretrained(DEFAULT_MODEL)
print(f"  classifier {DEFAULT_MODEL} ready")
import humanmonitor.engine
print("  models downloaded, project modules import OK")
PYEOF

mkdir -p sample_data/videos output

# config/*.json are site-specific and gitignored: seed them from the committed
# templates so a pull never collides with this machine's own calibration.
for f in sources lines; do
    [ -f "config/$f.json" ] || cp "config/$f.json.example" "config/$f.json"
done


echo
echo "Done. Next steps:"
echo "  1. Demo footage:   $PY download_demo_videos.py      (or put your own files in sample_data/videos/)"
echo "  2. Browser demo:   ./run_web.sh --mode video         then open http://localhost:8780"
echo "     Live cameras:   edit config/sources.json, then ./run_web.sh --mode live"
echo "  3. OpenCV windows: ./run_demo.sh --sources sample_data/videos/*.mp4"
echo "  Optional TensorRT: $PY export_tensorrt.py   then add  --model yolo11s.engine"
