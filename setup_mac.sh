#!/usr/bin/env bash
# Visitor Counting Console - one-time environment set-up for macOS (Apple Silicon).
# Creates .venv with Python 3.12, installs the dependencies and pre-downloads the model weights.
# Re-running is safe: existing packages are kept and only missing pieces are added.
set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found. Install it with:  curl -LsSf https://astral.sh/uv/install.sh | sh   (or: brew install uv)"
    exit 1
fi

if [ ! -x "$PY" ]; then
    echo "[1/4] Creating .venv (Python 3.12)"
    uv venv --python 3.12 .venv
else
    echo "[1/4] .venv already exists, reusing it"
fi

# macOS has no CUDA: the plain PyPI wheels of torch include Apple GPU (MPS) support.
echo "[2/4] Installing torch / torchvision"
uv pip install --python "$PY" torch torchvision

echo "[3/4] Installing requirements.txt"
uv pip install --python "$PY" -r requirements.txt

echo "[4/4] Pre-downloading model weights (detector + classifier) and checking the install"
"$PY" - <<'PYEOF'
import torch, cv2, ultralytics, transformers
print(f"  torch {torch.__version__}  (Apple GPU / MPS available: {torch.backends.mps.is_available()})")
print(f"  opencv {cv2.__version__}  ultralytics {ultralytics.__version__}  transformers {transformers.__version__}")
from ultralytics import YOLO
YOLO("yolo11s.pt")
from humanmonitor.classifier import DEFAULT_MODEL
from transformers import AutoModel, AutoTokenizer
AutoModel.from_pretrained(DEFAULT_MODEL)
AutoTokenizer.from_pretrained(DEFAULT_MODEL)
print(f"  classifier {DEFAULT_MODEL} ready")
import humanmonitor.detector, humanmonitor.classifier, humanmonitor.counter, humanmonitor.ui
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
echo "  1. Put your video files in sample_data/videos/ (or use RTSP URLs)."
echo "  2. Draw the trigger lines:   ./run_demo.sh --setup --sources sample_data/videos/<file>.mp4 ..."
echo "  3. Run the demo:             ./run_demo.sh --sources sample_data/videos/<file>.mp4 ..."
echo "See SETUP_MAC.md for details."
