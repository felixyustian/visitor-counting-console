#!/usr/bin/env bash
# Visitor Counting Console - macOS / Linux launcher (counterpart of run_demo.bat).
# --device auto selects CUDA (NVIDIA), Apple MPS or CPU; pass --device to override.
# Extra arguments are passed through, e.g.  ./run_demo.sh --setup   or   ./run_demo.sh --sources rtsp://... a.mp4
set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python
if [ ! -x "$PY" ]; then
    echo "Virtual environment not found. Run ./setup_mac.sh first (see SETUP_MAC.md)."
    exit 1
fi

# --device auto picks CUDA, then Apple MPS, then CPU; pass --device to override.

exec "$PY" app.py "$@"
