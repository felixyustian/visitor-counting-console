#!/usr/bin/env bash
# Browser console launcher (macOS / Linux; Windows: run_web.bat). Opens http://localhost:8780
#   ./run_web.sh                 # video mode (files listed in config/sources.json)
#   ./run_web.sh --mode live     # live mode (RTSP URLs listed in config/sources.json)
#   ./run_web.sh --sources a.mp4 rtsp://user:pass@host/stream   # explicit feeds
# Extra arguments are passed through to webapp.py (see --help).
set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python
if [ ! -x "$PY" ]; then
    echo "Virtual environment not found. Run ./setup_mac.sh first (see SETUP_MAC.md)."
    exit 1
fi

# --device auto selects CUDA (NVIDIA), Apple MPS or CPU; pass --device to override.
exec "$PY" webapp.py "$@"
