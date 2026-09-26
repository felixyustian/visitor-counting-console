"""Fetch the three demo videos used by `webapp.py --mode video` / `app.py`.

Video files are not committed (*.mp4 is git-ignored); this script downloads the
same public CCTV-style clips into sample_data/videos/ so every machine runs the
same demo. Starting trigger lines for them ship in config/lines.json.example.

    python download_demo_videos.py            # all three
    python download_demo_videos.py --force    # re-download

Needs yt-dlp (in requirements.txt) and ffmpeg on PATH (brew/apt/choco install ffmpeg).
Clips are limited to 720p to keep them small (~70 MB total).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

VIDEOS = [
    # file name                 YouTube id      description
    ("cam1_store_entrance.mp4", "iJ5u4aGp9EE", "indoor store entrance, fixed CCTV, 3 min"),
    ("cam2_street_corner.mp4",  "GJNjaRJWVP8", "street corner, fixed CCTV, 77 s"),
    ("cam3_retail_shop.mp4",    "KMJS66jBtVQ", "retail shop with entrance mat, fixed CCTV, 111 s"),
]
OUT_DIR = Path(__file__).parent / "sample_data" / "videos"


def yt_dlp_cmd() -> list[str]:
    """Prefer the yt-dlp installed next to this interpreter (kept current by requirements.txt)."""
    exe = Path(sys.executable).parent / ("yt-dlp.exe" if sys.platform == "win32" else "yt-dlp")
    if exe.exists():
        return [str(exe)]
    if shutil.which("yt-dlp"):
        return ["yt-dlp"]
    return [sys.executable, "-m", "yt_dlp"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="re-download existing files")
    ap.add_argument("--max-height", type=int, default=720)
    args = ap.parse_args()

    if not shutil.which("ffmpeg"):
        print("ffmpeg not found on PATH - install it first (brew install ffmpeg / apt install ffmpeg / choco install ffmpeg)")
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    failed = 0
    for name, vid, desc in VIDEOS:
        dest = OUT_DIR / name
        if dest.exists() and not args.force:
            print(f"[skip] {name} exists ({dest.stat().st_size / 1e6:.1f} MB)")
            continue
        print(f"[get ] {name}  <-  https://www.youtube.com/watch?v={vid}   ({desc})")
        h = args.max_height
        cmd = yt_dlp_cmd() + [
            "--no-warnings", "-q", "--no-playlist",
            "-f", f"bv*[height<={h}][ext=mp4]+ba[ext=m4a]/b[height<={h}][ext=mp4]/b[height<={h}]",
            "--merge-output-format", "mp4",
            "-o", str(dest),
            f"https://www.youtube.com/watch?v={vid}",
        ]
        if subprocess.run(cmd).returncode != 0 or not dest.exists():
            print(f"[fail] {name} - if this is a 403, update yt-dlp: pip install -U yt-dlp")
            failed += 1
        else:
            print(f"[ ok ] {name} ({dest.stat().st_size / 1e6:.1f} MB)")
    print("\nrun:  ./run_web.sh --mode video   (or run_web.bat --mode video)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
