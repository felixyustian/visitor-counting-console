"""Fetch the five demo videos used by `webapp.py --mode video` / `app.py`.

Video files are not committed (*.mp4 is git-ignored); this script downloads the
same public CCTV-style clips into sample_data/videos/ so every machine runs the
same demo. Starting trigger lines for them ship in config/lines.json.example.

    python download_demo_videos.py            # all five
    python download_demo_videos.py --force    # re-download

Needs yt-dlp (in requirements.txt) and ffmpeg on PATH (brew/apt/choco install ffmpeg).
Clips are limited to 720p to keep them small (~110 MB total).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# Each entry: output name, YouTube id, description, and optional clean-up applied
# after download. `trim` is (start, end) in seconds - several of these clips open
# or close on the uploader's title card, which would loop into the middle of a
# demo. `vf` is an ffmpeg filter chain, used here to remove a vendor watermark
# burnt into one clip's corner.
VIDEOS = [
    # file name                  YouTube id     description                                             trim        vf
    ("cam1_store_entrance.mp4",  "iJ5u4aGp9EE", "indoor store entrance, fixed CCTV, 3 min",             None,       None),
    ("cam2_street_corner.mp4",   "GJNjaRJWVP8", "street corner, fixed CCTV, 77 s",                      (7, 64),    None),
    ("cam3_retail_shop.mp4",     "KMJS66jBtVQ", "retail shop with entrance mat, fixed CCTV, 111 s",     None,       None),
    ("cam4_mall_doors.mp4",      "SvldnZ6qMGU", "automatic doors at a mall entrance, fixed CCTV, 121 s", None,      "delogo=x=752:y=2:w=204:h=62"),
    ("cam5_street_frontage.mp4", "66Ob1aJedHc", "roadside and footpath, fixed CCTV, 144 s",             (6, 132),   None),
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


def post_process(dest: Path, trim, vf) -> bool:
    """Re-encode `dest` in place with the trim and/or filter chain applied.

    -ss / -to go before -i so ffmpeg seeks instead of decoding and discarding.
    """
    pre = [] if trim is None else ["-ss", str(trim[0]), "-to", str(trim[1])]
    post = [] if vf is None else ["-vf", vf]
    tmp = dest.with_suffix(".tmp.mp4")
    cmd = ["ffmpeg", "-v", "error", *pre, "-i", str(dest), *post,
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-an", str(tmp), "-y"]
    if subprocess.run(cmd).returncode != 0 or not tmp.exists():
        print(f"[fail] post-processing {dest.name}")
        tmp.unlink(missing_ok=True)
        return False
    tmp.replace(dest)
    return True


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
    for name, vid, desc, trim, vf in VIDEOS:
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
            continue
        if (trim or vf) and not post_process(dest, trim, vf):
            failed += 1
            continue
        print(f"[ ok ] {name} ({dest.stat().st_size / 1e6:.1f} MB)")
    print("\nrun:  ./run_web.sh --mode video   (or run_web.bat --mode video)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
