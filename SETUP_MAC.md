# Running the demo on macOS (Apple Silicon)

> Cross-platform deployment (Ubuntu / Windows / macOS, services, RTSP set-up,
> troubleshooting) is covered in **DEPLOYMENT.md**; this file is the detailed
> macOS walkthrough.

The main `README.md` quick start targets Windows + NVIDIA CUDA. This guide is
the equivalent for this Mac (arm64, no CUDA). Two helper scripts do the work:

| script | purpose |
| --- | --- |
| `setup_mac.sh` | one-time environment set-up (venv, packages, model weights) |
| `run_demo.sh` | launcher – macOS counterpart of `run_demo.bat`, defaults to the Apple GPU |
| `run_web.sh` | browser console launcher – counterpart of `run_web.bat` (`--mode live` / `--mode video`), defaults to the Apple GPU |

## 1. Requirements

* macOS on Apple Silicon (tested: Darwin 27, M-series)
* [`uv`](https://docs.astral.sh/uv/) – `brew install uv` if missing
* Python 3.12 – `uv` downloads it automatically if not installed
* Internet access **once** for the model weights (`yolo11s.pt` ~19 MB,
  CLIP ~600 MB). No account or token needed.
* Video input: the three sample videos are **not** in the repo (`*.mp4` is
  gitignored). Bring your own files or RTSP URLs.

## 2. One-time set-up

```bash
cd ~/cv-demo-humanmonitor          # wherever the clone is
./setup_mac.sh
```

What it does:

1. `uv venv --python 3.12 .venv`
2. `uv pip install torch torchvision` – plain PyPI wheels (they include Apple
   GPU / MPS support; do **not** use the `cu128` index from the README)
3. `uv pip install -r requirements.txt`
4. Downloads `yolo11s.pt` and `openai/clip-vit-base-patch32` and checks that
   everything imports. Prints `MPS available: True` when the Apple GPU is usable.

Re-running the script is safe; it only adds what is missing.

## 3. Add the videos

Option A – the three public demo clips: `.venv/bin/python download_demo_videos.py`
(needs `brew install ffmpeg`). Trigger lines for them ship in `config/lines.json.example`,
copied to `config/lines.json` by `setup_mac.sh`.

Option A2 – the original three sample videos. Copy them to:

```
sample_data/videos/ShoppingMall.mp4
sample_data/videos/Checkpoint.mp4
sample_data/videos/Roadside.mp4
```

`config/lines.json` already contains trigger lines for these three (matched by
file name, so the Windows paths inside the file do not matter).

Option B – your own footage (up to 3 files and/or RTSP URLs). Put the files
anywhere; you will draw the trigger lines on first run (step 4).

## 4. Draw the trigger lines (once per new camera / file)

```bash
./run_demo.sh --setup --sources sample_data/videos/cam1.mp4 rtsp://user:pass@10.0.0.5:554/stream1
```

In the camera window: **left-click** the start and end of the line across the
entrance, **F** to flip the `IN` side, **S** to save, **E** to switch to
monitoring. Lines are stored in `config/lines.json`. Full key list in `README.md`.

## 5. Run the demo

```bash
# the three sample videos (if present in sample_data/videos/)
./run_demo.sh

# your own feeds
./run_demo.sh --sources sample_data/videos/cam1.mp4 sample_data/videos/cam2.mp4

# kiosk style: windows stay on top, paced at real-time speed
./run_demo.sh --topmost --realtime --sources ...
```

Four windows open (3 cameras + dashboard). **Q** / **Esc** quits, **R** resets
the counters, **P** pauses. Crossing events are appended to `output/events.csv`.

Any option from `python app.py --help` can be appended to `run_demo.sh`.

## 6. Device / performance notes

* `--device auto` (the default) picks CUDA, then Apple MPS, then CPU, so the
  launchers no longer need to force a device. Override with
  `./run_demo.sh --device cpu ...` if MPS misbehaves.
* Measured on this machine (three 720p frames per iteration):

  | stage | mps | cpu |
  | --- | --- | --- |
  | batched YOLO11s + 3× ByteTrack | ~24 ms | ~94 ms |
  | CLIP demographics (1 crop) | ~330 ms | ~40 ms |

  CLIP on MPS has a high per-call overhead; with real footage try both devices
  and keep whichever gives the higher `pipeline fps` on the dashboard.
* If it is still too slow: `--model yolo11n.pt` (smaller detector),
  `--stride 2` (skip file frames), `--classify-every 10`, or
  `--no-demographics` (counting only).

## 7. Troubleshooting

| symptom | fix |
| --- | --- |
| `Virtual environment not found` | run `./setup_mac.sh` |
| `Missing sample video: ...` | copy the videos to `sample_data/videos/` or pass `--sources` |
| `uv not found` | `brew install uv` |
| No windows appear / OpenCV GUI error | run from Terminal.app / iTerm, not over SSH or inside a headless session |
| Wrong entry direction (entries counted as exits) | `--setup`, hover the camera, press **F**, then **S** |
| Line does not load for a renamed file | lines are keyed by path, then by file name (`config.py`); re-draw with `--setup` |
| HF Hub "unauthenticated requests" warning | harmless; models are public |
