# Deployment guide – Visitor Counting Console

How to install, configure, run and operate the demo on **Ubuntu (NVIDIA)**,
**Windows (NVIDIA)** and **macOS (Apple Silicon)**. The browser console
(`webapp.py`) is the deployment target; the OpenCV-window app (`app.py`) is the
same pipeline for a single-screen kiosk and is covered where it differs.

Contents

1. [What gets deployed](#1-what-gets-deployed)
2. [Requirements](#2-requirements)
3. [Ubuntu 22.04 / 24.04 + NVIDIA](#3-ubuntu-2204--2404--nvidia)
4. [Windows 10 / 11 + NVIDIA](#4-windows-10--11--nvidia)
5. [macOS (Apple Silicon)](#5-macos-apple-silicon)
6. [Configuration](#6-configuration)
7. [Operating the console](#7-operating-the-console)
8. [Performance tuning](#8-performance-tuning)
9. [Security notes](#9-security-notes)
10. [Troubleshooting](#10-troubleshooting)
11. [Upgrade / uninstall](#11-upgrade--uninstall)

---

## 1. What gets deployed

```
CCTV (RTSP) or video files ──► webapp.py (FastAPI, port 8780)
                                  └─ humanmonitor.engine (background thread)
                                       YOLO11 person detection (batched, 3 feeds)
                                       ByteTrack tracking (one per feed)
                                       trigger-line entry / exit counting
                                       SigLIP 2 zero-shot gender + adult/child
                                  ├─ /stream/{0,1,2}  MJPEG with overlays  ──► 3 camera panels
                                  ├─ /ws              JSON state ~4 Hz     ──► crowd console panel
                                  └─ output/events.csv  one row per crossing
```

Everything runs in **one Python process on one machine** with a GPU; the
browser (any modern Chrome/Edge/Firefox on the same LAN) only renders. There is
no database, no cloud dependency and no account: the two model downloads
(`yolo11s.pt` 18 MB, `google/siglip2-base-patch16-256` ~1.4 GB) are public and
happen once during set-up.

Files that matter after installation:

| path | purpose | persists |
| --- | --- | --- |
| `.venv/` | Python 3.12 + all packages (~1–5 GB depending on the CUDA build) | recreate with the set-up script |
| `yolo11s.pt` (18 MB), `~/.cache/huggingface/` (~1.4 GB; `%USERPROFILE%\.cache\huggingface` on Windows) | model weights | keep |
| `config/sources.json` | the *live* (RTSP) and *video* (file) source sets | **site-specific – back up** |
| `config/lines.json` | trigger lines and counting areas drawn per camera | **site-specific – back up** |
| `config/site.json` | building hours, capacity, warning %, starting occupancy (set-up screen) | **site-specific – back up** |
| `sample_data/videos/` | demo clips (not in git - fetched per machine) | optional |
| `output/events.csv` | crossing log (append-only) | rotate / archive |
| `output/history.csv` | hourly / closing totals, corrections, recorded events | **keep – the HISTORY panel reads it** |
| `output/events/` | recorded clips (MP4) and snapshots (JPEG), ~10 MB per clip | archive / prune |

The three `config/*.json` files above are **written by the app and are not in
git**. Each is created on first use from the committed template beside it
(`config/sources.json.example`, `config/lines.json.example`); until then the app
reads the template directly, so a fresh clone runs with no set-up. Keeping them
untracked is what stops `git pull` on a deployment machine from colliding with
that machine's own calibration, and keeps camera passwords out of the
repository. Back them up with the `output/` folder.

## 2. Requirements

| | minimum | recommended |
| --- | --- | --- |
| GPU | NVIDIA with 4 GB VRAM (GTX 1650 / T4) or Apple M-series | RTX 3060 or better; 6 GB+ VRAM |
| CPU / RAM | 4 cores, 8 GB | 8 cores, 16 GB |
| Disk | 8 GB free | 20 GB (venv + weights + video archive) |
| OS | Ubuntu 22.04 / Windows 10 21H2 / macOS 14 | Ubuntu 24.04 / Windows 11 / macOS 15 |
| NVIDIA driver | ≥ 525 (`CUDA_INDEX=cu126`) | ≥ 570 (default `cu128` wheels) |
| Network | RTSP (TCP 554) reachable from the machine; outbound HTTPS **once** for the model download | wired LAN to the cameras; 1080p RTSP ≈ 4–8 Mbit/s per camera |
| Browser | Chrome / Edge / Firefox, current | 1920×1080 display for the 2×2 grid |

Three 1080p feeds on an RTX 3060 Ti run at ~11–14 pipeline fps (one iteration =
one frame from every camera, i.e. real time for 12–25 fps cameras); an M-series
Mac manages ~5–10 fps (demo/dev only, see §5). CPU-only works
(`CUDA_INDEX=cpu`) at 1–3 fps and is only useful for a smoke test.

Common tools on every platform:

* [`uv`](https://docs.astral.sh/uv/) – creates the venv and installs Python 3.12 itself if missing.
* `ffmpeg` – only for `download_demo_videos.py` (video mode); RTSP decoding uses OpenCV's bundled FFmpeg.
* `git` (or a zip of the repo).

---

## 3. Ubuntu 22.04 / 24.04 + NVIDIA

### 3.1 Driver

```bash
sudo ubuntu-drivers install            # or: sudo apt install nvidia-driver-570
sudo reboot
nvidia-smi                             # must list the GPU; note "CUDA Version" ≥ 12.6
```

### 3.2 Install

```bash
sudo apt install -y git ffmpeg
curl -LsSf https://astral.sh/uv/install.sh | sh && exec $SHELL     # uv
git clone <repo-url> /opt/cv-demo-humanmonitor && cd /opt/cv-demo-humanmonitor
./setup_ubuntu.sh                       # driver ≥ 570 (CUDA 12.8 wheels)
# CUDA_INDEX=cu126 ./setup_ubuntu.sh    # driver 525–569
# CUDA_INDEX=cpu   ./setup_ubuntu.sh    # no GPU (smoke test only)
```

The script installs `libgl1 libglib2.0-0 ffmpeg` (asks for sudo once), creates
`.venv`, installs torch/torchvision from the matching PyTorch index, the
requirements, downloads both models and prints
`CUDA available: True  (NVIDIA GeForce RTX ...)`. Re-running it is safe.

### 3.3 First run (video mode)

```bash
.venv/bin/python download_demo_videos.py     # three public CCTV clips, ~70 MB
./run_web.sh --mode video                    # http://<host>:8780
```

Open the console from any browser on the LAN. Trigger lines for the demo clips
are pre-configured; press **calibrate** on a panel to adjust (§6.2).

### 3.4 Go live

Either edit `config/sources.json` → `"live"` with the camera URLs (§6.1), or
start with an empty list and add the cameras on the page (⚙ / **+ add camera**
on each panel - they are saved back to `sources.json`), then

```bash
./run_web.sh --mode live --capacity 150
```

### 3.5 Run as a service (systemd)

`/etc/systemd/system/visitor-console.service`:

```ini
[Unit]
Description=Visitor Counting Console
After=network-online.target
Wants=network-online.target

[Service]
User=cvdemo
WorkingDirectory=/opt/cv-demo-humanmonitor
ExecStart=/opt/cv-demo-humanmonitor/.venv/bin/python webapp.py --mode live --capacity 150 --host 0.0.0.0 --port 8780
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1
# Environment=HF_HUB_OFFLINE=1   # after the first successful run, if the site has no internet

[Install]
WantedBy=multi-user.target
```

```bash
sudo useradd -r -d /opt/cv-demo-humanmonitor -s /usr/sbin/nologin cvdemo   # or use an existing user
sudo chown -R cvdemo: /opt/cv-demo-humanmonitor
sudo -u cvdemo HOME=/opt/cv-demo-humanmonitor .venv/bin/python -c "from humanmonitor.classifier import DEFAULT_MODEL; from transformers import AutoModel; AutoModel.from_pretrained(DEFAULT_MODEL)"  # cache under the service user
sudo systemctl daemon-reload && sudo systemctl enable --now visitor-console
sudo journalctl -u visitor-console -f          # logs: [init] ..., [event] ..., [web] ...
sudo ufw allow 8780/tcp                        # if ufw is enabled
```

Note the model cache lives under the **service user's** home; run the
pre-download once as that user (line 3) or point `HF_HOME` at a shared folder.

### 3.6 Docker alternative

Needs Docker ≥ 24 and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
(`sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`).

```bash
python3 download_demo_videos.py          # optional, on the host (needs yt-dlp + ffmpeg)
docker compose up --build -d             # video mode, http://<host>:8780
MODE=live CAPACITY=150 docker compose up -d
docker compose logs -f
```

`config/`, `output/` and `sample_data/` are bind-mounted from the host, so
calibration and the event log persist; weights are cached in the `models`
volume. The image is built from `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime`
(~7 GB). *The Docker path has not yet been exercised on a GPU host – treat the
first build as a validation step.*

### 3.7 TensorRT (optional, ~1.5–2× detector speed-up)

```bash
.venv/bin/python export_tensorrt.py          # installs tensorrt on first use, builds yolo11s.engine
./run_web.sh --mode live --model yolo11s.engine
```

Engines are tied to the GPU model and TensorRT version: rebuild after a driver
or GPU change, never copy between machines.

---

## 4. Windows 10 / 11 + NVIDIA

### 4.1 Driver and tools

1. Install the current NVIDIA Game Ready / Studio driver (≥ 570) and reboot; `nvidia-smi` in a terminal must list the GPU.
2. In an **Administrator** PowerShell:
   ```powershell
   winget install --id=astral-sh.uv -e
   winget install --id=Gyan.FFmpeg -e          # only for download_demo_videos.py
   winget install --id=Git.Git -e              # or download the repo as a zip
   ```
   Re-open the terminal so `uv` / `ffmpeg` are on `PATH`.

### 4.2 Install

```bat
git clone <repo-url> C:\cv-demo-humanmonitor
cd C:\cv-demo-humanmonitor
setup_windows.bat
REM older driver (525-569):   set CUDA_INDEX=cu126 && setup_windows.bat
REM no GPU:                    set CUDA_INDEX=cpu   && setup_windows.bat
```

Expect `CUDA available: True  (NVIDIA GeForce RTX ...)` at the end. Keep the
path short and without spaces (`C:\cv-demo-humanmonitor`) – some CUDA wheels
misbehave in long or non-ASCII paths.

### 4.3 First run

```bat
.venv\Scripts\python.exe download_demo_videos.py
run_web.bat --mode video
```

then open <http://localhost:8780>. On the first start Windows Defender Firewall
asks whether to allow Python on private networks – **Allow** if other PCs
should open the console. To do it explicitly:

```powershell
New-NetFirewallRule -DisplayName "Visitor console 8780" -Direction Inbound -Protocol TCP -LocalPort 8780 -Action Allow
```

### 4.4 Go live

Edit `config\sources.json` → `"live"` (§6.1) or add the cameras on the page (⚙ / **+ add camera**), then `run_web.bat --mode live --capacity 150`.

### 4.5 Auto-start at logon (Task Scheduler)

```powershell
$action  = New-ScheduledTaskAction -Execute "C:\cv-demo-humanmonitor\.venv\Scripts\python.exe" `
           -Argument "webapp.py --mode live --capacity 150" -WorkingDirectory "C:\cv-demo-humanmonitor"
$trigger = New-ScheduledTaskTrigger -AtLogOn
Register-ScheduledTask -TaskName "VisitorConsole" -Action $action -Trigger $trigger -RunLevel Highest
```

Logs go to the task's console window; to keep a file, run it through
`cmd /c "...python.exe webapp.py ... >> output\console.log 2>&1"`. A desktop
shortcut to `run_web.bat --mode live` is the simplest alternative for a demo PC.

### 4.6 Kiosk / OpenCV windows

`run_demo.bat --sources rtsp://... rtsp://... rtsp://... --topmost` opens four
always-on-top windows (dashboard top-left + 3 cameras) sized to the primary
screen. Keys are listed in `README.md`. TensorRT works the same as on Ubuntu
(`.venv\Scripts\python.exe export_tensorrt.py`, then `--model yolo11s.engine`).

---

## 5. macOS (Apple Silicon)

macOS has no CUDA; the pipeline runs on the Apple GPU (MPS). It is the
**development / demo-on-a-laptop** platform: three 720p files give ~5–10
pipeline fps, enough for a walkthrough, not for a site deployment.

```bash
brew install uv ffmpeg
git clone <repo-url> && cd cv-demo-humanmonitor
./setup_mac.sh                                # venv, MPS torch, models; prints "MPS available: True"
.venv/bin/python download_demo_videos.py
./run_web.sh --mode video                     # http://localhost:8780
./run_web.sh --mode live                      # RTSP from config/sources.json
./run_demo.sh --sources sample_data/videos/*.mp4     # OpenCV windows (run from Terminal, not SSH)
```

Notes

* `--device auto` selects MPS. CLIP on MPS has a high per-call overhead; if
  `pipeline fps` is low try `--device cpu` or `--no-demographics`.
* Do not build a TensorRT engine or use `CUDA_INDEX` here.
* The OpenCV windows need a logged-in desktop session (they will not open over SSH).
* `SETUP_MAC.md` has the longer walkthrough for this Mac.

---

## 6. Configuration

### 6.1 Sources – `config/sources.json`

Not in git: copied from `config/sources.json.example` the first time a camera is
saved on the page, or create it yourself with `cp config/sources.json.example
config/sources.json` (Windows: `copy config\sources.json.example
config\sources.json`) and edit the URLs.

```json
{
  "live": [
    "rtsp://admin:Secret%40123@192.168.1.101:554/Streaming/Channels/102",
    "rtsp://admin:Secret%40123@192.168.1.102:554/Streaming/Channels/102",
    "rtsp://admin:Secret%40123@192.168.1.103:554/Streaming/Channels/102"
  ],
  "video": [
    "sample_data/videos/cam1_store_entrance.mp4",
    "sample_data/videos/cam2_street_corner.mp4",
    "sample_data/videos/cam3_retail_shop.mp4"
  ]
}
```

* Up to `--max-cameras` entries per set (**6** by default; the shipped template lists five); an entry is a plain string or `{"src": "...", "name": "Main entrance"}` for a display name. `--sources a b c` on the command line overrides the file (files and URLs can be mixed).
* The list may be empty: cameras added on the page (⚙ on a panel / **+ add camera**) are written back here under the mode the server was started with, so they survive a restart. Empty slots are dropped when saving, so cameras move up to the first free slots.
* **Passwords with special characters go in as they are.** `@`, `:`, `$`, `%`,
  `&`, spaces and the rest are percent-encoded for you on the way to FFmpeg, so
  paste exactly what the camera's own page shows. The stored config keeps what
  you typed, so the ⚙ form shows it back unchanged.
  Three characters cannot be handled automatically and must be encoded by hand:
  **`/` → `%2F`, `?` → `%3F`, `#` → `%23`**. They end the URL's authority, which
  makes `rtsp://a/b@c` genuinely ambiguous - RFC 3986 reads it as host `a` with
  path `/b@c`, and no amount of guessing can tell the two apart. The page rejects
  such a URL with that message rather than failing to connect for unclear reasons.
* **Watch the shell**, not just the app. `$$` in a double-quoted or unquoted
  command-line argument is the shell's process id, so
  `--sources "rtsp://admin:Pa$$w0rd@..."` sends a different password every run.
  Use single quotes, or add the camera on the page instead.
* Typical URL formats (sub-stream = lower resolution, recommended when the GPU is small):

  | vendor | main stream | sub stream |
  | --- | --- | --- |
  | Hikvision / HiLook | `rtsp://u:p@IP:554/Streaming/Channels/101` | `.../Channels/102` |
  | Dahua / Imou | `rtsp://u:p@IP:554/cam/realmonitor?channel=1&subtype=0` | `subtype=1` |
  | Axis | `rtsp://u:p@IP/axis-media/media.amp` | `...media.amp?resolution=1280x720` |
  | Uniview | `rtsp://u:p@IP:554/unicast/c1/s0/live` | `.../s1/live` |
  | generic ONVIF | look up "Stream URI" in the camera's ONVIF/web UI | |

* Verify a URL before deploying: `ffplay -rtsp_transport tcp "rtsp://..."` (or VLC → Open Network Stream).
* RTSP is forced to **TCP**; streams reconnect automatically every 3 s when lost and show an *offline* alert meanwhile.

### 6.2 Trigger lines and counting areas – `config/lines.json`

Not in git: written the first time a line is saved. Until then the five demo
clips' starting lines are read from `config/lines.json.example`.

One line per camera across the entrance; a person is counted when their head
crosses it. In the browser: **calibrate** → click the two end points → **flip IN**
if the arrow points the wrong way → **save** → **done**. Counting is paused while
any panel is calibrating. Lines are stored in normalised coordinates keyed by the
source URL/path (files also match by name), so they survive a resolution change
and a moved folder. For RTSP cameras the line can only be drawn once the stream
is connected.

Placement tips: perpendicular to the walking direction, inside the door line
(not on the door itself), long enough to cover the whole opening, away from
where people queue or stand. A crossing still counts **8 % of the segment
length past each end**, drawn on the calibration overlay as a faded dashed
extension; saving a line within 4 % of the frame edge warns, because people can
cross it before the tracker has established them.

**Counting areas.** The same entry may carry a rectangle limiting what the
camera does, drawn with **draw area** on the panel:

```json
"sample_data/videos/cam1_store_entrance.mp4": {
  "p1": [0.2, 0.64], "p2": [0.78, 0.64], "entry_side": 1,
  "name": "cam1_store_entrance",
  "region": [0.55, 0.06, 0.99, 0.99],
  "region_role": "count_inside"
}
```

`region` is `[x1, y1, x2, y2]` normalised like everything else. `region_role`
is one of:

| role | effect |
| --- | --- |
| `count_inside` *(default)* | only people inside the box are tracked at all |
| `ignore_inside` | people inside are skipped – a window, a mirror, a poster |
| `classify_inside` | everyone is counted; only those inside are sexed and aged |

A box must be at least **5 % of the frame** on each side; anything smaller is
refused as a stray click.

`count_inside` **must contain the whole trigger line**, with room on both sides
for people to be picked up before they reach it: a person is only tracked once
inside the box, so one approaching from outside is first seen already at the
line and is seeded on the wrong side. The console warns when a box is saved
that does not. The other two roles leave tracking alone, so with them the line
may sit anywhere, including outside the box. An entry written before roles
existed, or one with an unrecognised `region_role`, loads as `count_inside`.

### 6.3 Building settings – `config/site.json`

Entered on the set-up screen the first time the page opens on a new day (or via
the *site* link): site name, date, opening / closing time (closing may be after
midnight), capacity (= 100 %), warning threshold %, starting occupancy. The
scheduler uses the server's local clock: the count resets to the starting
occupancy at opening time, an **hourly** row is written on the hour while open
and a **closing** row at closing time. Rows are never duplicated after a
restart. The file is per site and not in git.

### 6.4 Validating accuracy on site footage

Gender and adult/child accuracy is decided by camera geometry, not by this
code: a camera mounted high and angled at the tops of heads will do worse than
the published figures whatever model runs behind it. Measure it per site before
anyone quotes a number:

```bash
.venv/bin/python scripts/ab_demographics.py collect --out output/ab --sources <recorded clips>
.venv/bin/python scripts/ab_demographics.py sheet   --out output/ab   # label output/ab/labels.csv by hand
.venv/bin/python scripts/ab_demographics.py report  --out output/ab   # accuracy per model and per camera
```

`collect` saves the person crop of every crossing; `sheet` lays them out for
labelling (leave unclear ones blank). `votes` and `score` compare vote schemes
and models on the same labelled set. On the demo clips this gives 94.5% gender
and 100% adult/child for the live pipeline over 56 labelled crossings - treat
those as a method check, not a site prediction.

**Counting accuracy is not covered by this tool.** Validate entries and exits
against a manual tally at the door for an hour at each entrance.

### 6.5 Command-line options (`webapp.py --help`)

| option | default | notes |
| --- | --- | --- |
| `--mode live\|video` | `video` | which set of `sources.json` |
| `--host`, `--port` | `0.0.0.0`, `8780` | use `--host 127.0.0.1` to keep the console local |
| `--capacity N` | from `site.json` | overrides the capacity for this run; the UI edit persists to `site.json` |
| `--max-cameras N` | `6` | upper bound of camera panels |
| `--clip-pre`, `--clip-post` | `10`, `20` | seconds before / after a recorded event |
| `--history`, `--events-dir` | `output/history.csv`, `output/events/` | history CSV and clip folder |
| `--device` | `auto` | `cuda:0`, `mps`, `cpu` |
| `--classifier` | `google/siglip2-base-patch16-256` | gender/age model; `openai/clip-vit-base-patch32` is ~4x faster per crop and ~4 points worse on gender |
| `--model` | `yolo11s.pt` | `yolo11n.pt` faster / `yolo11m.pt` more accurate / `yolo11s.engine` TensorRT |
| `--imgsz` | `640` | detector input size; 960 for far-away people (slower) |
| `--conf` | `0.25` | detector confidence |
| `--initial-count N` | `100` | people assumed inside at start-up (site-wide baseline; `0` for a known-empty site) |
| `--debug-tracks [file]` | off | per-track decision log every `--debug-every` frames (default `output/tracks.csv`) |
| `--stride N` | auto | process every Nth frame of a **file** (streams always use the latest frame) |
| `--classify-every`, `--max-crops` | `15`, `6` | demographic workload per frame (lighter than CLIP-era defaults; SigLIP 2 needs fewer looks) |
| `--child-thresh` | `0.6` | P(child) needed to label a child |
| `--no-demographics` | off | counting only, roughly halves GPU load |
| `--stream-width`, `--stream-fps`, `--jpeg-quality` | `960`, `15`, `80` | MJPEG bandwidth to the browser |
| `--events` | `output/events.csv` | crossing log path |

---

## 7. Operating the console

For running a *demo* rather than a deployment — pre-flight checks, the order to
show things in, and the things that go wrong in front of an audience — see
**[DEMO_RUNBOOK.md](DEMO_RUNBOOK.md)**, which covers all three platforms.

* **Start / stop**: launcher in a terminal (Ctrl+C stops within ~3 s) or the service/task from §3.5 / §4.5.
* **Health check**: `curl -s http://localhost:8780/api/state | head -c 300` – contains `"error": null`, per-camera `"status": "LIVE"` and a rising `pipeline_fps`. Non-zero `"error"` means the pipeline thread died; restart and read the log.
* **Logs** (stdout): `[init]` set-up, `[event] <camera>: ENTRY/EXIT track #id gender= age=` per crossing, `[setup]` line changes, `[web]` server. Under systemd: `journalctl -u visitor-console`.
* **Alerts** in the console: over capacity (≥100 %), approaching (≥85 %), busy (≥60 %), camera offline, file ended, no trigger line, calibrating, paused, child entered. *ack* silences one until it clears and re-raises.
* **Day start**: confirm the set-up screen (date, hours, capacity, starting occupancy). The count then resets itself at opening time every day; *reset counters* in the footer does it manually. The CSVs are never cleared by the app.
* **Corrections**: the *site* link opens one form holding the head count, the capacity and the warning threshold - the numbers on the panel are a display, not click-to-edit. Changes are logged as `correction` rows.
* **Alarms**: red banner + red band on every camera frame at capacity, naming the camera that tipped the count; amber band at the warning threshold; per-camera *area capacity* alarms. A clip of the camera involved is recorded automatically (`output/events/`).
* **Recording**: **● record** on a panel saves a 30 s clip + snapshot and an `event` row with a note; open them from HISTORY → *Recorded events*.
* **Panels**: **+ camera** adds a panel (up to `--max-cameras`); ⚙ → *remove panel* deletes one. GPU load grows with every live camera - watch `pipeline fps`.
* **Swapping a feed**: a panel whose slot is paired in `sources.json` (a `live` and a `video` list of the same length) carries **▸ live** / **▸ video**; hovering it and pressing **`v`** does the same, **Shift+V** swaps all of them.
* **Counting areas**: **calibrate** → **draw area** → pick the role → drag a box (§6.2). While a box is being drawn the buttons that do not belong to that flow are disabled, because the button strip sits on top of the picture and a corner placed near the bottom of the frame would otherwise hit one of them. **clear area** removes it.
* **In area now**: each panel reports who is in the counted part of the frame right now, split M/F and A/C, independently of the trigger line - a presence figure, so it rises and falls, reports zero for a camera that is not delivering frames, and is not written to `history.csv`.
* **Event log**: `output/events.csv` – `timestamp, camera_id, camera_name, source, track_id, direction, gender, age, p_male, p_child, n_obs`. Archive/rotate it (e.g. weekly `mv` + restart); a day of a busy entrance is a few hundred KB.
* **Capacity** set in the UI is not saved – put the permanent value in the launcher (`--capacity`).
* **Multiple viewers** are fine; each open browser costs three extra MJPEG encodes on the CPU, so reduce `--stream-width 640 --stream-fps 10` when several screens are connected.

## 8. Performance tuning

Watch `pipeline fps` in the console footer (each iteration processes one frame from every camera). Target ≥ 10 fps for reliable head-crossing detection at walking pace.

| symptom | first try | then |
| --- | --- | --- |
| low fps, GPU 100 % | `--model yolo11n.pt` | TensorRT (§3.7), `--imgsz 512`, camera sub-streams |
| low fps, GPU idle, CPU 100 % | camera sub-streams (decode cost), `--stream-width 640` | fewer viewers, `--stride 2` (files) |
| demographics slow (MPS/CPU) | `--classifier openai/clip-vit-base-patch32` (~4x faster per crop) | `--classify-every 25 --max-crops 4`, or `--no-demographics` |
| missed crossings on fast walkers | raise fps as above | shorten the line's dead zone: not configurable – place the line where people walk slower |
| people far from the camera not detected | `--imgsz 960`, `--conf 0.25` | move/zoom the camera; ≥ 48 px person height is needed for demographics |

Reference: RTX 3060 Ti, three 1080p files, defaults → ~11–14 fps (README *Performance*); M-series MacBook → 5–10 fps; CPU → 1–3 fps.

## 9. Security notes

* The console has **no authentication**. Bind it to `--host 127.0.0.1` when only the local browser needs it, or keep it on an isolated CCTV VLAN / behind a reverse proxy with auth (nginx `auth_basic`, or the site's SSO proxy) for shared access. Do not expose port 8780 to the internet.
* `config/sources.json` contains camera credentials in clear text: restrict it (`chmod 600`, owner = service user) and use a dedicated read-only camera account.
* Frames are processed in memory and never written to disk; the only persisted personal data are aggregate counts and `events.csv` (no images, track IDs are session-local). Decide the retention period for the CSV with the site.
* Outbound network access is only needed once (model download); set `HF_HUB_OFFLINE=1` afterwards to keep the process from contacting the Hub.
* Weights come from public sources (Ultralytics GitHub release, HuggingFace `google/siglip2-base-patch16-256`, Apache-2.0) – pin the versions in `requirements.txt` for a reproducible install and keep the `.venv` off any auto-update.

## 10. Troubleshooting

| symptom | cause / fix |
| --- | --- |
| `CUDA available: False` after set-up | driver too old for the wheels → `CUDA_INDEX=cu126 ./setup_ubuntu.sh` (or `set CUDA_INDEX=cu126`), reboot after driver install; `nvidia-smi` must work first |
| `torch ... not compiled with CUDA` / `device=cpu` on an NVIDIA box | venv was created with the CPU wheels → delete `.venv` and re-run the set-up script |
| `OFFLINE (cannot open)` for a camera | wrong URL/credentials, camera blocks a 4th client, or UDP-only camera → test with `ffplay -rtsp_transport tcp` (single-quote the URL so the shell does not eat a `$`), or use the sub-stream |
| stream connects then drops every few seconds | network MTU/packet loss on Wi-Fi → wired LAN, sub-stream, check camera "max connections" |
| `libGL.so.1: cannot open shared object` (Ubuntu) | `sudo apt install libgl1 libglib2.0-0` (the set-up script does this) |
| `Address already in use` | another instance running → `pkill -f webapp.py` / Task Manager, or `--port 8781` |
| console shows *connection lost – reconnecting* | server stopped or a proxy that does not pass WebSockets (`/ws`) → check the process; nginx needs `proxy_set_header Upgrade $http_upgrade; Connection "upgrade"` |
| camera panel black, dot red, `NO SIGNAL` | source offline; the MJPEG stream resumes automatically once the RTSP reconnects |
| `HTTP Error 403` from `download_demo_videos.py` | outdated yt-dlp → `.venv/bin/python -m pip install -U yt-dlp` (or `uv pip install -U yt-dlp`) |
| HF Hub *unauthenticated requests* warning | harmless, models are public |
| Model download blocked by a proxy | set `HTTPS_PROXY`, or copy `yolo11s.pt` + `~/.cache/huggingface/hub/models--google--siglip2-base-patch16-256` from another machine and set `HF_HUB_OFFLINE=1` |
| entries and exits swapped | **calibrate** → **flip IN** → **save** |
| a camera stops counting after an area was drawn | the area's role is `count_inside` and the trigger line is not inside it, so people are first seen already at the line → enlarge the box to cover the whole line with room on both sides, or switch the role to `classify_inside` (§6.2). Saving such a box warns |
| everyone reads `??` on one camera | a `classify_inside` area that nobody walks through, or one covering only distant figures → move or enlarge it, or **clear area** |
| a person is counted twice / not at all | line on the door itself or in a queue area → move it; fps below ~10 → §8 |
| OpenCV windows do not open | headless session (SSH, service) → the browser console is the headless option; on macOS run from Terminal |
| Windows: `python.exe` blocked by SmartScreen / AV | allow `.venv\Scripts\python.exe`; keep the folder out of OneDrive-synced paths |
| `pipeline failed: ...` in the log | read the traceback above it; most common are a decode error on a corrupt file (`--no-loop` to stop at the end) and out-of-memory on a 4 GB GPU (`--model yolo11n.pt`, sub-streams) |

## 11. Upgrade / uninstall

**Upgrade**: `git pull` (or replace the folder, keeping `config/`, `output/` and
`sample_data/`), re-run the platform set-up script (installs only what
changed), restart the service. `config/lines.json` and `sources.json` are
forward-compatible, and since they are untracked a pull leaves them alone.

*One-off, when upgrading a machine cloned before these files were untracked*:
that machine still tracks them, so the pull that removes them reports either
`Your local changes ... would be overwritten by merge` or deletes them. Save
them first, pull, then put them back:

```bash
cp config/lines.json   ~/lines.site.bak.json      # if present
cp config/sources.json ~/sources.site.bak.json
git checkout -- config/lines.json config/sources.json   # only if the pull was refused
git pull
cp ~/lines.site.bak.json   config/lines.json      # restore this site's calibration
cp ~/sources.site.bak.json config/sources.json
```

After that the files stay put and no later pull touches them.

**Uninstall**: stop the service/task, delete the repo folder, optionally
`rm -rf ~/.cache/huggingface/hub/models--openai--clip-vit-base-patch32`
(Windows: `%USERPROFILE%\.cache\huggingface`). Nothing is installed outside the
folder except `uv`, `ffmpeg` and the NVIDIA driver.
