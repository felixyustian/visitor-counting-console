# Visitor Counting Console

Counts people entering and leaving a venue from the CCTV that is already on the
wall, and breaks the count down by gender and age group. No turnstile, no new
hardware, no app on anyone's phone — a trigger line drawn on each camera's
picture and a GPU.

![The console counting three feeds](docs/console.jpg)

*Three clips running live: 13 visitors so far, 11 still inside, split by
male/female and adult/child. Camera 1 has a counting area drawn over the shop
floor, reporting `4 HERE 2M 2F 4A 0C` — who is standing in it right now, without
waiting for anyone to cross the line.*

```
CCTV feeds ──► YOLO11 person detection (one batched pass for all cameras)
           ──► ByteTrack tracking (one tracker per camera)
           ──► head point crosses the camera's trigger line  ──► entry (+1) / exit (−1)
           ──► SigLIP 2 zero-shot gender & adult/child, voted over the whole track
           ──► dashboard counters + output/events.csv
```

Two front-ends over one engine: OpenCV windows for a kiosk (`app.py`), or a
FastAPI browser console with MJPEG overlays and a ~4 Hz WebSocket state feed
(`webapp.py`). Up to six cameras, added and removed on the page.

### The decisions worth reading

Counting people is easy to do badly and the failures are quiet, so most of the
work here is in the parts that stop a plausible-looking number from being wrong.

| | |
| --- | --- |
| **Gender and age are voted differently** | Gender takes the last 3 looks plus one at the moment of crossing counted double; age averages everything. The classifier's gender read swings early in a track and settles at the line, while "child" is noisy per frame and smooths out. **90.9% → 94.5%** and **96.4% → 100%** on 56 hand-labelled crossings against averaging everything ([details](#demographic-classifier)) |
| **Visitors and occupancy are different numbers** | Someone who walks in and out has still visited, so visitors-today only rises while occupancy is net and falls again. Conflating them is the most common way a footfall dashboard misleads |
| **The line counts 8% past each end** | So a line drawn a little short of a doorway still catches the edges — and the overlay *draws* that reach, because an invisible tolerance is indistinguishable from a bug when the numbers look off |
| **A drawn area has a role** | *Count only inside*, *ignore inside*, or *classify only inside*. Restricting what is tracked and restricting what is classified are different needs; only the first constrains where the line may go, and the console warns when it is violated |
| **Nothing is assumed** | The opening occupancy defaults to 0 rather than a guess, because a non-zero baseline has to be split 50/50 male/female by assumption and that guess would propagate into every demographic figure |
| **Runtime config is not in git** | Every line dragged on a frame is written back to `config/`. Those files are gitignored and seeded from committed templates, so a deployment machine's own calibration never collides with a `git pull` |

## Quick start

Full install / service / RTSP / troubleshooting instructions per platform are in
**[DEPLOYMENT.md](DEPLOYMENT.md)**, and **[DEMO_RUNBOOK.md](DEMO_RUNBOOK.md)** is
the short version for actually running a demo on Ubuntu, Windows or macOS —
pre-flight checks, how to start and stop on each, what to show in order, and
what to do when something misbehaves in front of an audience. In short: one set-up script per platform; each creates `.venv` (Python 3.12 via `uv`),
installs the right PyTorch build, the requirements and pre-downloads the model
weights. Re-running is safe.

| platform | GPU | set-up | run (browser console) | run (OpenCV windows) |
| --- | --- | --- | --- | --- |
| Ubuntu 22.04 / 24.04 | NVIDIA CUDA | `./setup_ubuntu.sh` | `./run_web.sh --mode video` | `./run_demo.sh --sources ...` |
| Windows 10 / 11 | NVIDIA CUDA | `setup_windows.bat` | `run_web.bat --mode video` | `run_demo.bat --sources ...` |
| macOS (Apple Silicon) | Apple MPS | `./setup_mac.sh` | `./run_web.sh --mode video` | `./run_demo.sh --sources ...` |
| Ubuntu + Docker | NVIDIA CUDA | – | `docker compose up --build` | – |

`--device auto` (the default) picks CUDA, then Apple MPS, then CPU. On CUDA the
detector and CLIP run in FP16 automatically. Prerequisites: [`uv`](https://docs.astral.sh/uv/)
(`curl -LsSf https://astral.sh/uv/install.sh | sh` / `winget install astral-sh.uv`),
an NVIDIA driver ≥ 570 for the default CUDA 12.8 wheels (`CUDA_INDEX=cu126` for
older drivers, `CUDA_INDEX=cpu` without a GPU) and `ffmpeg` for the demo footage.

```bash
./setup_ubuntu.sh                                 # or setup_windows.bat / ./setup_mac.sh
.venv/bin/python download_demo_videos.py          # three public CCTV clips -> sample_data/videos/
./run_web.sh --mode video                         # http://localhost:8780
./run_web.sh --mode live                          # RTSP URLs from config/sources.json
./run_demo.sh --sources rtsp://user:pass@10.0.0.5:554/stream1 rtsp://10.0.0.6/live entrance.mp4
```

Model weights (`yolo11s.pt` from GitHub, `google/siglip2-base-patch16-256` from
the HuggingFace hub) download on first run. **No account or token is required** –
both are public. Video files loop forever by default (`--no-loop` to stop at
the end); RTSP streams reconnect automatically.

### Demo footage

`download_demo_videos.py` fetches three public fixed-camera clips (720p, ~70 MB)
with `yt-dlp`; they are not kept in git, so each machine downloads its own copy.
`config/lines.json.example` already holds a starting trigger line for each
(adjust with *calibrate* in the browser):

| file | view | source |
| --- | --- | --- |
| `cam1_store_entrance.mp4` | indoor store entrance door, 3 min | youtube.com/watch?v=iJ5u4aGp9EE |
| `cam2_street_corner.mp4` | street corner pavement, 77 s | youtube.com/watch?v=GJNjaRJWVP8 |
| `cam3_retail_shop.mp4` | retail shop with entrance mat, 111 s | youtube.com/watch?v=KMJS66jBtVQ |

These are third-party YouTube uploads used for an internal demo only; for a
customer-facing or published demo use the site's own footage or a CC0 source
(Pexels / Pixabay clips are licence-clean but short and rarely fixed-camera).

### NVIDIA: TensorRT and Docker

* **TensorRT** – `python export_tensorrt.py` builds `yolo11s.engine` (FP16,
  batch 3) on the target GPU; run with `--model yolo11s.engine`. Engines are
  GPU- and TensorRT-version specific, so export on each machine.
* **Docker (Ubuntu + NVIDIA Container Toolkit)** – `docker compose up --build`
  builds from `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime` and serves the
  browser console on port 8780 with the GPU attached; `MODE=live` / `CAPACITY=150`
  environment variables select the source set and room capacity. Weights are
  cached in the `models` volume; `config/`, `output/` and `sample_data/` are
  bind-mounted so calibration and the event log persist on the host.

## Browser console (crowd management panel + camera panels)

`webapp.py` runs the same pipeline as `app.py` but renders in the browser: a
**crowd management console** in the top-left tile and one panel per camera
(1–6, added and removed on the page; the grid adapts). Dark operations theme.

```
CCTV feeds ──► humanmonitor.engine (YOLO11 + ByteTrack + line counter + CLIP)
           ──► /stream/{cam}   MJPEG with overlays (encoded once, shared)  ──► camera panels
           ──► /ws             JSON snapshot ~4 Hz                          ──► crowd console
           ──► output/history.csv  hourly / closing / corrections / events
```

Two source sets in `config/sources.json`, selected at launch (the file is
written by the app; until it exists the committed
`config/sources.json.example` is read instead):

```powershell
run_web.bat --mode live      # "live":  RTSP URLs                (Windows)
run_web.bat --mode video     # "video": video files             (default)
./run_web.sh --mode live     # macOS / Linux equivalents
./run_web.sh --sources rtsp://user:pass@10.0.0.5/live a.mp4   # explicit feeds
```

Then open **http://localhost:8780** (`--port`, `--max-cameras 6`, `--stream-width /
--stream-fps / --jpeg-quality` for bandwidth, `--clip-pre / --clip-post` for
recordings).

**Building day set-up.** The first time the page opens on a new day it asks for
the site name, date, opening and closing time, capacity, warning threshold (%)
and the starting occupancy before showing the console; the *site* link reopens
it. Settings live in `config/site.json`. At opening time the count restarts
from the starting occupancy; while open an **hourly** row and at closing time a
**closing** row go to `output/history.csv`.

**Information panel.** Light-grey, high-contrast card, three columns side by
side so the whole site fits in one view without scrolling:

| column | headline | beneath it |
| --- | --- | --- |
| **VISITORS TODAY** | `TOTAL VISITORS` - everyone who entered since opening | male/female and adult/child of those visitors, plus a *historical →* link |
| **CURRENT OCCUPANCY** *(NORMAL / BUSY / CROWDED / OVER badge)* | `PEOPLE INSIDE` - who is inside now | male/female and adult/child of the people inside |
| **CAMERAS DEPLOYED** | one card per camera | that entrance's net, its male/female and adult/child, and its area capacity if set |

**The two headline figures are different things, deliberately.** VISITORS TODAY
only ever rises during the day, because someone who walks in and out has still
visited;
CURRENT OCCUPANCY is the net figure and falls again when they leave. Each box
sums to the two tiles beneath it. Under them sits a single line of context -
capacity, warning threshold, opening hours and the day's in/out - then
**ALERTS**, and **⤢** expands the console to the cameras, crossings, hourly
rows and closing totals.

Corrections and settings both live behind the *site* link: the head count (e.g.
after walking the floor), the capacity and the warning threshold. The numbers
themselves are no longer click-to-edit - the dashed underlines and the "click a
number to edit" hint were removed as clutter - so the panel now reads as a
display and every change goes through one form. Corrections are logged as
`correction` rows in `history.csv`.

**Per-entrance cards.** The CAMERAS DEPLOYED column gives every camera a card
carrying the net for that entrance and its splits, so a headline number can
always be read back to the doors it came from. Clicking a card focuses the
whole panel on that camera (click again for the whole site).

**One camera per row, whatever the count.** Every name, net and split is the same
size however many cameras there are, so the column scans top to bottom. The list
scrolls inside its own fixed-height box rather than growing: adding a sixth or
seventh camera lengthens the list instead of pushing the alerts off the panel.
An earlier version paired the cards into two columns past four cameras, which
shrank the text and turned the column into a grid of small tiles.

A camera that drops off **hides its numbers and shows `OFFLINE`** instead - the
figures it had were frozen at the moment the link dropped, and showing them
would imply they were still live. The
same numbers are drawn into each camera frame, bottom right, so a recorded clip
carries them.

**Alarms.** At the warning threshold an amber band appears at the top of the
console; at capacity a red **OVER CAPACITY** banner names the camera whose entry
tipped the count, every camera frame gets a red alarm band (also in recorded
clips), the tipping camera is highlighted, and a clip of it is recorded
automatically. Each camera can also carry an **area capacity** of its own
(⚙ → *area capacity*): its net count is shown as `area n / cap` in the panel
header and raises an *AREA OVER CAPACITY* alarm on that panel.

**Cameras on the page.** ⚙ on a panel (or **+ add camera** in an empty one)
takes an RTSP/HTTP URL or a server-side video path, a display name and the area
capacity; *connect* swaps the feed at run time and saves it to
`config/sources.json` (under the mode the server was started with). *remove
camera* empties the slot, *remove panel* deletes it, **+ camera** in the
console header adds one (up to `--max-cameras`). Credentials are masked in the
UI afterwards.

**Passwords go in as typed.** `@`, `:`, `$`, `%`, `&` and spaces are structural
in a URL, and are percent-encoded for you on the way to FFmpeg - paste whatever
the camera's own page shows. `/`, `?` and `#` are the exception and must be
encoded by hand (`%2F`, `%3F`, `%23`): they end the URL's authority, so
`rtsp://a/b@c` is indistinguishable from host `a` with path `/b@c`. The form
says so rather than failing to connect for unclear reasons. Note this is about
the *page*: on a command line, `$$` is the shell's process id, so single-quote
any URL you pass with `--sources`.

**Swapping a panel between its video file and its live feed.** Give
`config/sources.json` a `live` and a `video` list of the same length and slot
*i* pairs with the other list's entry *i*; the panel then carries a
**▸ live** / **▸ video** button. Hovering a panel and pressing **`v`** does the
same, **Shift+V** swaps every panel at once. A camera pointed at a quiet corridor produces an empty frame, and an empty
frame demonstrates nothing - so a demo runs on the clips.

**Trigger lines on the frames.** *calibrate* loads the saved line onto the
frame with draggable end points; click two points to draw a new one, *flip IN*
to reverse the entry direction, *save*, *done*. Counting is paused while any
panel is calibrating. Lines go to `config/lines.json`, shared with `app.py`.

The draft overlay draws two things that used to be invisible and made a line
look as though it counted in the wrong place: the picture's real edge (the
panel is usually wider than the frame, so the image sits letterboxed inside
it), and the **8 % of the segment length past each end that still counts** -
shown as a faded dashed extension. A line saved within 4 % of the frame edge
warns: people can cross it before the tracker has established them.

**Counting areas.**

![A counting area drawn over the shop floor](docs/counting-area.jpg)

*draw area* limits what a camera does to a rectangle. Pick
what the area is *for* from the dropdown beside the button, then drag a box
across the picture (or click two opposite corners - drag when the box needs to
reach an edge, since the button strip sits on the picture and only the start of
a drag has to miss it):

| role | effect | may the line sit outside it? |
| --- | --- | --- |
| **count only inside** | only people inside are tracked at all; outside is dimmed | **no** - saving warns |
| **ignore inside** | people inside are skipped - a window, a mirror, a poster | yes |
| **classify only inside** | everyone is counted, but only those inside are sexed and aged | yes |

A box must be at least **5 % of the frame** on each side; smaller is refused as
a stray click.

*count only inside* must contain the whole line with room on both sides: a
person is only tracked once inside the box, so one approaching from outside is
first seen already at the line and their side is seeded wrongly. The other two
leave tracking alone, which is what makes a line outside the box valid for
them. *classify only inside* is the one to reach for on a wide scene - put it
where people are large and well lit and the classifier's crop budget stops
being spent on distant figures; tracks outside it still count, they just read
`??`.

**What an area holds right now.** An area reports its own population without
waiting for anyone to cross the line - drawn under the box label on the frame
(`COUNTING AREA` / `3 HERE  1M 2F  3A 0C`) and repeated in the panel as
**in area now**, or **in view now** with no area drawn. A trailing `?` counts
people the classifier has not decided yet, so the splits always add up to the
total. It is a *presence* figure, not a crossing figure: it rises and falls as
people walk through shot, it is the number to point at when a clip starts
mid-scene and the crossing counters are still at zero, and a camera that stops
delivering frames reports zero rather than the last thing it saw. It is not
written to `history.csv`, which stays crossing-derived.

**Recording events.** **● record** on a camera saves the last 10 s and the next
20 s of that camera as an MP4 plus a snapshot to `output/events/` and adds an
*event* row (with your note) to `history.csv`; over-capacity alarms do the same
automatically. Clips are linked from the HISTORY section.

**HISTORY section.** Last closing totals (visitors, entries/exits, male/female,
adult/child), today's hourly rows, the previous days' closing totals and the
recorded events. `history.csv` columns: `timestamp, kind, date, occupancy,
entries, exits, male, female, adult, child, visitors, v_male, v_female, v_adult,
v_child, capacity, pct, cameras, note, clip` (`male`..`child` are net inside,
`visitors`/`v_*` are cumulative entries; a file with the older layout is moved
aside and re-written on start-up).

Other controls: *acknowledge all* / *ack* silence an alert until it clears and
re-raises, *pause* / *reset counters* in the footer, **⤢** expands any panel.

### Demographic classifier

`--classifier` (alias `--clip`) selects the zero-shot gender / adult-child model:

| model | gender | adult/child | per-crop cost (MPS) | pipeline (3×720p, MPS) |
| --- | --- | --- | --- | --- |
| `google/siglip2-base-patch16-256` (default) | **98.2 %** | 89.3 % | ~15 ms | 14.7 fps |
| `google/siglip2-base-patch16-224` | – | – | ~12 ms | ~15 fps |
| `openai/clip-vit-base-patch32` | 94.5 % | 89.3 % | ~4 ms | 16.5 fps |

Accuracy measured on 56 hand-labelled crossings from the demo clips with
`scripts/ab_demographics.py`; both models share the same prompt ensembles and
thresholds. SigLIP 2 is ~4× more expensive per crop (256 vision tokens vs 49),
so the default schedule is lighter (`--classify-every 15 --max-crops 6`, a new
track is still classified the moment it appears) — that keeps the pipeline
within ~10 % of CLIP's frame rate while gaining ~4 points of gender accuracy.
On an NVIDIA GPU the per-crop cost is far smaller and the schedule can be
tightened again. Use `--classifier openai/clip-vit-base-patch32` to go back.

### Validating accuracy on your own footage

Every accuracy figure above comes from `scripts/ab_demographics.py`, run on the
demo clips. Site cameras differ - mounting height and angle decide how well
gender and age work - so re-run it on real footage before trusting the numbers:

```bash
.venv/bin/python scripts/ab_demographics.py collect --out output/ab --sources <clips>
.venv/bin/python scripts/ab_demographics.py sheet   --out output/ab   # label output/ab/labels.csv
.venv/bin/python scripts/ab_demographics.py votes   --out output/ab   # compare vote schemes
.venv/bin/python scripts/ab_demographics.py score   --out output/ab   # compare models
.venv/bin/python scripts/ab_demographics.py report  --out output/ab   # accuracy per model and camera
```

`collect` saves the person crop of every crossing plus what the pipeline decided;
`sheet` lays them out in numbered contact sheets for hand labelling (leave a row
blank when the crop is too unclear to call). Counting accuracy - entries and
exits - is not measured here: check that against a manual tally at the door.

## Trigger line set-up (OpenCV windows)

Each camera needs a trigger line across the entrance. Lines are stored in
`config/lines.json`, keyed by the video path or RTSP URL, in normalised
coordinates so they survive a resolution change. Starting lines for the three
sample videos ship in `config/lines.json.example`, which is read until the first
line is saved.

1. Start with `--setup`, or press **E** in the running app (counting pauses).
2. **Left-click** the start point and the end point of the line on a camera window.
3. A green **triangle** perpendicular to the line marks the *inside*: it points the
   way a person walks to be counted as an **entry**; crossing against it is an
   **exit**. **Right-click** that camera window (or press **F**) reverses it.
   **C** clears the points you have drawn.
4. Press **S** (or **s**) to save. Press **E** to return to monitoring.
   **X** deletes the saved line of the camera under the mouse.

## Counting rules

* A person is counted when the **head point** of their track crosses the line.
  The head is the top-centre of the body box (8 % down), or the mean of the
  facial keypoints when a pose model is used (`--model yolo11s-pose.pt`).
* Crossing toward the `IN` side is an **entry (+1)**, the reverse is an
  **exit (−1)**. Occupancy and its gender/age splits move with the same sign,
  so they show who is inside now; the browser console's VISITORS TODAY counts
  entries only and never falls.
* `--initial-count` (default **0**) is the number of people assumed already
  inside at start-up; `visitors = initial + entries − exits`. It defaults to 0
  so that **no gender or age figure ever carries an assumption** - a non-zero
  baseline has to be split by guesswork (50/50 male/female, 80/20 adult/child)
  and that guess would propagate into the demographics. Set it on the day
  set-up screen, or pass it, when a real opening head count is known. The
  baseline is site-wide, so per-camera rows still start at 0 and show only the
  flow through that entrance, and **R** restarts from the baseline.
* A **counting area** can restrict a camera to part of the frame, either to
  what is tracked at all or to what is classified - see *Counting areas* under
  the browser console above. Whoever a tracking area excludes is not counted,
  not classified and not drawn, so they never reach these rules.
* No double counting:
  * one event per side transition per track ID – a person standing on the line
    is never counted twice;
  * a dead-zone (1 % of the frame) around the line plus a 3-frame confirmation
    suppress detector jitter;
  * the crossing must happen within the drawn segment, plus a tolerance of
    **8 % of its length past each end** so a line drawn a little short of a
    doorway still catches the edges of it (the browser draws that reach as a
    faded dashed extension, so it is not a surprise in the numbers);
  * a brand-new track ID crossing at the same spot right after another one
    (tracker ID switch) is ignored;
  * track IDs stay unique across cameras and across file loops.
* Gender and age are **voted over the track**, but not the same way:
  **gender** uses the last 3 classifications plus one taken at the instant of
  crossing (counted double) - the view at the line is the informative one, and the
  classifier's read swings a lot earlier in a track (median P(male) spread 0.49);
  **age** averages every observation, because "child" is a noisy per-frame call
  that smooths out. Measured on 56 hand-labelled crossings from the demo clips:
  gender 90.9% -> 94.5%, adult/child 96.4% -> 100% against the previous
  average-everything vote (`scripts/ab_demographics.py votes` reproduces this).
* Dashboard window: total visitors (entries − exits), male / female,
  adult / child, and a per-camera table (status, fps, in/out, M/F, A/C).
* `output/events.csv`: one row per crossing –
  `timestamp, camera_id, camera_name, source, track_id, direction, gender, age, p_male, p_child, n_obs` –
  the raw dataset that can be aggregated hourly / daily / weekly / monthly.

## Useful options

```
--model yolo11n.pt | yolo11s.pt | yolo11m.pt | yolo11s-pose.pt   detector (pose = keypoint head)
--conf 0.25              detector confidence (0.35 is ~25% faster but drops small / occluded people)
--stride 0               process every Nth file frame (0 = auto: ~30 processed fps, i.e. 2 for 60 fps files)
--classify-every 15      re-run demographics on each track every N processed frames
--min-crop-height 48     skip demographics for people smaller than this (px)
--child-thresh 0.6       P(child) needed to count a child
--initial-count 0        people assumed already inside at start-up (0 = count only what is seen)
--realtime               pace files at their (strided) frame rate instead of as fast as possible
--topmost                keep the four windows above other applications (kiosk / demo)
--no-loop                stop at the end of a file instead of looping
--no-demographics        counting only
--device cpu             force CPU
```

## Performance (RTX 3060 Ti, three 1080p files)

| stage | cost |
| --- | --- |
| batched YOLO11s detection + 3x ByteTrack | ~40 ms per iteration |
| classifier on the due crops of all cameras | ~10–25 ms (CLIP-era measurement) |
| pipeline throughput | ~11–14 iterations/s (each iteration = one frame from every camera) |

Measured with CLIP and the old classification schedule; SigLIP 2 costs ~4x more
per crop, offset by the lighter schedule (`--classify-every 15 --max-crops 6`).
On an M-series Mac the same three 720p clips run at 14.7–16.5 fps either way.

Video decoding runs in one thread per file, the detection/classification
pipeline in its own thread and rendering in the main thread. For live RTSP
cameras (typically 12–25 fps) the pipeline keeps up in real time; 60 fps files
are processed at every 2nd frame by default.

## Layout

```
humanmonitor/
  engine.py      headless pipeline shared by both front-ends (sources -> detector -> counter -> CLIP -> stats)
  sources.py     file / RTSP capture (threaded decode, stride, looping, auto-reconnect)
  detector.py    batched YOLO11 detection + per-camera ByteTrack, head-point estimation
  classifier.py  SigLIP 2 / CLIP zero-shot gender & adult/child on person crops
  counter.py     trigger-line crossing logic, per-track demographic vote
  stats.py       counters + CSV event log
  dashboard.py   OpenCV dashboard rendering
  ui.py          2x2 window grid, overlays
  config.py      trigger line + counting area store (config/lines.json)
web/             browser console (index.html, app.css, app.js)
config/                  *.json here are written by the app and gitignored;
                         each is seeded from the .example committed beside it
  lines.json     trigger lines and counting areas per source
  sources.json   "live" (RTSP) and "video" (file) source sets for webapp.py
  site.json      building hours, capacity, warning %, starting occupancy
  bytetrack.yaml tracker settings (tuned against the samples: 221 -> 57 IDs per clip)
app.py           OpenCV-windows front-end          run_demo.sh / run_demo.bat
webapp.py        browser front-end (FastAPI)       run_web.sh  / run_web.bat
setup_ubuntu.sh / setup_windows.bat / setup_mac.sh   one-time environment set-up
download_demo_videos.py   fetch the five demo clips
export_tensorrt.py        YOLO -> TensorRT engine (NVIDIA)
scripts/ab_demographics.py   accuracy evaluation on real crossings (see below)
design/console_panel_v1.drawio   editable draw.io of the console panel layout
docs/                     screenshots used by this README
Dockerfile / docker-compose.yml   CUDA container for the browser console
```

## License

MIT — see [LICENSE](LICENSE).

**Read [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before using this
commercially.** MIT covers this code and reads as "do what you like", but the
detector is **Ultralytics YOLO11, which is AGPL-3.0** — and this is a network
service. An AGPL component inside a service you offer to others can oblige you
to release that service's source under the AGPL, or to buy a commercial license
from Ultralytics. A permissive license on your own code says nothing about what
it imports. Swapping the detector is a contained change: `humanmonitor/detector.py`
is the only file that touches it.

## Known limits

Worth stating plainly, because each of these is a place the numbers get worse:

* **Counting is the solid part; demographics are the weak part.** Counting needs
  the camera to see heads cleanly across the line — overhead or high-angle
  entrance cameras work best, a camera looking along a corridor is much harder.
* **Zero-shot classification on full-body CCTV crops is a baseline, not a
  product.** It will not match a purpose-trained attribute model. The classifier
  sits behind a one-method interface (`classify(crops)`), so swapping in a
  PA-100K / PETA attribute network, or a face-based model where faces are
  actually visible, is a contained change.
* **Small and distant people are counted but not classified** until they come
  closer than `--min-crop-height`. They land in the unknown buckets rather than
  being guessed at, which is why the splits always reconcile to the total.
* **Per-site configuration is deliberately tiny**: the trigger line, and
  optionally `--conf` / `--min-crop-height`. One pipeline runs everywhere, so
  there is no per-site model to retrain or drift.
