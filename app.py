"""Visitor Counting Console - multi-camera demo (OpenCV windows).

Monitors up to three CCTV feeds (RTSP URLs or video files) in a 2x2 grid of
windows (dashboard top-left + 3 feeds). People are detected and tracked; when a
tracked person's *head* crosses the per-camera trigger line the visitor,
gender and adult/child counters are updated (entry = +1, exit = -1).

For the browser version of the same demo see webapp.py.

Run:
    python app.py                                   # the three sample videos
    python app.py --sources rtsp://cam1/live a.mp4  # your own feeds (max 3)
    python app.py --setup                           # start in line set-up mode

Keys (any window focused):
    E  toggle set-up mode            S  save trigger line(s) drawn in set-up mode
    F  reverse IN arrow (hovered cam) C  clear the points drawn on the hovered cam
    X  delete saved line (hovered cam, set-up mode)
    R  reset counters                P  pause / resume
    Q / Esc  quit
Set-up mode mouse: left-click start & end point of the line, right-click reverses
the IN arrow of that camera. The triangle points the way someone walks to be
counted as an ENTRY; crossing against it is an EXIT.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from humanmonitor.classifier import DEFAULT_MODEL as DEFAULT_CLASSIFIER
from humanmonitor.config import DEFAULT_CONFIG_PATH
from humanmonitor.dashboard import render_dashboard
from humanmonitor.engine import Camera, Engine, EngineConfig, MAX_CAMERAS
from humanmonitor import ui

DEFAULT_SOURCES = [
    "sample_data/shoppingmall_interior01.mp4",
    "sample_data/shop_interior01.mp4",
    "sample_data/seaside_walking01.mp4",
]


class App:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.engine = Engine(EngineConfig.from_args(args))
        self.hover_idx: Optional[int] = None
        # scale, ox, oy of each camera's letterboxed view (for mouse -> frame mapping)
        self.transform: dict[int, tuple[float, int, int]] = {}

        self.cam_windows = [f"CAM {cam.idx + 1} - {cam.name}" for cam in self.engine.cameras]
        self.dash_window = "DASHBOARD - Visitor Analytics"
        # grid order: dashboard top-left, then cameras (top-right, bottom-left, bottom-right)
        names = [self.dash_window] + self.cam_windows
        while len(names) < 4:
            names.append(f"CAM {len(names)} - (no source)")
        self.grid = ui.WindowGrid(names)
        self.window_names = names
        for cam in self.engine.cameras:
            cv2.setMouseCallback(self.cam_windows[cam.idx], self._on_mouse, cam.idx)
        if args.topmost:
            for n in names:
                cv2.setWindowProperty(n, cv2.WND_PROP_TOPMOST, 1)
        if self.engine.setup_mode:
            print("[setup] left-click two points on a camera window, right-click to reverse the IN arrow, press S to save")

    # ------------------------------------------------------------- mouse
    def view_to_frame(self, cam: Camera, x: int, y: int) -> tuple[int, int]:
        scale, ox, oy = self.transform.get(cam.idx, (1.0, 0, 0))
        fx = int(round((x - ox) / scale))
        fy = int(round((y - oy) / scale))
        fx = min(max(fx, 0), max(0, cam.source.width - 1))
        fy = min(max(fy, 0), max(0, cam.source.height - 1))
        return fx, fy

    def _on_mouse(self, event, x, y, flags, cam_idx):
        cam = self.engine.cameras[cam_idx]
        self.hover_idx = cam_idx
        if not self.engine.setup_mode:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            p = self.view_to_frame(cam, x, y)
            if len(cam.setup_points) >= 2:
                cam.setup_points = [p]
            else:
                cam.setup_points.append(p)
            print(f"[setup] {cam.name}: point {len(cam.setup_points)} at {p}")
        elif event == cv2.EVENT_RBUTTONDOWN:
            # reverse the IN arrow of the camera under the cursor (same as F, but per-window)
            self.engine.flip_entry_side(cam)

    def _hovered(self) -> Camera:
        return self.engine.cameras[self.hover_idx] if self.hover_idx is not None else self.engine.cameras[0]

    # ------------------------------------------------------------- render
    def render_camera(self, cam: Camera) -> None:
        e = self.engine
        w, h = self.grid.size(cam.idx + 1)
        win = self.cam_windows[cam.idx]
        img = e.annotated_frame(cam)
        if img is None:
            canvas = np.full((h, w, 3), 16, np.uint8)
            ui.draw_banner(canvas, [f"{cam.name}: NO SIGNAL  ({cam.source.status()})"], color=(60, 120, 255))
            cv2.imshow(win, canvas)
            return

        c = e.stats.per_camera[cam.idx]
        lines = [f"CAM {cam.idx + 1}  {cam.name}   {cam.source.status()}   {cam.fps:4.1f} fps   "
                 f"in {c.entries}  out {c.exits}   M {c.male} F {c.female}   A {c.adult} C {c.child}"]
        if e.setup_mode:
            lines.append(f"SETUP: L-click 2 points ({len(cam.setup_points)}/2)   R-click / F reverse IN arrow   C clear   S save   X delete   E done")
        elif cam.line is None:
            lines.append("no trigger line - press E and click two points to define one")
        if e.paused:
            lines.append("PAUSED")
        ui.draw_banner(img, lines, color=(0, 200, 255) if e.setup_mode else (255, 255, 255))

        canvas, scale, ox, oy = ui.letterbox(img, w, h)
        self.transform[cam.idx] = (scale, ox, oy)
        cv2.imshow(win, canvas)

    def render_dashboard(self) -> None:
        e = self.engine
        w, h = self.grid.size(0)
        img = render_dashboard(e.stats, (w, h), e.camera_rows(), e.setup_mode, e.pipeline_fps, e.device)
        cv2.imshow(self.dash_window, img)
        for i in range(len(e.cameras), 3):  # placeholder windows
            pw, ph = self.grid.size(i + 1)
            blank = np.full((ph, pw, 3), 16, np.uint8)
            ui.draw_banner(blank, [f"CAM {i + 1}: no source configured"], color=(120, 120, 120))
            cv2.imshow(self.window_names[i + 1], blank)

    # --------------------------------------------------------------- loop
    def run(self) -> None:
        e = self.engine
        e.start()
        rendered: dict[int, object] = {}
        last_dash = 0.0
        try:
            while True:
                if e.error is not None:
                    raise SystemExit(f"pipeline failed: {e.error!r}")
                for cam in e.cameras:
                    # redraw a camera only when the pipeline produced a new frame (or in set-up mode,
                    # where the mouse changes the overlay) - saves CPU for the pipeline thread
                    snap = cam.display
                    if e.setup_mode or rendered.get(cam.idx) is not snap:
                        self.render_camera(cam)
                        rendered[cam.idx] = snap
                now = time.perf_counter()
                if now - last_dash > 0.1:  # dashboard at ~10 Hz is plenty
                    self.render_dashboard()
                    last_dash = now
                key = cv2.waitKey(10) & 0xFF
                if not self._handle_key(key):
                    break
                # stop if the user closed every camera window
                if all(cv2.getWindowProperty(w, cv2.WND_PROP_VISIBLE) < 1 for w in self.cam_windows):
                    break
        finally:
            self.shutdown()

    def _handle_key(self, key: int) -> bool:
        e = self.engine
        if key in (27, ord("q"), ord("Q")):
            return False
        if key in (ord("e"), ord("E")):
            e.set_setup_mode(not e.setup_mode)
        elif key in (ord("s"), ord("S")):
            if e.setup_mode:
                e.save_lines()
            else:
                print("[info] press E to enter set-up mode before saving a line")
        elif key in (ord("c"), ord("C")) and e.setup_mode:
            e.clear_setup_points(self._hovered())
        elif key in (ord("f"), ord("F")) and e.setup_mode:
            e.flip_entry_side(self._hovered())
            print("[setup] press S to save")
        elif key in (ord("x"), ord("X")) and e.setup_mode:
            e.delete_line(self._hovered())
        elif key in (ord("r"), ord("R")):
            e.reset_counts()
        elif key in (ord("p"), ord("P"), ord(" ")):
            e.paused = not e.paused
        return True

    def shutdown(self) -> None:
        self.engine.shutdown()
        cv2.destroyAllWindows()


def add_engine_args(p: argparse.ArgumentParser) -> None:
    """Pipeline options shared by app.py and webapp.py."""
    p.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="trigger line config JSON")
    p.add_argument("--events", default="output/events.csv", help="CSV log of crossing events")
    p.add_argument("--model", default="yolo11s.pt", help="Ultralytics detector weights (use *-pose.pt for keypoint-based head)")
    p.add_argument("--classifier", "--clip", dest="clip", default=DEFAULT_CLASSIFIER,
                   help="zero-shot gender/age model: a google/siglip2-* name (default, more accurate) "
                        "or openai/clip-vit-base-patch32 (~4x faster per crop, ~4 points worse on gender)")
    p.add_argument("--device", default="auto", help="auto | cuda:0 | mps | cpu")
    p.add_argument("--tracker", default="config/bytetrack.yaml", help="Ultralytics tracker config (falls back to built-in bytetrack.yaml)")
    p.add_argument("--conf", type=float, default=0.25, help="detector confidence threshold")
    p.add_argument("--imgsz", type=int, default=640, help="detector input size")
    p.add_argument("--gender-thresh", type=float, default=0.5, help="P(male) at/above which a track is male")
    p.add_argument("--child-thresh", type=float, default=0.6, help="P(child) at/above which a track is a child")
    p.add_argument("--debug-tracks", nargs="?", const="output/tracks.csv", default=None,
                   help="log every track's live decision to this CSV (default output/tracks.csv), so a track "
                        "seen on screen can be looked up afterwards by its id")
    p.add_argument("--debug-every", type=int, default=15, help="write a --debug-tracks row every N processed frames")
    p.add_argument("--initial-count", type=int, default=None,
                   help="people assumed already inside at start-up (default 0 - count only what the cameras see). "
                        "A non-zero value is an assumption, not a measurement: it is split 50/50 male/female and "
                        "80/20 adult/child and shown as 'assumed' in the browser console")
    p.add_argument("--classify-every", type=int, default=15, help="re-classify each track every N processed frames (a new track is always classified at once)")
    p.add_argument("--max-crops", type=int, default=6, help="max person crops classified per camera per frame")
    p.add_argument("--min-crop-height", type=int, default=48, help="skip demographics on people shorter than this (px)")
    p.add_argument("--stride", type=int, default=0, help="process every Nth file frame (0 = auto, ~30 fps)")
    p.add_argument("--setup", action="store_true", help="start in trigger-line set-up mode")
    p.add_argument("--realtime", action="store_true", help="pace processing at the files' (strided) frame rate")
    p.add_argument("--no-loop", action="store_true", help="do not loop video files")
    p.add_argument("--no-demographics", action="store_true", help="counting only (skip gender/age model)")


def check_sources(p: argparse.ArgumentParser, sources: list[str]) -> None:
    if len(sources) > MAX_CAMERAS:
        print(f"[warn] only the first {MAX_CAMERAS} sources are used")
    for s in sources[:MAX_CAMERAS]:
        if "://" not in s and not Path(s).exists():
            p.error(f"source not found: {s}")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sources", nargs="+", default=DEFAULT_SOURCES, help="up to 3 RTSP URLs or video files")
    add_engine_args(p)
    p.add_argument("--topmost", action="store_true", help="keep the four windows above other applications")
    args = p.parse_args(argv)
    check_sources(p, args.sources)
    # webapp.py keeps None to mean "use the baseline saved in config/site.json"; here there is no
    # site config, so an unset baseline means count only what the cameras actually see.
    if args.initial_count is None:
        args.initial_count = 0
    return args


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)  # events show up immediately when piped to a log
    App(parse_args()).run()
