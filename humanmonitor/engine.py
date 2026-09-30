"""Headless multi-camera counting engine shared by the OpenCV app and the web UI.

Owns the sources, detector, classifier, per-camera line counters, statistics
and the background pipeline thread. Front-ends (app.py windows, webapp.py
browser) only read `Camera.display` / `Stats` and call the set-up / control
methods; they never touch the models directly.
"""
from __future__ import annotations

import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .classifier import DEFAULT_MODEL as DEFAULT_CLASSIFIER, DemographicClassifier, crop_person
from .config import LineConfigStore, Region, TriggerLine, is_url
from .counter import CROSSING_WEIGHT, CrossEvent, LineCounter, Revision, UNKNOWN
from .detector import MultiStreamPersonTracker, Track
from .sources import Frame, NullSource, VideoSource
from .stats import EventLog, Stats, TrackLog
from . import ui

MAX_CAMERAS = 3          # default number of slots (app.py windows); webapp.py raises it with --max-cameras


@dataclass
class EngineConfig:
    sources: list                      # up to MAX_CAMERAS entries: "path-or-url" or {"src": ..., "name": ...}
    config: str = "config/lines.json"
    events: str = "output/events.csv"
    model: str = "yolo11s.pt"
    clip: str = DEFAULT_CLASSIFIER   # zero-shot gender / age model (SigLIP 2 by default; CLIP still selectable)
    device: str = "auto"
    tracker: str = "config/bytetrack.yaml"
    conf: float = 0.25
    imgsz: int = 640
    gender_thresh: float = 0.5
    child_thresh: float = 0.6
    classify_every: int = 15   # re-classify a track every N frames (SigLIP2 needs fewer looks than CLIP)
    max_crops: int = 6
    min_crop_height: int = 48
    stride: int = 0
    setup: bool = False
    realtime: bool = False
    no_loop: bool = False
    no_demographics: bool = False
    initial_count: int = 0            # people assumed inside at start-up (site-wide); 0 = count only what is seen
    debug_tracks: Optional[str] = None  # CSV path for a per-track decision log
    debug_every: int = 15
    max_cameras: int = MAX_CAMERAS   # upper bound of camera slots (web UI can add up to this many)
    min_slots: int = MAX_CAMERAS     # slots created at start-up even when fewer sources are given

    @classmethod
    def from_args(cls, args) -> "EngineConfig":
        """Build from an argparse namespace that has (at least) the fields above."""
        names = cls.__dataclass_fields__
        return cls(**{k: getattr(args, k) for k in names if hasattr(args, k)})


_next_cam_id = [0]


def _new_cam_id() -> int:
    _next_cam_id[0] += 1
    return _next_cam_id[0]


PRESENT_ZERO = {"n": 0, "male": 0, "female": 0, "adult": 0, "child": 0,
                "gender_unknown": 0, "age_unknown": 0}


@dataclass
class Camera:
    idx: int
    source: VideoSource | NullSource
    label: Optional[str] = None          # display name chosen by the operator (else the source's own)
    cid: int = field(default_factory=_new_cam_id)   # stable id for the web API (idx shifts when slots are removed)
    area_capacity: int = 0               # optional capacity of the area behind this entrance (0 = none)
    banner: Optional[str] = None         # alarm text drawn across the frame (set by the console)
    editing: bool = False                # web UI is editing the line: the client draws it, the server does not
    line: Optional[TriggerLine] = None
    counter: Optional[LineCounter] = None
    # set-up state
    setup_points: list[tuple[int, int]] = field(default_factory=list)
    setup_entry_side: int = 1
    line_dirty: bool = False
    # optional rectangle limiting where this camera counts or classifies (see config.ROLES)
    region: Optional[Region] = None
    # the other feed this slot can show: the video file when live, the RTSP URL
    # when on video. Lets the page toggle a panel without retyping the source
    # None when the slot has no counterpart.
    alt_src: Optional[str] = None
    # who is in the counted part of the frame right now, split M/F and A/C. Not a
    # crossing figure: it is what the camera can see, so it is right from the first
    # frame instead of waiting for someone to walk over the line.
    present: dict = field(default_factory=lambda: dict(PRESENT_ZERO))
    # pipeline -> render hand-off (assigned atomically as a tuple)
    display: Optional[tuple[Frame, list[Track]]] = None
    flash: dict[int, tuple[int, str]] = field(default_factory=dict)
    fps: float = 0.0
    _t_last: float = 0.0

    @property
    def name(self) -> str:
        """Display name. An RTSP URL makes a poor label - `192.168.1.62:554/
        Streaming/Channels/101` is what BY LOCATION was showing - so a live feed
        with no label falls back to its slot number instead - an operator wants
        "Camera 2", not an address. A file keeps its stem, which already reads well."""
        if self.label:
            return self.label
        return self.source.name if not is_url(self.source.src) else f"Camera {self.idx + 1}"

    def tick_fps(self) -> None:
        now = time.perf_counter()
        if self._t_last:
            inst = 1.0 / max(1e-6, now - self._t_last)
            self.fps = inst if not self.fps else 0.9 * self.fps + 0.1 * inst
        self._t_last = now


@dataclass
class EventRecord:
    """A crossing event enriched with camera / wall-clock context for the UIs."""
    time: float
    cam_idx: int
    cam_name: str
    event: CrossEvent
    crop: Optional[np.ndarray] = None       # tight person crop at the crossing (for audits / evaluation)
    context: Optional[np.ndarray] = None    # wider crop around the person (scene context)


def pick_device(requested: str) -> str:
    import torch

    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda:0"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return requested


class Engine:
    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg
        self.device = pick_device(cfg.device)
        self.setup_mode = bool(cfg.setup)
        self.paused = False
        self.pipeline_fps = 0.0
        self.started_at = time.time()
        self.lock = threading.RLock()        # guards counters / lines between pipeline and UI
        self._stop = threading.Event()
        self.error: Optional[BaseException] = None
        self.recent: deque[EventRecord] = deque(maxlen=50)
        self.revisions: deque[tuple[float, int, Revision]] = deque(maxlen=50)
        self.listeners: list = []            # callables invoked with an EventRecord
        self.keep_crops: bool = False        # attach person crops to EventRecords (evaluation tooling)

        self.store = LineConfigStore.load(cfg.config)
        tracker_cfg = cfg.tracker if Path(cfg.tracker).exists() else "bytetrack.yaml"
        print(f"[init] device={self.device}  detector={cfg.model}  tracker={tracker_cfg}  classifier={cfg.clip}")

        t0 = time.time()
        self.classifier = None if cfg.no_demographics else DemographicClassifier(cfg.clip, self.device)
        print(f"[init] classifier ready ({time.time() - t0:.1f}s)")

        # fixed slots at start-up (empty ones hold a NullSource); the web UI adds / removes slots later
        self.cameras: list[Camera] = []
        given = list(cfg.sources[:cfg.max_cameras])
        entries = given + [None] * max(0, min(cfg.min_slots, cfg.max_cameras) - len(given))
        for i, entry in enumerate(entries):
            src, label = self._split_entry(entry)
            cam = Camera(i, NullSource(), label=None)
            if src:
                cam.source = self._open_source(src)
                cam.label = label
                cam.area_capacity = self._entry_area(entry)
                cam.alt_src = self._entry_alt(entry)
                self._load_line(cam)
                vs = cam.source
                print(f"[init] camera {i + 1}: {src}  {vs.width}x{vs.height} @ {vs.fps:.1f}fps  "
                      f"processing every {vs.stride} frame(s)  {vs.status()}")
            else:
                print(f"[init] camera {i + 1}: empty slot")
            self.cameras.append(cam)

        self.detector = MultiStreamPersonTracker(len(self.cameras), cfg.model, self.device,
                                                 conf=cfg.conf, imgsz=cfg.imgsz, tracker=tracker_cfg)
        for cam in self.cameras:
            if cam.counter is not None:
                self._align_counter_memory(cam.counter)
        self.stats = Stats([c.name for c in self.cameras], initial_count=cfg.initial_count)
        self.log = EventLog(cfg.events)
        self.track_log = TrackLog(cfg.debug_tracks) if cfg.debug_tracks else None
        if self.track_log is not None:
            print(f"[init] logging every track's decision to {cfg.debug_tracks} (every {cfg.debug_every} frames)")

        # warm-up (CUDA kernels / cudnn autotune) so the first frames are not slow
        self.detector.update([np.zeros((cam.source.height or 720, cam.source.width or 1280, 3), np.uint8) for cam in self.cameras])
        for i in range(len(self.cameras)):
            self.detector.reset(i)
        if self.classifier is not None:
            self.classifier.classify([np.zeros((128, 64, 3), np.uint8)])
        if self.setup_mode:
            print("[setup] set-up mode: draw a trigger line on each camera, then save")

        self._worker = threading.Thread(target=self._pipeline_loop, name="pipeline", daemon=True)

    # ----------------------------------------------------------- sources
    @staticmethod
    def _split_entry(entry) -> tuple[str, Optional[str]]:
        """A sources.json entry is either "path-or-url" or {"src": ..., "name": ...}."""
        if entry is None:
            return "", None
        if isinstance(entry, dict):
            return str(entry.get("src", "")).strip(), (str(entry["name"]).strip() or None) if entry.get("name") else None
        return str(entry).strip(), None

    @staticmethod
    def _entry_alt(entry) -> Optional[str]:
        """The paired feed stored alongside a slot, if any."""
        return (str(entry.get("alt", "")).strip() or None) if isinstance(entry, dict) else None

    @staticmethod
    def _entry_area(entry) -> int:
        return int(entry.get("area_capacity") or 0) if isinstance(entry, dict) else 0

    def _open_source(self, src: str) -> VideoSource:
        return VideoSource(src, loop=not self.cfg.no_loop, stride=self.cfg.stride)

    def replace_source(self, cam: Camera, src: Optional[str], label: Optional[str] = None, area_capacity: int = 0) -> None:
        """Swap the feed of one slot at run time (web UI). src=None empties the slot.

        Raises FileNotFoundError for a video file that cannot be opened; an RTSP URL
        that is unreachable is accepted and keeps reconnecting in the background.
        """
        new = self._open_source(src) if src else NullSource()
        with self.lock:
            old = cam.source
            cam.source = new
            cam.label = label
            cam.area_capacity = max(0, int(area_capacity))
            cam.banner = None
            cam.line, cam.counter = None, None
            cam.setup_points.clear()
            cam.line_dirty = False
            cam.display = None
            cam.present = dict(PRESENT_ZERO)
            cam.flash.clear()
            cam.fps, cam._t_last = 0.0, 0.0
            self.detector.reset(cam.idx)
            self.stats.per_camera[cam.idx].reset()
            self.stats.camera_names[cam.idx] = cam.name
            if src:
                self._load_line(cam)   # deferred until the stream connects if needed
        # releasing an RTSP source can block inside FFmpeg while it is reconnecting - do it off the request thread
        threading.Thread(target=old.release, name=f"release-{cam.idx}", daemon=True).start()
        print(f"[source] camera {cam.idx + 1}: {'cleared' if not src else src + '  ' + cam.source.status()}")

    def camera_by_id(self, cid: int) -> Optional[Camera]:
        return next((c for c in self.cameras if c.cid == cid), None)

    def add_slot(self) -> Camera:
        """Append an empty camera slot (web UI '+ camera')."""
        with self.lock:
            if len(self.cameras) >= self.cfg.max_cameras:
                raise ValueError(f"maximum of {self.cfg.max_cameras} cameras reached")
            cam = Camera(len(self.cameras), NullSource())
            self.cameras.append(cam)
            self.detector.add_stream()
            self.stats.add_camera(cam.name)
        print(f"[source] slot {cam.idx + 1} added")
        return cam

    def remove_slot(self, cam: Camera) -> None:
        """Remove a camera slot entirely; the remaining slots keep their order and are re-indexed."""
        with self.lock:
            if len(self.cameras) <= 1:
                raise ValueError("at least one camera slot must remain")
            i = cam.idx
            old = cam.source
            del self.cameras[i]
            self.detector.remove_stream(i)
            self.stats.remove_camera(i)
            for j, c in enumerate(self.cameras):
                c.idx = j
        threading.Thread(target=old.release, name=f"release-{cam.cid}", daemon=True).start()
        print(f"[source] slot {i + 1} removed")

    def swap_source(self, cam: Camera) -> str:
        """Switch a slot to its paired feed and keep the old one as the new pair.

        The trigger line is keyed by source in lines.json, so each feed keeps its
        own calibration across the swap; the label and area capacity stay with
        the slot. Raises ValueError when the slot has no counterpart.
        """
        if not cam.alt_src:
            raise ValueError("this camera has no paired feed to switch to")
        previous, target = cam.source.src, cam.alt_src
        label, area = cam.label, cam.area_capacity
        self.replace_source(cam, target, label, area)
        cam.alt_src = previous
        return target

    def set_region(self, cam: Camera, region: Optional[Region]) -> None:
        """Set or clear the counting region for this camera's source, and persist it."""
        with self.lock:
            cam.region = region.normalised() if region else None
            self.store.set_region(cam.source.src, cam.region)
            self.store.save()
            if cam.counter is not None:
                cam.counter.reset_tracks()      # states seeded outside the new region are meaningless

    def set_occupancy(self, n: int) -> int:
        with self.lock:
            return self.stats.set_occupancy(n)

    def source_entries(self) -> list:
        """Current slots in sources.json form (strings, or {src, name} when a label is set)."""
        out = []
        for cam in self.cameras:
            if cam.source.is_null:
                continue
            if cam.label or cam.area_capacity or cam.alt_src:
                entry = {"src": cam.source.src, "name": cam.label, "area_capacity": cam.area_capacity}
                if cam.alt_src:
                    entry["alt"] = cam.alt_src
                out.append(entry)
            else:
                out.append(cam.source.src)
        return out

    # ------------------------------------------------------------- lines
    def _load_line(self, cam: Camera) -> bool:
        """Apply the stored line once the source resolution is known. Returns True if applied."""
        cam.region = self.store.get_region(cam.source.src)   # independent of the line
        line = self.store.get(cam.source.src)
        if line is None:
            print(f"[init] no trigger line configured for {cam.source.src} - use set-up mode to draw one")
            return True   # nothing to wait for
        if not (cam.source.width and cam.source.height):
            return False  # stream not connected yet - retried from the pipeline
        cam.line = line
        cam.counter = self._new_counter(cam, line)
        cam.setup_entry_side = line.entry_side
        return True

    def _new_counter(self, cam: Camera, line: TriggerLine) -> LineCounter:
        lc = LineCounter(line, cam.source.width, cam.source.height,
                         gender_thresh=self.cfg.gender_thresh, child_thresh=self.cfg.child_thresh)
        lc.configure_timing(cam.source.effective_fps)
        self._align_counter_memory(lc)
        return lc

    def _align_counter_memory(self, lc: LineCounter) -> None:
        """Remember a person's side of the line for as long as the tracker keeps their ID hidden.

        Forgetting sooner meant a child who reappeared after being blocked by an adult kept
        their ID but had lost their side, so the crossing could not be counted.
        """
        detector = getattr(self, "detector", None)   # counters are first built before the detector
        if detector is not None:
            lc.stale_frames = max(lc.stale_frames, detector.max_time_lost + 10)

    def save_lines(self) -> int:
        """Commit the draft points / flipped sides of every camera. Returns the number saved."""
        saved = 0
        with self.lock:
            for cam in self.cameras:
                if len(cam.setup_points) == 2:
                    w, h = cam.source.width, cam.source.height
                    line = TriggerLine.from_pixels(cam.setup_points[0], cam.setup_points[1], w, h, cam.setup_entry_side, cam.name)
                    self._apply_line(cam, line)
                    cam.setup_points.clear()
                    saved += 1
                elif cam.line_dirty and cam.line is not None:
                    cam.line.entry_side = cam.setup_entry_side
                    self._apply_line(cam, cam.line)
                    saved += 1
                cam.line_dirty = False
        if saved:
            self.store.save()
            print(f"[setup] saved {saved} trigger line(s) to {self.store.path}")
        else:
            print("[setup] nothing to save - click two points on a camera first")
        return saved

    def set_line(self, cam: Camera, p1: tuple[float, float], p2: tuple[float, float], entry_side: int) -> TriggerLine:
        """Save a line given in normalised coordinates (web UI path)."""
        line = TriggerLine(p1=(float(p1[0]), float(p1[1])), p2=(float(p2[0]), float(p2[1])),
                           entry_side=1 if entry_side >= 0 else -1, name=cam.name)
        with self.lock:
            self._apply_line(cam, line)
            cam.setup_points.clear()
            cam.setup_entry_side = line.entry_side
            cam.line_dirty = False
        self.store.save()
        print(f"[setup] {cam.name}: trigger line saved to {self.store.path}")
        return line

    def _apply_line(self, cam: Camera, line: TriggerLine) -> None:
        cam.line = line
        self.store.set(cam.source.src, line)
        if cam.counter is None:
            cam.counter = self._new_counter(cam, line)
        else:
            cam.counter.set_line(line)

    def flip_entry_side(self, cam: Camera, persist: bool = False) -> None:
        cam.setup_entry_side = -cam.setup_entry_side
        if len(cam.setup_points) < 2 and cam.line is not None:
            cam.line_dirty = True
        if persist and cam.line is not None and len(cam.setup_points) < 2:
            with self.lock:
                cam.line.entry_side = cam.setup_entry_side
                self._apply_line(cam, cam.line)
                cam.line_dirty = False
            self.store.save()
        print(f"[setup] {cam.name}: IN arrow reversed -> entry_side={cam.setup_entry_side}")

    def clear_setup_points(self, cam: Camera) -> None:
        if cam.setup_points:
            cam.setup_points.clear()
            print(f"[setup] {cam.name}: points cleared")

    def delete_line(self, cam: Camera) -> None:
        if cam.line is None:
            return
        with self.lock:
            cam.line, cam.counter = None, None
            cam.setup_points.clear()
            self.store.remove(cam.source.src)
        self.store.save()
        print(f"[setup] {cam.name}: trigger line deleted")

    def set_setup_mode(self, on: bool) -> None:
        self.setup_mode = bool(on)
        if not self.setup_mode:
            for cam in self.cameras:
                cam.setup_points.clear()
                cam.setup_entry_side = cam.line.entry_side if cam.line else 1
                cam.line_dirty = False
        print(f"[mode] {'SETUP' if self.setup_mode else 'MONITOR'}")

    def reset_counts(self) -> None:
        with self.lock:
            self.stats.reset()
            for cam in self.cameras:
                if cam.counter:
                    cam.counter.reset_tracks()
                cam.flash.clear()
            self.recent.clear()
            self.revisions.clear()
        print("[info] counters reset")

    # ------------------------------------------------------------ pipeline
    def start(self) -> None:
        self._worker.start()

    def _pipeline_loop(self) -> None:
        t_prev = time.perf_counter()
        try:
            while not self._stop.is_set():
                if self.paused:
                    time.sleep(0.05)
                    t_prev = time.perf_counter()
                    continue
                frames = [cam.source.read() for cam in self.cameras]
                if all(f is None for f in frames):
                    time.sleep(0.005)
                    continue
                with self.lock:
                    self._step(frames)
                now = time.perf_counter()
                if self.cfg.realtime:
                    target = 1.0 / max(1.0, max(c.source.effective_fps for c in self.cameras))
                    rest = target - (now - t_prev)
                    if rest > 0:
                        time.sleep(rest)
                        now = time.perf_counter()
                inst = 1.0 / max(1e-6, now - t_prev)
                self.pipeline_fps = inst if not self.pipeline_fps else 0.9 * self.pipeline_fps + 0.1 * inst
                t_prev = now
        except BaseException as e:  # surfaced by the front-end
            self.error = e
            traceback.print_exc()

    def _step(self, frames: list[Optional[Frame]]) -> None:
        for cam, f in zip(self.cameras, frames):
            if f is None:
                continue
            cam.tick_fps()
            if cam.line is None and cam.counter is None and cam.source.width and not cam.source.is_null:
                # a stream that was offline at start-up: apply its stored line now
                if self.store.get(cam.source.src) is not None:
                    self._load_line(cam)
            if f.looped:
                self.detector.reset(cam.idx)
                if cam.counter:
                    cam.counter.reset_tracks()
                cam.flash.clear()

        tracks_all = self.detector.update([f.image if f is not None else None for f in frames])
        # Apply a tracking region before anything else looks at the tracks: whoever
        # it drops is then not classified, not counted and not drawn. Done here rather
        # than per camera below because _classify() runs across all cameras at once
        # and its crop budget is part of what a region is meant to protect.
        tracks_all = [self._within_region(cam, ts) for cam, ts in zip(self.cameras, tracks_all)]

        if self.classifier is not None:
            self._classify(frames, tracks_all)

        for cam, f, tracks in zip(self.cameras, frames, tracks_all):
            if f is None:
                continue
            if cam.counter is not None and not self.setup_mode:
                for ev in cam.counter.update(tracks, f.index):
                    # The view at the line is the most informative one this track will give, so take a
                    # fresh look now and let it outweigh the earlier ones (see counter.CROSSING_WEIGHT).
                    if self.classifier is not None:
                        self._classify_crossing(cam, f, tracks, ev.track_id)
                        st = cam.counter.get_state(ev.track_id)
                        if st and st.n_obs:
                            ev.gender, ev.age = st.decide(cam.counter.gender_thresh, cam.counter.child_thresh)
                            ev.p_male, ev.p_child, ev.n_obs = st.p_male, st.p_child, st.n_obs
                            # the counters are about to be moved by THIS decision, so it is also the
                            # one a later revision must undo - _commit_crossing recorded the vote from
                            # before the crossing look, and revising against that stale value would
                            # decrement a bucket that was never incremented
                            st.last_gender, st.last_age = ev.gender, ev.age
                    self.stats.apply(cam.idx, ev)
                    self.log.write(cam.idx, cam.name, cam.source.src, ev)
                    cam.flash[ev.track_id] = (f.index, "ENTRY" if ev.direction > 0 else "EXIT")
                    rec = EventRecord(time.time(), cam.idx, cam.name, ev)
                    if self.keep_crops:
                        t_ev = next((t for t in tracks if t.track_id == ev.track_id), None)
                        if t_ev is not None:
                            rec.crop = crop_person(f.image, t_ev.box)
                            rec.context = crop_person(f.image, t_ev.box, margin=0.6)
                    self.recent.appendleft(rec)
                    for cb in self.listeners:
                        cb(rec)
                    print(f"[event] {cam.name}: {'ENTRY' if ev.direction > 0 else 'EXIT '} track #{ev.track_id} "
                          f"gender={ev.gender} age={ev.age} (p_male={ev.p_male:.2f} p_child={ev.p_child:.2f} n={ev.n_obs})")
                # A crossing is decided the instant the head crosses, when the person has
                # only just walked into frame. Once the track has been watched for a few
                # seconds the vote is much better informed, so move the count if it changed.
                for rev in cam.counter.collect_revisions(f.index):
                    self.stats.revise(cam.idx, rev)
                    self.log.write_revision(cam.idx, cam.name, cam.source.src, rev)
                    self.revisions.appendleft((time.time(), cam.idx, rev))
                    for rec in self.recent:   # keep the UI event feed consistent with the counters
                        if rec.cam_idx == cam.idx and rec.event.track_id == rev.track_id:
                            rec.event.gender, rec.event.age = rev.gender, rev.age
                            break
                    print(f"[revise] {cam.name}: track #{rev.track_id} "
                          f"{rev.old_gender}/{rev.old_age} -> {rev.gender}/{rev.age} "
                          f"(p_male={rev.p_male:.2f} p_child={rev.p_child:.2f} n={rev.n_obs})")
            if self.track_log is not None and cam.counter is not None and f.index % self.cfg.debug_every == 0:
                self.track_log.write(cam.idx, cam.name, f.index, tracks, cam.counter)
            cam.present = self._present_tally(cam, tracks)
            cam.display = (f, tracks)

    def _present_tally(self, cam: Camera, tracks: list[Track]) -> dict:
        """Everyone the camera is counting at this instant, split M/F and A/C.

        The trigger line does not come into it. With a count-inside area drawn,
        `tracks` is exactly the people standing in the box, so the box reports its
        own population from the first frame - nobody has to cross anything to be
        counted. Whoever the classifier has not decided yet lands in the unknown
        buckets rather than being guessed at, so the parts always sum to `n`.
        """
        out = dict(PRESENT_ZERO)
        c = cam.counter
        for t in tracks:
            out["n"] += 1
            st = c.get_state(t.track_id) if c is not None else None
            g, a = st.decide(c.gender_thresh, c.child_thresh) if (st and c) else (UNKNOWN, UNKNOWN)
            out["male" if g == "M" else "female" if g == "F" else "gender_unknown"] += 1
            out["adult" if a == "A" else "child" if a == "C" else "age_unknown"] += 1
        return out

    def _within_region(self, cam: Camera, tracks: list[Track]) -> list[Track]:
        """The tracks a region keeps: those inside it, those outside it, or all of them.

        Which of the three depends on the region's role, and a classify-only region
        keeps everyone. The head point decides, because that is what the trigger line
        tests too, so a person cannot be inside for the crossing test and outside here.
        """
        r = cam.region
        if r is None or not tracks or not r.affects_tracking:
            return tracks                       # classify-only regions leave tracking alone
        w, h = cam.source.width, cam.source.height
        if not (w and h):                       # resolution not known yet
            return tracks
        return [t for t in tracks if r.keeps(t.head, w, h)]

    def _due_tracks(self, cam: Camera, f: Frame, tracks: list[Track], force: bool) -> list[tuple[int, Track]]:
        c = self.cfg
        due = []
        for t in tracks:
            st = cam.counter.get_state(t.track_id)
            n_obs = st.n_obs if st else 0
            last = st.last_classified if st else -10_000
            if force or n_obs == 0 or f.index - last >= c.classify_every:
                if t.height >= c.min_crop_height or force:
                    if not self._classifiable(cam, t):
                        continue
                    due.append((n_obs, t))
        due.sort(key=lambda x: x[0])  # least-observed first
        return due[: c.max_crops]

    def _classifiable(self, cam: Camera, t: Track) -> bool:
        """A classify-only region spends the crop budget on the part of the frame
        where people are worth looking at; they are still tracked and counted
        everywhere. Every classify path goes through here, the crossing included:
        a crossing look carries CROSSING_WEIGHT, so letting it through would
        decide the demographics from exactly the crop the box excluded."""
        r = cam.region
        if r is None or r.affects_tracking:
            return True
        w, h = cam.source.width, cam.source.height
        if not (w and h):
            return True
        return r.contains(t.head, w, h)

    def _classify(self, frames: list[Optional[Frame]], tracks_all: list[list[Track]]) -> None:
        """One CLIP batch for all cameras' due crops."""
        crops, owners = [], []
        for cam, f, tracks in zip(self.cameras, frames, tracks_all):
            if f is None or cam.counter is None or not tracks:
                continue
            for _, t in self._due_tracks(cam, f, tracks, force=False):
                c = crop_person(f.image, t.box)
                if c is not None:
                    crops.append(c)
                    owners.append((cam, f, t))
        if not crops:
            return
        for (cam, f, t), (p_male, p_child) in zip(owners, self.classifier.classify(crops)):
            cam.counter.ensure_state(t.track_id, f.index).add_observation(p_male, p_child, f.index, t.height)

    def _classify_crossing(self, cam: Camera, f: Frame, tracks: list[Track], track_id: int) -> None:
        """Classify the person's crop at the moment they cross, weighted above the earlier looks."""
        t = next((t for t in tracks if t.track_id == track_id), None)
        if t is None or not self._classifiable(cam, t):
            return
        crop = crop_person(f.image, t.box)
        if crop is None:
            return
        p_male, p_child = self.classifier.classify([crop])[0]
        cam.counter.ensure_state(track_id, f.index).add_observation(
            p_male, p_child, f.index, t.height, weight=CROSSING_WEIGHT)

    def _classify_tracks(self, cam: Camera, f: Frame, tracks: list[Track], force: bool = False) -> None:
        crops, sel = [], []
        for _, t in self._due_tracks(cam, f, tracks, force):
            c = crop_person(f.image, t.box)
            if c is not None:
                crops.append(c)
                sel.append(t)
        if crops:
            for t, (p_male, p_child) in zip(sel, self.classifier.classify(crops)):
                cam.counter.ensure_state(t.track_id, f.index).add_observation(p_male, p_child, f.index, t.height)

    # ------------------------------------------------------------- render
    def annotated_frame(self, cam: Camera, draft: bool = True) -> Optional[np.ndarray]:
        """Copy of the latest frame with line / track overlays, or None before the first frame."""
        snap = cam.display
        if snap is None:
            return None
        frame, tracks = snap
        img = frame.image.copy()
        # The draft overlay in the browser owns the picture while a panel is being
        # calibrated; otherwise the area is drawn whether or not a line exists, so
        # a camera set up area-first still shows what it is doing.
        settled = not cam.editing and not (self.setup_mode and len(cam.setup_points) == 2)
        if settled and cam.region is not None:
            ui.draw_region(img, cam.region, cam.present)
        if settled and cam.line is not None:
            shown = TriggerLine(cam.line.p1, cam.line.p2,
                                cam.setup_entry_side if self.setup_mode else cam.line.entry_side, cam.line.name)
            ui.draw_line(img, shown)
        if self.setup_mode and draft:
            ui.draw_setup_points(img, cam.setup_points, cam.setup_entry_side)
        ui.draw_tracks(img, tracks, cam.counter, cam.flash, frame.index)
        if cam.banner:
            ui.draw_alarm(img, cam.banner)
        return img

    # --------------------------------------------------------------- misc
    def camera_rows(self) -> list[dict]:
        return [{
            "name": cam.name,
            "status": cam.source.status(),
            "fps": cam.fps,
            "ok": cam.source.connected,
            "has_line": cam.line is not None,
        } for cam in self.cameras]

    @property
    def stopping(self) -> bool:
        return self._stop.is_set()

    def shutdown(self) -> None:
        self._stop.set()
        if self._worker.is_alive():
            self._worker.join(timeout=5.0)
        for cam in self.cameras:
            cam.source.release()
        self.log.close()
        if self.track_log is not None:
            self.track_log.close()
        t = self.stats.total
        print(f"\n[summary] visitors={t.visitors} (entries {t.entries}, exits {t.exits})  "
              f"male={t.male} female={t.female}  adult={t.adult} child={t.child}")
        print(f"[summary] events written to {self.log.path}")
