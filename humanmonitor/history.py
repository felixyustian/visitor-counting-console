"""Historical records for the web console.

* HistoryLog  - output/history.csv: hourly / opening / closing snapshots of the
                counters, manual corrections and recorded events.
* Scheduler   - writes the hourly rows, the closing row, and resets the count to
                the baseline at opening time (building hours from SiteConfig).
* FrameStore  - encodes each camera's annotated frame once (JPEG) for every
                MJPEG viewer and keeps a rolling buffer for clip recording.
* Recorder    - saves a clip (pre + post seconds) and a snapshot of one camera
                to output/events/ and logs the event in the history CSV.
"""
from __future__ import annotations

import csv
import json
import threading
import time
from collections import deque
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from .site import SiteConfig

HISTORY_HEADER = ["timestamp", "kind", "date", "occupancy", "entries", "exits", "male", "female",
                  "adult", "child", "visitors", "v_male", "v_female", "v_adult", "v_child",
                  "capacity", "pct", "cameras", "note", "clip"]


class HistoryLog:
    """Append-only CSV with an in-memory copy for the HISTORY panel."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows: list[dict] = []
        self.lock = threading.Lock()
        if self.path.exists() and self.path.stat().st_size:
            with self.path.open(newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                header = reader.fieldnames or []
                for row in reader:
                    self.rows.append({k: row.get(k, "") for k in HISTORY_HEADER})
            if list(header) != HISTORY_HEADER:
                # older column layout: keep the old file aside and start a fresh one (rows stay in memory)
                old = self.path.with_name(f"{self.path.stem}_{datetime.now():%Y%m%d_%H%M%S}{self.path.suffix}")
                self.path.rename(old)
                print(f"[history] {self.path.name} had an older column layout - moved to {old.name}")
                with self.path.open("w", newline="", encoding="utf-8") as fh:
                    w = csv.DictWriter(fh, fieldnames=HISTORY_HEADER)
                    w.writeheader()
                    w.writerows(self.rows)
        else:
            with self.path.open("w", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerow(HISTORY_HEADER)

    def write(self, kind: str, totals: dict, capacity: int, cameras: list[dict],
              note: str = "", clip: str = "", when: Optional[datetime] = None) -> dict:
        when = when or datetime.now()
        occ = max(0, int(totals.get("visitors", 0)))
        row = {
            "timestamp": when.strftime("%Y-%m-%dT%H:%M:%S"),
            "kind": kind,
            "date": when.date().isoformat(),
            "occupancy": occ,
            "entries": totals.get("entries", 0), "exits": totals.get("exits", 0),
            "male": totals.get("male", 0), "female": totals.get("female", 0),
            "adult": totals.get("adult", 0), "child": totals.get("child", 0),
            "visitors": totals.get("total", totals.get("entries", 0)),
            "v_male": totals.get("v_male", 0), "v_female": totals.get("v_female", 0),
            "v_adult": totals.get("v_adult", 0), "v_child": totals.get("v_child", 0),
            "capacity": capacity,
            "pct": round(100.0 * occ / capacity, 1) if capacity else 0,
            "cameras": " | ".join(f"{c['name']}: in {c['counts']['entries']} out {c['counts']['exits']}"
                                  for c in cameras if c.get("kind") != "none"),
            "note": note,
            "clip": clip,
        }
        with self.lock:
            with self.path.open("a", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, fieldnames=HISTORY_HEADER).writerow(row)
            self.rows.append({k: str(v) for k, v in row.items()})
        return row

    def has(self, kind: str, day: str, hour: Optional[int] = None) -> bool:
        with self.lock:
            for r in reversed(self.rows):
                if r["kind"] == kind and r["date"] == day and (hour is None or int(r["timestamp"][11:13]) == hour):
                    return True
        return False

    def summary(self) -> dict:
        today = date.today().isoformat()
        with self.lock:
            rows = list(self.rows)
        closings = [r for r in rows if r["kind"] == "closing"]
        return {
            "last_closing": closings[-1] if closings else None,
            "today": [r for r in rows if r["date"] == today and r["kind"] in ("opening", "hourly", "closing", "correction")][-30:],
            "days": closings[-7:],
            "events": [r for r in rows if r["kind"] == "event"][-12:],
            "rows": len(rows),
        }


class Scheduler:
    """Hourly snapshots, closing row and opening reset, driven by the building hours."""

    def __init__(self, site: SiteConfig, log: HistoryLog, snapshot: Callable[[], dict],
                 reset: Callable[[], None], tick: float = 5.0):
        self.site, self.log, self.snapshot, self.reset = site, log, snapshot, reset
        self.tick = tick
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="scheduler", daemon=True)
        self._last_check = datetime.now()
        self.last_action: Optional[str] = None

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _write(self, kind: str, note: str = "", when: Optional[datetime] = None) -> None:
        st = self.snapshot()
        self.log.write(kind, st["totals"], st["capacity"], st["cameras"], note=note, when=when)
        self.last_action = f"{kind} {(when or datetime.now()):%H:%M}"
        print(f"[history] {kind} row written{(' - ' + note) if note else ''}")

    def _loop(self) -> None:
        while not self._stop.wait(self.tick):
            try:
                self._check(datetime.now())
            except Exception as e:  # never let a bad time string kill the timer
                print(f"[history] scheduler error: {e!r}")

    def _check(self, now: datetime) -> None:
        prev, self._last_check = self._last_check, now
        if not self.site.configured:
            return
        for day in (now.date() - timedelta(days=1), now.date()):
            opening, closing = self.site.open_at(day), self.site.close_at(day)
            # opening: reset to the baseline once per operating day
            if prev < opening <= now and not self.log.has("opening", day.isoformat()):
                self.reset()
                self._write("opening", f"count reset to baseline {self.site.baseline}", when=now)
            # closing: final totals of the day
            if prev < closing <= now and not self.log.has("closing", day.isoformat()):
                self._write("closing", "closing time", when=now)
            # hourly while open (top of each hour strictly between opening and closing)
            if opening < now < closing:
                top = now.replace(minute=0, second=0, microsecond=0)
                if prev < top <= now and top > opening and not self.log.has("hourly", top.date().isoformat(), top.hour):
                    self._write("hourly", when=top)


class FrameStore:
    """One JPEG encode per camera per frame, shared by all MJPEG viewers, with a rolling buffer for clips."""

    def __init__(self, engine, width: int = 960, fps: float = 15.0, quality: int = 80, keep_seconds: float = 12.0):
        self.engine, self.width, self.fps, self.quality, self.keep = engine, width, fps, quality, keep_seconds
        self.latest: dict[int, tuple[int, float, bytes]] = {}     # cid -> (seq, t, jpeg)
        self.rings: dict[int, deque] = {}                          # cid -> deque[(t, jpeg)]
        self._snaps: dict[int, object] = {}
        self._seq = 0
        self.lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="frames", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def encode(self, img: np.ndarray) -> Optional[bytes]:
        h, w = img.shape[:2]
        if w > self.width:
            img = cv2.resize(img, (self.width, int(h * self.width / w)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        return buf.tobytes() if ok else None

    def _loop(self) -> None:
        interval = 1.0 / max(1.0, self.fps)
        while not self._stop.is_set():
            t0 = time.perf_counter()
            for cam in list(self.engine.cameras):
                snap = cam.display
                if snap is None or (self._snaps.get(cam.cid) is snap and not self.engine.setup_mode):
                    continue
                img = self.engine.annotated_frame(cam, draft=False)
                if img is None:
                    continue
                data = self.encode(img)
                if data is None:
                    continue
                now = time.time()
                with self.lock:
                    self._seq += 1
                    self.latest[cam.cid] = (self._seq, now, data)
                    ring = self.rings.setdefault(cam.cid, deque())
                    ring.append((now, data))
                    while ring and now - ring[0][0] > self.keep:
                        ring.popleft()
                self._snaps[cam.cid] = snap
            # forget slots that were removed
            live = {c.cid for c in self.engine.cameras}
            for cid in [c for c in list(self.latest) if c not in live]:
                with self.lock:
                    self.latest.pop(cid, None); self.rings.pop(cid, None)
                self._snaps.pop(cid, None)
            rest = interval - (time.perf_counter() - t0)
            if rest > 0:
                time.sleep(rest)

    def get(self, cid: int) -> Optional[tuple[int, float, bytes]]:
        with self.lock:
            return self.latest.get(cid)

    def ring_copy(self, cid: int, seconds: float) -> list[tuple[float, bytes]]:
        cutoff = time.time() - seconds
        with self.lock:
            return [(t, d) for t, d in self.rings.get(cid, ()) if t >= cutoff]


class Recorder:
    """Saves clips + snapshots of one camera to output/events and logs them in the history CSV."""

    def __init__(self, engine, frames: FrameStore, log: HistoryLog, snapshot: Callable[[], dict],
                 out_dir: Path | str = Path("output") / "events", pre: float = 10.0, post: float = 20.0):
        self.engine, self.frames, self.log, self.snapshot = engine, frames, log, snapshot
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.pre, self.post = pre, post
        self.active: dict[int, dict] = {}       # cid -> recording in progress
        self.recent: deque[dict] = deque(maxlen=20)
        self.lock = threading.Lock()

    def is_recording(self, cid: int) -> bool:
        return cid in self.active

    def record(self, cam, note: str = "", kind: str = "manual") -> dict:
        """Start a recording of this camera in the background; returns its descriptor."""
        with self.lock:
            if cam.cid in self.active:
                return self.active[cam.cid]
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in cam.name)[:32]
            base = f"{stamp}_cam{cam.idx + 1}_{safe}"
            rec = {"cid": cam.cid, "cam_name": cam.name, "started": time.time(), "note": note, "kind": kind,
                   "clip": f"{base}.mp4", "snapshot": f"{base}.jpg", "done": False, "frames": 0}
            self.active[cam.cid] = rec
        threading.Thread(target=self._run, args=(cam, rec), name=f"record-{cam.cid}", daemon=True).start()
        return rec

    def _run(self, cam, rec: dict) -> None:
        try:
            frames = self.frames.ring_copy(cam.cid, self.pre)
            latest = self.frames.get(cam.cid)
            if latest is not None:
                (self.out_dir / rec["snapshot"]).write_bytes(latest[2])
            last_seq = latest[0] if latest else -1
            end = time.time() + self.post
            while time.time() < end and not self.engine.stopping:
                cur = self.frames.get(cam.cid)
                if cur is not None and cur[0] != last_seq:
                    frames.append((cur[1], cur[2]))
                    last_seq = cur[0]
                time.sleep(0.03)
            rec["frames"] = len(frames)
            if len(frames) >= 2:
                self._write_clip(self.out_dir / rec["clip"], frames)
            else:
                rec["clip"] = ""
        except Exception as e:
            print(f"[record] failed: {e!r}")
            rec["clip"] = ""
        finally:
            rec["done"] = True
            rec["ended"] = time.time()
            st = self.snapshot()
            note = f"{rec['kind']}: {rec['note']}" if rec["note"] else rec["kind"]
            self.log.write("event", st["totals"], st["capacity"], st["cameras"],
                           note=f"{cam.name} - {note}", clip=rec["clip"] or rec["snapshot"])
            with self.lock:
                self.active.pop(cam.cid, None)
                self.recent.appendleft(rec)
            print(f"[record] {cam.name}: {rec['frames']} frames -> {rec['clip'] or rec['snapshot']}")

    @staticmethod
    def _write_clip(path: Path, frames: list[tuple[float, bytes]]) -> None:
        first = cv2.imdecode(np.frombuffer(frames[0][1], np.uint8), cv2.IMREAD_COLOR)
        h, w = first.shape[:2]
        span = max(0.5, frames[-1][0] - frames[0][0])
        fps = max(2.0, min(30.0, (len(frames) - 1) / span))
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for _, data in frames:
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                if img.shape[:2] != (h, w):
                    img = cv2.resize(img, (w, h))
                vw.write(img)
        vw.release()

    def public(self) -> list[dict]:
        with self.lock:
            cur = [dict(r) for r in self.active.values()]
            done = [dict(r) for r in self.recent]
        return cur + done
