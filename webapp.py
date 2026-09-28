"""Visitor Counting Console - browser console (crowd panel + live camera panels).

Same pipeline as app.py (humanmonitor.engine), rendered in the browser: a crowd
management console panel plus one panel per camera (MJPEG), fed over a WebSocket.
Cameras are added / removed on the page (up to --max-cameras), trigger lines are
drawn on the frames, building hours drive the hourly / closing history rows and
the opening reset, and events can be recorded as clips.

Run:
    python webapp.py --mode video          # the files listed in config/sources.json
    python webapp.py --mode live           # the RTSP URLs listed in config/sources.json
    python webapp.py --sources a.mp4 rtsp://cam/live   # explicit feeds
then open http://localhost:8780

Files written: config/sources.json (cameras), config/lines.json (trigger lines),
config/site.json (building settings), output/events.csv (crossings),
output/history.csv (hourly / closing / events), output/events/ (clips).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app import add_engine_args, check_sources
from humanmonitor.config import (ROLES, ROLE_COUNT_INSIDE, Region, is_url,
                                 template_of, unencodable_password)
from humanmonitor.engine import PRESENT_ZERO, Camera, Engine, EngineConfig, EventRecord
from humanmonitor.history import FrameStore, HistoryLog, Recorder, Scheduler
from humanmonitor.site import SiteConfig

WEB_DIR = Path(__file__).parent / "web"
DEFAULT_SOURCES_CONFIG = Path("config") / "sources.json"


def mask_credentials(src: str) -> str:
    return re.sub(r"://[^@/]+@", "://***@", src) if is_url(src) else src


def read_sources(path: Path) -> dict:
    """sources.json is site-specific and gitignored; fall back to the committed template."""
    for p in (path, template_of(path)):
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                # do not fall through to the template: that would silently
                # discard the cameras in the file the operator is editing
                print(f"[web] {p} is not valid JSON ({e}) - ignoring it")
                return {}
    return {}


def load_sources(mode: str, path: Path, limit: int) -> list:
    """Entries of the mode's list: "path-or-url" or {"src", "name", "area_capacity"}. Empty = add cameras on the page."""
    data = read_sources(path)
    if not data:
        print(f"[web] no {path} or {template_of(path)} - starting with empty camera slots")
        return []
    return list(data.get(mode) or [])[:limit]


def save_sources(mode: str, path: Path, entries: list) -> None:
    # seeded from the template on first save, so editing one mode on the page
    # does not drop the other mode's list
    data = read_sources(path)
    data.pop("_comment", None)
    data[mode] = entries
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


# a track is seeded from the side it is first seen on (humanmonitor/counter.py),
# so a person who first appears already past the line is seeded on the wrong side
# and their next step registers a crossing that did not happen. That is what a
# line drawn hard against the frame edge causes - seen live on 23 Sep (03:54:
# "never detected yet, then you already [counted]"). Warn, do not block: an
# entrance really can sit at the edge of the view.
EDGE_MARGIN = 0.04          # fraction of the frame; ~50 px on a 1280-wide stream


def edge_warning(p1, p2) -> Optional[str]:
    near = [n for n, p in (("start", p1), ("end", p2))
            if min(p[0], 1 - p[0], p[1], 1 - p[1]) < EDGE_MARGIN]
    if not near:
        return None
    which = " and ".join(near)
    return (f"the {which} of the line {'sits' if len(near) == 1 else 'sit'} within "
            f"{EDGE_MARGIN:.0%} of the frame edge - people may cross before the tracker "
            f"has picked them up, which miscounts. Pull it further into the frame if you can.")


def region_warning(region, line) -> Optional[str]:
    """A count-inside region that does not contain the whole trigger line breaks counting.

    A person is only tracked while inside such a region, so one approaching from
    outside is first seen already at or past the line: their side is seeded on
    the wrong side and the crossing is missed or invented. The other two roles
    leave tracking alone, so the line may sit anywhere - including outside the box.
    """
    if line is None or region.role != ROLE_COUNT_INSIDE:
        return None
    r = region.normalised()
    outside = [n for n, (x, y) in (("start", line.p1), ("end", line.p2))
               if not (r.x1 <= x <= r.x2 and r.y1 <= y <= r.y2)]
    if outside:
        which = " and ".join(outside)
        return (f"the {which} of the trigger line {'is' if len(outside) == 1 else 'are'} outside "
                f"this region - crossings there cannot be counted. Cover the whole line, with room "
                f"on both sides for people to be picked up before they reach it.")
    # the line is inside, but is there approach room on both sides?
    thin = min(abs(line.p1[1] - r.y1), abs(line.p1[1] - r.y2),
               abs(line.p2[1] - r.y1), abs(line.p2[1] - r.y2))
    if thin < 0.06:
        return ("the region ends very close to the trigger line - people may not be picked up "
                "before they cross it. Leave room on both sides.")
    return None


def entry_src(entry) -> str:
    return str(entry.get("src", "")) if isinstance(entry, dict) else str(entry)


class Console:
    """Web-only state: site settings, alarms, occupancy trend, history, recordings."""

    def __init__(self, engine: Engine, mode: str, site: SiteConfig, sources_path: Optional[Path],
                 history: HistoryLog, frames: FrameStore):
        self.engine = engine
        self.mode = mode                      # sources.json key that page edits are saved under
        self.site = site
        self.sources_path = sources_path
        self.history_log = history
        self.frames = frames
        self.recorder: Optional[Recorder] = None
        self.session = f"{mode}_{time.strftime('%Y%m%d_%H%M%S')}"
        self.acked: dict[str, float] = {}         # alert id -> time acknowledged
        self.raised: dict[str, float] = {}        # alert id -> first time seen (active)
        self.trend: deque[tuple[float, int]] = deque(maxlen=240)   # (t, occupancy) every 5 s = 20 min
        self._last_sample = 0.0
        self.over: Optional[dict] = None          # active site over-capacity alarm {since, cid, cam_name}
        self.area_over: dict[int, dict] = {}      # cid -> {since}
        self.lock = threading.Lock()
        engine.stats.set_baseline(site.baseline)

    # ---------------------------------------------------------------- levels
    @property
    def capacity(self) -> int:
        return self.site.capacity

    def level(self) -> tuple[float, str]:
        occ = max(0, self.engine.stats.total.visitors)
        frac = occ / self.capacity
        if frac >= 1.0:
            return frac, "over"
        if frac >= self.site.warn_pct / 100.0:
            return frac, "crowded"
        if frac >= 0.6:
            return frac, "busy"
        return frac, "normal"

    def _last_entry_cam(self) -> Optional[Camera]:
        for rec in self.engine.recent:
            if rec.event.direction > 0:
                cams = self.engine.cameras
                return cams[rec.cam_idx] if rec.cam_idx < len(cams) and cams[rec.cam_idx].name == rec.cam_name else None
        return None

    def _update_alarms(self) -> None:
        """Track the site and per-area over-capacity alarms and paint the camera banners."""
        e = self.engine
        frac, lvl = self.level()
        now = time.time()
        with self.lock:
            if lvl == "over" and self.over is None:
                cam = self._last_entry_cam()
                self.over = {"since": now, "cid": cam.cid if cam else None, "cam_name": cam.name if cam else None,
                             "occupancy": e.stats.total.visitors}
                print(f"[alarm] site over capacity ({e.stats.total.visitors}/{self.capacity})"
                      f"{' - tipped at ' + cam.name if cam else ''}")
                if cam is not None and self.recorder is not None:
                    self.recorder.record(cam, f"site over capacity {e.stats.total.visitors}/{self.capacity}", kind="auto")
            elif lvl != "over" and self.over is not None:
                self.over = None
                self.acked.pop("capacity_over", None)
            for cam in e.cameras:
                occ = e.stats.per_camera[cam.idx].visitors if cam.idx < len(e.stats.per_camera) else 0
                is_over = cam.area_capacity > 0 and occ >= cam.area_capacity and not cam.source.is_null
                if is_over and cam.cid not in self.area_over:
                    self.area_over[cam.cid] = {"since": now, "occupancy": occ}
                    print(f"[alarm] area over capacity at {cam.name} ({occ}/{cam.area_capacity})")
                    if self.recorder is not None:
                        self.recorder.record(cam, f"area over capacity {occ}/{cam.area_capacity}", kind="auto")
                elif not is_over and cam.cid in self.area_over:
                    self.area_over.pop(cam.cid, None)
                    self.acked.pop(f"area_over_{cam.cid}", None)
                # banner text drawn into the frame (and therefore into recorded clips)
                if cam.cid in self.area_over:
                    cam.banner = f"AREA OVER CAPACITY  {occ} / {cam.area_capacity}"
                elif self.over is not None:
                    tipped = self.over.get("cid") == cam.cid
                    cam.banner = f"SITE OVER CAPACITY  {e.stats.total.visitors} / {self.capacity}" + ("  - TRIGGERED HERE" if tipped else "")
                else:
                    cam.banner = None

    # ------------------------------------------------------------- alerts
    def _current_alerts(self) -> list[dict]:
        e = self.engine
        out = []
        frac, lvl = self.level()
        occ = e.stats.total.visitors
        if lvl == "over":
            where = f" - triggered at {self.over['cam_name']}" if self.over and self.over.get("cam_name") else ""
            out.append({"id": "capacity_over", "level": "critical",
                        "title": f"OVER CAPACITY: {occ} inside, limit {self.capacity}{where}",
                        "detail": "crowd control - stop entries"})
        elif lvl == "crowded":
            out.append({"id": "capacity_crowded", "level": "warning",
                        "title": f"Approaching capacity: {occ} of {self.capacity} ({frac:.0%})",
                        "detail": f"crowd control - warning threshold {self.site.warn_pct}%"})
        elif lvl == "busy":
            out.append({"id": "capacity_busy", "level": "info",
                        "title": f"Busy: {occ} of {self.capacity} ({frac:.0%})",
                        "detail": "occupancy"})
        for cam in e.cameras:
            if cam.source.is_null:
                continue
            if cam.cid in self.area_over:
                c = e.stats.per_camera[cam.idx].visitors
                out.append({"id": f"area_over_{cam.cid}", "level": "critical", "cid": cam.cid,
                            "title": f"AREA OVER CAPACITY at {cam.name}: {c} of {cam.area_capacity}",
                            "detail": f"area - {cam.name}"})
            if not cam.source.connected:
                out.append({"id": f"cam{cam.cid}_offline", "level": "critical", "cid": cam.cid,
                            "title": f"{cam.name} is offline ({cam.source.last_error or 'no signal'})",
                            "detail": f"camera health - {cam.name}"})
            elif cam.source.ended:
                out.append({"id": f"cam{cam.cid}_ended", "level": "info", "cid": cam.cid,
                            "title": f"{cam.name}: video file ended", "detail": f"playback - {cam.name}"})
            if cam.line is None:
                out.append({"id": f"cam{cam.cid}_noline", "level": "warning", "cid": cam.cid,
                            "title": f"{cam.name} has no trigger line - not counting",
                            "detail": f"configuration - {cam.name} - press calibrate"})
        if self.site.configured and not self.site.is_open():
            out.append({"id": "closed", "level": "info",
                        "title": f"Outside building hours ({self.site.open}-{self.site.close}) - counting continues",
                        "detail": "schedule"})
        if e.setup_mode:
            out.append({"id": "setup_mode", "level": "info", "title": "Calibration in progress - counting paused",
                        "detail": "set-up mode"})
        if e.paused:
            out.append({"id": "paused", "level": "info", "title": "Pipeline paused", "detail": "control"})
        now = time.time()
        for rec in e.recent:
            if now - rec.time > 60:
                break
            if rec.event.direction > 0 and rec.event.age == "C":
                out.append({"id": f"child_{rec.cam_idx}_{rec.event.track_id}_{int(rec.time)}", "level": "info",
                            "title": f"Child entered at {rec.cam_name}", "detail": f"demographics - {rec.cam_name}"})
        return out

    def alerts(self) -> list[dict]:
        now = time.time()
        cur = self._current_alerts()
        ids = {a["id"] for a in cur}
        with self.lock:
            for a in cur:
                self.raised.setdefault(a["id"], now)
            for k in [k for k in self.raised if k not in ids]:
                self.raised.pop(k)
                self.acked.pop(k, None)   # a cleared alert re-raises fresh next time
            for a in cur:
                a["since"] = self.raised[a["id"]]
                a["acked"] = a["id"] in self.acked
        order = {"critical": 0, "warning": 1, "info": 2}
        cur.sort(key=lambda a: (a["acked"], order[a["level"]], -a["since"]))
        return cur

    def ack(self, alert_id: Optional[str]) -> None:
        with self.lock:
            targets = [alert_id] if alert_id else list(self.raised)
            for k in targets:
                if k in self.raised:
                    self.acked[k] = time.time()

    # ----------------------------------------------------------- snapshot
    def sample_trend(self) -> None:
        now = time.time()
        if now - self._last_sample >= 5.0:
            self.trend.append((now, max(0, self.engine.stats.total.visitors)))
            self._last_sample = now

    @staticmethod
    def _counts(c) -> dict:
        return {"visitors": c.visitors, "initial": getattr(c, "initial", 0), "entries": c.entries, "exits": c.exits,
                "male": c.male, "female": c.female, "adult": c.adult, "child": c.child,
                "gender_unknown": c.gender_unknown, "age_unknown": c.age_unknown,
                "total": c.entries, "v_male": c.v_male, "v_female": c.v_female, "v_adult": c.v_adult, "v_child": c.v_child,
                # raw male/female/adult/child above are the running buckets (what history.csv records);
                # these are the same people scaled to the occupancy, so the panel always reconciles
                "in_male": c.inside_split()[0], "in_female": c.inside_split()[1],
                "in_adult": c.inside_split()[2], "in_child": c.inside_split()[3],
                # net = entered - left, and its split: all detected, no assumed opening count
                "net": c.entries - c.exits,
                "n_male": c.net_split()[0], "n_female": c.net_split()[1],
                "n_adult": c.net_split()[2], "n_child": c.net_split()[3]}

    @staticmethod
    def _event(rec: EventRecord) -> dict:
        ev = rec.event
        return {"t": rec.time, "cam": rec.cam_idx, "cam_name": rec.cam_name, "track": ev.track_id,
                "dir": ev.direction, "gender": ev.gender, "age": ev.age,
                "p_male": round(ev.p_male, 2), "p_child": round(ev.p_child, 2), "n_obs": ev.n_obs}

    def camera_dict(self, cam: Camera) -> dict:
        e = self.engine
        src, line = cam.source, cam.line
        counts = self._counts(e.stats.per_camera[cam.idx]) if cam.idx < len(e.stats.per_camera) else self._counts(type(e.stats.total)())
        return {
            "cid": cam.cid, "idx": cam.idx,
            "name": cam.name, "label": cam.label,
            "src": mask_credentials(src.src),
            "kind": "none" if src.is_null else ("live" if src.is_stream else "file"),
            "status": src.status(), "ok": src.connected,
            "fps": round(cam.fps, 1), "w": src.width, "h": src.height,
            "has_line": line is not None,
            "line": None if line is None else {"p1": list(line.p1), "p2": list(line.p2),
                                               "entry_side": cam.setup_entry_side if e.setup_mode else line.entry_side},
            "editing": cam.editing,
            "area_capacity": cam.area_capacity,
            "region": None if cam.region is None else
                      [cam.region.x1, cam.region.y1, cam.region.x2, cam.region.y2],
            "region_role": None if cam.region is None else cam.region.role,
            # live count of who is in the counted part of the frame, split M/F and A/C.
            # Independent of the trigger line, so a drawn area reports its own
            # population before anyone has crossed anything. A camera that is not
            # delivering frames reports zero rather than the last thing it saw:
            # unlike the crossing totals, this one is only ever "right now".
            "present": dict(cam.present) if src.connected else dict(PRESENT_ZERO),
            "alt_src": mask_credentials(cam.alt_src) if cam.alt_src else None,
            "alt_kind": None if not cam.alt_src else ("live" if is_url(cam.alt_src) else "file"),
            "area_over": cam.cid in self.area_over,
            "tipped": bool(self.over and self.over.get("cid") == cam.cid),
            "recording": bool(self.recorder and self.recorder.is_recording(cam.cid)),
            "counts": counts,
        }

    def snapshot(self, with_history: bool = True) -> dict:
        e = self.engine
        self.sample_trend()
        self._update_alarms()
        frac, lvl = self.level()
        cams = [self.camera_dict(c) for c in e.cameras]
        kinds = {c["kind"] for c in cams}
        out = {
            "mode": "live" if "live" in kinds else ("video" if "file" in kinds else "empty"),
            "config_mode": self.mode,
            "session": self.session,
            "device": e.device,
            "pipeline_fps": round(e.pipeline_fps, 1),
            "uptime": int(time.time() - e.started_at),
            "paused": e.paused,
            "setup_mode": e.setup_mode,
            "max_cameras": e.cfg.max_cameras,
            "site": self.site.public(),
            "capacity": self.capacity,
            "occupancy": {"current": max(0, e.stats.total.visitors), "capacity": self.capacity,
                          "fraction": round(frac, 3), "pct": round(100 * frac), "level": lvl,
                          "warn_pct": self.site.warn_pct, "over": self.over},
            "totals": self._counts(e.stats.total),
            "cameras": cams,
            "alerts": self.alerts(),
            "events": [self._event(r) for r in list(e.recent)[:30]],
            "trend": [[int(t), n] for t, n in self.trend],
            "revisions": len(e.revisions),
            "recordings": self.recorder.public() if self.recorder else [],
            "error": repr(e.error) if e.error else None,
        }
        if with_history:
            out["history"] = self.history_log.summary()
        return out

    def log_snapshot(self) -> dict:
        """Cheap snapshot for the history log (no alerts / history recursion)."""
        e = self.engine
        return {"totals": self._counts(e.stats.total), "capacity": self.capacity,
                "cameras": [self.camera_dict(c) for c in e.cameras]}


# ---------------------------------------------------------------- server
def build_app(engine: Engine, console: Console, frames: FrameStore, should_stop=lambda: False) -> FastAPI:
    app = FastAPI(title="Visitor Analytics console")
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")
    events_dir = console.recorder.out_dir if console.recorder else Path("output") / "events"
    events_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/events", StaticFiles(directory=str(events_dir)), name="events")

    def cam_or_404(cid: int) -> Camera:
        cam = engine.camera_by_id(cid)
        if cam is None:
            raise HTTPException(404, "no such camera")
        return cam

    @app.get("/")
    def index():
        return FileResponse(WEB_DIR / "index.html")

    # ---- video ----
    def mjpeg(cid: int):
        last_seq = -1
        blank_for = None   # the source object the current blank frame was drawn for
        while not (engine.stopping or should_stop()):   # end the response on shutdown instead of being cancelled
            cam = engine.camera_by_id(cid)
            if cam is None:
                return
            cur = frames.get(cid)
            if cur is None or cam.display is None:
                if blank_for is not cam.source:
                    label = "no camera - add one with the source button" if cam.source.is_null else f"{cam.name}: {cam.source.status()}"
                    data = frames.encode(cv2_blank(frames.width, label))
                    if data:
                        yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n"
                    blank_for = cam.source
                time.sleep(0.05)
                continue
            seq, _, data = cur
            if seq == last_seq:
                time.sleep(0.01)
                continue
            last_seq = seq
            blank_for = None
            yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n"

    @app.get("/stream/{cid}")
    def stream(cid: int):
        cam_or_404(cid)
        return StreamingResponse(mjpeg(cid), media_type="multipart/x-mixed-replace; boundary=frame",
                                 headers={"Cache-Control": "no-cache, no-store", "Pragma": "no-cache"})

    @app.get("/snapshot/{cid}.jpg")
    def snapshot_jpg(cid: int):
        cam = cam_or_404(cid)
        img = engine.annotated_frame(cam, draft=False)
        if img is None:
            raise HTTPException(503, "no frame yet")
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return StreamingResponse(iter([buf.tobytes()]), media_type="image/jpeg")

    # ---- state ----
    @app.get("/api/state")
    def state():
        return console.snapshot()

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        n = 0
        try:
            while True:
                await sock.send_text(json.dumps(console.snapshot(with_history=(n % 20 == 0))))
                n += 1
                await asyncio.sleep(0.25)
        except (WebSocketDisconnect, RuntimeError):
            pass

    @app.get("/api/history")
    def history():
        return console.history_log.summary()

    # ---- site / occupancy ----
    @app.post("/api/site")
    def site(body: dict):
        try:
            changed = console.site.update(body)
        except (ValueError, TypeError) as e:
            raise HTTPException(400, f"invalid site settings: {e}")
        if "baseline" in changed:
            engine.stats.set_baseline(console.site.baseline)
        if body.get("apply_baseline"):
            engine.reset_counts()
            console.history_log.write("opening", **_log_kw(console), note=f"day started - count set to baseline {console.site.baseline}")
        console.site.save()
        return {"ok": True, "site": console.site.public(), "changed": changed}

    @app.post("/api/occupancy")
    def occupancy(body: dict):
        try:
            n = int(body["current"])
        except (KeyError, ValueError, TypeError):
            raise HTTPException(400, "current must be an integer")
        delta = engine.set_occupancy(n)
        console.history_log.write("correction", **_log_kw(console),
                                  note=f"occupancy corrected to {max(0, n)} ({delta:+d})" + (f" - {body['note']}" if body.get("note") else ""))
        return {"ok": True, "occupancy": engine.stats.total.visitors, "delta": delta}

    # ---- controls ----
    @app.post("/api/reset")
    def reset():
        engine.reset_counts()
        with console.lock:
            console.trend.clear()
        console.history_log.write("correction", **_log_kw(console), note=f"counters reset to baseline {console.site.baseline}")
        return {"ok": True}

    @app.post("/api/pause")
    def pause(body: dict):
        engine.paused = bool(body.get("paused", not engine.paused))
        return {"ok": True, "paused": engine.paused}

    @app.post("/api/setup")
    def setup(body: dict):
        engine.set_setup_mode(bool(body.get("on", not engine.setup_mode)))
        if not engine.setup_mode:
            for c in engine.cameras:
                c.editing = False
        return {"ok": True, "setup_mode": engine.setup_mode}

    @app.post("/api/alerts/ack")
    def ack(body: dict):
        console.ack(body.get("id"))
        return {"ok": True}

    # ---- cameras / sources ----
    def persist_sources():
        if console.sources_path is not None:
            save_sources(console.mode, console.sources_path, engine.source_entries())

    @app.post("/api/cameras")
    def add_camera():
        try:
            cam = engine.add_slot()
        except ValueError as e:
            raise HTTPException(409, str(e))
        return {"ok": True, "camera": console.camera_dict(cam)}

    @app.delete("/api/cameras/{cid}")
    def remove_camera(cid: int):
        cam = cam_or_404(cid)
        try:
            engine.remove_slot(cam)
        except ValueError as e:
            raise HTTPException(409, str(e))
        persist_sources()
        return {"ok": True}

    @app.post("/api/cameras/{cid}/source")
    def set_source(cid: int, body: dict):
        cam = cam_or_404(cid)
        src = str(body.get("src", "")).strip()
        name = str(body.get("name", "")).strip() or None
        try:
            area = max(0, int(body.get("area_capacity") or 0))
        except (ValueError, TypeError):
            raise HTTPException(400, "area_capacity must be an integer")
        if not src:
            raise HTTPException(400, "src (RTSP/HTTP URL or a video file path on the server) is required")
        if not is_url(src) and not Path(src).exists():
            raise HTTPException(400, f"video file not found on the server: {src}")
        if unencodable_password(src):
            # A `/`, `?` or `#` in the password ends the authority, so the URL is
            # genuinely ambiguous - `rtsp://a/b@c` is host `a` with path `/b@c` as
            # far as RFC 3986 is concerned. Everything else we encode silently.
            raise HTTPException(400,
                "the password appears to contain / ? or # - those cannot be worked out "
                "from the URL and must be percent-encoded by hand: / becomes %2F, "
                "? becomes %3F, # becomes %23. Other characters, including @ : $ % "
                "and spaces, can be pasted as they are.")
        if any(c is not cam and not c.source.is_null and c.source.src == src for c in engine.cameras):
            raise HTTPException(409, "that source is already assigned to another camera")
        if not cam.source.is_null and cam.source.src == src:
            # same feed: only the label / area capacity change - no reconnect
            cam.label, cam.area_capacity = name, area
            engine.stats.camera_names[cam.idx] = cam.name
        else:
            try:
                engine.replace_source(cam, src, name, area)
            except FileNotFoundError as e:
                raise HTTPException(400, str(e))
            if cam.alt_src == src:        # the pair would now point at itself
                cam.alt_src = None
        persist_sources()
        return {"ok": True, "camera": console.camera_dict(cam)}

    @app.post("/api/cameras/{cid}/swap")
    def swap_source(cid: int):
        """Switch this panel between its video file and its live feed."""
        cam = cam_or_404(cid)
        target = cam.alt_src
        if target and any(c is not cam and not c.source.is_null and c.source.src == target
                          for c in engine.cameras):
            raise HTTPException(409, "the paired feed is already assigned to another camera")
        try:
            engine.swap_source(cam)
        except ValueError as e:
            raise HTTPException(409, str(e))
        except FileNotFoundError as e:
            raise HTTPException(400, str(e))
        persist_sources()
        return {"ok": True, "camera": console.camera_dict(cam)}

    @app.delete("/api/cameras/{cid}/source")
    def clear_source(cid: int):
        cam = cam_or_404(cid)
        engine.replace_source(cam, None)
        persist_sources()
        return {"ok": True}

    # ---- trigger lines ----
    @app.post("/api/cameras/{cid}/editing")
    def editing(cid: int, body: dict):
        cam = cam_or_404(cid)
        cam.editing = bool(body.get("on", True))
        if cam.editing and not engine.setup_mode:
            engine.set_setup_mode(True)
        if not cam.editing and not any(c.editing for c in engine.cameras):
            engine.set_setup_mode(False)
        return {"ok": True, "editing": cam.editing, "setup_mode": engine.setup_mode}

    @app.post("/api/cameras/{cid}/line")
    def set_line(cid: int, body: dict):
        cam = cam_or_404(cid)
        try:
            p1 = (float(body["p1"][0]), float(body["p1"][1]))
            p2 = (float(body["p2"][0]), float(body["p2"][1]))
            side = int(body.get("entry_side", cam.setup_entry_side))
        except (KeyError, ValueError, TypeError, IndexError):
            raise HTTPException(400, "body must have p1, p2 (normalised [x, y]) and optional entry_side")
        if not (cam.source.width and cam.source.height):
            raise HTTPException(409, "camera resolution unknown - wait for the stream to connect")
        line = engine.set_line(cam, p1, p2, side)
        return {"ok": True, "line": {"p1": list(line.p1), "p2": list(line.p2), "entry_side": line.entry_side},
                "warning": edge_warning(line.p1, line.p2)}

    @app.post("/api/cameras/{cid}/region")
    def set_region(cid: int, body: dict):
        """Limit where this camera counts people. Body: x1, y1, x2, y2 normalised."""
        cam = cam_or_404(cid)
        try:
            role = str(body.get("role", ROLE_COUNT_INSIDE))
            if role not in ROLES:
                raise HTTPException(400, f"role must be one of {', '.join(ROLES)}")
            r = Region(float(body["x1"]), float(body["y1"]), float(body["x2"]), float(body["y2"]), role)
        except (KeyError, ValueError, TypeError):
            raise HTTPException(400, "body must have x1, y1, x2, y2 (normalised 0..1)")
        r = r.normalised()
        if (r.x2 - r.x1) < 0.05 or (r.y2 - r.y1) < 0.05:
            raise HTTPException(400, "region is too small to be useful")
        engine.set_region(cam, r)
        return {"ok": True, "region": [r.x1, r.y1, r.x2, r.y2], "role": r.role,
                "warning": region_warning(r, cam.line)}

    @app.delete("/api/cameras/{cid}/region")
    def clear_region(cid: int):
        cam = cam_or_404(cid)
        engine.set_region(cam, None)
        return {"ok": True}

    @app.post("/api/cameras/{cid}/flip")
    def flip(cid: int):
        cam = cam_or_404(cid)
        engine.flip_entry_side(cam, persist=True)
        return {"ok": True, "entry_side": cam.setup_entry_side}

    @app.delete("/api/cameras/{cid}/line")
    def delete_line(cid: int):
        cam = cam_or_404(cid)
        engine.delete_line(cam)
        return {"ok": True}

    # ---- recording ----
    @app.post("/api/cameras/{cid}/record")
    def record(cid: int, body: dict):
        cam = cam_or_404(cid)
        if cam.source.is_null or cam.display is None:
            raise HTTPException(409, "no frames from this camera yet")
        rec = console.recorder.record(cam, str(body.get("note", "")).strip(), kind="manual")
        return {"ok": True, "recording": rec}

    return app


def _log_kw(console: Console) -> dict:
    st = console.log_snapshot()
    return {"totals": st["totals"], "capacity": st["capacity"], "cameras": st["cameras"]}


def cv2_blank(width: int, label: str):
    import numpy as np

    img = np.full((int(width * 9 / 16), width, 3), 16, np.uint8)
    cv2.putText(img, label[:60], (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 140, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "NO SIGNAL", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1, cv2.LINE_AA)
    return img


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["live", "video"], default="video",
                   help="which source set of --sources-config to run (default: video)")
    p.add_argument("--sources", nargs="+", help="explicit feeds: RTSP URLs and/or video files; overrides --mode")
    p.add_argument("--sources-config", default=str(DEFAULT_SOURCES_CONFIG), help="JSON with the live / video source sets")
    p.add_argument("--site-config", default="config/site.json", help="building settings entered on the set-up screen")
    p.add_argument("--history", default="output/history.csv", help="hourly / closing / event history CSV")
    p.add_argument("--events-dir", default="output/events", help="recorded clips and snapshots")
    p.add_argument("--max-cameras", type=int, default=6, help="upper bound of camera panels")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8780)
    p.add_argument("--capacity", type=int, default=None, help="room capacity (overrides config/site.json)")
    p.add_argument("--stream-width", type=int, default=960, help="max width of the MJPEG camera streams")
    p.add_argument("--stream-fps", type=float, default=15.0, help="max frame rate of the MJPEG camera streams")
    p.add_argument("--jpeg-quality", type=int, default=80)
    p.add_argument("--clip-pre", type=float, default=10.0, help="seconds kept before a recorded event")
    p.add_argument("--clip-post", type=float, default=20.0, help="seconds recorded after a recorded event")
    add_engine_args(p)
    args = p.parse_args(argv)
    if args.sources:
        args.mode = "live" if any(is_url(s) for s in args.sources) else "video"
    else:
        args.sources = load_sources(args.mode, Path(args.sources_config), args.max_cameras)
    check_sources(p, [entry_src(e) for e in args.sources])
    args.min_slots = min(3, args.max_cameras)   # at least three panels at start-up (empty ones say "add camera")
    return args


def pair_alt_sources(engine, mode: str, path: Path) -> None:
    """Give each slot the other mode's feed at the same index as its pair.

    sources.json keeps "live" and "video" as parallel lists, so slot i's
    counterpart is simply the other list's entry i. Only slots that do not
    already carry an explicit "alt" are seeded, and a pair is never the feed the
    slot is already showing.
    """
    other = read_sources(path).get("video" if mode == "live" else "live") or []
    for cam in engine.cameras:
        if cam.alt_src or cam.source.is_null or cam.idx >= len(other):
            continue
        alt = entry_src(other[cam.idx]).strip()
        if alt and alt != cam.source.src:
            cam.alt_src = alt


def main(argv=None) -> None:
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args(argv)
    site = SiteConfig.load(args.site_config)
    if args.capacity:
        site.capacity = max(1, args.capacity)
    if args.initial_count is not None:       # an explicit --initial-count overrides the saved baseline
        site.baseline = max(0, args.initial_count)
    args.initial_count = site.baseline       # the engine starts from the site's baseline (0 unless set)
    engine = Engine(EngineConfig.from_args(args))
    pair_alt_sources(engine, args.mode, Path(args.sources_config))
    history = HistoryLog(args.history)
    frames = FrameStore(engine, args.stream_width, args.stream_fps, args.jpeg_quality,
                        keep_seconds=args.clip_pre + 2)
    console = Console(engine, args.mode, site, Path(args.sources_config), history, frames)
    console.recorder = Recorder(engine, frames, history, console.log_snapshot, args.events_dir, args.clip_pre, args.clip_post)
    scheduler = Scheduler(site, history, console.log_snapshot, engine.reset_counts)
    server_ref: list[uvicorn.Server] = []
    app = build_app(engine, console, frames, should_stop=lambda: bool(server_ref and server_ref[0].should_exit))
    server = uvicorn.Server(uvicorn.Config(app, host=args.host, port=args.port, log_level="warning",
                                           ws_ping_interval=20, timeout_graceful_shutdown=3))
    server_ref.append(server)
    engine.start()
    frames.start()
    scheduler.start()
    print(f"[web] {args.mode.upper()} mode - open http://localhost:{args.port}  (Ctrl+C to stop)")
    print(f"[web] site: {site.name}  hours {site.open}-{site.close}  capacity {site.capacity}  baseline {site.baseline}"
          f"  {'configured for ' + site.date if site.configured else 'NOT CONFIGURED - the page asks for the building hours first'}")
    try:
        server.run()
    finally:
        scheduler.stop()
        frames.stop()
        # releasing an RTSP source that is mid-reconnect can block inside FFmpeg; do not hang on Ctrl+C
        t = threading.Thread(target=engine.shutdown, daemon=True)
        t.start()
        t.join(timeout=8.0)
        if t.is_alive():
            print("[web] shutdown timed out - forcing exit")
            os._exit(0)


if __name__ == "__main__":
    main()
