"""Persistent per-source trigger-line configuration.

Lines are stored in normalised (0..1) image coordinates so they survive a
change of stream resolution. Each entry is keyed by the source identity: the
RTSP/HTTP URL as-is, or the normalised absolute path for a video file.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

DEFAULT_CONFIG_PATH = Path("config") / "lines.json"


def template_of(path: Path | str) -> Path:
    """The committed `<name>.example` that seeds a missing site-specific file."""
    return Path(str(path) + ".example")


def is_url(src: str) -> bool:
    s = src.lower()
    return s.startswith(("rtsp://", "rtsps://", "http://", "https://", "rtmp://", "udp://", "tcp://"))


def source_key(src: str) -> str:
    """Stable identity for a source used as the config key."""
    if is_url(src):
        return src.strip()
    p = Path(src).expanduser()
    try:
        p = p.resolve()
    except OSError:
        p = p.absolute()
    return os.path.normcase(str(p)).replace("\\", "/")


def source_name(src: str) -> str:
    """Short human-readable label for a source."""
    if is_url(src):
        # rtsp://user:pass@host:554/stream -> host:554/stream
        s = src.split("://", 1)[-1]
        s = s.split("@")[-1]
        return s[:40]
    return Path(src).stem


@dataclass
class TriggerLine:
    """A trigger line in normalised coordinates.

    entry_side: which side (+1 or -1, sign of the 2D cross product of
    (p2-p1) x (head-p1)) is considered *inside* the centre. A head moving from
    the other side onto this side is an entry, the reverse is an exit.
    """

    p1: tuple[float, float]
    p2: tuple[float, float]
    entry_side: int = 1
    name: str = ""

    def to_pixels(self, w: int, h: int) -> tuple[tuple[int, int], tuple[int, int]]:
        return (
            (int(round(self.p1[0] * w)), int(round(self.p1[1] * h))),
            (int(round(self.p2[0] * w)), int(round(self.p2[1] * h))),
        )

    @staticmethod
    def from_pixels(
        p1: tuple[int, int], p2: tuple[int, int], w: int, h: int, entry_side: int = 1, name: str = ""
    ) -> "TriggerLine":
        return TriggerLine(
            p1=(p1[0] / w, p1[1] / h),
            p2=(p2[0] / w, p2[1] / h),
            entry_side=1 if entry_side >= 0 else -1,
            name=name,
        )


# What a region does. The trigger line must sit inside a COUNT_INSIDE region -
# a person approaching from outside it is not tracked until they enter, so they
# are first seen already at the line. The other two roles leave tracking alone,
# so the line may be anywhere.
ROLE_COUNT_INSIDE = "count_inside"       # only people inside are tracked at all
ROLE_IGNORE_INSIDE = "ignore_inside"     # people inside are ignored (a window, a mirror, a poster)
ROLE_CLASSIFY_INSIDE = "classify_inside" # everyone is counted; only those inside are classified
ROLES = (ROLE_COUNT_INSIDE, ROLE_IGNORE_INSIDE, ROLE_CLASSIFY_INSIDE)


@dataclass
class Region:
    """A rectangle in normalised coordinates, and what it is for (see ROLES)."""

    x1: float
    y1: float
    x2: float
    y2: float
    role: str = ROLE_COUNT_INSIDE

    @property
    def affects_tracking(self) -> bool:
        """True when the region decides who is tracked, rather than only who is classified."""
        return self.role in (ROLE_COUNT_INSIDE, ROLE_IGNORE_INSIDE)

    def keeps(self, pt: tuple[float, float], w: int, h: int) -> bool:
        """Whether a track at `pt` survives this region, for tracking roles."""
        inside = self.contains(pt, w, h)
        return inside if self.role == ROLE_COUNT_INSIDE else not inside

    def normalised(self) -> "Region":
        """Corners in any order in, top-left/bottom-right out."""
        return Region(min(self.x1, self.x2), min(self.y1, self.y2),
                      max(self.x1, self.x2), max(self.y1, self.y2), self.role)

    def contains(self, pt: tuple[float, float], w: int, h: int) -> bool:
        """`pt` is in pixels, as the tracker reports it."""
        r = self.normalised()
        return (r.x1 * w) <= pt[0] <= (r.x2 * w) and (r.y1 * h) <= pt[1] <= (r.y2 * h)

    def to_pixels(self, w: int, h: int) -> tuple[tuple[int, int], tuple[int, int]]:
        r = self.normalised()
        return ((int(round(r.x1 * w)), int(round(r.y1 * h))),
                (int(round(r.x2 * w)), int(round(r.y2 * h))))


@dataclass
class LineConfigStore:
    path: Path = field(default=DEFAULT_CONFIG_PATH)
    lines: dict[str, TriggerLine] = field(default_factory=dict)
    regions: dict[str, Region] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | str = DEFAULT_CONFIG_PATH) -> "LineConfigStore":
        path = Path(path)
        store = cls(path=path)
        # lines.json is site-specific and gitignored; a fresh clone has only the
        # committed template. Read that instead, but keep saving to the real path.
        src = path if path.exists() else template_of(path)
        if src.exists():
            try:
                data = json.loads(src.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                raise SystemExit(f"Config file {src} is not valid JSON: {e}")
            for key, item in data.get("sources", {}).items():
                store.lines[key] = TriggerLine(
                    p1=tuple(item["p1"]),
                    p2=tuple(item["p2"]),
                    entry_side=int(item.get("entry_side", 1)),
                    name=item.get("name", ""),
                )
                r = item.get("region")
                if r:
                    role = item.get("region_role", ROLE_COUNT_INSIDE)
                    store.regions[key] = Region(float(r[0]), float(r[1]), float(r[2]), float(r[3]),
                                                role if role in ROLES else ROLE_COUNT_INSIDE)
        return store

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "coordinates": "normalised 0..1 (x/width, y/height)",
            "sources": {k: {**asdict(v), **({"region": [r.x1, r.y1, r.x2, r.y2],
                                             "region_role": r.role}
                                            if (r := self.regions.get(k)) else {})}
                        for k, v in self.lines.items()},
        }
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    def get(self, src: str) -> Optional[TriggerLine]:
        key = source_key(src)
        if key in self.lines:
            return self.lines[key]
        # Fallback: match on file name so a moved repo still finds its line.
        if not is_url(src):
            base = Path(src).name.lower()
            for k, v in self.lines.items():
                if k.rsplit("/", 1)[-1] == base:
                    return v
        return None

    def set(self, src: str, line: TriggerLine) -> None:
        line.name = line.name or source_name(src)
        self.lines[source_key(src)] = line

    def remove(self, src: str) -> None:
        self.lines.pop(source_key(src), None)
        self.regions.pop(source_key(src), None)

    def get_region(self, src: str) -> Optional[Region]:
        key = source_key(src)
        if key in self.regions:
            return self.regions[key]
        if not is_url(src):                      # same file-name fallback as get()
            base = Path(src).name.lower()
            for k, v in self.regions.items():
                if k.rsplit("/", 1)[-1] == base:
                    return v
        return None

    def set_region(self, src: str, region: Optional[Region]) -> None:
        key = source_key(src)
        if region is None:
            self.regions.pop(key, None)
        else:
            self.regions[key] = region.normalised()
