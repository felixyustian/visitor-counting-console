"""Building / site settings entered on the web set-up screen (config/site.json).

Date, opening and closing time, capacity, warning threshold and the occupancy
baseline the count restarts from at opening time.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path

DEFAULT_SITE_PATH = Path("config") / "site.json"


def _parse_hhmm(s: str) -> time:
    h, m = str(s).strip().split(":")[:2]
    return time(int(h), int(m))


@dataclass
class SiteConfig:
    name: str = "Site"
    date: str = ""                 # YYYY-MM-DD of the operating day the settings were confirmed for
    open: str = "08:00"
    close: str = "22:00"
    capacity: int = 100
    warn_pct: int = 85             # amber "approaching capacity" threshold, % of capacity
    baseline: int = 0              # occupancy assumed at opening time
    path: Path = field(default=DEFAULT_SITE_PATH, repr=False, compare=False)

    # ------------------------------------------------------------ persistence
    @classmethod
    def load(cls, path: Path | str = DEFAULT_SITE_PATH) -> "SiteConfig":
        path = Path(path)
        cfg = cls(path=path)
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                data = {}
            for k in ("name", "date", "open", "close", "capacity", "warn_pct", "baseline"):
                if k in data:
                    setattr(cfg, k, data[k])
        cfg.capacity = max(1, int(cfg.capacity))
        cfg.warn_pct = min(100, max(1, int(cfg.warn_pct)))
        cfg.baseline = max(0, int(cfg.baseline))
        return cfg

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {k: v for k, v in asdict(self).items() if k != "path"}
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    def update(self, body: dict) -> list[str]:
        """Apply validated fields from the web form; returns the names that changed."""
        changed = []
        for k in ("name", "date", "open", "close"):
            if k in body and body[k] is not None:
                v = str(body[k]).strip()
                if k in ("open", "close"):
                    _parse_hhmm(v)                     # raises ValueError on bad input
                    v = _parse_hhmm(v).strftime("%H:%M")
                if k == "date" and v:
                    datetime.strptime(v, "%Y-%m-%d")
                if getattr(self, k) != v:
                    setattr(self, k, v); changed.append(k)
        for k, lo, hi in (("capacity", 1, 10 ** 6), ("warn_pct", 1, 100), ("baseline", 0, 10 ** 6)):
            if k in body and body[k] is not None:
                v = int(body[k])
                if not lo <= v <= hi:
                    raise ValueError(f"{k} must be between {lo} and {hi}")
                if getattr(self, k) != v:
                    setattr(self, k, v); changed.append(k)
        return changed

    # ---------------------------------------------------------------- times
    @property
    def configured(self) -> bool:
        return bool(self.date)

    def configured_for_today(self) -> bool:
        return self.date == date.today().isoformat()

    def open_at(self, day: date) -> datetime:
        return datetime.combine(day, _parse_hhmm(self.open))

    def close_at(self, day: date) -> datetime:
        c = datetime.combine(day, _parse_hhmm(self.close))
        if c <= self.open_at(day):          # closes after midnight
            c += timedelta(days=1)
        return c

    def is_open(self, now: datetime | None = None) -> bool:
        now = now or datetime.now()
        for day in (now.date() - timedelta(days=1), now.date()):
            if self.open_at(day) <= now < self.close_at(day):
                return True
        return False

    def public(self) -> dict:
        return {k: v for k, v in asdict(self).items() if k != "path"} | {
            "configured": self.configured,
            "configured_today": self.configured_for_today(),
            "is_open": self.is_open(),
        }
