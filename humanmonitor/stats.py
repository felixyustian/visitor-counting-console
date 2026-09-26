"""Aggregated visitor statistics and CSV event log."""
from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .counter import CrossEvent, Revision, UNKNOWN


# Assumed make-up of the people already inside when the demo starts. The head count
# is unknown in practice (nobody counted them in), so the baseline is an explicit
# assumption rather than a measurement - see --initial-count.
INITIAL_MALE_FRACTION = 0.5    # male : female = 50 : 50
INITIAL_ADULT_FRACTION = 0.8   # adult : child = 80 : 20


@dataclass
class Counts:
    entries: int = 0
    exits: int = 0
    male: int = 0
    female: int = 0
    adult: int = 0
    child: int = 0
    gender_unknown: int = 0
    age_unknown: int = 0
    initial: int = 0           # occupancy assumed at start-up, before any crossing
    # cumulative visitors (entries) by demographic - totals for the day, never decremented by exits
    v_male: int = 0
    v_female: int = 0
    v_adult: int = 0
    v_child: int = 0

    @property
    def visitors(self) -> int:
        """Net visitors currently inside (assumed start-up occupancy + entries - exits)."""
        return self.initial + self.entries - self.exits

    def seed(self, initial: int) -> None:
        """Assume `initial` people are already inside, split by the fractions above.

        The splits are rounded so that male + female and adult + child each add up
        to `initial` exactly, however the fractions divide.
        """
        self.initial = initial
        self.male = round(initial * INITIAL_MALE_FRACTION)
        self.female = initial - self.male
        self.adult = round(initial * INITIAL_ADULT_FRACTION)
        self.child = initial - self.adult

    def apply(self, ev: CrossEvent) -> None:
        d = ev.direction
        if d > 0:
            self.entries += 1
            self._visitor(ev.gender, ev.age, +1)
        else:
            self.exits += 1
        if ev.gender == "M":
            self.male += d
        elif ev.gender == "F":
            self.female += d
        else:
            self.gender_unknown += d
        if ev.age == "A":
            self.adult += d
        elif ev.age == "C":
            self.child += d
        else:
            self.age_unknown += d
        # counters never go below zero (an exit of someone who entered before start-up)
        self._clamp()

    def _visitor(self, gender: str, age: str, sign: int) -> None:
        if gender == "M":
            self.v_male += sign
        elif gender == "F":
            self.v_female += sign
        if age == "A":
            self.v_adult += sign
        elif age == "C":
            self.v_child += sign

    @property
    def visitors_total(self) -> int:
        """Visitors who entered through the counted entrances (cumulative entries)."""
        return self.entries

    def revise(self, rev: Revision) -> None:
        """Move a crossing already counted from its old bucket to the re-decided one."""
        d = rev.direction
        if d > 0:
            self._visitor(rev.old_gender, rev.old_age, -1)
            self._visitor(rev.gender, rev.age, +1)
        for label, sign in ((rev.old_gender, -d), (rev.gender, d)):
            if label == "M":
                self.male += sign
            elif label == "F":
                self.female += sign
            else:
                self.gender_unknown += sign
        for label, sign in ((rev.old_age, -d), (rev.age, d)):
            if label == "A":
                self.adult += sign
            elif label == "C":
                self.child += sign
            else:
                self.age_unknown += sign
        self._clamp()

    def _split_to(self, n: int) -> tuple[int, int, int, int]:
        """Share out `n` people using the measured gender / age mix (see inside_split)."""
        if n <= 0:
            return 0, 0, 0, 0
        g, a = self.male + self.female, self.adult + self.child
        male = round(n * self.male / g) if g else (n + 1) // 2
        adult = round(n * self.adult / a) if a else n
        return male, n - male, adult, n - adult

    def net_split(self) -> tuple[int, int, int, int]:
        """Gender / age of `entries - exits`, summing to it. Everything here was detected:
        unlike inside_split it carries none of the assumed opening count."""
        return self._split_to(self.entries - self.exits)

    def inside_split(self) -> tuple[int, int, int, int]:
        """(male, female, adult, child) of the people currently inside, summing to `visitors`.

        The raw buckets are incremented on entry and decremented on exit independently, so they
        drift away from the occupancy whenever a person is classified one way walking in and the
        other way walking out - and because they are clamped at zero, the error only accumulates
        upward. A panel that shows 0 inside next to "male 8, female 2" is simply wrong.

        Nothing links a person's exit back to their entry (they are separate tracks), so the
        honest reading is a proportion: apply the measured mix of the people inside to the
        occupancy. The parts then always add up to the headline number.
        """
        return self._split_to(max(0, self.visitors))

    def _clamp(self) -> None:
        for k in ("male", "female", "adult", "child", "gender_unknown", "age_unknown", "v_male", "v_female", "v_adult", "v_child"):
            if getattr(self, k) < 0:
                setattr(self, k, 0)

    def reset(self) -> None:
        initial = self.initial
        for k in self.__dataclass_fields__:
            setattr(self, k, 0)
        self.seed(initial)   # R restarts the count from the baseline, not from zero


@dataclass
class Stats:
    camera_names: list[str]
    initial_count: int = 0     # people assumed inside at start-up (site-wide, not per camera)
    total: Counts = field(default_factory=Counts)
    per_camera: list[Counts] = field(init=False)
    started_at: float = field(default_factory=time.time)
    last_event: Optional[tuple[float, int, CrossEvent]] = None

    def __post_init__(self) -> None:
        self.per_camera = [Counts() for _ in self.camera_names]
        # Only the site total carries the baseline: a per-camera row counts the flow
        # through that one entrance, so seeding each of them would multiply the
        # assumed occupancy by the number of cameras.
        self.total.seed(self.initial_count)

    def apply(self, cam_idx: int, ev: CrossEvent) -> None:
        self.total.apply(ev)
        self.per_camera[cam_idx].apply(ev)
        self.last_event = (time.time(), cam_idx, ev)

    def revise(self, cam_idx: int, rev: Revision) -> None:
        self.total.revise(rev)
        self.per_camera[cam_idx].revise(rev)

    def add_camera(self, name: str) -> None:
        self.camera_names.append(name)
        self.per_camera.append(Counts())

    def remove_camera(self, cam_idx: int) -> None:
        del self.camera_names[cam_idx]
        del self.per_camera[cam_idx]

    def set_occupancy(self, n: int) -> int:
        """Manual correction (e.g. after a head count): move the baseline so visitors == n.

        The demographic buckets absorb the difference in the assumed split, the same
        way the start-up baseline is split. Returns the applied delta.
        """
        n = max(0, int(n))
        delta = n - self.total.visitors
        self.total.initial += delta       # running offset only; reset() still restores initial_count
        m = round(delta * INITIAL_MALE_FRACTION)
        a = round(delta * INITIAL_ADULT_FRACTION)
        self.total.male += m
        self.total.female += delta - m
        self.total.adult += a
        self.total.child += delta - a
        self.total._clamp()
        return delta

    def set_baseline(self, n: int) -> None:
        """Change the baseline that reset() restores (the running count is untouched)."""
        self.initial_count = max(0, int(n))

    def reset(self) -> None:
        self.total.initial = self.initial_count   # restart from the configured baseline, not a corrected offset
        self.total.reset()
        for c in self.per_camera:
            c.reset()
        self.started_at = time.time()
        self.last_event = None


class TrackLog:
    """Per-track diagnostic CSV: what the app currently believes about everyone on screen.

    Only crossings reach events.csv, so a track pointed at on screen ("#3042 is a child")
    could not be checked afterwards. This records every track periodically, keyed by the
    id drawn on its box.
    """

    HEADER = ["timestamp", "frame", "camera_id", "camera_name", "track_id", "x1", "y1", "x2", "y2",
              "det_conf", "n_obs", "p_male", "p_child", "gender", "age"]

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.path.exists() or self.path.stat().st_size == 0
        self._fh = self.path.open("a", newline="", encoding="utf-8")
        self._w = csv.writer(self._fh)
        if new:
            self._w.writerow(self.HEADER)
            self._fh.flush()

    def write(self, cam_idx: int, cam_name: str, frame_idx: int, tracks, counter) -> None:
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        for t in tracks:
            st = counter.get_state(t.track_id)
            g, a = st.decide(counter.gender_thresh, counter.child_thresh) if st else (UNKNOWN, UNKNOWN)
            x1, y1, x2, y2 = t.box
            self._w.writerow([ts, frame_idx, cam_idx, cam_name, t.track_id, x1, y1, x2, y2,
                              f"{t.conf:.3f}", st.n_obs if st else 0,
                              f"{st.p_male:.3f}" if st else "", f"{st.p_child:.3f}" if st else "",
                              {"M": "male", "F": "female"}.get(g, "unknown"),
                              {"A": "adult", "C": "child"}.get(a, "unknown")])
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


class EventLog:
    """Append-only CSV of crossing events (the 'visitor analytics dataset').

    `event` is "crossing" for the row written when the head crosses the line, and
    "revision" for a later row carrying the same crossing's re-decided demographics
    once the track had been observed long enough. A revision supersedes the earlier
    crossing row for that (camera_id, track_id, direction); the entry/exit itself is
    never revised, so counting entries means counting event == "crossing".
    """

    HEADER = ["timestamp", "camera_id", "camera_name", "source", "track_id", "direction",
              "gender", "age", "p_male", "p_child", "n_obs", "event"]

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._rotate_if_stale()
        new = not self.path.exists() or self.path.stat().st_size == 0
        self._fh = self.path.open("a", newline="", encoding="utf-8")
        self._w = csv.writer(self._fh)
        if new:
            self._w.writerow(self.HEADER)
            self._fh.flush()

    def _rotate_if_stale(self) -> None:
        """Move an existing log aside if it was written with a different set of columns."""
        if not self.path.exists() or self.path.stat().st_size == 0:
            return
        try:
            with self.path.open("r", newline="", encoding="utf-8") as fh:
                header = next(csv.reader(fh), None)
        except OSError:
            return
        if header == self.HEADER:
            return
        backup = self.path.with_suffix(f".{time.strftime('%Y%m%d-%H%M%S')}.csv")
        self.path.replace(backup)
        print(f"[events] {self.path} used older columns - moved to {backup.name}")

    def _row(self, cam_idx: int, cam_name: str, source: str, track_id: int, direction: int,
             gender: str, age: str, p_male: float, p_child: float, n_obs: int, kind: str) -> None:
        self._w.writerow([
            time.strftime("%Y-%m-%dT%H:%M:%S"),
            cam_idx,
            cam_name,
            source,
            track_id,
            "entry" if direction > 0 else "exit",
            {"M": "male", "F": "female"}.get(gender, "unknown"),
            {"A": "adult", "C": "child"}.get(age, "unknown"),
            f"{p_male:.3f}",
            f"{p_child:.3f}",
            n_obs,
            kind,
        ])
        self._fh.flush()

    def write(self, cam_idx: int, cam_name: str, source: str, ev: CrossEvent) -> None:
        self._row(cam_idx, cam_name, source, ev.track_id, ev.direction,
                  ev.gender, ev.age, ev.p_male, ev.p_child, ev.n_obs, "crossing")

    def write_revision(self, cam_idx: int, cam_name: str, source: str, rev: Revision) -> None:
        self._row(cam_idx, cam_name, source, rev.track_id, rev.direction,
                  rev.gender, rev.age, rev.p_male, rev.p_child, rev.n_obs, "revision")

    def close(self) -> None:
        self._fh.close()
