"""Trigger-line crossing logic with per-track state and demographic voting.

A crossing is registered only when the *head point* of a tracked person moves
from one stable side of the line to the other. Robustness measures:

* Hysteresis dead-zone: a side is only "stable" when the head is more than
  `hysteresis_px` away from the line, so jitter on the line cannot toggle it.
* Segment extent: the head must cross within the drawn segment (with a small
  tolerance), not on its infinite extension.
* One event per transition per track ID, so a person standing on the line or
  re-detected in the same place is never counted twice.
* Duplicate suppression: a brand-new track ID that crosses in the same
  direction, at the same place, right after another track did is treated as a
  tracker ID switch and ignored.

Each track also accumulates classifier probabilities; the decision reported on
the crossing event is the average over all observations of that track.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import ClassVar, Optional

from .config import TriggerLine
from .detector import Track

UNKNOWN = "?"

# How a track's demographic vote is formed.
#
# Averaging every observation of a track is worse than it sounds: the classifier's
# read swings a lot across a track (median P(male) spread 0.49 over the evaluation
# set, and half the tracks cross the decision boundary at some point), because early
# frames catch people far away, side-on or half-occluded. What matters for gender is
# the view *at the line*, so gender votes over the most recent looks only.
#
# Age behaves the opposite way: "child" is a noisy per-frame call (a crouching adult, a
# person carrying a bag) that averages out over a whole track, and cutting age down to
# the recent window made it worse in a live run (96.4% -> 91.1%). So age keeps the mean
# over every observation. Measured on 56 hand-labelled crossings
# (scripts/ab_demographics.py votes), with the crossing-moment look included:
#
#   all observations for both      gender 92.7%   age 100.0%
#   recent window for both         gender 98.2%   age  98.2%
#   recent gender + all-obs age    gender 98.2%   age 100.0%   <- implemented
#
# Crop-size weighting was tried first and is not the win (89.1% gender).
RECENT_OBS = 3               # recent observations kept for the gender vote
CROSSING_WEIGHT = 2.0        # the look taken at the moment of crossing counts double


@dataclass
class TrackState:
    keep_history: ClassVar[bool] = False   # switch used by the evaluation tooling

    track_id: int
    first_seen: int
    last_seen: int
    stable_side: Optional[int] = None
    pending_side: Optional[int] = None   # side seen on the far side of the line, not yet confirmed
    pending_count: int = 0
    last_event_frame: int = -10_000
    last_head: Optional[tuple[float, float]] = None
    last_height: float = 0.0            # box height at last sighting, the scale for plausible movement
    # demographic vote (see the notes above): gender from the recent window, age from all of them
    n_obs: int = 0                       # total classifications of this track
    recent: deque = field(default_factory=lambda: deque(maxlen=RECENT_OBS + 1))
    sum_child_all: float = 0.0           # running total of P(child) over every observation
    last_classified: int = -10_000
    obs_log: Optional[list] = None       # full per-observation history, only when TrackState.keep_history is on
    # last emitted event (for exit-decrement bookkeeping and late revision)
    last_gender: str = UNKNOWN
    last_age: str = UNKNOWN
    last_direction: int = 0          # 0 = this track has never crossed
    revised: bool = False            # its matured vote has already been checked

    def add_observation(self, p_male: float, p_child: float, frame_idx: int, height: float = 0.0,
                        weight: float = 1.0) -> None:
        self.n_obs += 1
        self.recent.append((weight, p_male, p_child))
        self.sum_child_all += p_child
        self.last_classified = frame_idx
        if TrackState.keep_history:
            if self.obs_log is None:
                self.obs_log = []
            self.obs_log.append((frame_idx, round(p_male, 4), round(p_child, 4), round(height, 1)))

    @property
    def p_male(self) -> float:
        """Gender: weighted vote over the most recent looks (the ones taken near the line)."""
        if not self.recent:
            return 0.5
        tw = sum(o[0] for o in self.recent) or 1.0
        return sum(o[0] * o[1] for o in self.recent) / tw

    @property
    def p_child(self) -> float:
        """Age: mean over every observation - a noisy per-frame call that averages out."""
        return self.sum_child_all / self.n_obs if self.n_obs else 0.5

    def decide(self, gender_thresh: float = 0.5, child_thresh: float = 0.5) -> tuple[str, str]:
        if self.n_obs == 0:
            return UNKNOWN, UNKNOWN
        gender = "M" if self.p_male >= gender_thresh else "F"
        age = "C" if self.p_child >= child_thresh else "A"
        return gender, age


@dataclass
class CrossEvent:
    track_id: int
    direction: int         # +1 entry, -1 exit
    gender: str            # "M", "F" or "?"
    age: str               # "A", "C" or "?"
    head: tuple[float, float]
    frame_idx: int
    p_male: float
    p_child: float
    n_obs: int


@dataclass
class Revision:
    """A crossing's demographics re-decided once the track had been seen properly.

    The vote at the line is taken the moment the head crosses, when the person has
    just walked into frame and has only a handful of observations. Tracks keep being
    classified afterwards, so the estimate improves; when it improves enough to change
    the decision, the counters move from the bucket that was credited to the right one.
    """

    track_id: int
    direction: int         # the direction of the crossing being revised
    old_gender: str
    old_age: str
    gender: str
    age: str
    p_male: float
    p_child: float
    n_obs: int


@dataclass
class LineCounter:
    line: TriggerLine
    width: int
    height: int
    hysteresis_px: float = 6.0           # dead-zone half-width; recomputed from frame size in set_line
    confirm_frames: int = 3              # head must be on the new side this many consecutive frames
    segment_tolerance: float = 0.08      # fraction of segment length beyond the end points
    stale_frames: int = 90               # forget a track after this many frames unseen
    dup_window_frames: int = 20          # ID-switch suppression window
    dup_radius_px: float = 60.0
    new_track_frames: int = 6            # a track younger than this is "new" for dedup
    revise_frames: int = 90              # keep refining a crossing's vote for this long after it
    max_jump_bodies: float = 1.5         # a track re-found further than this many body heights away is
                                         # treated as a possible ID switch, not as the same person moving
    stitch_frames: int = 30              # a new ID can continue a person who vanished at most this long ago...
    stitch_bodies: float = 1.0           # ...within this many of their body heights of where they vanished...
    stitch_height_ratio: float = 1.33    # ...and at a similar size (so a child's history never passes to a parent)
    gender_thresh: float = 0.5
    child_thresh: float = 0.5
    states: dict[int, TrackState] = field(default_factory=dict)
    recent_events: list[CrossEvent] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.set_line(self.line)

    def set_line(self, line: TriggerLine) -> None:
        self.line = line
        (x1, y1), (x2, y2) = line.to_pixels(self.width, self.height)
        self.p1 = (float(x1), float(y1))
        self.p2 = (float(x2), float(y2))
        self.dx = self.p2[0] - self.p1[0]
        self.dy = self.p2[1] - self.p1[1]
        self.length = math.hypot(self.dx, self.dy) or 1.0
        self.hysteresis_px = max(4.0, 0.01 * max(self.width, self.height))
        self.states.clear()

    def configure_timing(self, fps: float) -> None:
        """Scale frame-based thresholds to the source frame rate (defaults assume ~30 fps)."""
        fps = fps if fps and fps > 1 else 30.0
        self.confirm_frames = max(2, int(round(0.1 * fps)))
        self.stale_frames = max(30, int(round(3.0 * fps)))
        self.dup_window_frames = max(8, int(round(0.7 * fps)))
        self.new_track_frames = max(3, int(round(0.2 * fps)))
        self.revise_frames = max(30, int(round(3.0 * fps)))

    # ------------------------------------------------------------- geometry
    def signed_distance(self, pt: tuple[float, float]) -> float:
        # positive on the "left" of p1->p2 (in image coords with y down this is visually the right side)
        return (self.dx * (pt[1] - self.p1[1]) - self.dy * (pt[0] - self.p1[0])) / self.length

    def projection(self, pt: tuple[float, float]) -> float:
        return ((pt[0] - self.p1[0]) * self.dx + (pt[1] - self.p1[1]) * self.dy) / (self.length ** 2)

    def side(self, pt: tuple[float, float]) -> int:
        d = self.signed_distance(pt)
        if abs(d) < self.hysteresis_px:
            return 0
        return 1 if d > 0 else -1

    # --------------------------------------------------------------- update
    def update(self, tracks: list[Track], frame_idx: int) -> list[CrossEvent]:
        events: list[CrossEvent] = []
        seen: set[int] = set()

        self._stitch_new_ids(tracks, frame_idx)
        for t in tracks:
            seen.add(t.track_id)
            st = self.states.get(t.track_id)
            if st is None:
                st = TrackState(t.track_id, first_seen=frame_idx, last_seen=frame_idx)
                self.states[t.track_id] = st
            if self._implausible_reappearance(st, t, frame_idx):
                # The tracker kept this ID alive while the person was hidden and has now re-found it
                # somewhere they could not have walked to - most likely on a different person. Do not
                # let that jump register as a crossing: re-seed the side from the new position.
                st.stable_side = None
                st.pending_side, st.pending_count = None, 0
            st.last_seen = frame_idx

            s = self.side(t.head)
            if st.stable_side is None:
                # Seed the side from the *raw* sign, even inside the dead-zone. A person is
                # often only picked up as they step through the door, i.e. straddling the
                # line; discarding those frames would seed the side once they are already
                # inside and their entry would never be counted at all.
                if s != 0:
                    st.stable_side = s
                else:
                    d = self.signed_distance(t.head)
                    if d != 0.0:
                        st.stable_side = 1 if d > 0 else -1
            elif s != 0:
                if s != st.stable_side:
                    # on the far side: confirm over several frames before committing
                    if st.pending_side == s:
                        st.pending_count += 1
                    else:
                        st.pending_side, st.pending_count = s, 1
                    if st.pending_count >= self.confirm_frames:
                        ev = self._commit_crossing(st, t, s, frame_idx)
                        if ev is not None:
                            events.append(ev)
                        st.stable_side = s
                        st.pending_side, st.pending_count = None, 0
                else:
                    st.pending_side, st.pending_count = None, 0  # came back: cancel pending
            st.last_head = t.head
            st.last_height = float(t.box[3] - t.box[1])

        # forget stale tracks
        stale = [tid for tid, st in self.states.items() if frame_idx - st.last_seen > self.stale_frames]
        for tid in stale:
            del self.states[tid]
        self.recent_events = [e for e in self.recent_events if frame_idx - e.frame_idx <= self.dup_window_frames]
        return events

    def _commit_crossing(self, st: TrackState, t: Track, new_side: int, frame_idx: int) -> Optional[CrossEvent]:
        # must have crossed within the drawn segment, not on its infinite extension
        u = self.projection(t.head)
        tol = self.segment_tolerance
        if not (-tol <= u <= 1 + tol):
            return None
        # NB: a quick re-crossing is *not* suppressed here - dropping it would silently flip the
        # stable side and leave the entry/exit counters inconsistent. Jitter is handled by the
        # dead-zone and the confirm_frames debounce instead.
        direction = 1 if new_side == self.line.entry_side else -1
        gender, age = st.decide(self.gender_thresh, self.child_thresh)
        ev = CrossEvent(t.track_id, direction, gender, age, t.head, frame_idx, st.p_male, st.p_child, st.n_obs)
        if self._is_duplicate(ev, st):
            return None
        self.recent_events.append(ev)
        st.last_event_frame = frame_idx
        st.last_gender, st.last_age = gender, age
        st.last_direction, st.revised = direction, False
        return ev

    def _stitch_new_ids(self, tracks: list[Track], frame_idx: int) -> None:
        """Let a new track ID carry on the state of a person who just vanished at the same spot.

        When someone is blocked (a child walking behind a parent) the tracker can lose them and
        give them a fresh ID when they reappear. Without this, the fresh ID has no side of the line
        (a crossing made while hidden is never counted) and no demographic history (a crossing
        already counted can no longer be revised). The new ID inherits that state only when it
        appears soon after, close to where the person vanished, and at a similar size.
        """
        present = {t.track_id for t in tracks}
        vanished = [st for tid, st in self.states.items()
                    if tid not in present and st.last_head is not None
                    and 1 <= frame_idx - st.last_seen <= self.stitch_frames]
        if not vanished:
            return
        pairs = []
        for t in tracks:
            if t.track_id in self.states:
                continue
            h_new = float(t.box[3] - t.box[1])
            for st in vanished:
                body = max(st.last_height, 1.0)
                ratio = max(h_new, body) / max(min(h_new, body), 1.0)
                dist = math.dist(t.head, st.last_head)
                if dist <= self.stitch_bodies * body and ratio <= self.stitch_height_ratio:
                    pairs.append((dist / body, t, st))
        used_t, used_st = set(), set()
        for _, t, st in sorted(pairs, key=lambda x: x[0]):          # closest first, each side used once
            if t.track_id in used_t or st.track_id in used_st:
                continue
            used_t.add(t.track_id); used_st.add(st.track_id)
            del self.states[st.track_id]
            st.track_id = t.track_id
            self.states[t.track_id] = st

    def _implausible_reappearance(self, st: TrackState, t: Track, frame_idx: int) -> bool:
        """True when a track returns after a gap further away than the same person could plausibly be.

        A child hidden behind a parent for a moment reappears less than a body height from where
        they were last seen, so their crossing still counts. A lost track whose predicted box drifted
        onto a stranger reappears several body heights away; counting that would be a false crossing.
        """
        if st.last_head is None or frame_idx - st.last_seen <= 1:
            return False
        body = max(st.last_height, float(t.box[3] - t.box[1]), 1.0)
        return math.dist(t.head, st.last_head) > self.max_jump_bodies * body

    def collect_revisions(self, frame_idx: int) -> list[Revision]:
        """Re-decide crossings whose track has now been observed long enough.

        Checked once per crossing, `revise_frames` after it, so a track cannot
        oscillate the counters. Returns only the decisions that actually changed.
        """
        out: list[Revision] = []
        for st in self.states.values():
            if st.last_direction == 0 or st.revised:
                continue
            if frame_idx - st.last_event_frame < self.revise_frames:
                continue
            st.revised = True
            gender, age = st.decide(self.gender_thresh, self.child_thresh)
            if gender == st.last_gender and age == st.last_age:
                continue
            out.append(Revision(st.track_id, st.last_direction, st.last_gender, st.last_age,
                                gender, age, st.p_male, st.p_child, st.n_obs))
            st.last_gender, st.last_age = gender, age
        return out

    def _is_duplicate(self, ev: CrossEvent, st: TrackState) -> bool:
        """Heuristic for tracker ID switches mid-crossing."""
        if ev.frame_idx - st.first_seen > self.new_track_frames:
            return False
        for old in self.recent_events:
            if old.track_id == ev.track_id or old.direction != ev.direction:
                continue
            if math.dist(old.head, ev.head) <= self.dup_radius_px:
                return True
        return False

    def get_state(self, track_id: int) -> Optional[TrackState]:
        return self.states.get(track_id)

    def ensure_state(self, track_id: int, frame_idx: int) -> TrackState:
        """Return the state for a track, creating it (without side info) if needed."""
        st = self.states.get(track_id)
        if st is None:
            st = TrackState(track_id, first_seen=frame_idx, last_seen=frame_idx)
            self.states[track_id] = st
        return st

    def reset_tracks(self) -> None:
        self.states.clear()
        self.recent_events.clear()
