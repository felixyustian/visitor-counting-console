"""Window layout (2x2 grid of separate OpenCV windows) and camera-frame overlays."""
from __future__ import annotations

import math
import platform
from dataclasses import dataclass

import cv2
import numpy as np

from .config import ROLE_CLASSIFY_INSIDE, ROLE_IGNORE_INSIDE, Region, TriggerLine
from .counter import LineCounter, TrackState, UNKNOWN
from .detector import Track
from . import dashboard as D


def screen_size() -> tuple[int, int]:
    if platform.system() == "Windows":
        try:
            import ctypes

            u = ctypes.windll.user32
            u.SetProcessDPIAware()
            return int(u.GetSystemMetrics(0)), int(u.GetSystemMetrics(1))
        except Exception:
            pass
    if platform.system() == "Darwin":
        # tkinter and OpenCV's Cocoa windows do not coexist in one process (Tcl crashes,
        # corrupted torch calls), so ask the Finder for the desktop size instead.
        try:
            import subprocess

            out = subprocess.run(["osascript", "-e", 'tell application "Finder" to get bounds of window of desktop'],
                                 capture_output=True, text=True, timeout=5).stdout
            x0, y0, x1, y1 = (int(v) for v in out.strip().split(", "))
            return x1 - x0, y1 - y0
        except Exception:
            return 1440, 900
    try:
        import tkinter as tk

        r = tk.Tk()
        r.withdraw()
        w, h = r.winfo_screenwidth(), r.winfo_screenheight()
        r.destroy()
        return w, h
    except Exception:
        return 1920, 1080


@dataclass
class Cell:
    x: int
    y: int
    w: int
    h: int


class WindowGrid:
    """Positions four WINDOW_NORMAL windows in a 2x2 grid on the primary screen."""

    TITLE_BAR = 38   # approx. window chrome height (px)
    BORDER = 8
    TASKBAR = 48

    def __init__(self, names: list[str], screen: tuple[int, int] | None = None):
        assert len(names) <= 4
        sw, sh = screen or screen_size()
        cw = sw // 2
        ch = (sh - self.TASKBAR) // 2
        self.cells: list[Cell] = []
        for i, name in enumerate(names):
            col, row = i % 2, i // 2
            cell = Cell(col * cw + self.BORDER // 2, row * ch, cw - self.BORDER, ch - self.TITLE_BAR - self.BORDER // 2)
            self.cells.append(cell)
            cv2.namedWindow(name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow(name, cell.w, cell.h)
            cv2.moveWindow(name, cell.x, cell.y)
        self.names = names

    def size(self, i: int) -> tuple[int, int]:
        return self.cells[i].w, self.cells[i].h


def letterbox(img: np.ndarray, w: int, h: int) -> tuple[np.ndarray, float, int, int]:
    """Fit img into (w, h) keeping aspect ratio. Returns canvas, scale, x-offset, y-offset."""
    ih, iw = img.shape[:2]
    scale = min(w / iw, h / ih)
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((h, w, 3), 16, dtype=np.uint8)
    ox, oy = (w - nw) // 2, (h - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    return canvas, scale, ox, oy


# ------------------------------------------------------------------ overlays
def _gender_color(g: str):
    return {"M": D.MALE, "F": D.FEMALE}.get(g, (200, 200, 200))


def draw_entry_arrow(
    img: np.ndarray,
    p1: tuple[int, int],
    p2: tuple[int, int],
    entry_side: int,
    color=(60, 220, 60),
) -> tuple[int, int]:
    """Filled triangle on the line's midpoint normal, apex pointing at the IN side.

    The normal (-dy, dx) is the +1 side of the cross product used by
    LineCounter.signed_distance, so the apex always points the way a person
    must walk to be counted as an entry. Returns the apex, for label placement.
    """
    h, w = img.shape[:2]
    mx, my = (p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0
    dx, dy = float(p2[0] - p1[0]), float(p2[1] - p1[1])
    n = (dx * dx + dy * dy) ** 0.5 or 1.0
    sgn = 1.0 if entry_side >= 0 else -1.0
    nx, ny = sgn * -dy / n, sgn * dx / n          # unit normal toward the entry side
    tx, ty = -ny, nx                             # unit tangent, for the triangle base

    L = max(30.0, min(w, h) * 0.085)             # stem + head, from the line to the apex
    head = max(12.0, L * 0.5)                    # triangle height
    half = max(7.0, head * 0.55)                 # half the base width
    bx, by = mx + nx * (L - head), my + ny * (L - head)
    tip = (int(round(mx + nx * L)), int(round(my + ny * L)))
    tri = np.array(
        [
            [tip[0], tip[1]],
            [int(round(bx + tx * half)), int(round(by + ty * half))],
            [int(round(bx - tx * half)), int(round(by - ty * half))],
        ],
        dtype=np.int32,
    )
    # stem, then the head - each outlined in black first so it reads on any background
    cv2.line(img, (int(mx), int(my)), (int(round(bx)), int(round(by))), (0, 0, 0), 5, cv2.LINE_AA)
    cv2.line(img, (int(mx), int(my)), (int(round(bx)), int(round(by))), color, 2, cv2.LINE_AA)
    cv2.drawContours(img, [tri], 0, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.fillConvexPoly(img, tri, color, cv2.LINE_AA)
    return tip


def _dashed(img: np.ndarray, a, b, color, thickness: int, dash: int = 9, gap: int = 7) -> None:
    """A dashed segment; cv2 has no dash support."""
    ax, ay = a
    bx, by = b
    length = math.hypot(bx - ax, by - ay)
    if length < 1:
        return
    ux, uy = (bx - ax) / length, (by - ay) / length
    pos = 0.0
    while pos < length:
        end = min(pos + dash, length)
        cv2.line(img,
                 (int(round(ax + ux * pos)), int(round(ay + uy * pos))),
                 (int(round(ax + ux * end)), int(round(ay + uy * end))),
                 color, thickness, cv2.LINE_AA)
        pos = end + gap


def draw_line(img: np.ndarray, line: TriggerLine, color=(60, 220, 60), label: str = "TRIGGER LINE", thickness: int = 3):
    h, w = img.shape[:2]
    p1, p2 = line.to_pixels(w, h)
    # A crossing still counts a little past each end (LineCounter.segment_tolerance),
    # so draw that reach rather than leave it to surprise whoever reads the counts.
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    seg = math.hypot(dx, dy)
    if seg >= 1:
        ex, ey = dx / seg * seg * LineCounter.segment_tolerance, dy / seg * seg * LineCounter.segment_tolerance
        faint = tuple(int(c * 0.55) for c in color)
        for a, b in (((p1[0] - ex, p1[1] - ey), p1), (p2, (p2[0] + ex, p2[1] + ey))):
            _dashed(img, a, b, faint, max(1, thickness - 1))
        # a tick square across each reach end, so the limit is unambiguous
        tx, ty = -dy / seg * 6, dx / seg * 6
        for cx, cy in ((p1[0] - ex, p1[1] - ey), (p2[0] + ex, p2[1] + ey)):
            cv2.line(img, (int(round(cx - tx)), int(round(cy - ty))),
                          (int(round(cx + tx)), int(round(cy + ty))), faint, max(1, thickness - 1), cv2.LINE_AA)
    cv2.line(img, p1, p2, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)
    for p in (p1, p2):
        cv2.circle(img, p, 6, color, -1, cv2.LINE_AA)
    # entry-direction marker: triangle perpendicular to the line, apex on the IN side
    tip = draw_entry_arrow(img, p1, p2, line.entry_side, color)
    s = max(0.6, min(w, h) / 900.0)              # the overlay is letterboxed down for display
    D.text(img, "IN", (tip[0] + int(10 * s), tip[1] + int(8 * s)), s, (0, 0, 0), int(4 * s) + 1)
    D.text(img, "IN", (tip[0] + int(10 * s), tip[1] + int(8 * s)), s, color, max(2, int(2 * s)))
    D.text(img, label, (p1[0] + 8, p1[1] - 10), 0.55, color, 2)


REGION_STYLE = {
    ROLE_IGNORE_INSIDE:   ((80, 80, 235), "IGNORED AREA", True),      # dim inside
    ROLE_CLASSIFY_INSIDE: ((235, 180, 60), "CLASSIFY HERE", None),    # dim neither
}


def present_line(p: dict) -> str:
    """`5 HERE  3M 2F  4A 1C`, with a trailing `?` count when some are undecided."""
    if not p or not p.get("n"):
        return "0 HERE"
    s = f"{p['n']} HERE  {p['male']}M {p['female']}F  {p['adult']}A {p['child']}C"
    unknown = max(p.get("gender_unknown", 0), p.get("age_unknown", 0))
    return f"{s}  {unknown}?" if unknown else s


def draw_region(img: np.ndarray, region: Region, present: dict | None = None) -> None:
    """The rectangle and what it is for; the dimmed part is the part not used.

    `present` is the live tally of who is inside the counted part of the frame,
    drawn under the label so the area reports its own population on the picture
    (and therefore in any recorded clip) without waiting for a line crossing.
    """
    color, label, dim_inside = REGION_STYLE.get(region.role, ((0, 200, 255), "COUNTING AREA", False))
    h, w = img.shape[:2]
    (x1, y1), (x2, y2) = region.to_pixels(w, h)
    if dim_inside is not None:
        shade = img.copy()
        cv2.rectangle(shade, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.rectangle(shade, (x1, y1), (x2, y2), (255, 255, 255), -1)
        sel = (shade[:, :, 0] != 0) if dim_inside else (shade[:, :, 0] == 0)
        img[sel] = (img[sel] * 0.55).astype(img.dtype)
    _dashed(img, (x1, y1), (x2, y1), color, 2)
    _dashed(img, (x2, y1), (x2, y2), color, 2)
    _dashed(img, (x2, y2), (x1, y2), color, 2)
    _dashed(img, (x1, y2), (x1, y1), color, 2)
    # Label and tally go on a plate INSIDE the top-left corner. Outside it they are
    # pushed off the picture by a box drawn against the top edge, and over a busy
    # shop floor the text is unreadable without something behind it.
    rows = [(label, 0.5, 2)] + ([(present_line(present), 0.55, 2)] if present is not None else [])
    sizes = [cv2.getTextSize(s, D.F, sc, th)[0] for s, sc, th in rows]
    pad, gap = 7, 6
    pw = max(w_ for w_, _ in sizes) + pad * 2
    ph = sum(h_ for _, h_ in sizes) + gap * (len(rows) - 1) + pad * 2
    px = min(max(x1, 0), max(0, w - pw))
    py = min(max(y1, 0), max(0, h - ph))
    plate = img[py:py + ph, px:px + pw]
    if plate.size:
        cv2.addWeighted(plate, 0.35, np.zeros_like(plate), 0.65, 0, plate)
    ty = py + pad
    for (s, sc, th), (_, th_px) in zip(rows, sizes):
        ty += th_px
        D.text(img, s, (px + pad, ty), sc, color, th)
        ty += gap


def draw_setup_points(img: np.ndarray, pts: list[tuple[int, int]], entry_side: int):
    for p in pts:
        cv2.circle(img, p, 8, (0, 200, 255), 2, cv2.LINE_AA)
        cv2.circle(img, p, 2, (0, 200, 255), -1, cv2.LINE_AA)
    if len(pts) == 2:
        h, w = img.shape[:2]
        tl = TriggerLine.from_pixels(pts[0], pts[1], w, h, entry_side)
        draw_line(img, tl, color=(0, 200, 255), label="NEW LINE (press S to save)")


CHILD_ICON = (120, 190, 235)   # teddy-bear tan (BGR)


def draw_child_icon(img: np.ndarray, cx: int, cy: int, r: int) -> None:
    """Small teddy-bear head marking a track decided to be a child.

    Drawn from primitives (no image asset to ship or load). Every shape is
    stroked in black first so the icon stays readable over a busy CCTV frame.
    """
    ear = max(2, int(round(r * 0.45)))
    eo = int(round(r * 0.78))               # ear offset from the head centre
    for sx in (-1, 1):                      # ears
        e = (cx + sx * eo, cy - eo)
        cv2.circle(img, e, ear + 1, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(img, e, ear, CHILD_ICON, -1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), r + 1, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), r, CHILD_ICON, -1, cv2.LINE_AA)
    # muzzle
    mr = max(2, int(round(r * 0.42)))
    cv2.circle(img, (cx, cy + int(r * 0.32)), mr, (245, 245, 250), -1, cv2.LINE_AA)
    # eyes and nose
    ex, ey = max(1, int(round(r * 0.36))), int(round(r * 0.22))
    d = max(1, int(round(r * 0.16)))
    for sx in (-1, 1):
        cv2.circle(img, (cx + sx * ex, cy - ey), d, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy + int(r * 0.18)), d, (0, 0, 0), -1, cv2.LINE_AA)


def draw_tracks(img: np.ndarray, tracks: list[Track], counter: LineCounter | None, flash: dict[int, tuple[int, str]], frame_idx: int):
    for t in tracks:
        st: TrackState | None = counter.get_state(t.track_id) if counter else None
        # Decide with the counter's own thresholds, never decide()'s defaults: otherwise a
        # track with 0.5 <= p_child < --child-thresh is drawn as a child (teddy bear and all)
        # but counted as an adult when it crosses, which reads as a broken counter.
        if st is None:
            g, a = UNKNOWN, UNKNOWN
        elif counter is not None:
            g, a = st.decide(counter.gender_thresh, counter.child_thresh)
        else:
            g, a = st.decide()
        col = _gender_color(g)
        x1, y1, x2, y2 = t.box
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
        if a == "C":
            # teddy bear on the top-right corner, kept inside the frame
            r = int(min(16, max(7, (x2 - x1) * 0.16)))
            icx = min(img.shape[1] - r - 2, max(r + 2, x2 - r))
            icy = min(img.shape[0] - r - 2, max(int(r * 1.8), y1 + r))
            draw_child_icon(img, icx, icy, r)
        # head marker
        hx, hy = int(t.head[0]), int(t.head[1])
        cv2.circle(img, (hx, hy), 5, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, (hx, hy), 5, (0, 0, 0), 1, cv2.LINE_AA)
        # label
        a_word = {"A": "Adult", "C": "Child"}.get(a, "?")
        g_word = {"M": "M", "F": "F"}.get(g, "?")
        conf = ""
        if st and st.n_obs:
            conf = f" {max(st.p_male, 1 - st.p_male):.0%}/{max(st.p_child, 1 - st.p_child):.0%}"
        label = f"#{t.track_id} {g_word} {a_word}{conf}"
        (tw, th), _ = cv2.getTextSize(label, D.F, 0.5, 1)
        ly = max(th + 4, y1 - 4)
        cv2.rectangle(img, (x1, ly - th - 4), (x1 + tw + 6, ly + 2), col, -1)
        D.text(img, label, (x1 + 3, ly - 2), 0.5, (0, 0, 0), 1)
        # crossing flash
        if t.track_id in flash:
            f_idx, word = flash[t.track_id]
            if frame_idx - f_idx < 20:
                fcol = (60, 220, 60) if word == "ENTRY" else (60, 120, 255)
                D.text(img, word, (hx - 30, hy - 14), 0.8, (0, 0, 0), 4)
                D.text(img, word, (hx - 30, hy - 14), 0.8, fcol, 2)


def draw_banner(img: np.ndarray, lines: list[str], color=(255, 255, 255), bg=(0, 0, 0)):
    """Semi-transparent banner in the top-left with one or more lines of text."""
    h, w = img.shape[:2]
    scale = max(0.45, min(0.8, w / 1600))
    pad = 8
    lh = int(28 * scale / 0.6)
    bh = pad * 2 + lh * len(lines)
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, bh), bg, -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    y = pad + int(lh * 0.75)
    for s in lines:
        D.text(img, s, (pad, y), scale, color, 1)
        y += lh


def draw_alarm(img: np.ndarray, text: str) -> None:
    """Red alarm band across the top of a camera frame plus a red border (also lands in recorded clips)."""
    h, w = img.shape[:2]
    bh = max(28, int(h * 0.07))
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, bh), (40, 40, 220), -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
    scale = max(0.6, min(1.1, w / 1400))
    D.text_center(img, text.upper(), w // 2, int(bh * 0.7), scale, (255, 255, 255), 2, D.FD)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (40, 40, 220), max(4, int(min(w, h) * 0.008)))
