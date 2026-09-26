"""OpenCV-rendered statistics dashboard (the top-left window of the 2x2 grid)."""
from __future__ import annotations

import time

import cv2
import numpy as np

from .stats import Stats

# BGR palette
BG = (24, 26, 32)
PANEL = (38, 41, 50)
PANEL_EDGE = (64, 68, 80)
TEXT = (235, 235, 235)
MUTED = (150, 155, 165)
ACCENT = (90, 200, 120)     # green
MALE = (255, 170, 60)       # blue (BGR)
FEMALE = (200, 90, 230)     # magenta
ADULT = (80, 200, 230)      # yellow
CHILD = (255, 120, 120)     # light blue
WARN = (60, 120, 255)       # orange
UNKNOWN_COL = (120, 120, 120)

F = cv2.FONT_HERSHEY_SIMPLEX
FD = cv2.FONT_HERSHEY_DUPLEX


def text(img, s, org, scale=0.6, color=TEXT, thick=1, font=F):
    cv2.putText(img, s, (int(org[0]), int(org[1])), font, scale, color, thick, cv2.LINE_AA)


def text_size(s, scale, thick=1, font=F):
    (w, h), _ = cv2.getTextSize(s, font, scale, thick)
    return w, h


def text_center(img, s, cx, y, scale, color, thick=1, font=F):
    w, _ = text_size(s, scale, thick, font)
    text(img, s, (cx - w // 2, y), scale, color, thick, font)


def text_right(img, s, right_x, y, scale, color, thick=1, font=F):
    w, _ = text_size(s, scale, thick, font)
    text(img, s, (right_x - w, y), scale, color, thick, font)


def panel(img, x1, y1, x2, y2, title: str | None = None, s: float = 1.0):
    cv2.rectangle(img, (x1, y1), (x2, y2), PANEL, -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), PANEL_EDGE, 1)
    if title:
        text(img, title, (x1 + int(16 * s), y1 + int(26 * s)), 0.6 * s, MUTED, 1)


def bar(img, x1, y1, x2, y2, frac_left: float, col_left, col_right):
    frac_left = min(1.0, max(0.0, frac_left))
    cv2.rectangle(img, (x1, y1), (x2, y2), (60, 62, 70), -1)
    xm = int(x1 + (x2 - x1) * frac_left)
    if xm > x1:
        cv2.rectangle(img, (x1, y1), (xm, y2), col_left, -1)
    if xm < x2:
        cv2.rectangle(img, (xm, y1), (x2, y2), col_right, -1)


def fmt(n: int) -> str:
    return f"{n:,}"


def render_dashboard(
    stats: Stats,
    size: tuple[int, int],
    cam_rows: list[dict],
    setup_mode: bool,
    loop_fps: float,
    device: str,
) -> np.ndarray:
    W, H = size
    img = np.full((H, W, 3), BG, dtype=np.uint8)
    s = min(W / 960.0, H / 540.0)   # scale relative to a 960x540 design
    m = int(14 * s)
    t = stats.total

    # ------------------------------------------------------------ header
    y_title = int(0.065 * H)
    y_sub = int(0.11 * H)
    text(img, "VISITOR COUNTING CONSOLE", (m, y_title), 0.9 * s, TEXT, 2, FD)
    text(img, "Entry-line counting  |  gender  |  adult / child", (m, y_sub), 0.5 * s, MUTED, 1)
    text_right(img, time.strftime("%Y-%m-%d  %H:%M:%S"), W - m, y_title, 0.62 * s, TEXT, 1)
    elapsed = int(time.time() - stats.started_at)
    text_right(img, f"running {elapsed // 3600:02d}:{(elapsed // 60) % 60:02d}:{elapsed % 60:02d}   "
                    f"{loop_fps:4.1f} pipeline fps   {device}", W - m, y_sub, 0.48 * s, MUTED, 1)

    y = int(0.14 * H)
    if setup_mode:
        bh = int(0.055 * H)
        cv2.rectangle(img, (0, y), (W, y + bh), WARN, -1)
        text_center(img, "SETUP MODE - counting paused.   L-click: line points   R-click / F: reverse IN arrow   C: clear   S: save   E: exit",
                    W // 2, y + int(bh * 0.68), 0.5 * s, (20, 20, 20), 1)
        y += bh + m // 2

    # ------------------------------------------------- total visitors tile
    ty1, ty2 = y, y + int(0.25 * H)
    panel(img, m, ty1, W - m, ty2, "VISITOR COUNTING", s)
    caption = ("Total number of visitors (entries - exits)" if not t.initial
               else f"Total inside (assumed {fmt(t.initial)} at start-up + entries - exits)")
    text(img, caption, (m + int(16 * s), ty1 + int(48 * s)), 0.5 * s, MUTED)
    text(img, fmt(t.visitors), (m + int(16 * s), ty2 - int(18 * s)), 2.6 * s, ACCENT, int(max(2, 5 * s)), FD)
    text_right(img, f"entries  {fmt(t.entries)}", W - m - int(16 * s), ty1 + int(0.42 * (ty2 - ty1)), 0.7 * s, TEXT, 1)
    text_right(img, f"exits    {fmt(t.exits)}", W - m - int(16 * s), ty1 + int(0.65 * (ty2 - ty1)), 0.7 * s, TEXT, 1)
    if stats.last_event:
        ts, ci, ev = stats.last_event
        if time.time() - ts < 8:
            word = "ENTRY" if ev.direction > 0 else "EXIT"
            g = {"M": "male", "F": "female"}.get(ev.gender, "unknown")
            a = {"A": "adult", "C": "child"}.get(ev.age, "unknown")
            text_right(img, f"last: {word}  {g}  {a}  @ {cam_rows[ci]['name'][:18]}", W - m - int(16 * s),
                       ty2 - int(12 * s), 0.45 * s, ACCENT)

    # ---------------------------------------------------- gender / age tiles
    ty1 = ty2 + m
    ty2 = ty1 + int(0.27 * H)
    half = (W - 3 * m) // 2
    th = ty2 - ty1

    def two_col_tile(x1, title, l_lab, l_val, l_col, r_lab, r_val, r_col, unknown):
        x2 = x1 + half
        panel(img, x1, ty1, x2, ty2, title, s)
        cx_l = x1 + half // 4
        cx_r = x1 + 3 * half // 4
        text_center(img, l_lab, cx_l, ty1 + int(0.36 * th), 0.6 * s, l_col, 1)
        text_center(img, r_lab, cx_r, ty1 + int(0.36 * th), 0.6 * s, r_col, 1)
        text_center(img, fmt(l_val), cx_l, ty1 + int(0.72 * th), 1.8 * s, l_col, int(max(2, 3 * s)), FD)
        text_center(img, fmt(r_val), cx_r, ty1 + int(0.72 * th), 1.8 * s, r_col, int(max(2, 3 * s)), FD)
        cv2.line(img, (x1 + half // 2, ty1 + int(0.25 * th)), (x1 + half // 2, ty1 + int(0.78 * th)), PANEL_EDGE, 1)
        tot = l_val + r_val
        bar(img, x1 + int(16 * s), ty2 - int(0.16 * th), x2 - int(16 * s), ty2 - int(0.09 * th),
            (l_val / tot) if tot else 0.5, l_col, r_col)
        if unknown:
            text_right(img, f"unknown {unknown}", x2 - int(16 * s), ty1 + int(26 * s), 0.45 * s, UNKNOWN_COL)

    g_title, a_title = "GENDER DETECTION COUNT", "AGE GROUP"
    if t.initial:   # make clear the tiles include an assumption, not only measurements
        g_title += "  (incl. assumed 50/50 baseline)"
        a_title += "  (incl. assumed 80/20 baseline)"
    two_col_tile(m, g_title, "MALE", t.male, MALE, "FEMALE", t.female, FEMALE, t.gender_unknown)
    two_col_tile(2 * m + half, a_title, "ADULT", t.adult, ADULT, "CHILD", t.child, CHILD, t.age_unknown)

    # -------------------------------------------------- per camera table
    ty1 = ty2 + m
    ty2 = H - m - int(16 * s)
    panel(img, m, ty1, W - m, ty2, "PER CAMERA / ENTRANCE", s)
    inner = W - 2 * m
    cols = [m + int(inner * f) for f in (0.02, 0.22, 0.38, 0.47, 0.60, 0.74, 0.87)]
    hdr = ["camera", "status", "fps", "visitors", "in / out", "M / F", "A / C"]
    avail = ty2 - ty1 - int(40 * s)
    n_rows = max(1, len(cam_rows))
    row_h = min(int(24 * s), avail // (n_rows + 1))
    y = ty1 + int(26 * s) + row_h
    for cx, h in zip(cols, hdr):
        text(img, h, (cx, y), 0.42 * s, MUTED)
    for i, row in enumerate(cam_rows):
        c = stats.per_camera[i]
        y += row_h
        col = TEXT if row.get("ok", True) else WARN
        status = row["status"] if row.get("has_line", True) else "NO LINE (E)"
        text(img, row["name"][:18], (cols[0], y), 0.5 * s, col)
        text(img, status[:14], (cols[1], y), 0.45 * s, col if row.get("has_line", True) else WARN)
        text(img, f"{row['fps']:.1f}", (cols[2], y), 0.45 * s, col)
        text(img, fmt(c.visitors), (cols[3], y), 0.55 * s, ACCENT, 1)
        text(img, f"{c.entries} / {c.exits}", (cols[4], y), 0.45 * s, col)
        text(img, f"{c.male} / {c.female}", (cols[5], y), 0.45 * s, col)
        text(img, f"{c.adult} / {c.child}", (cols[6], y), 0.45 * s, col)

    text_right(img, "E setup   S save line   R reset   P pause   Q quit", W - m, H - int(5 * s), 0.4 * s, MUTED)
    return img
