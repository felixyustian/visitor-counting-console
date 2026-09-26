"""Video sources: local files and RTSP/HTTP streams behind one interface.

* Files are decoded in a background thread into a small ordered queue so the
  decode cost overlaps with inference. Every `stride`-th frame is delivered
  (stride is chosen automatically for high-frame-rate files) and the file
  loops back to frame 0 when it ends (unless loop=False).
* Streams are read in a background thread that keeps only the latest frame,
  so processing never lags behind real time, and reconnect automatically.
"""
from __future__ import annotations

import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional

# Prefer TCP for RTSP so frames are not corrupted by UDP loss. Must be set before
# cv2 opens a capture. `stimeout` was removed from FFmpeg in favour of `timeout`;
# keep both so the socket timeout applies whichever build is installed.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|timeout;5000000|stimeout;5000000",
)

# How long a capture may take to open or to deliver a frame. Without these an
# unreachable camera holds the call for OpenCV's 30 s default, which is long
# enough to hang the page's "connect" button (measured: 30.1 s -> 4.0 s).
OPEN_TIMEOUT_MS = 5000
READ_TIMEOUT_MS = 5000

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from .config import is_url, source_name  # noqa: E402


@dataclass
class Frame:
    image: np.ndarray
    index: int            # running counter of *delivered* frames for this source
    timestamp: float      # wall clock when the frame was read
    looped: bool = False  # True on the first frame after a file wrapped around


class NullSource:
    """Placeholder for an empty camera slot (no file / stream configured yet)."""

    src = ""
    name = "no camera"
    is_stream = False
    is_null = True
    loop = False
    width = height = 0
    fps = 0.0
    frame_count = 0
    stride = 1
    effective_fps = 30.0
    ended = False
    connected = False
    last_error = "no source"

    def read(self) -> Optional[Frame]:
        return None

    def release(self) -> None:
        pass

    def status(self) -> str:
        return "NO SOURCE"


class VideoSource:
    is_null = False

    def __init__(self, src: str, loop: bool = True, stride: int = 0, reconnect_delay: float = 3.0, prefetch: int = 4):
        self.src = src
        self.name = source_name(src)
        self.is_stream = is_url(src)
        self.loop = loop and not self.is_stream
        self.reconnect_delay = reconnect_delay
        self.width = 0
        self.height = 0
        self.fps = 0.0
        self.frame_count = 0
        self.ended = False
        self.connected = False
        self.last_error = ""
        self._index = 0

        self._cap: Optional[cv2.VideoCapture] = None
        self._stop = threading.Event()
        if self.is_stream:
            # Do not open here: the caller is usually an HTTP request handler, and
            # an unreachable camera would hold it for the whole open timeout. The
            # reader thread opens instead and keeps retrying; until it succeeds the
            # source reports OFFLINE, which is what the page already renders.
            self.last_error = "connecting"
        else:
            self._open()

        # stride: 0 = auto (aim for ~30 processed fps on high-fps files)
        if self.is_stream:
            self.stride = 1
        else:
            self.stride = stride if stride > 0 else max(1, int(round(self.fps / 30.0)))
        self.effective_fps = (self.fps / self.stride) if self.fps else 30.0

        self._lock = threading.Lock()
        self._latest: Optional[Frame] = None                       # streams
        self._queue: queue.Queue[Frame] = queue.Queue(maxsize=prefetch)  # files
        target = self._stream_loop if self.is_stream else self._file_loop
        self._thread = threading.Thread(target=target, name=f"reader-{self.name}", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ open
    def _open(self) -> bool:
        if self._cap is not None:
            self._cap.release()
        try:
            cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG,
                                   [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, OPEN_TIMEOUT_MS,
                                    cv2.CAP_PROP_READ_TIMEOUT_MSEC, READ_TIMEOUT_MS])
        except (AttributeError, TypeError):        # older OpenCV without the params overload
            cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            self.connected = False
            self.last_error = "cannot open"
            self._cap = cap
            if not self.is_stream:
                raise FileNotFoundError(f"Cannot open video source: {self.src}")
            return False
        if self.is_stream:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap = cap
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not self.is_stream else 0
        if self.is_stream:
            # a stream is opened by its reader thread, after __init__ has already
            # computed this from a then-unknown fps; correct it now
            self.effective_fps = self.fps or 30.0
        self.connected = True
        self.last_error = ""
        return True

    # ---------------------------------------------------------------- stream
    def _stream_loop(self) -> None:
        while not self._stop.is_set():
            if not self.connected:
                if not self._open():
                    time.sleep(self.reconnect_delay)
                    continue
            ok, img = self._cap.read()
            if not ok or img is None:
                self.connected = False
                self.last_error = "stream lost, reconnecting"
                time.sleep(self.reconnect_delay)
                continue
            self._index += 1
            with self._lock:
                self._latest = Frame(img, self._index, time.time())

    # ------------------------------------------------------------------ file
    def _read_strided(self) -> tuple[bool, Optional[np.ndarray]]:
        """Skip stride-1 frames cheaply (decode without colour conversion), then read one.

        Used for the first frame after a loop too, so every pass processes the same
        source frames (stride-1, 2*stride-1, ...). Reading straight after the rewind
        used to shift a stride-2 file from odd frames on the first pass to even frames
        on every later one, so replays were not replays.
        """
        for _ in range(self.stride - 1):
            self._cap.grab()
        ok, img = self._cap.read()
        return ok and img is not None, img

    def _file_loop(self) -> None:
        looped = False
        while not self._stop.is_set():
            ok, img = self._read_strided()
            if not ok:
                if not self.loop:
                    self.ended = True
                    return
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, img = self._read_strided()
                if not ok:
                    self._open()
                    ok, img = self._read_strided()
                    if not ok:
                        self.ended = True
                        return
                looped = True
            self._index += 1
            frame = Frame(img, self._index, time.time(), looped=looped)
            looped = False
            while not self._stop.is_set():
                try:
                    self._queue.put(frame, timeout=0.2)
                    break
                except queue.Full:
                    continue

    # ------------------------------------------------------------------ read
    def read(self) -> Optional[Frame]:
        """Next frame, or None if nothing new is available yet / the source ended."""
        if self.is_stream:
            with self._lock:
                f = self._latest
                self._latest = None
            return f
        try:
            return self._queue.get(timeout=0.05)
        except queue.Empty:
            return None

    def release(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()

    def status(self) -> str:
        if self.is_stream:
            return "LIVE" if self.connected else f"OFFLINE ({self.last_error})"
        if self.ended:
            return "ENDED"
        return "FILE" if self.stride == 1 else f"FILE 1/{self.stride}"
