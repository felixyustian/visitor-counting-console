"""Person detection + multi-object tracking.

All camera frames of one iteration are run through YOLO in a single batched
forward pass; a separate ByteTrack instance per camera then associates the
detections into tracks. (Ultralytics' own `model.track()` shares one tracker
across a batch of images, which is why the trackers are driven directly.)

If a *-pose* model is used the head point is estimated from the facial
keypoints (nose, eyes, ears); otherwise from the top of the bounding box.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

# Facial keypoint indices in the COCO-17 layout used by YOLO pose models.
_FACE_KPTS = (0, 1, 2, 3, 4)  # nose, l-eye, r-eye, l-ear, r-ear


@dataclass
class Track:
    track_id: int
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 in pixels
    conf: float
    head: tuple[float, float]       # estimated head centre in pixels
    head_from_pose: bool = False

    @property
    def height(self) -> int:
        return self.box[3] - self.box[1]

    @property
    def width(self) -> int:
        return self.box[2] - self.box[0]


def head_point(box, kxy: Optional[np.ndarray] = None, kconf: Optional[np.ndarray] = None) -> tuple[tuple[float, float], bool]:
    x1, y1, x2, y2 = box
    if kxy is not None:
        pts = []
        for k in _FACE_KPTS:
            if (kconf is None or kconf[k] > 0.5) and (kxy[k][0] > 0 or kxy[k][1] > 0):
                pts.append(kxy[k])
        if pts:
            p = np.mean(np.asarray(pts), axis=0)
            return (float(p[0]), float(p[1])), True
    # Fallback: the head centre sits roughly 8% of the body height below the top edge.
    return (float((x1 + x2) / 2.0), float(y1 + 0.08 * (y2 - y1))), False


class _Detections:
    """Minimal Results-like view over plain arrays - all ByteTrack reads is conf / xywh / cls.

    Lets a detector that is not Ultralytics (RF-DETR) drive the same tracker.
    """

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xyxy = xyxy
        self.conf = conf
        self.cls = cls
        if len(xyxy):
            x1, y1, x2, y2 = xyxy.T
            self.xywh = np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=1)
        else:
            self.xywh = np.zeros((0, 4), np.float32)

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, mask) -> "_Detections":
        return _Detections(self.xyxy[mask], self.conf[mask], self.cls[mask])


# RF-DETR reports COCO's 91-class indexing, where person is 1 (YOLO's 80-class map uses 0).
RFDETR_PERSON_CLASS = 1
_RFDETR_SIZES = {"nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium", "large": "RFDETRLarge"}


def _load_rfdetr(weights: str):
    """`rfdetr-medium` / `rfdetr-small` ... -> the matching RF-DETR class (Apache-2.0)."""
    import rfdetr

    size = next((k for k in _RFDETR_SIZES if k in weights.lower()), "medium")
    return getattr(rfdetr, _RFDETR_SIZES[size])(), size


class MultiStreamPersonTracker:
    """Batched YOLO person detector with one ByteTrack instance per stream."""

    def __init__(
        self,
        n_streams: int,
        weights: str = "yolo11s.pt",
        device: str = "cuda",
        conf: float = 0.35,
        iou: float = 0.6,
        imgsz: int = 640,
        tracker: str = "bytetrack.yaml",
        half: bool = True,
    ):
        from ultralytics.trackers import BYTETracker, BOTSORT
        from ultralytics.utils import YAML, IterableSimpleNamespace
        from ultralytics.utils.checks import check_yaml

        self.backend = "rfdetr" if "rfdetr" in weights.lower() else "yolo"
        if self.backend == "rfdetr":
            self.model, size = _load_rfdetr(weights)
            print(f"[init] detector backend: RF-DETR {size} (Apache-2.0)")
        else:
            from ultralytics import YOLO

            self.model = YOLO(weights)
        self.device = device
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.quantize = 16 if (half and device.startswith("cuda")) else 32
        self.is_pose = "pose" in weights.lower() and self.backend == "yolo"

        cfg = IterableSimpleNamespace(**YAML.load(check_yaml(tracker)))
        cfg.device = device
        cls = {"bytetrack": BYTETracker, "botsort": BOTSORT}.get(cfg.tracker_type)
        if cls is None:
            raise ValueError(f"unsupported tracker_type {cfg.tracker_type!r} (use bytetrack or botsort)")
        # BYTE keeps an existing track alive on detections too weak to start one (a person
        # half-hidden behind someone, or at the edge of the detector's confidence). Filtering
        # the detector at `conf` threw those boxes away before the tracker saw them, so that
        # second association stage never ran. The detector now reports down to the tracker's
        # low threshold, and `conf` becomes the score a detection needs to count as a strong
        # match; new tracks still need `new_track_thresh`, so weak boxes never start a track.
        cfg.track_high_thresh = conf
        self.det_conf = min(conf, float(cfg.track_low_thresh))
        self._tracker_cls, self._tracker_cfg = cls, cfg
        self.trackers = [cls(cfg) for _ in range(n_streams)]
        # how long a hidden person keeps their ID, in processed frames
        self.max_time_lost = int(self.trackers[0].max_frames_lost) if self.trackers else 0

    def update(self, images: Sequence[Optional[np.ndarray]]) -> list[list[Track]]:
        """Run detection on all non-None images (one batch) and update each stream's tracker.

        Returns one list of tracks per input position; None inputs yield [].
        """
        out: list[list[Track]] = [[] for _ in images]
        valid = [i for i, im in enumerate(images) if im is not None]
        if not valid:
            return out
        if self.backend == "rfdetr":
            return self._update_rfdetr(images, valid, out)
        results = self.model.predict(
            [images[i] for i in valid],
            classes=[0],  # COCO person
            conf=self.det_conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            quantize=self.quantize,
            verbose=False,
        )
        for r, i in zip(results, valid):
            det = r.boxes.cpu().numpy()
            tracks = self.trackers[i].update(det, r.orig_img)  # (n, 8): x1,y1,x2,y2,id,conf,cls,det_idx
            if len(tracks) == 0:
                continue
            kxy = kconf = None
            if self.is_pose and r.keypoints is not None:
                kxy_all = r.keypoints.xy.cpu().numpy()
                kconf_all = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else None
                idx = tracks[:, -1].astype(int)
                kxy = kxy_all[idx]
                kconf = kconf_all[idx] if kconf_all is not None else None
            for j, row in enumerate(tracks):
                x1, y1, x2, y2, tid, score = row[:6]
                head, from_pose = head_point((x1, y1, x2, y2), kxy[j] if kxy is not None else None,
                                             kconf[j] if kconf is not None else None)
                out[i].append(Track(int(tid), (int(x1), int(y1), int(x2), int(y2)), float(score), head, from_pose))
        return out

    def _update_rfdetr(self, images, valid, out) -> list[list[Track]]:
        """RF-DETR path: detect, keep people, then drive the same ByteTrack instances."""
        import cv2

        rgb = [cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB) for i in valid]
        preds = self.model.predict(rgb, threshold=self.det_conf)
        if not isinstance(preds, list):
            preds = [preds]
        for pred, i, frame in zip(preds, valid, rgb):
            keep = pred.class_id == RFDETR_PERSON_CLASS
            det = _Detections(pred.xyxy[keep].astype(np.float32),
                              pred.confidence[keep].astype(np.float32),
                              np.zeros(int(keep.sum()), np.float32))   # single class downstream
            tracks = self.trackers[i].update(det, images[i])
            for row in tracks:
                x1, y1, x2, y2, tid, score = row[:6]
                head, from_pose = head_point((x1, y1, x2, y2))
                out[i].append(Track(int(tid), (int(x1), int(y1), int(x2), int(y2)), float(score), head, from_pose))
        return out

    def add_stream(self) -> int:
        """Append a tracker for a new camera slot; returns its index."""
        self.trackers.append(self._tracker_cls(self._tracker_cfg))
        return len(self.trackers) - 1

    def remove_stream(self, i: int) -> None:
        del self.trackers[i]

    def reset(self, i: int) -> None:
        """Forget all tracks of one stream (e.g. when its file loops) *without* resetting the
        global track-ID counter, so IDs stay unique across the other streams."""
        t = self.trackers[i]
        t.tracked_stracks, t.lost_stracks, t.removed_stracks = [], [], []
        t.frame_id = 0
        t.kalman_filter = t.get_kalmanfilter()
