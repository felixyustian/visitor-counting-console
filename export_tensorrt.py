"""Export the YOLO detector to a TensorRT engine for NVIDIA GPUs (Ubuntu / Windows).

    python export_tensorrt.py                       # yolo11s.pt -> yolo11s.engine (FP16)
    python export_tensorrt.py --model yolo11m.pt --int8
then run with  --model yolo11s.engine  (app.py / webapp.py load .engine files directly).

Engines are specific to the GPU + TensorRT version they were built on: rebuild on
each machine. Needs the `tensorrt` package (pip install tensorrt) - Ultralytics
installs it on demand on the first export if it is missing.
"""
from __future__ import annotations

import argparse

from ultralytics import YOLO


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="yolo11s.pt")
    p.add_argument("--imgsz", type=int, default=640, help="must match the --imgsz used at run time")
    p.add_argument("--batch", type=int, default=3, help="max batch = number of cameras")
    p.add_argument("--int8", action="store_true", help="INT8 instead of FP16 (needs calibration images, slower export)")
    p.add_argument("--device", default="0")
    a = p.parse_args()
    out = YOLO(a.model).export(format="engine", imgsz=a.imgsz, batch=a.batch, half=not a.int8, int8=a.int8,
                               dynamic=True, device=a.device, verbose=False)
    print(f"\nexported {out}\nrun:  ./run_web.sh --model {out}   (or run_web.bat --model {out})")


if __name__ == "__main__":
    main()
