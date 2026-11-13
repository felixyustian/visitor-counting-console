"""A trained CMIO classifier (Chinese / Malay / Indian / Others), loaded from a checkpoint.

Mirrored from the working repository, where it is the console's CMIO source.
It is a model with "race classifier" written on it, so the bar it has to clear
before anything it says reaches a customer is high: the checkpoint shipped
alongside it (models/cmio.pt, see models/cmio.source.txt) is PROVISIONAL and
has not been validated against hand-labelled crops. Treat every CMIO figure
the console shows as unvalidated until it has been.

WHAT A CHECKPOINT MUST DO

Exactly one thing, and the contract is small on purpose so that whoever trains
it can work against it without reading this build:

    input    one person crop, RGB, uint8, any size. It is letterboxed to
             INPUT_PX and scaled to 0-1, then normalised with IMAGENET_MEAN /
             IMAGENET_STD. Letterboxed, not stretched: a person is tall and
             squashing them to a square is a distortion the training set will
             not have contained.

    output   a float tensor of 4 or 5 logits, in this order:

                 0 Chinese   1 Malay   2 Indian   3 Others   [4 unsure]

             Logits, not probabilities - softmax happens here, so a checkpoint
             exported with a softmax baked in will be softmaxed twice and
             report everything at once flatter and less confident than it is.
             The load-time probe below refuses such a checkpoint.

The fifth class is optional and worth having. A four-way softmax cannot say
"I cannot tell"; it can only spread, and spread mass is read downstream as a
low-confidence answer rather than as an abstention. Trained with an explicit
unsure class - on crops the labellers themselves could not call - it can
decline, which on this axis is the answer most often correct.

FORMAT

TorchScript (`.pt` from torch.jit.script/trace) or ONNX (`.onnx`). Both are
self-describing and neither needs the training code present, which matters
because this build should not grow a dependency on whatever framework the model
was trained in. A bare state_dict is refused: it carries no architecture, so
loading one means hard-coding the model class here and a retrain with a
different backbone silently loads wrong weights.

WHERE IT RUNS

On the pipeline thread, beside the gender / age encoder: every crop the encoder
is shown is also shown to this, so CMIO gets the same looks per track, the same
double-weighted look at the line and the same late revision as gender and age
(see counter.TrackState.vote_race). It answers the CMIO question only; gender
and age still come from the zero-shot encoder. At a few milliseconds a crop it
does not need a queue of its own.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

# The order is the contract. It is the Department of Statistics order, the same
# one counter.RACE_ORDER uses, and it is not negotiable per checkpoint: a model
# that emits them in another order is not a different configuration, it is
# wrong, and silently so - every Malay would be reported Chinese.
CMIO_ORDER = ("CN", "MY", "IN", "OT")
UNSURE = "?"

INPUT_PX = int(os.environ.get("RACE_INPUT_PX", "224"))
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Below this the crop carries too little to be worth a prediction. On typical
# entrance cameras the face is a small fraction of the 224px the checkpoint was
# trained at, so a model asked about a 24px smudge is being asked to invent. Returning no answer is the honest
# result and the counter already knows what to do with one.
MIN_CROP_PX = int(os.environ.get("RACE_MIN_CROP_PX", "48"))


def _is_state_dict(path: str, torch) -> bool:
    """Is this file a bare dict of tensors rather than a scripted module?

    weights_only=True so that answering the question cannot execute anything
    the file asks for - the checkpoint is being examined precisely because it
    did not load, and a file of unknown provenance is the wrong thing to
    unpickle.
    """
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except BaseException:
        return False
    return isinstance(obj, dict) and any(hasattr(v, "shape") for v in obj.values())


def letterbox(crop: np.ndarray, px: int) -> np.ndarray:
    """Fit `crop` into px x px on grey, keeping its aspect ratio.

    Grey rather than black because the pad is not evidence: a black border is a
    strong edge the first convolution responds to, and a model trained on one
    pad colour and served another has been handed a different image.
    """
    import cv2

    h, w = crop.shape[:2]
    scale = px / max(h, w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    small = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA
                       if scale < 1 else cv2.INTER_LINEAR)
    out = np.full((px, px, 3), 114, dtype=np.uint8)
    top, left = (px - nh) // 2, (px - nw) // 2
    out[top:top + nh, left:left + nw] = small
    return out


def preprocess(crop_rgb: np.ndarray, px: int = INPUT_PX) -> np.ndarray:
    """One crop -> (1, 3, px, px) float32, normalised. The contract, in code."""
    boxed = letterbox(crop_rgb, px).astype(np.float32) / 255.0
    boxed = (boxed - np.array(IMAGENET_MEAN, np.float32)) / np.array(IMAGENET_STD, np.float32)
    return np.ascontiguousarray(boxed.transpose(2, 0, 1)[None])


class LocalRaceClassifier:
    """A checkpoint that answers CMIO.

    `predict` returns {code: probability} over CMIO (plus "?" for a five-class
    checkpoint), which is what counter.TrackState.vote_race takes: the soft
    vote, the agreement threshold and the revision all work from it.
    """

    def __init__(self, path: str, device: str = "cuda:0", input_px: int = INPUT_PX):
        self.path = path
        self.device = device
        self.input_px = input_px
        self.classes: tuple = ()
        self.backend = ""
        self.error: Optional[str] = None
        self._model = None
        self._session = None
        self._load()

    # ---------------------------------------------------------------- loading
    def _load(self) -> None:
        if not os.path.exists(self.path):
            self.error = f"no such checkpoint: {self.path}"
            return
        try:
            if self.path.endswith(".onnx"):
                self._load_onnx()
            else:
                self._load_torchscript()
        except BaseException as problem:
            self.error = f"{type(problem).__name__}: {problem}"
            self._model = self._session = None
            return
        n = self._probe()
        if self.error:
            return
        self.classes = tuple(CMIO_ORDER) + ((UNSURE,) if n == 5 else ())
        print(f"[race] trained CMIO classifier ready ({self.backend}, {n} classes, "
              f"{self.input_px}px: {os.path.basename(self.path)})")

    def _load_torchscript(self) -> None:
        import torch

        try:
            obj = torch.jit.load(self.path, map_location=self.device)
        except RuntimeError as problem:
            # torch.jit.load on a plain state_dict reports a miniz error about
            # a corrupted file, which sends whoever trained it looking for a bad
            # transfer. Name the real problem instead.
            if _is_state_dict(self.path, torch):
                raise RuntimeError(
                    "this is a state_dict, not a TorchScript module. It carries weights "
                    "but no architecture, so loading it would mean hard-coding the model "
                    "class here - and a retrain with a different backbone would then load "
                    "the wrong weights silently. Re-export it:\n"
                    "    torch.jit.save(torch.jit.script(model.eval()), 'cmio.pt')\n"
                    "or export to ONNX.") from problem
            raise
        obj.eval()
        self._model = obj
        self._torch = torch
        self.backend = "torchscript"

    def _load_onnx(self) -> None:
        import onnxruntime as ort

        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if str(self.device).startswith("cuda") else ["CPUExecutionProvider"])
        self._session = ort.InferenceSession(self.path, providers=providers)
        self._in_name = self._session.get_inputs()[0].name
        self.backend = "onnx"

    def _probe(self) -> int:
        """Run one grey frame through and check the output shape is the contract.

        Done at load rather than on the first real crop, because the first real
        crop happens with a demo running and a wall full of people, and a model
        that emits three classes or twelve should fail before then, loudly.
        """
        probe = np.full((self.input_px, self.input_px, 3), 114, np.uint8)
        try:
            raw = self._forward(preprocess(probe, self.input_px))
        except BaseException as problem:
            self.error = f"the checkpoint did not run: {type(problem).__name__}: {problem}"
            return 0
        flat = np.asarray(raw, dtype=np.float64).reshape(-1)
        if flat.size not in (4, 5):
            self.error = (f"the checkpoint emits {flat.size} values; the contract is 4 "
                          f"(CN MY IN OT) or 5 (+ unsure). See humanmonitor/race_model.py")
            return 0
        # Probabilities already, rather than logits? Softmaxing them again
        # flattens the distribution and no exception is ever raised, so the
        # symptom would be a model that mysteriously never exceeds ~0.3.
        if flat.min() >= 0.0 and abs(flat.sum() - 1.0) < 1e-3:
            self.error = ("the checkpoint emits probabilities (they sum to 1), not logits. "
                          "Export without the final softmax - applying it twice makes every "
                          "answer look less confident than the model actually is.")
            return 0
        return int(flat.size)

    # --------------------------------------------------------------- running
    def _forward(self, batch: np.ndarray) -> np.ndarray:
        if self._session is not None:
            return self._session.run(None, {self._in_name: batch.astype(np.float32)})[0]
        t = self._torch.from_numpy(batch).to(self.device)
        with self._torch.no_grad():
            return self._model(t).float().cpu().numpy()

    @property
    def ready(self) -> bool:
        return self.error is None and (self._model is not None or self._session is not None)

    def predict(self, crop_rgb: np.ndarray) -> Optional[dict]:
        """{code: probability} for one crop, or None when it should not be asked.

        None rather than a flat distribution for a crop below MIN_CROP_PX: "too
        small to ask" and "asked and undecided" are different facts, and the
        counter records only the second. A tiny crop that returned a flat
        distribution would quietly vote against every group it saw.
        """
        if not self.ready or crop_rgb is None:
            return None
        if min(crop_rgb.shape[:2]) < MIN_CROP_PX:
            return None
        logits = np.asarray(self._forward(preprocess(crop_rgb, self.input_px)),
                            dtype=np.float64).reshape(-1)
        e = np.exp(logits - logits.max())
        probs = e / e.sum()
        return {code: float(p) for code, p in zip(self.classes, probs)}

    def stats(self) -> dict:
        return {"path": self.path, "backend": self.backend, "ready": self.ready,
                "error": self.error, "classes": list(self.classes),
                "input_px": self.input_px, "min_crop_px": MIN_CROP_PX}
