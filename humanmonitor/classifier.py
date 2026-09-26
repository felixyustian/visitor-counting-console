"""Gender and adult/child classification on person crops.

Zero-shot image-text matching with prompt ensembles. Each crop yields P(male)
and P(child); the caller accumulates these over a track's lifetime so that the
final decision at the trigger line is a vote over many frames rather than a
single view.

Two model families are supported, chosen by the model name:

* **SigLIP 2** (default, `google/siglip2-*`, Apache-2.0) - sigmoid-trained, and
  measurably better than CLIP on person attributes from CCTV viewpoints
  (98% vs 94% gender on the crossing set collected by scripts/ab_demographics.py).
* **CLIP** (`openai/clip-vit-base-patch32`) - the original model, kept so the
  two can be compared on site footage.

Both are public weights, no auth required. The model is pluggable: anything
exposing classify(crops) -> list[(p_male, p_child)] can replace it (e.g. a
PA-100K attribute network or InsightFace gender-age).
"""
from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

MALE_PROMPTS = [
    "a photo of a man",
    "a photo of a boy",
    "a cctv image of a man walking",
    "a male person seen from a surveillance camera",
    "a man wearing casual clothes",
]
FEMALE_PROMPTS = [
    "a photo of a woman",
    "a photo of a girl",
    "a cctv image of a woman walking",
    "a female person seen from a surveillance camera",
    "a woman wearing casual clothes",
]
ADULT_PROMPTS = [
    "a photo of an adult",
    "a photo of a grown man or grown woman",
    "a cctv image of an adult person walking",
    "a tall adult pedestrian",
    "an elderly person walking",
]
CHILD_PROMPTS = [
    "a photo of a child",
    "a photo of a small kid",
    "a cctv image of a young child walking",
    "a short little boy or girl",
    "a toddler or primary school child",
]

def _as_tensor(out):
    """transformers>=5 returns a ModelOutput from get_*_features; older versions a tensor."""
    if hasattr(out, "pooler_output"):
        return out.pooler_output
    if hasattr(out, "text_embeds"):
        return out.text_embeds
    if hasattr(out, "image_embeds"):
        return out.image_embeds
    return out


_CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
_SIGLIP_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
_SIGLIP_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)

DEFAULT_MODEL = "google/siglip2-base-patch16-256"


def is_siglip(model_name: str) -> bool:
    return "siglip" in model_name.lower()


class DemographicClassifier:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cuda", half: bool = True):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = device
        self.model_name = model_name
        self.is_siglip = is_siglip(model_name)
        self.dtype = torch.float16 if (half and device.startswith("cuda")) else torch.float32
        self.model = AutoModel.from_pretrained(model_name).to(device, dtype=self.dtype).eval()
        tok = AutoTokenizer.from_pretrained(model_name)
        # SigLIP was trained with every caption padded to 64 tokens; CLIP pads to the longest prompt.
        tok_kwargs = {"padding": "max_length", "max_length": 64} if self.is_siglip else {"padding": True}

        with torch.no_grad():
            def embed(prompts: list[str]):
                t = tok(prompts, return_tensors="pt", **tok_kwargs).to(device)
                e = _as_tensor(self.model.get_text_features(**t))
                return e / e.norm(dim=-1, keepdim=True)

            self.gender_text = torch.cat([embed(MALE_PROMPTS), embed(FEMALE_PROMPTS)])
            self.n_male = len(MALE_PROMPTS)
            self.age_text = torch.cat([embed(ADULT_PROMPTS), embed(CHILD_PROMPTS)])
            self.n_adult = len(ADULT_PROMPTS)
        self.logit_scale = float(self.model.logit_scale.exp().item())
        # SigLIP's sigmoid head carries a learned bias; CLIP has none.
        self.logit_bias = float(getattr(self.model, "logit_bias", torch.zeros(())).item()) if self.is_siglip else 0.0
        self.input_size = self.model.config.vision_config.image_size
        mean, std = (_SIGLIP_MEAN, _SIGLIP_STD) if self.is_siglip else (_CLIP_MEAN, _CLIP_STD)
        self._mean = torch.tensor(mean, device=device, dtype=self.dtype).view(1, 3, 1, 1)
        self._std = torch.tensor(std, device=device, dtype=self.dtype).view(1, 3, 1, 1)

    # ------------------------------------------------------------ preprocess
    def _prep(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Letterbox to a square and resize to the model input size (uint8 RGB HWC; normalised on the GPU)."""
        s = self.input_size
        h, w = crop_bgr.shape[:2]
        side = max(h, w)
        canvas = np.full((side, side, 3), 114, dtype=np.uint8)
        oy, ox = (side - h) // 2, (side - w) // 2
        canvas[oy:oy + h, ox:ox + w] = crop_bgr
        img = cv2.resize(canvas, (s, s), interpolation=cv2.INTER_LINEAR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # -------------------------------------------------------------- classify
    def classify(self, crops: Sequence[np.ndarray]) -> list[tuple[float, float]]:
        """Return [(p_male, p_child), ...] for each BGR crop."""
        if len(crops) == 0:
            return []
        torch = self.torch
        batch = np.stack([self._prep(c) for c in crops])  # (n, s, s, 3) uint8
        with torch.no_grad():
            x = torch.from_numpy(batch).to(self.device).permute(0, 3, 1, 2).to(self.dtype) / 255.0
            x = (x - self._mean) / self._std
            f = _as_tensor(self.model.get_image_features(pixel_values=x))
            f = f / f.norm(dim=-1, keepdim=True)
            # Both families score cosine similarity; only the scaling differs (SigLIP adds a bias).
            # The two-way softmax that follows is what turns the scores into P(male) / P(child),
            # so the prompt ensembles and thresholds stay the same across models.
            g = (self.logit_scale * (f @ self.gender_text.T) + self.logit_bias).float().softmax(dim=-1)
            a = (self.logit_scale * (f @ self.age_text.T) + self.logit_bias).float().softmax(dim=-1)
            p_male = g[:, : self.n_male].sum(dim=-1)
            p_child = a[:, self.n_adult:].sum(dim=-1)
        return list(zip(p_male.cpu().tolist(), p_child.cpu().tolist()))


def crop_person(image: np.ndarray, box: tuple[int, int, int, int], margin: float = 0.08) -> np.ndarray | None:
    """Crop a person box with a small margin; returns None if degenerate."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    mx, my = int(bw * margin), int(bh * margin)
    x1, y1 = max(0, x1 - mx), max(0, y1 - my)
    x2, y2 = min(w, x2 + mx), min(h, y2 + my)
    if x2 - x1 < 8 or y2 - y1 < 16:
        return None
    return image[y1:y2, x1:x2]
