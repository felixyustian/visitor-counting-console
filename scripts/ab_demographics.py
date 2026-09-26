"""A/B test of demographic classifiers on real crossings: CLIP vs SigLIP 2 vs Qwen2.5-VL.

Answers "would a stronger encoder or a VLM improve gender / adult-child?" with
numbers from this site's own footage instead of a guess.

    python scripts/ab_demographics.py collect  --out output/ab  [--sources a.mp4 b.mp4 ...]
        runs the counting engine over the videos once (no loop) and saves the person
        crop of every crossing + what the live pipeline decided (crossings.csv)
    python scripts/ab_demographics.py score    --out output/ab  [--skip-vlm]
        runs CLIP (single crop + the live vote), SigLIP 2 and Qwen2.5-VL on every crop
        -> predictions.csv
    python scripts/ab_demographics.py sheet    --out output/ab
        contact sheets (numbered) + labels.csv template for hand labelling
    python scripts/ab_demographics.py report   --out output/ab
        accuracy per model / per camera against labels.csv (rows left blank are skipped)
    python scripts/ab_demographics.py votes    --out output/ab
        replays each track's recorded observations under different vote schemes
        (unweighted / crop-height weighted / top-N largest) against the same labels

Models download from the HuggingFace hub on first use (SigLIP 2 base ~1.4 GB,
Qwen2.5-VL-3B ~7 GB). --vlm Qwen/Qwen2.5-VL-7B-Instruct on a 16 GB+ GPU.

Typical run on a site's own footage:

    python scripts/ab_demographics.py collect --out output/ab --sources <clips or RTSP>
    python scripts/ab_demographics.py sheet   --out output/ab      # then label output/ab/labels.csv
    python scripts/ab_demographics.py votes   --out output/ab      # which vote scheme is best here
    python scripts/ab_demographics.py score   --out output/ab      # which model is best here
    python scripts/ab_demographics.py report  --out output/ab

`collect` is deterministic for video files: the same clips produce the same
crossings, so labels carry over between runs and A/B comparisons are exact.
Leave a row blank in labels.csv when the crop is too unclear to call - unlabelled
rows are skipped rather than guessed.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

GENDER_WORD = {"M": "male", "F": "female", "?": "unknown"}
AGE_WORD = {"A": "adult", "C": "child", "?": "unknown"}


# ----------------------------------------------------------------------------- collect
def cmd_collect(a: argparse.Namespace) -> None:
    from humanmonitor.counter import TrackState
    from humanmonitor.engine import Engine, EngineConfig

    TrackState.keep_history = True          # record every classification so vote schemes can be replayed offline
    out = Path(a.out); (out / "crops").mkdir(parents=True, exist_ok=True); (out / "context").mkdir(exist_ok=True)
    cfg = EngineConfig(sources=a.sources, no_loop=True, initial_count=0, device=a.device, conf=a.conf, config=a.config,
                       max_cameras=max(3, len(a.sources)), min_slots=len(a.sources))
    eng = Engine(cfg)
    eng.keep_crops = True
    rows: list[dict] = []
    idx = [0]

    def on_event(rec):
        if rec.crop is None:
            return
        i = idx[0]; idx[0] += 1
        name = f"{i:04d}_cam{rec.cam_idx + 1}_t{rec.event.track_id}"
        cv2.imwrite(str(out / "crops" / f"{name}.jpg"), rec.crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if rec.context is not None:
            cv2.imwrite(str(out / "context" / f"{name}.jpg"), rec.context, [cv2.IMWRITE_JPEG_QUALITY, 92])
        ev = rec.event
        rows.append({"id": i, "file": f"{name}.jpg", "camera": rec.cam_name, "track": ev.track_id,
                     "direction": "entry" if ev.direction > 0 else "exit",
                     "crop_w": rec.crop.shape[1], "crop_h": rec.crop.shape[0],
                     "live_gender": ev.gender, "live_age": ev.age,
                     "live_p_male": round(ev.p_male, 3), "live_p_child": round(ev.p_child, 3), "live_n_obs": ev.n_obs})

    obs_history: dict[int, list] = {}

    def on_event_obs(rec):
        """Snapshot the track's observation history at the moment of the crossing."""
        if rec.crop is None:
            return
        cam = eng.cameras[rec.cam_idx] if rec.cam_idx < len(eng.cameras) else None
        st = cam.counter.get_state(rec.event.track_id) if cam and cam.counter else None
        obs_history[idx[0] - 1] = list(st.obs_log or []) if st else []

    eng.listeners.append(on_event)
    eng.listeners.append(on_event_obs)
    eng.start()
    t0 = time.time()
    print(f"[collect] processing {len(eng.cameras)} video(s) once - this takes about the clips' total length")
    while not all(c.source.ended for c in eng.cameras) and eng.error is None:
        time.sleep(1.0)
        if int(time.time() - t0) % 30 == 0:
            print(f"[collect] {len(rows)} crossings so far, {eng.pipeline_fps:.1f} fps", flush=True)
    time.sleep(2.0)
    # apply the pipeline's later revisions so 'live' = what the counters finally held
    revised = {(cam_idx, rev.track_id): rev for _, cam_idx, rev in eng.revisions}
    names = {c.idx: c.name for c in eng.cameras}
    for r in rows:
        key = next((k for k in revised if names.get(k[0]) == r["camera"] and k[1] == r["track"]), None)
        if key:
            r["live_gender"], r["live_age"] = revised[key].gender, revised[key].age
            r["live_p_male"], r["live_p_child"], r["live_n_obs"] = round(revised[key].p_male, 3), round(revised[key].p_child, 3), revised[key].n_obs
    eng.shutdown()
    with (out / "crossings.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    (out / "observations.json").write_text(json.dumps({str(k): v for k, v in obs_history.items()}))
    n_obs = sum(len(v) for v in obs_history.values())
    print(f"[collect] {len(rows)} crossings ({n_obs} observations) -> {out / 'crossings.csv'}")


# ------------------------------------------------------------------------------- score
def _need(path: Path, how: str) -> Path:
    """Exit with a usable message instead of a traceback when a step was skipped."""
    if not path.exists():
        raise SystemExit(f"{path} not found - run `python scripts/ab_demographics.py {how}` first")
    return path


def _load(out: Path) -> list[dict]:
    with _need(out / "crossings.csv", f"collect --out {out}").open() as fh:
        return list(csv.DictReader(fh))


def score_clip(rows, out: Path, device: str) -> dict:
    from humanmonitor.classifier import DemographicClassifier
    clf = DemographicClassifier("openai/clip-vit-base-patch32", device)
    res = {}
    t0 = time.time()
    for i in range(0, len(rows), 16):
        batch = rows[i:i + 16]
        crops = [cv2.imread(str(out / "crops" / r["file"])) for r in batch]
        for r, (pm, pc) in zip(batch, clf.classify(crops)):
            res[r["id"]] = ("M" if pm >= 0.5 else "F", "C" if pc >= 0.6 else "A", pm, pc)
    print(f"[score] CLIP B/32 single-crop: {len(rows)} crops in {time.time() - t0:.1f}s")
    return res


def score_siglip(rows, out: Path, device: str, model_name: str) -> dict:
    """SigLIP 2 zero-shot with the same prompt sets as classifier.py (sigmoid scores -> two-way softmax)."""
    import torch
    from transformers import AutoModel, AutoProcessor
    from humanmonitor.classifier import MALE_PROMPTS, FEMALE_PROMPTS, ADULT_PROMPTS, CHILD_PROMPTS, _as_tensor
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    proc = AutoProcessor.from_pretrained(model_name)
    prompts = MALE_PROMPTS + FEMALE_PROMPTS + ADULT_PROMPTS + CHILD_PROMPTS
    nm, nf, na = len(MALE_PROMPTS), len(FEMALE_PROMPTS), len(ADULT_PROMPTS)
    with torch.no_grad():
        t = proc(text=prompts, padding="max_length", max_length=64, return_tensors="pt").to(device)
        text = _as_tensor(model.get_text_features(**t))
        text = text / text.norm(dim=-1, keepdim=True)
        scale, bias = model.logit_scale.exp(), model.logit_bias
    res = {}
    t0 = time.time()
    for i in range(0, len(rows), 16):
        batch = rows[i:i + 16]
        imgs = [cv2.cvtColor(cv2.imread(str(out / "crops" / r["file"])), cv2.COLOR_BGR2RGB) for r in batch]
        with torch.no_grad():
            x = proc(images=imgs, return_tensors="pt").to(device)
            f = _as_tensor(model.get_image_features(**x))
            f = f / f.norm(dim=-1, keepdim=True)
            logits = f @ text.T * scale + bias
            g = torch.softmax(logits[:, : nm + nf], dim=-1); a_ = torch.softmax(logits[:, nm + nf:], dim=-1)
            pm = g[:, :nm].sum(-1); pc = a_[:, na:].sum(-1)
        for r, m, c in zip(batch, pm.tolist(), pc.tolist()):
            res[r["id"]] = ("M" if m >= 0.5 else "F", "C" if c >= 0.6 else "A", m, c)
    print(f"[score] SigLIP2 ({model_name.split('/')[-1]}): {len(rows)} crops in {time.time() - t0:.1f}s")
    return res


def score_vlm(rows, out: Path, device: str, model_name: str, folder: str) -> dict:
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    dtype = torch.float16 if device != "cpu" else torch.float32
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, dtype=dtype).to(device).eval()
    proc = AutoProcessor.from_pretrained(model_name, min_pixels=64 * 28 * 28, max_pixels=384 * 28 * 28)
    question = ("This is a CCTV crop of one person. Answer with JSON only: "
                '{"gender": "male" or "female", "age": "adult" or "child"}. '
                "A child is under about 12 years old. If unsure, give your best guess.")
    res = {}
    t0 = time.time()
    for n, r in enumerate(rows):
        img = Image.open(out / folder / r["file"]).convert("RGB")
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=[img], return_tensors="pt").to(device)
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=32, do_sample=False)
        ans = proc.batch_decode(ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].lower()
        g = "M" if re.search(r"\bmale", ans) and not re.search(r"\bfemale", ans) else ("F" if "female" in ans else "?")
        a_ = "C" if "child" in ans else ("A" if "adult" in ans else "?")
        res[r["id"]] = (g, a_, ans.strip().replace("\n", " ")[:80], "")
        if n % 20 == 0:
            print(f"[score] VLM {n}/{len(rows)}  {ans.strip()[:60]!r}  ({(time.time() - t0) / max(1, n):.1f}s/crop)", flush=True)
    print(f"[score] {model_name.split('/')[-1]} on {folder}: {len(rows)} crops in {time.time() - t0:.0f}s")
    return res


def cmd_score(a: argparse.Namespace) -> None:
    out = Path(a.out); rows = _load(out)
    preds = {r["id"]: {} for r in rows}
    for r in rows:
        preds[r["id"]]["live"] = (r["live_gender"], r["live_age"])
    for name, res in (("clip", score_clip(rows, out, a.device)),
                      ("siglip2", score_siglip(rows, out, a.device, a.siglip))):
        for k, v in res.items():
            preds[k][name] = v[:2]; preds[k][name + "_p"] = (round(v[2], 3), round(v[3], 3))
    if not a.skip_vlm:
        for name, folder in (("vlm_crop", "crops"), ("vlm_context", "context")):
            for k, v in score_vlm(rows, out, a.device, a.vlm, folder).items():
                preds[k][name] = v[:2]; preds[k][name + "_raw"] = v[2]
    with (out / "predictions.csv").open("w", newline="") as fh:
        cols = ["id", "file", "camera", "direction", "crop_h", "live_gender", "live_age", "live_n_obs",
                "clip_gender", "clip_age", "clip_p_male", "clip_p_child",
                "siglip2_gender", "siglip2_age", "siglip2_p_male", "siglip2_p_child",
                "vlm_crop_gender", "vlm_crop_age", "vlm_context_gender", "vlm_context_age", "vlm_raw"]
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        for r in rows:
            p = preds[r["id"]]
            w.writerow({"id": r["id"], "file": r["file"], "camera": r["camera"], "direction": r["direction"], "crop_h": r["crop_h"],
                        "live_gender": r["live_gender"], "live_age": r["live_age"], "live_n_obs": r["live_n_obs"],
                        "clip_gender": p["clip"][0], "clip_age": p["clip"][1], "clip_p_male": p["clip_p"][0], "clip_p_child": p["clip_p"][1],
                        "siglip2_gender": p["siglip2"][0], "siglip2_age": p["siglip2"][1], "siglip2_p_male": p["siglip2_p"][0], "siglip2_p_child": p["siglip2_p"][1],
                        "vlm_crop_gender": p.get("vlm_crop", ("", ""))[0], "vlm_crop_age": p.get("vlm_crop", ("", ""))[1],
                        "vlm_context_gender": p.get("vlm_context", ("", ""))[0], "vlm_context_age": p.get("vlm_context", ("", ""))[1],
                        "vlm_raw": p.get("vlm_context_raw", "")})
    print(f"[score] -> {out / 'predictions.csv'}")


# ------------------------------------------------------------------------------- sheet
def cmd_sheet(a: argparse.Namespace) -> None:
    out = Path(a.out); rows = _load(out)
    (out / "sheets").mkdir(exist_ok=True)
    per, cols, cell = 36, 6, (200, 420)
    for s in range(0, len(rows), per):
        chunk = rows[s:s + per]
        tiles = []
        for r in chunk:
            im = cv2.imread(str(out / "crops" / r["file"]))   # the tight crop is what the models see
            h, w = im.shape[:2]; sc = min(cell[0] / w, (cell[1] - 26) / h)
            im = cv2.resize(im, (max(1, int(w * sc)), max(1, int(h * sc))))
            canvas = np.full((cell[1], cell[0], 3), 40, np.uint8)
            canvas[26:26 + im.shape[0], (cell[0] - im.shape[1]) // 2:(cell[0] - im.shape[1]) // 2 + im.shape[1]] = im
            cv2.putText(canvas, f"#{r['id']}  cam{r['camera'][3:4] if r['camera'].startswith('cam') else ''} {r['direction']}", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            tiles.append(canvas)
        while len(tiles) % cols:
            tiles.append(np.full((cell[1], cell[0], 3), 40, np.uint8))
        grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
        cv2.imwrite(str(out / "sheets" / f"sheet_{s // per:02d}.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
    tpl = out / "labels.csv"
    if not tpl.exists():
        with tpl.open("w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["id", "file", "camera", "gender", "age", "note"])
            for r in rows:
                w.writerow([r["id"], r["file"], r["camera"], "", "", ""])
    print(f"[sheet] {len(rows)} crops in {out / 'sheets'} - fill gender (M/F) and age (A/C) in {tpl}; leave unclear ones blank")


# ------------------------------------------------------------------------------ report
def cmd_report(a: argparse.Namespace) -> None:
    out = Path(a.out)
    with _need(out / "predictions.csv", f"score --out {out}").open() as fh:
        preds = {r["id"]: r for r in csv.DictReader(fh)}
    with _need(out / "labels.csv", f"sheet --out {out}").open() as fh:
        labels = [r for r in csv.DictReader(fh) if r["gender"].strip().upper() in ("M", "F") or r["age"].strip().upper() in ("A", "C")]
    if not labels:
        print("no labels yet - fill output/ab/labels.csv first (gender M/F, age A/C)"); return
    models = [("live pipeline (track vote)", "live"), ("CLIP B/32 single crop", "clip"), ("SigLIP 2 single crop", "siglip2"),
              ("Qwen2.5-VL crop", "vlm_crop"), ("Qwen2.5-VL context crop", "vlm_context")]
    cams = sorted({r["camera"] for r in labels})
    lines = []

    def acc(keyfn, attr, lab_key):
        n = ok = 0
        for r in labels:
            truth = r[lab_key].strip().upper()
            if truth not in ("M", "F", "A", "C"):
                continue
            p = keyfn(preds[r["id"]])
            if not p:
                continue
            n += 1; ok += p == truth
        return ok, n

    lines.append(f"labelled crossings: {len(labels)} (gender {sum(r['gender'].strip().upper() in 'MF' and r['gender'].strip() != '' for r in labels)}, age {sum(r['age'].strip().upper() in 'AC' and r['age'].strip() != '' for r in labels)})")
    lines.append(""); lines.append(f"{'model':30s} {'gender acc':>12s} {'age acc':>12s}   " + "  ".join(f"{c[:18]:>18s}" for c in cams))
    for title, key in models:
        if not any(preds[r["id"]].get(f"{key}_gender") for r in labels):
            continue
        g = acc(lambda p: p.get(f"{key}_gender", ""), "gender", "gender")
        ag = acc(lambda p: p.get(f"{key}_age", ""), "age", "age")
        percam = []
        for c in cams:
            sub = [r for r in labels if r["camera"] == c]
            gg = sum(preds[r["id"]].get(f"{key}_gender") == r["gender"].strip().upper() for r in sub if r["gender"].strip())
            gn = sum(1 for r in sub if r["gender"].strip())
            aa = sum(preds[r["id"]].get(f"{key}_age") == r["age"].strip().upper() for r in sub if r["age"].strip())
            an = sum(1 for r in sub if r["age"].strip())
            percam.append(f"g {100 * gg / gn if gn else 0:3.0f}% a {100 * aa / an if an else 0:3.0f}%")
        lines.append(f"{title:30s} {100 * g[0] / max(1, g[1]):9.1f}% ({g[1]:3d}) {100 * ag[0] / max(1, ag[1]):9.1f}% ({ag[1]:3d})   " + "  ".join(f"{s:>18s}" for s in percam))
    # class balance and small-crop note
    lines.append("")
    gm = sum(r["gender"].strip().upper() == "M" for r in labels); gf = sum(r["gender"].strip().upper() == "F" for r in labels)
    aa_ = sum(r["age"].strip().upper() == "A" for r in labels); ac = sum(r["age"].strip().upper() == "C" for r in labels)
    lines.append(f"label balance: male {gm} / female {gf}   adult {aa_} / child {ac}   (majority-class baseline: gender {100 * max(gm, gf) / max(1, gm + gf):.0f}%, age {100 * max(aa_, ac) / max(1, aa_ + ac):.0f}%)")
    small = [r for r in labels if int(preds[r["id"]]["crop_h"]) < 120]
    lines.append(f"crops under 120 px tall: {len(small)} of {len(labels)}")
    text = "\n".join(lines)
    (out / "report.txt").write_text(text + "\n"); print(text)


# ------------------------------------------------------------------------- votes
def _vote(obs: list, scheme: str, param: float) -> tuple[float, float]:
    """Combine one track's observations [(frame, p_male, p_child, height), ...] into a vote."""
    if not obs:
        return 0.5, 0.5
    if scheme == "topn":
        obs = sorted(obs, key=lambda o: -o[3])[: max(1, int(param))]
        ws = [1.0] * len(obs)
    elif scheme == "lastn":
        obs = obs[-max(1, int(param)):]
        ws = [1.0] * len(obs)
    else:                                   # power: 0 = unweighted mean, 1 = linear in height, 2 = area-like
        ws = [max(o[3], 32.0) ** param for o in obs]
    tw = sum(ws) or 1.0
    return (sum(w * o[1] for w, o in zip(ws, obs)) / tw,
            sum(w * o[2] for w, o in zip(ws, obs)) / tw)


def cmd_votes(a: argparse.Namespace) -> None:
    out = Path(a.out)
    obs_all = json.loads(_need(out / "observations.json", f"collect --out {out}").read_text())
    labels = [r for r in csv.DictReader(_need(out / "labels.csv", f"sheet --out {out}").open()) if r["gender"].strip() or r["age"].strip()]
    if not labels:
        print("no labels yet - run 'sheet' and fill labels.csv first"); return
    schemes = [("unweighted mean (before)", "power", 0.0), ("weighted by height", "power", 1.0),
               ("weighted by height^2", "power", 2.0), ("largest crop only", "topn", 1),
               ("3 largest crops", "topn", 3), ("5 largest crops", "topn", 5), ("last 5 crops", "lastn", 5)]
    print(f"{'vote scheme':26s} {'gender':>10s} {'age':>10s}   (n = {len(labels)} labelled crossings)")
    best = None
    for name, scheme, param in schemes:
        g_ok = g_n = a_ok = a_n = 0
        for r in labels:
            obs = obs_all.get(str(r["id"]), [])
            pm, pc = _vote(obs, scheme, param)
            if r["gender"].strip():
                g_n += 1; g_ok += ("M" if pm >= a.gender_thresh else "F") == r["gender"].strip().upper()
            if r["age"].strip():
                a_n += 1; a_ok += ("C" if pc >= a.child_thresh else "A") == r["age"].strip().upper()
        g = 100 * g_ok / max(1, g_n); ag = 100 * a_ok / max(1, a_n)
        print(f"{name:26s} {g:9.1f}% {ag:9.1f}%")
        if best is None or g + ag > best[0]:
            best = (g + ag, name)
    print(f"\nbest combined: {best[1]}")
    # how many observations per track, and how much of the weight the biggest crop carries
    sizes = [len(obs_all.get(str(r["id"]), [])) for r in labels]
    print(f"observations per track: median {sorted(sizes)[len(sizes)//2]}, min {min(sizes)}, max {max(sizes)}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect"); c.add_argument("--out", default="output/ab")
    c.add_argument("--sources", nargs="+", default=["sample_data/videos/cam1_store_entrance.mp4", "sample_data/videos/cam2_street_corner.mp4", "sample_data/videos/cam3_retail_shop.mp4"])
    c.add_argument("--device", default="auto"); c.add_argument("--conf", type=float, default=0.25)
    c.add_argument("--config", default="config/lines.json", help="trigger line config")
    s = sub.add_parser("score"); s.add_argument("--out", default="output/ab"); s.add_argument("--device", default=None)
    s.add_argument("--siglip", default="google/siglip2-base-patch16-256"); s.add_argument("--vlm", default="Qwen/Qwen2.5-VL-3B-Instruct")
    s.add_argument("--skip-vlm", action="store_true")
    sh = sub.add_parser("sheet"); sh.add_argument("--out", default="output/ab")
    r = sub.add_parser("report"); r.add_argument("--out", default="output/ab")
    v = sub.add_parser("votes"); v.add_argument("--out", default="output/ab")
    v.add_argument("--gender-thresh", type=float, default=0.5); v.add_argument("--child-thresh", type=float, default=0.6)
    a = p.parse_args()
    if a.cmd == "score" and a.device is None:
        from humanmonitor.engine import pick_device
        a.device = pick_device("auto")
    {"collect": cmd_collect, "score": cmd_score, "sheet": cmd_sheet, "report": cmd_report, "votes": cmd_votes}[a.cmd](a)


if __name__ == "__main__":
    main()
