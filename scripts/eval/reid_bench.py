#!/usr/bin/env python3
"""Pairwise-cosine benchmark of person-ReID ONNX models on Market-1501.

Protocol
--------
- Sample ~600 images (default: 60 person ids x 10 images) with a fixed seed
  from Market-1501 ``query/`` + ``bounding_box_test/`` (person id = integer
  prefix of the filename; ids <= 0 -- junk/distractors -- are excluded).
- Embed every image with each model (preprocessing identical to
  ``app/ml/reid.py``: resize 256x128 (WxH), BGR->RGB, /255, ImageNet
  mean/std), L2-normalize, and score **all** pairs (i<j) with cosine
  similarity.
- Report same-person vs different-person distribution stats and
  FMR/TPR at fixed thresholds as JSON lines on stdout (and ``--out``).

Usage
-----
    python reid_bench.py --market-root /bench/testsets/Market-1501-v15.09.15 \
        --model name=path.onnx [--model name=path.onnx --in-domain ...] \
        --out results.jsonl

Each ``--model`` is ``name=path`` optionally followed by ``:in_domain`` to
flag a row whose training data includes Market-1501 (not a valid
cross-domain comparison row). Static-batch graphs (e.g. the deployed
osnet_x0_25 baseline, batch 16) are handled by padding the final batch.
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
INPUT_W, INPUT_H = 128, 256


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--market-root", required=True, type=Path)
    p.add_argument("--model", action="append", required=True,
                   metavar="NAME=PATH[:in_domain]",
                   help="model spec, repeatable")
    p.add_argument("--n-images", type=int, default=600)
    p.add_argument("--per-pid", type=int, default=10)
    p.add_argument("--min-pid-images", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--thresholds", type=float, nargs="+",
                   default=[0.55, 0.65, 0.74])
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--out", type=Path, default=None,
                   help="append JSON lines here as well as stdout")
    return p.parse_args()


def sample_images(market_root: Path, n_images: int, per_pid: int,
                  min_pid_images: int, seed: int):
    """Fixed-seed sample: person ids with >= min_pid_images images in
    query+bbox_test, per_pid images per id, exactly n_images total."""
    by_pid = {}
    for sub in ("query", "bounding_box_test"):
        d = market_root / sub
        for fname in sorted(os.listdir(d)):
            if not fname.lower().endswith(".jpg"):
                continue
            pid = int(fname.split("_")[0])
            if pid <= 0:  # junk (-1) and distractor (0000)
                continue
            by_pid.setdefault(pid, []).append(str(d / fname))

    rng = random.Random(seed)
    eligible = sorted(pid for pid, imgs in by_pid.items()
                      if len(imgs) >= max(min_pid_images, per_pid))
    if not eligible:
        raise SystemExit("no person ids with enough images")
    rng.shuffle(eligible)

    n_pids = n_images // per_pid
    if n_pids * per_pid != n_images:
        raise SystemExit("n-images must be divisible by per-pid")
    if len(eligible) < n_pids:
        raise SystemExit(f"only {len(eligible)} eligible ids, need {n_pids}")

    picked = []
    for pid in eligible[:n_pids]:
        imgs = sorted(by_pid[pid])
        picked.extend(rng.sample(imgs, per_pid))
    pids = [int(Path(p).name.split("_")[0]) for p in picked]
    return picked, pids


def preprocess(path: str) -> np.ndarray:
    import cv2

    img = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR, matches app input
    if img is None:
        raise SystemExit(f"failed to read {path}")
    img = cv2.resize(img, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
    img = img[:, :, ::-1].astype(np.float32) / 255.0  # BGR -> RGB
    img = (img - MEAN) / STD
    return img.transpose(2, 0, 1)


def embed(model_path: Path, batches_x: list, batch_hint: int):
    import onnxruntime as ort

    sess = ort.InferenceSession(str(model_path),
                                providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    static = inp.shape[0] if isinstance(inp.shape[0], int) and inp.shape[0] > 0 else None
    bs = static or batch_hint

    outs = []
    t0 = time.time()
    x_all = np.concatenate(batches_x, axis=0)
    for i in range(0, len(x_all), bs):
        chunk = x_all[i:i + bs]
        n = len(chunk)
        if static and n < bs:  # pad static-batch graph, drop padding rows
            pad = np.zeros((bs - n,) + chunk.shape[1:], dtype=np.float32)
            chunk = np.concatenate([chunk, pad], axis=0)
        out = sess.run(None, {inp.name: chunk})[0]
        outs.append(out[:n])
    emb = np.concatenate(outs, axis=0).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    return emb, time.time() - t0


def evaluate(emb: np.ndarray, pids: list, thresholds):
    n = len(pids)
    sim = emb @ emb.T
    iu = np.triu_indices(n, k=1)
    s = sim[iu]
    pid_arr = np.asarray(pids)
    same_mask = pid_arr[iu[0]] == pid_arr[iu[1]]
    same = s[same_mask]
    diff = s[~same_mask]

    row = {
        "n_images": int(n),
        "n_same_pairs": int(same.size),
        "n_diff_pairs": int(diff.size),
        "same_p50": float(np.percentile(same, 50)),
        "same_p10": float(np.percentile(same, 10)),
        "diff_p50": float(np.percentile(diff, 50)),
        "diff_p99": float(np.percentile(diff, 99)),
        "diff_max": float(diff.max()),
    }
    for t in thresholds:
        tag = f"{t:.2f}"
        row[f"tpr_{tag}"] = float((same >= t).mean())   # same pair scores >= t
        row[f"fmr_{tag}"] = float((diff >= t).mean())   # diff pair scores >= t
    return row


def main():
    args = parse_args()
    picked, pids = sample_images(args.market_root, args.n_images,
                                 args.per_pid, args.min_pid_images, args.seed)

    meta = {
        "type": "meta",
        "dataset": args.market_root.name,
        "dirs": ["query", "bounding_box_test"],
        "seed": args.seed,
        "n_images": len(picked),
        "n_pids": len(set(pids)),
        "per_pid": args.per_pid,
        "thresholds": args.thresholds,
        "preprocess": "resize 256x128 WxH, BGR->RGB, /255, "
                      "(x-ImageNet-mean)/ImageNet-std (app/ml/reid.py)",
        "distance": "cosine on L2-normalized embeddings; all i<j pairs",
    }

    def emit(obj):
        line = json.dumps(obj)
        print(line, flush=True)
        if args.out:
            with open(args.out, "a") as f:
                f.write(line + "\n")

    emit(meta)

    batches = [np.stack([preprocess(p) for p in picked[i:i + args.batch]])
               for i in range(0, len(picked), args.batch)]

    for spec in args.model:
        in_domain = spec.endswith(":in_domain")
        spec = spec[: -len(":in_domain")] if in_domain else spec
        name, path = spec.split("=", 1)
        path = Path(path)
        t0 = time.time()
        emb, infer_s = embed(path, batches, args.batch)
        row = evaluate(emb, pids, args.thresholds)
        row.update({
            "type": "result",
            "model": name,
            "path": str(path),
            "size_mb": round(path.stat().st_size / 1e6, 2),
            "dim": int(emb.shape[1]),
            "in_domain": in_domain,
            "infer_seconds": round(infer_s, 1),
            "wall_seconds": round(time.time() - t0, 1),
        })
        emit(row)

    print("BENCH DONE", flush=True)


if __name__ == "__main__":
    main()
