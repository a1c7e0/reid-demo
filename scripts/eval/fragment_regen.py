"""Regenerate the LFW bench set with the SHIPPED face model and measure how
fragmented the clustering is, plus whether a safe complete-link merge pass
recovers the fragments without mixing anyone.

Why this exists: the user reports the app splits faces that look identical to
the eye (often blurry ones). Purity is fine - nobody is mixed - but the same
person lands in several clusters. Two candidate fixes are measured here:

  * a complete-link merge pass on top of the shipped clustering (two clusters
    fold together ONLY when every cross pair is >= bar, so it cannot mix anyone
    at that bar - the exact inverse of the "Mixed grouping?" breakup)
  * a lower join bar (t_good), which merges more but does risk mixing

Output is in the project's own units: mixed pairs (what a human would have to
break up by hand), split pairs, and same-person recall. A bar is worth
shipping when recall rises while mixed pairs stay flat.

Runs INSIDE the app image (models + app code present):

    docker run --rm --cpus=1 --memory=1200m \
      -v $PWD:/work -v $PWD/testsets/lfw_flat:/lfw:ro reid-gallery:latest \
      python /work/scripts/eval/fragment_regen.py
"""
from __future__ import annotations

import glob
import json
import random
import sys
import time

import numpy as np

sys.path.insert(0, "/app")
from app import ente_core as core          # noqa: E402
from app.ml.face import FaceEngine, align5  # noqa: E402

MODEL = "/models/LVFace-T_Glint360K.onnx"
OUT = "/work/testsets/out/fragment_lfw.npz"


def load_files(limit: int, min_per_id: int = 6):
    """Sample LFW: keep identities with enough photos so splits are measurable,
    then take up to 8 photos each."""
    by_id: dict[str, list[str]] = {}
    for p in sorted(glob.glob("/lfw/*.jpg")):
        name = p.rsplit("/", 1)[1][:-4]
        ident = name.rsplit("_", 1)[0]
        by_id.setdefault(ident, []).append(p)
    ids = [i for i, v in by_id.items() if len(v) >= min_per_id]
    random.seed(7)
    random.shuffle(ids)
    out = []
    for i in ids:
        out.extend(sorted(by_id[i])[:8])
        if len(out) >= limit:
            break
    return out


def embed(files):
    """The production path: FaceEngine.detect already aligns and embeds with
    whatever model the app ships, so the bench measures the SHIPPED stack
    rather than a re-implementation of it."""
    import cv2

    eng = FaceEngine()
    emb, lab, photo = [], [], []
    for k, p in enumerate(files):
        img = cv2.imread(p)
        if img is None:
            continue
        dets = eng.detect(img)
        if not dets:
            continue
        d = max(dets, key=lambda x: (x["box"][2] - x["box"][0]) * (x["box"][3] - x["box"][1]))
        v = np.asarray(d["feat"], dtype=np.float32)
        n = float(np.linalg.norm(v))
        if n < 1e-6:
            continue
        emb.append(v / n)
        lab.append(p.rsplit("/", 1)[1][:-4].rsplit("_", 1)[0])
        photo.append(p.rsplit("/", 1)[1][:-4])
        if (k + 1) % 100 == 0:
            print(f"  embedded {k + 1}/{len(files)}", flush=True)
    return np.asarray(emb, dtype=np.float32), np.asarray(lab), np.asarray(photo)


def pair_counts(clu, lab):
    iu = np.triu_indices(len(clu), 1)
    sl = lab[iu[0]] == lab[iu[1]]
    sc = clu[iu[0]] == clu[iu[1]]
    fm = int((~sl & sc).sum())          # mixed: one cluster, two people
    fs = int((sl & ~sc).sum())          # split: one person, two clusters
    keep = int((sl & sc).sum())
    return fm, fs, keep / max(1, keep + fs)


def main():
    files = load_files(2400)
    print(f"# sampling {len(files)} LFW faces", flush=True)
    t0 = time.time()
    emb, lab, photo = embed(files)
    print(f"# embedded {len(emb)} faces in {time.time() - t0:.0f}s", flush=True)
    try:
        import os
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        np.savez(OUT, emb=emb, lab=lab, photo=photo)
    except OSError as e:
        print(f"# (could not cache embeddings: {e})", flush=True)

    n = len(emb)
    zero = np.zeros(n, dtype=np.float32)
    good = np.full(n, 150.0, dtype=np.float32)     # above the blur gate
    strong = np.full(n, 0.99, dtype=np.float32)    # above the score gate

    print("#")
    print(f"# {n} faces, {len(set(lab))} identities, shipped model LVFace-T")
    print(f"{'t_good':>7} {'bar':>6} {'mixed':>7} {'split':>7} {'recall':>7} {'clusters':>9}")
    for t_good, t_bad in ((0.64, 0.72), (0.58, 0.66), (0.52, 0.60)):
        ids, _ = core.cluster_faces(
            emb, good, strong, zero, photo, zero,
            core.EnteParams(t_good, t_bad), one_per_photo=True, gram=emb @ emb.T)
        for bar in (None, 0.50, 0.55, 0.60):
            clu = ids if bar is None else core.complete_link_merge(emb, ids, bar, photo)
            fm, fs, rec = pair_counts(np.asarray(clu), lab)
            print(f"{t_good:>7.2f} {'none' if bar is None else f'{bar:.2f}':>6} "
                  f"{fm:>7} {fs:>7} {rec:>7.3f} {len(set(clu)):>9}", flush=True)


if __name__ == "__main__":
    main()
