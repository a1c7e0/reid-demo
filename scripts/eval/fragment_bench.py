"""Can the fragmentation be recovered WITHOUT introducing mixing?

The user's complaint: two visually near-identical (often blurry) faces end up
in different clusters. Purity is right, but to the eye it is over-split. Two
candidate fixes:

  1. lower the join bar (t_good) - merges more, but mixes some people
  2. a complete-link merge pass on top of the shipped clustering: two clusters
     fold together ONLY when every cross pair is >= bar, which cannot mix
     anyone at that bar (the inverse of the "Mixed grouping?" breakup)

Measured on LFW 500x10 (public, the project's tuning set) in the project's own
units: mixed pairs (FM, what the user would have to break up by hand), split
pairs (FS), and same-person recall. A bar is worth shipping when recall rises
while FM stays flat.

    python3 scripts/eval/fragment_bench.py
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path("/home/a1c7e0/workspace/reid-gallery")
sys.path.insert(0, str(ROOT))
from app import ente_core as core  # noqa: E402

WORK = pathlib.Path.home() / ".hermes/cache/scratch"
NPZ = WORK / "memb_500x10_w600k_mbf.npz"


def z_norm(a: np.ndarray) -> np.ndarray:
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-9)


def pair_counts(clu: np.ndarray, lab: np.ndarray):
    """FM = mixed pairs (one cluster, two people). FS = split pairs (one
    person, two clusters). recall = same-person pairs kept together."""
    iu = np.triu_indices(len(clu), 1)
    same_lab = lab[iu[0]] == lab[iu[1]]
    same_clu = clu[iu[0]] == clu[iu[1]]
    fm = int((~same_lab & same_clu).sum())
    fs = int((same_lab & ~same_clu).sum())
    keep = int((same_lab & same_clu).sum())
    rec = keep / max(1, keep + fs)
    return fm, fs, rec


def bench(bar, t_good: float, t_bad: float, photo: np.ndarray, emb: np.ndarray):
    n = len(emb)
    zero = np.zeros(n, dtype=np.float32)
    good = np.full(n, 150.0, dtype=np.float32)   # all faces above the blur gate
    strong = np.full(n, 0.99, dtype=np.float32)  # ...and above the score gate
    t0 = time.time()
    ids, nclu = core.cluster_faces(
        emb, good, strong, zero.copy(), photo, zero.copy(),
        core.EnteParams(t_good, t_bad), one_per_photo=True, gram=emb @ emb.T)
    merged = ids if bar is None else core.complete_link_merge(emb, ids, bar, photo)
    return merged, nclu, (time.time() - t0) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t-good", type=float, default=0.64)
    ap.add_argument("--t-bad", type=float, default=0.72)
    ap.add_argument("--model", default="w600k_mbf")
    args = ap.parse_args()

    z = np.load(WORK / "bench_lfw.npz")   # LFW with ground-truth labels
    emb = z_norm(z["emb"].astype(np.float32))
    lab = z["label"].astype(object)
    photo = z["photo"].astype(str)

    print("# fragment recovery: complete-link merge on top of shipped clustering")
    print(f"#   LFW labelled set ({len(emb)} faces), t_good={args.t_good} "
          f"t_bad={args.t_bad}")
    print("#   a bar is worth shipping when recall rises while mixed pairs stay flat")
    print("#")
    print(f"{'bar':>6} {'mixed':>7} {'split':>7} {'recall':>7} {'clusters':>9} {'e2e_ms':>7}")
    for bar in (None, 0.45, 0.50, 0.55):
        clu, nclu, e2e = bench(bar, args.t_good, args.t_bad, photo, emb)
        fm, fs, rec = pair_counts(np.asarray(clu), lab)
        print(f"{'none' if bar is None else f'{bar:.2f}':>6} {fm:>7} {fs:>7} "
              f"{rec:>7.3f} {len(set(clu)):>9} {e2e:>7.0f}")


if __name__ == "__main__":
    main()
