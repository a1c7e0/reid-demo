"""Line calibration under the PRODUCT objective (clustering, not pairs).

Asymmetric cost (the production preference):
  FM  (different identities in one cluster) = the accident -> heavy
  FS  (same identity split over clusters)   = acceptable -> free
  UN  (unassigned)                          = neutral

Data: identity-labeled images with embeddings already cached:
  /s2/memb_<model>.npy + /s2/model_bench_crops.npz   (LFW sample)
  /s2/hemb_<model>.npy + /s2/hard_crops.npz          (CFP, KinFaceW,
      CALFW, CPLFW, TinyFace crops)
Identity comes from the file path. Runs the REAL app core (ente_core)
over each set at a threshold grid and scores cluster structure.

Usage: python calib_bench.py <model> [t_good ...]
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/app")

from app.ente_core import EnteParams, cluster_faces, complete_link_merge

MODEL = sys.argv[1] if len(sys.argv) > 1 else "arcface_r50"
MERGE = "--merge" in sys.argv
MERGE_BAR = float(__import__("os").environ.get("MERGE_BAR", "0"))
if MERGE:
    sys.argv = [a for a in sys.argv if a != "--merge"]
GRID = [float(x) for x in sys.argv[2:]] or [round(x, 2) for x in np.arange(0.55, 0.86, 0.05)]


def identity_of(path: str) -> str:
    p = path.replace("\\", "/")
    name = os.path.basename(p)
    if "/cfp/" in p:
        parts = p.split("/")
        return parts[-3]                      # Data/Images/<id>/frontal/x.jpg
    if "/kinfacew/" in p:
        return name.rsplit("_", 1)[0]         # fd_024_1.jpg -> fd_024
    if "/tinyface/" in p:
        return name.split("_")[0]
    return name.rsplit("_", 2)[0] if name.count("_") >= 2 else name.rsplit("_", 1)[0]


def load_sets():
    sets = {}
    m = np.load("/s2/model_bench_crops.npz", allow_pickle=True)
    emb = np.load(f"/s2/memb_{MODEL}.npy")
    labels = [str(x) for x in m["label"]]
    sets["lfw"] = (emb, np.array(labels), np.arange(len(labels)))
    h = np.load("/s2/hard_crops.npz", allow_pickle=True)
    hemb = np.load(f"/s2/hemb_{MODEL}.npy")
    paths = [str(x) for x in h["paths"]]
    ids = [identity_of(p) for p in paths]
    groups = {}
    for i, p in enumerate(paths):
        key = "kinfacew" if "/kinfacew/" in p else ("cfp" if "/cfp/" in p else
              ("calfw" if "/calfw/" in p else ("cplfw" if "/cplfw/" in p else "tinyface")))
        groups.setdefault(key, []).append(i)
    for key, idx in groups.items():
        sets[key] = (hemb[idx], np.array([ids[i] for i in idx]),
                     np.arange(len(idx)))
    return sets


def score_clusters(cid, labels):
    n = len(labels)
    FM = FS = HIT = same = 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = cid[i], cid[j]
            if labels[i] == labels[j]:
                same += 1
                if a >= 0 and a == b:
                    HIT += 1
                elif a >= 0 and b >= 0 and a != b:
                    FS += 1
            else:
                if a >= 0 and a == b:
                    FM += 1
    sizes = {}
    for c in cid:
        if c >= 0:
            sizes[c] = sizes.get(c, 0) + 1
    return FM, FS, HIT, same, len(sizes), (max(sizes.values()) if sizes else 0)


def main():
    sets = load_sets()
    print("sets:", {k: (v[0].shape, len(set(v[1].tolist()))) for k, v in sets.items()})
    for tg in GRID:
        tb = round(tg + 0.08, 2)
        tot = dict(FM=0, FS=0, HIT=0, same=0)
        per = {}
        for name, (emb, labels, photo) in sets.items():
            n = len(emb)
            blur = np.full(n, 10000.0)
            score = np.full(n, 0.95)
            sideways = np.zeros(n, dtype=bool)
            times = photo.astype(np.float64)
            cid, _ = cluster_faces(emb, blur, score, sideways, photo, times,
                                   EnteParams(tg, tb))
            FM, FS, HIT, same, persons, biggest = score_clusters(cid, labels)
            per[name] = dict(FM=FM, FS=FS, HIT=HIT, persons=persons, biggest=biggest)
            tot["FM"] += FM
            tot["FS"] += FS
            tot["HIT"] += HIT
            tot["same"] += same
            if MERGE:
                cid2 = complete_link_merge(emb, cid.copy(), MERGE_BAR or tg, photo)
                FM2, FS2, HIT2, same2, persons2, biggest2 = score_clusters(cid2, labels)
                per[name] = dict(FM=FM2, FS=FS2, HIT=HIT2, persons=persons2, biggest=biggest2)
                tot["FM"] = tot["FM"] - FM + FM2
                tot["FS"] = tot["FS"] - FS + FS2
                tot["HIT"] = tot["HIT"] - HIT + HIT2
        print(json.dumps(dict(t_good=tg, t_bad=tb, merge=MERGE, total=tot, per=per)), flush=True)


if __name__ == "__main__":
    main()
