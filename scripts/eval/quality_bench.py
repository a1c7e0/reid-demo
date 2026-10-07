"""Does the QUALITY layer earn its keep? (public data only)

The main bake-off passed score=0.95 / sideways=False everywhere, so the
quality-adaptive threshold and the clusterable gate were effectively
inactive. LFW's crop meta carries the REAL detector score and the real 5
landmarks, so this isolates the quality layer with production-consistent
inputs:

    blur      - Laplacian variance of the aligned crop (face.py's formula)
    score     - detector confidence recorded at detection time
    sideways  - Ente's nose-extends-past-eye-line test on those landmarks

Configs differ only in the quality layer; t_good/t_bad are swept alongside
it because a quality gate only makes sense together with the bar it feeds.

Reports per config: impure units %, recall, unassigned count, FM pairs.
"""
import json
import sys

import numpy as np

sys.path.insert(0, "/app")

from app.ente_core import (EnteParams, cluster_faces, face_direction,
                           is_bad_face, is_clusterable)

T_GOODS = [0.55, 0.60, 0.65, 0.70]
GAPS = [0.00, 0.04, 0.08, 0.12]


def load_lfw():
    m = np.load("/s2/model_bench_crops.npz", allow_pickle=True)
    emb = np.load("/s2/memb_mobilefacenet_w600k.npy")
    crops = m["crops"]
    lab = np.array([str(x) for x in m["label"]])
    meta = json.loads(str(m["meta"]))          # [[score, [5 x,y]], ...]

    import cv2
    blur = np.empty(len(crops), dtype=np.float32)
    score = np.empty(len(crops), dtype=np.float32)
    sid = np.zeros(len(crops), dtype=bool)
    for i, c in enumerate(crops):
        g = cv2.cvtColor(c, cv2.COLOR_RGB2GRAY) if c.ndim == 3 else c
        blur[i] = cv2.Laplacian(g, cv2.CV_64F).var()
        s, lm = meta[i]
        score[i] = float(s)
        if lm and len(lm) == 5:
            # production sideways: the same faceDirection() the app uses
            sid[i] = face_direction(np.asarray(lm, dtype=np.float64)) != "straight"
    return emb, lab, blur, score, sid


def score_clusters(cid, lab):
    codes, inv = np.unique(lab, return_inverse=True)
    K = len(codes)
    valid = cid >= 0
    n = len(cid)
    _, allc = np.unique(inv, return_counts=True)
    same_pairs = int((allc * (allc - 1) // 2).sum())
    total_pairs = n * (n - 1) // 2
    diff_pairs = total_pairs - same_pairs
    FM = HIT = impure_units = clustered = 0
    if valid.any():
        v = inv[valid]
        code = cid[valid].astype(np.int64) * K + v
        u, cnt = np.unique(code, return_counts=True)
        cl_of = u // K
        uniq_cl, cl_cnt = np.unique(cid[valid], return_counts=True)
        starts = np.flatnonzero(np.r_[True, cl_of[1:] != cl_of[:-1]])
        distinct = np.diff(np.r_[starts, len(u)])
        same_in = int((cnt * (cnt - 1) // 2).sum())
        any_in = int((cl_cnt * (cl_cnt - 1) // 2).sum())
        FM, HIT = any_in - same_in, same_in
        impure_units = int(cl_cnt[distinct > 1].sum())
        clustered = int(valid.sum())
    return dict(FM=FM, HIT=HIT, same=same_pairs, diff=diff_pairs,
                impure_units=impure_units, clustered=clustered,
                unassigned=int((cid < 0).sum()),
                recall=HIT / same_pairs if same_pairs else 0.0,
                impure_pct=100 * impure_units / max(clustered, 1))


def main():
    emb, lab, blur, score, sid = load_lfw()
    n = len(emb)
    photo = np.arange(n)
    times = photo.astype(np.float64)
    print(json.dumps(dict(n=n, bad=int(sum(is_bad_face(blur[i], score[i], sid[i])
                                            for i in range(n))),
                          not_clusterable=int(sum(not is_clusterable(blur[i], score[i])
                                                  for i in range(n))),
                          sid=int(sid.sum()),
                          score_below_085=int((score < 0.85).sum()))),
          file=sys.stderr)

    G = emb @ emb.T
    rows = []
    for quality in ("off", "real"):
        if quality == "off":
            sc = np.full(n, 0.95); sw = np.zeros(n, dtype=bool)
        else:
            sc = score; sw = sid
        for tg in T_GOODS:
            for gap in GAPS:
                cid, _ = cluster_faces(emb, blur, sc, sw, photo, times,
                                       EnteParams(tg, round(tg + gap, 2)),
                                       gram=G)
                s = score_clusters(cid, lab)
                rows.append(dict(quality=quality, t_good=tg, t_bad=round(tg + gap, 2),
                                 **{k: (round(v, 4) if isinstance(v, float) else v)
                                    for k, v in s.items()}))
    for r in rows:
        print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
