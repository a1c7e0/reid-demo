"""A/B benchmark for the body-attach fusion layer (public re-id data only).

`fusion.body_attach` lets a body detection join a FACE-anchored person when
its embedding clears BODY_JOIN_T, inside a time window, and only if that
person is not already present in the same photo. Attached vectors then join
the person's evidence (chaining). None of that is measured by
`market_reid.py` (plain retrieval rank/mAP), and the user's album is not
used for measurement - so this simulates the gallery regime on Market-1501:

    each sampled identity contributes
      anchors  - images whose body vector stands in for "a photo where we
                 saw this person's face" (the seed evidence)
      queries  - the rest, standing in for faceless/back-view detections

Objective is the same asymmetric one as the face side: attaching the wrong
person is a MIX (heavy), failing to attach is a split (free). The headline
numbers are therefore `wrong` share among attaches and the share of persons
that end up holding more than one identity.

Dimensions swept:
    -t        BODY_JOIN_T (0.60 .. 0.82)
    -agg      how a query is scored against a person: max (shipped) / mean /
              median / top2   (max is single-linkage-ish: one look-alike
              vector is enough, so it is the risky one)
    -chain    whether an attached vector is added to the person's evidence
              (shipped: yes) or the seed set stays frozen
    -anchors  1 or 3 face-photos per identity
    -cooc     virtual group photos: two identities share a photo, which
              exercises the co-occurrence block (synthetic time window: on/off)

Usage (inside the app container):
    python /app/scripts/eval/fusion_bench.py --stage core
    python /app/scripts/eval/fusion_bench.py --stage model
    python /app/scripts/eval/fusion_bench.py --stage cooc
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/app")

MARKET = Path("/t/Market-1501-v15.09.15")
OUT = Path("/s2")
PAT = re.compile(r"^(-?\d+)_c\d+")
SEED = 7
N_ID = 300          # identities sampled from the Market test split
IMG_PER_ID = 6      # images taken per identity (rest are unused)
TIME_GAP = 6 * 3600  # BODY_MAX_GAP_S


# ------------------------------------------------------------------ data ---
def sample_market(n_id: int = N_ID, per_id: int = IMG_PER_ID):
    d = MARKET / "bounding_box_test"
    by_id: dict[int, list[str]] = {}
    for p in sorted(d.glob("*.jpg")):
        m = PAT.search(p.name)
        if not m:
            continue
        pid = int(m.group(1))
        if pid <= 0:
            continue
        by_id.setdefault(pid, []).append(str(p))
    rng = np.random.RandomState(SEED)
    ids = sorted(k for k, v in by_id.items() if len(v) >= per_id)
    ids = sorted(rng.choice(ids, n_id, replace=False))
    labs, paths = [], []
    for pid in ids:
        pick = rng.choice(by_id[pid], per_id, replace=False)
        for p in pick:
            labs.append(pid)
            paths.append(str(p))
    return np.asarray(labs), paths


def embed(model_path: str) -> np.ndarray:
    """Cached: production ReId preprocessing, L2-normalized (N, 512)."""
    from PIL import Image, ImageOps
    import cv2
    from app.ml.reid import ReId

    labs, paths = sample_market()
    cache = OUT / f"fusion_emb_{Path(model_path).stem}.npy"
    lab_cache = OUT / "fusion_labs.npy"
    if cache.exists() and lab_cache.exists() and np.array_equal(
            np.load(lab_cache), labs):
        print(f"  [fusion] cache hit {cache.name}", flush=True)
        return np.load(cache)
    print(f"  [fusion] embedding {len(paths)} crops with {Path(model_path).name}",
          flush=True)
    reid = ReId(model_path)
    out = []
    for i in range(0, len(paths), 64):
        buf = []
        for p in paths[i:i + 64]:
            with Image.open(p) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                buf.append(cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR))
        out.append(reid.embed(buf))
        if (i // 64) % 10 == 0:
            print(f"    {i}/{len(paths)}", flush=True)
    emb = np.concatenate(out)
    np.save(cache, emb)
    np.save(lab_cache, labs)
    return emb


# --------------------------------------------------------------- simulate ---
def agg_scores(q: np.ndarray, vecs: np.ndarray, how: str) -> float:
    s = q @ vecs.T
    if how == "max":
        return float(s.max())
    if how == "mean":
        return float(s.mean())
    if how == "median":
        return float(np.median(s))
    if how == "top2":
        return float(np.sort(s)[-2:].mean()) if len(s) >= 2 else float(s.max())
    raise ValueError(how)


def simulate(emb: np.ndarray, labs: np.ndarray, *, t: float, how: str,
             chain: bool, n_anchor: int, cooc: bool, time_win: bool,
             margin: float = 0.0, gap: int = TIME_GAP,
             seed: int = SEED) -> dict:
    """Replicates fusion.body_attach over a simulated gallery."""
    rng = np.random.RandomState(seed)
    ids = np.unique(labs)
    # anchors / queries per identity
    person_vecs: dict[int, np.ndarray] = {}
    person_photos: dict[int, set] = {}
    person_times: dict[int, list[float]] = {}
    queries: list[tuple[int, np.ndarray, int, float]] = []

    # simulated timestamps: identities live in "sessions" of a few hours,
    # spread across days - body evidence decays with the real 6h window
    time_of: dict[int, float] = {}
    for k, pid in enumerate(ids):
        # identities OVERLAP in time: everyone is out within the same 3-day
        # span, each for a ~2h outing. Separating identities in time would
        # hand the 6h window a free win - in a real album different people
        # are photographed on the same day, which is exactly when mixing
        # can happen.
        base = float(rng.uniform(0.0, 3 * 24 * 3600))
        span = float(rng.uniform(0.5, 2.0) * 3600)
        for j in range(IMG_PER_ID):
            time_of[(pid, j)] = base + (j / max(IMG_PER_ID - 1, 1)) * span

    photo_of: dict[int, int] = {}           # global image idx -> photo id
    next_photo = 0
    # optional virtual group photos: pair up identities so two people share
    # one photo (that is the rule the co-occurrence block exists for)
    order = rng.permutation(len(emb))
    idx_to_slot: dict[int, tuple[int, int]] = {}
    seen: dict[int, int] = {}
    for gi, ei in enumerate(order):
        pid = int(labs[ei])
        slot = seen.get(pid, 0)
        if slot < IMG_PER_ID:
            idx_to_slot[ei] = (pid, slot)
            seen[pid] = slot + 1
    slots = list(idx_to_slot.items())
    rng.shuffle(slots)
    i = 0
    while i < len(slots):
        if cooc and i + 1 < len(slots):
            # two different identities share one virtual photo
            photo_of[slots[i][0]] = next_photo
            photo_of[slots[i + 1][0]] = next_photo
            next_photo += 1
            i += 2
        else:
            photo_of[slots[i][0]] = next_photo
            next_photo += 1
            i += 1

    for ei, (pid, slot) in idx_to_slot.items():
        v = emb[ei]
        ph = photo_of.get(ei, ei)
        tt = time_of[(pid, slot)]
        if slot < n_anchor:
            person_vecs.setdefault(pid, []).append(v)
            person_photos.setdefault(pid, set()).add(ph)
            person_times.setdefault(pid, []).append(tt)
        else:
            queries.append((pid, v, ph, tt))

    for pid in list(person_vecs):
        person_vecs[pid] = np.stack(person_vecs[pid])

    # greedy attach in a fixed order (production uses rowid order)
    rng2 = np.random.RandomState(seed + 1)
    rng2.shuffle(queries)
    right = wrong = miss = 0
    tainted: set[int] = set()
    for pid_true, v, ph, tt in queries:
        best, best_pid, second = -2.0, None, -2.0
        for pid, vecs in person_vecs.items():
            if ph in person_photos.get(pid, ()):
                continue                       # co-occurrence hard rule
            if time_win and gap > 0 and person_times.get(pid):
                mid = float(np.mean(person_times[pid]))
                if abs(tt - mid) > gap:
                    continue                   # time-window bound
            s = agg_scores(v, vecs, how)
            if s > best:
                second = best
                best, best_pid = s, pid
            elif s > second:
                second = s
        if best_pid is None or best <= t:
            miss += 1
            continue
        if margin > 0 and (best - second) < margin:
            miss += 1                          # ambiguous: leave unassigned
            continue
        if best_pid == pid_true:
            right += 1
        else:
            wrong += 1
            tainted.add(best_pid)
        if chain:
            person_vecs[best_pid] = np.concatenate(
                [person_vecs[best_pid], v[None, :]], axis=0)
            person_photos.setdefault(best_pid, set()).add(ph)
            person_times.setdefault(best_pid, []).append(tt)

    # visible mixing: a person carrying at least one attach of a stranger
    return dict(right=right, wrong=wrong, miss=miss,
                attached=right + wrong,
                wrong_share=round(wrong / max(right + wrong, 1), 4),
                coverage=round(right / max(len(queries), 1), 4),
                persons=len(person_vecs),
                impure=len(tainted),
                impure_pct=round(100 * len(tainted) / max(len(person_vecs), 1), 2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage",
                    choices=["core", "cooc", "tune", "model", "final", "gap"],
                    default="core")
    ap.add_argument("--model", action="append", default=None)
    args = ap.parse_args()

    default_models = ["/models/osnet_x0_25_msmt17.onnx"]
    models = args.model or default_models

    labs = None
    for model in models:
        emb = embed(model)
        if labs is None:
            labs = np.load(OUT / "fusion_labs.npy")
        print(f"[fusion] model={Path(model).name} emb={emb.shape}", flush=True)

        if args.stage == "core":
            grid = [dict(t=t, how=h, chain=c, n_anchor=a, cooc=False,
                         time_win=False, margin=0.0)
                    for t in (0.60, 0.65, 0.70, 0.74, 0.78, 0.82)
                    for h in ("max", "mean", "median", "top2")
                    for c in (True, False)
                    for a in (1, 3)]
        elif args.stage == "tune":
            # production regime (group photos + time window + 3 anchors),
            # refined around the leading configs, with the ambiguity margin
            grid = [dict(t=t, how=h, chain=True, n_anchor=3, cooc=True,
                         time_win=True, margin=m)
                    for t in (0.74, 0.76, 0.78, 0.80, 0.82, 0.84)
                    for h in ("max", "mean", "top2")
                    for m in (0.0, 0.03, 0.06)]
        elif args.stage == "gap":
            # how wide a capture-time window should body evidence stay valid?
            # TIME_GAP is a module constant, so the sweep patches it per run
            grid = [dict(t=0.74, how="max", chain=True, n_anchor=3, cooc=True,
                         time_win=g > 0, margin=m, gap=g)
                    for g in (1 * 3600, 3 * 3600, 6 * 3600, 12 * 3600,
                              24 * 3600, 0)
                    for m in (0.05, 0.06, 0.07)]
        elif args.stage == "final":
            # margin knee + body-embedder A/B, all in the production regime
            grid = [dict(t=t, how="max", chain=True, n_anchor=3, cooc=True,
                         time_win=True, margin=m)
                    for t in (0.72, 0.74, 0.76)
                    for m in (0.0, 0.02, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10)]
        elif args.stage == "model":
            # same regime: which body embedder attaches safest?
            grid = [dict(t=t, how=h, chain=True, n_anchor=3, cooc=True,
                         time_win=True, margin=0.0)
                    for t in (0.74, 0.78, 0.82)
                    for h in ("max", "mean")]
        else:  # cooc: production-shaped - group photos and/or the time window
            grid = [dict(t=t, how=h, chain=True, n_anchor=a, cooc=c,
                         time_win=w, margin=0.0)
                    for t in (0.74, 0.78, 0.82)
                    for h in ("max", "mean", "top2")
                    for a in (1, 3)
                    for c in (False, True)
                    for w in (False, True)]

        for g in grid:
            res = simulate(emb, labs, **g)
            res.update(model=Path(model).name, **g)
            print("FUSION " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
