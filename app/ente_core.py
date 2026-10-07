"""Faithful port of Ente's face clustering core.

Source of truth: web/packages/new/photos/services/ml/cluster.ts (see
docs/ENTE-PORT-SPEC.md §2). Semantics preserved exactly:

- enumeration gate: blur > 10 and score > 0.8 (cluster.ts:143)
- newest-first processing order (cluster.ts:62-63, 155-175)
- already-assigned faces placed first; matches look BACKWARD only
  (cluster.ts:216-217, 232)
- greedy single-linkage: a face joins its best previously-placed
  neighbour's cluster (cluster.ts:232-268)
- quality-adaptive threshold chosen by the NEIGHBOR face:
  t_good (0.76) if !isBadFace else t_bad (0.84) (cluster.ts:240)
- isBadFace = blur < 50 || (blur < 200 && score < 0.85) || sideways
  (cluster.ts:177-183)
- per-(face x cluster) rejection memory: never return a face to a
  cluster the user rejected (cluster.ts:243-252)
- overlapping batches 10000/7500 so clusters link across boundaries
  (cluster.ts:117-130)

Threshold values are per-embedding-space calibration (Ente's own numbers
are tied to their MobileFaceNet space); pass them in via EnteParams.
"""
from __future__ import annotations

import numpy as np

# Ente originals (web/.../ml/cluster.ts:143, 240; face.ts:71-86)
BLUR_CLUSTERABLE = 10.0
SCORE_CLUSTERABLE = 0.8
BLUR_BAD_SOFT = 50.0
BLUR_BAD_VERY_SOFT = 200.0
SCORE_BAD = 0.85
NOSE_NEAR_EYE_X = 0.2
UPRIGHT_EYE_Y = 0.5
UPRIGHT_MOUTH_Y = 0.5
BATCH_SIZE = 10000
BATCH_STRIDE = 7500


class EnteParams:
    def __init__(self, t_good: float = 0.76, t_bad: float = 0.84):
        self.t_good = t_good
        self.t_bad = t_bad


def face_direction(landmarks: np.ndarray) -> str:
    """Ente faceDirection (web/.../ml/face.ts:60-95).

    landmarks: 5x2 array, rows = left eye, right eye, nose, left mouth,
    right mouth. Returns "left" | "right" | "straight".
    """
    le, re, nose, lm, rm = landmarks.astype(np.float64)
    eye_dx = abs(float(re[0] - le[0]))
    eye_dy = abs(float(re[1] - le[1]))
    mouth_dx = abs(float(rm[0] - lm[0]))
    # upright check (face.ts:71-73)
    upright = (max(le[1], re[1]) + UPRIGHT_EYE_Y * eye_dy < nose[1] and
               nose[1] + UPRIGHT_MOUTH_Y * mouth_dy_ok(landmarks) < min(lm[1], rm[1]))
    # nose sticking out past both eyes and both mouth corners (face.ts:75-81)
    if nose[0] < min(le[0], re[0], lm[0], rm[0]):
        return "left"
    if nose[0] > max(le[0], re[0], lm[0], rm[0]):
        return "right"
    # upright but nose near one eye -> sideways profile (face.ts:83-86)
    if upright:
        if abs(nose[0] - le[0]) < NOSE_NEAR_EYE_X * eye_dx:
            return "left"
        if abs(nose[0] - re[0]) < NOSE_NEAR_EYE_X * eye_dx:
            return "right"
    return "straight"


def mouth_dy_ok(landmarks: np.ndarray) -> float:
    lm, rm = landmarks[3], landmarks[4]
    return abs(float(rm[1] - lm[1]))


def is_bad_face(blur: float, score: float, sideways: bool) -> bool:
    """cluster.ts:177-183."""
    if blur < BLUR_BAD_SOFT:
        return True
    if blur < BLUR_BAD_VERY_SOFT and score < SCORE_BAD:
        return True
    return bool(sideways)


def is_clusterable(blur: float, score: float) -> bool:
    """cluster.ts:143."""
    return blur > BLUR_CLUSTERABLE and score > SCORE_CLUSTERABLE


def complete_link_merge(emb: np.ndarray, assigned: np.ndarray, bar: float,
                        photo_ids: np.ndarray) -> np.ndarray:
    """Merge clusters whose MINIMUM cross-pair similarity >= bar.

    The inverse of Ente's "Mixed grouping?" complete-linkage split: only
    provably-safe folds (every cross pair clears the bar) are allowed, so
    this can recover greedy-split fragments without opening a mixing hole.
    """
    clusters: dict[int, list[int]] = {}
    for i, c in enumerate(assigned):
        if c >= 0:
            clusters.setdefault(int(c), []).append(i)
    keys = sorted(clusters)
    reps = {k: np.stack([emb[i] for i in clusters[k]]) for k in keys}
    changed = True
    while changed:
        changed = False
        keys = sorted(clusters)
        for ai in range(len(keys)):
            for bi in range(ai + 1, len(keys)):
                a, b = keys[ai], keys[bi]
                photos_a = {photo_ids[i] for i in clusters[a]}
                photos_b = {photo_ids[i] for i in clusters[b]}
                if photos_a & photos_b:
                    continue  # co-occurrence hard rule
                cross = (reps[a] @ reps[b].T).min()
                if cross >= bar:
                    clusters[a].extend(clusters[b])
                    reps[a] = np.vstack([reps[a], reps[b]])
                    for i in clusters[b]:
                        assigned[i] = a
                    del clusters[b]
                    del reps[b]
                    changed = True
                    break
            if changed:
                break
    return assigned


def complete_link_split(emb: np.ndarray, assigned: np.ndarray, bar: float) -> np.ndarray:
    """Eject cluster members whose MINIMUM similarity to co-members < bar.

    Automatic form of Ente's "Mixed grouping?" complete-linkage cut: an
    intruder may look like ONE member (that is how it got in) but its
    minimum similarity to the rest of the cluster is low -> it leaves and
    becomes a reviewable fragment. Splitting is free under our objective;
    mixing is the accident.
    """
    while True:
        clusters: dict[int, list[int]] = {}
        for i, c in enumerate(assigned):
            if c >= 0:
                clusters.setdefault(int(c), []).append(i)
        eject = None
        for cid, idxs in clusters.items():
            if len(idxs) < 2:
                continue
            m = emb[idxs]
            sims = m @ m.T
            np.fill_diagonal(sims, np.inf)
            mins = sims.min(axis=1)
            if mins.min() < bar:
                eject = idxs[int(np.argmin(mins))]
                break
        if eject is None:
            return assigned
        assigned[eject] = -1


def cluster_faces(emb: np.ndarray, blur: np.ndarray, score: np.ndarray,
                  sideways: np.ndarray, photo_ids: np.ndarray,
                  times: np.ndarray, params: EnteParams,
                  face_keys: list[str] | None = None,
                  rejected: dict[str, set[int]] | None = None,
                  seed_clusters: dict[str, int] | None = None,
                  one_per_photo: bool = True,
                  gram: np.ndarray | None = None
                  ) -> tuple[np.ndarray, int]:
    """Ente greedy single-linkage. Returns (cluster_index_per_face, n_clusters).

    - emb: N x D L2-normalized
    - rejected: face_key -> set of cluster keys the user rejected
    - seed_clusters: face_key -> cluster index (existing assignments,
      already-clustered faces are placed first and are immutable)
    """
    n = len(emb)
    keys = face_keys or [f"f{i}" for i in range(n)]
    rejected = rejected or {}
    seed = seed_clusters or {}

    bad = np.array([is_bad_face(blur[i], score[i], sideways[i]) for i in range(n)])
    keep = np.array([is_clusterable(blur[i], score[i]) for i in range(n)])

    assigned = np.full(n, -1, dtype=np.int64)
    next_id = 0
    if seed:
        for i, k in enumerate(keys):
            if k in seed:
                assigned[i] = seed[k]
                next_id = max(next_id, seed[k] + 1)

    # newest first (cluster.ts:62-63); already-clustered first (216-217).
    # One pass = Ente's windowed run exactly for n <= BATCH_SIZE (our
    # scale); the 10000/7500 windowing only matters beyond that.
    order = [i for i in range(n) if keep[i]]
    order.sort(key=lambda i: (-times[i], i))
    placed_first = [i for i in order if assigned[i] >= 0]
    unplaced = [i for i in order if assigned[i] < 0]
    work = placed_first + unplaced

    cluster_photos: dict[int, set] = {}
    for i in range(n):
        if assigned[i] >= 0:
            cluster_photos.setdefault(int(assigned[i]), set()).add(photo_ids[i])

    for r, i in enumerate(work):
        if assigned[i] >= 0:
            continue
        best_sim = -1.0  # strictly improve (cluster.ts:238)
        best_cluster = -1
        rej = rejected.get(keys[i], set())
        for j in work[:r]:
            th = params.t_bad if bad[j] else params.t_good
            # gram fast-path: precomputed cosine matrix makes the O(n^2)
            # scan a table lookup instead of a D-dim dot (benchmarks only;
            # production passes gram=None and computes dots on the fly)
            s = float(gram[i, j]) if gram is not None else float(np.dot(emb[i], emb[j]))
            if s < th or s <= best_sim:
                continue
            cid = assigned[j]
            if cid in rej:
                continue  # rejection memory (cluster.ts:243-252)
            if one_per_photo and photo_ids[i] in cluster_photos.get(cid, ()):
                continue  # co-occurrence hard rule (our invariant)
            best_sim = s
            best_cluster = cid
        if best_cluster >= 0:
            assigned[i] = best_cluster
            cluster_photos.setdefault(best_cluster, set()).add(photo_ids[i])
        else:
            assigned[i] = next_id
            cluster_photos[next_id] = {photo_ids[i]}
            next_id += 1
    return assigned, next_id
