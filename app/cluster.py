"""Incremental identity clustering.

Signals, in the spirit of Google Photos' person grouping:

* face embeddings (SFace)          -> strong identity anchors
* body ReID embeddings (OSNet)     -> works for backs / no visible face
* capture-time proximity           -> photos taken seconds apart likely
                                      contain the same people

Design principle (user requirement): **prefer splitting over merging** —
a person split into two clusters is easy to merge by hand, a cluster that
mixes two people is hard to take apart. Everything below is biased that way.

One recluster pass performs, in order:

0. demote faces below the quality gates (detection score, size in the
   original image, size on the 640 detection canvas) — they never drive
   identity; their detections re-enter the body pass.
0b. drop body-only persons: identity must be anchored by a face. Unbound
   body clusters are too error-prone to stand on their own (user
   decision), so their members are released and may only claim
   FACE-anchored persons below.
1. co-occurrence split: two disjoint detections (or two faces) of one
   person in the SAME photo prove different people — members are
   partitioned so no photo ever holds two conflicting members of one
   person (IoU >= CO_DUP_IOU counts as a duplicate detection instead).
2. coherence eviction: members whose similarity to the mean of their OWN
   cluster's other faces drops below FACE_GROUP_SIM are evicted. The
   criterion is identical to the join rule below, which makes
   join/evict symmetric — an evictee can never rejoin the cluster that
   rejected it, so the pass cannot oscillate. Evictees re-group among
   themselves in step 4 (this is how mush clusters fall apart).
3. aggregates + per-photo presence maps are seeded.
4. unassigned faces match existing persons or form new groups
   (FACE_GROUP_SIM). Co-occurrence blocks matching into a person that
   already appears in the photo — passers-by cannot join.
5. detections carrying an assigned face inherit it ('face') unless that
   person already appears in the photo.
6. remaining detections score against every FACE-anchored person:
       score = BODY_W * cos(body, person_mean_body)
             + TIME_W * exp(-dt / half-life)
   with a hard window BODY_MAX_GAP_S (5h — no claim across a larger
   gap, no time-free claims) and the co-occurrence gate. Only
   'body+time' claims exist. Detections that match nobody simply stay
   unassigned; body-only clusters are never created.
7. cross-person merges (co-occurrence is a hard blocker): face-cluster
   pairs at FACE_MERGE_SIM where every member of the combined cluster
   still passes the eviction criterion against the combined mean (so a
   merge can never be undone by step 2 — no oscillation).
8. empty persons are dropped.
"""
import json
import logging
import threading

import numpy as np

from . import config, db

log = logging.getLogger("gallery.cluster")

_lock = threading.Lock()

# two boxes in one photo overlapping at least this much are duplicate
# detections of one human; below it, two boxes = two different people
CO_DUP_IOU = 0.3


def _unit(vec: bytes | None) -> np.ndarray | None:
    if vec is None:
        return None
    v = np.frombuffer(vec, dtype=np.float32)
    if v.size == 0 or not np.all(np.isfinite(v)):
        return None
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return None
    return v / n


_HEADS: dict[int, list[tuple[float, float, float]]] = {}  # photo_id -> [(x,y,r)]
_LIMBS: dict[int, list[tuple[float, float]]] = {}         # photo_id -> [(x,y)]


def _load_heads(c) -> None:
    """Pose anchors per photo: head centers (informational) and wrist/ankle
    endpoints (used for the face false-positive veto)."""
    _HEADS.clear()
    _LIMBS.clear()
    for r in c.execute("SELECT photo_id, head FROM detections WHERE head IS NOT NULL"):
        try:
            p = json.loads(r["head"])
        except Exception:
            continue
        if isinstance(p, dict):
            h = p.get("h")
            if h:
                _HEADS.setdefault(r["photo_id"], []).append(
                    (float(h[0]), float(h[1]), float(h[2])))
            for x, y in (p.get("l") or []):
                _LIMBS.setdefault(r["photo_id"], []).append((float(x), float(y)))
        else:  # legacy format: bare [x, y, r]
            _HEADS.setdefault(r["photo_id"], []).append(
                (float(p[0]), float(p[1]), float(p[2])))


def recluster() -> dict:
    with _lock:
        try:
            return _recluster_locked()
        except Exception:
            # never leave a half-applied pass inside this connection's
            # transaction — the next request would inherit it
            try:
                db.conn().rollback()
            except Exception:
                log.exception("rollback after failed recluster also failed")
            raise


# ---------------------------------------------------------------- passes ---


def _recluster_locked() -> dict:
    c = db.conn()
    _load_heads(c)
    faces = [
        dict(r)
        for r in c.execute(
            "SELECT f.id, f.person_id, f.face_emb, f.score, f.quality, f.yaw, f.reviewed, "
            "f.box, f.photo_id, "
            "p.width AS w, p.height AS h, COALESCE(p.taken_at,0) AS t "
            "FROM faces f JOIN photos p ON p.id=f.photo_id ORDER BY t, f.id"
        ).fetchall()
    ]
    dets = [
        dict(r)
        for r in c.execute(
            "SELECT d.id, d.person_id, d.face_id, d.body_emb, d.reason, d.box, d.conf, "
            "d.personness, d.track_id, d.photo_id, p.width AS w, p.height AS h, "
            "COALESCE(p.taken_at,0) AS t "
            "FROM detections d JOIN photos p ON p.id=d.photo_id ORDER BY t, d.id"
        ).fetchall()
    ]
    # ground truth from manual labeling: named persons and persons carrying
    # manually labeled members are anchors — clustering never undoes them
    names = {r["id"]: r["name"] for r in c.execute("SELECT id, name FROM persons")}
    anchored = {r[0] for r in c.execute(
        "SELECT DISTINCT person_id FROM detections "
        "WHERE person_id IS NOT NULL AND reason='manual'")}
    anchored |= {r[0] for r in c.execute(
        "SELECT DISTINCT person_id FROM faces "
        "WHERE person_id IS NOT NULL AND reviewed=1")}
    trusted = anchored | {p for p, n in names.items() if n}
    stats = {"faces_demoted": 0, "persons_cooc_split": 0, "faces_evicted": 0,
             "persons_body_dropped": 0, "claims_revalidated": 0, "persons_merged": 0,
             "faces_assigned": 0, "dets_assigned": 0, "persons_created": 0}

    # --- Ente-core engine (ported from ente cluster.ts): a complete       ---
    # --- alternative grouping path, kept switchable for comparison        ---
    if config.CLUSTER_ENGINE in ("ente", "immich"):
        from . import cluster_ente, fusion
        st = cluster_ente.recluster_ente(stats)
        st = fusion.body_attach(st)   # body connects face-anchored persons
        c.commit()
        return st

# The deprecated self-developed grouping algorithm used to live here. It was
# measured against the Ente port and lost (see docs/RESEARCH-LOG.md 1); it is
# removed from this demo build. CLUSTER_ENGINE now always resolves to the
# Ente core plus body attach.
