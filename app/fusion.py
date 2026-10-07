"""Re-id fusion layer: body evidence CONNECTS face-anchored persons.

The confirmed semantics (design invariants):
- identity is anchored ONLY by faces (a person exists because faces say so)
- body evidence may CONNECT face-anchored persons and attach back-view /
  faceless detections to them ("back views only hang, never anchor")
- hard rules: one person per photo (co-occurrence), time-window bound,
  and a body-join threshold calibrated on Market-1501

Run AFTER the face clustering pass.
"""
from __future__ import annotations

import numpy as np

from . import config, db


def _vec(b):
    if b is None:
        return None
    v = np.frombuffer(b, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-6 else None


def body_attach(stats: dict) -> dict:
    if not config.BODY_ATTACH_ENABLED:
        return stats
    dets = db.query(
        "SELECT d.id, d.photo_id, d.body_emb, d.person_id, d.face_id, p.taken_at AS t "
        "FROM detections d JOIN photos p ON p.id=d.photo_id")
    face_person = {r["id"]: r["person_id"] for r in db.query(
        "SELECT id, person_id FROM faces WHERE person_id IS NOT NULL")}
    # person -> body evidence vectors. Seed via the EXACT face link: a det
    # whose linked face belongs to a person carries that person's body
    # evidence (the identity anchor comes from the face, per the invariants).
    person_vecs: dict[int, list[np.ndarray]] = {}
    person_photos: dict[int, set] = {}
    person_times: dict[int, list[float]] = {}
    for r in dets:
        pid = r["person_id"]
        if pid is None and r["face_id"] is not None:
            pid = face_person.get(r["face_id"])
        if pid is None:
            continue
        v = _vec(r["body_emb"])
        if v is None:
            continue
        person_vecs.setdefault(pid, []).append(v)
        person_photos.setdefault(pid, set()).add(r["photo_id"])
        person_times.setdefault(pid, []).append(float(r["t"] or 0.0))
    # faces also count as photo occupancy and time anchors for their person
    for r in db.query("SELECT person_id, photo_id, p.taken_at AS t FROM faces f "
                      "JOIN photos p ON p.id=f.photo_id WHERE person_id IS NOT NULL"):
        person_photos.setdefault(r["person_id"], set()).add(r["photo_id"])
        person_times.setdefault(r["person_id"], []).append(float(r["t"] or 0.0))
    if not person_vecs:
        return stats

    attached = 0
    for r in dets:
        if r["person_id"] is not None:
            continue
        v = _vec(r["body_emb"])
        if v is None:
            continue
        # best AND runner-up among the candidates that survived the
        # co-occurrence and time-window gates: the margin gate only means
        # something against the competition that was actually allowed.
        best, second, best_pid = -2.0, -2.0, None
        for pid, vecs in person_vecs.items():
            if r["photo_id"] in person_photos.get(pid, ()):
                continue  # co-occurrence hard rule
            if config.BODY_MAX_GAP_S > 0 and person_times.get(pid):
                mid = sum(person_times[pid]) / len(person_times[pid])
                if abs((r["t"] or 0.0) - mid) > config.BODY_MAX_GAP_S:
                    continue  # time-window bound
            s = max(float(np.dot(v, pv)) for pv in vecs)
            if s > best:
                second, best, best_pid = best, s, pid
            elif s > second:
                second = s
        if best_pid is None or best < config.BODY_JOIN_T:
            continue
        gap = best - second
        if config.BODY_MARGIN > 0 and gap < config.BODY_MARGIN:
            continue  # ambiguous: two people look equally close - stays unassigned
        db.execute("UPDATE detections SET person_id=?, reason='body+time', "
                   "score=?, margin=? WHERE id=?",
                   (best_pid, best, gap, r["id"]))
        person_vecs[best_pid].append(v)
        person_photos.setdefault(best_pid, set()).add(r["photo_id"])
        attached += 1
    # A face may only inherit its box's person when a HUMAN said so. The
    # derived 'body+time' claims this pass writes must never anchor a face:
    # "back views only hang, they never anchor". Without this guard a wrong
    # body attach silently assigns an identity to a face, and once that face
    # is named the next recluster pins it as a seed.
    for r in db.query("SELECT id, face_id, person_id FROM detections "
                      "WHERE face_id IS NOT NULL AND person_id IS NOT NULL "
                      "AND reason='manual'"):
        db.execute("UPDATE faces SET person_id=? WHERE id=? AND person_id IS NULL",
                  (r["person_id"], r["face_id"]))
    if attached:
        db.conn().commit()
    stats["body_attached"] = stats.get("body_attached", 0) + attached
    return stats
