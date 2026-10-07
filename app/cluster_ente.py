"""Ente face clustering wired to the gallery DB.

Algorithm = app/ente_core.py (faithful port of Ente cluster.ts, see
docs/ENTE-PORT-SPEC.md). This module loads faces from SQLite, runs the
core, and applies assignments.

Rebuild semantics (Ente's "full-rebuild + reconcile", spec 2.7):
- NAMED persons (Ente cgroups) and user-reviewed faces seed the run and
  are immutable - human decisions are permanent (spec 3.3)
- anonymous clusters are re-derived from scratch every pass
- per-(face x person) rejection memory from rejected_faces (spec 2.3/2.8)

Face evidence only: this is the Ente face algorithm. Body/re-id is a
separate fusion layer that may only CONNECT face-anchored persons.

One deviation, a confirmed design invariant of ours: two faces from the
same photo can never share a person (co-occurrence hard rule).
"""
from __future__ import annotations

import numpy as np

from . import config, db
from .ente_core import (EnteParams, cluster_faces, complete_link_merge,
                        complete_link_split)


def _vec(b):
    if b is None:
        return None
    v = np.frombuffer(b, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-6 else None


def _load() -> list[dict]:
    rows = db.query(
        "SELECT f.id, f.photo_id, f.box, f.score, f.face_emb, f.blur, f.sideways, "
        "f.person_id, f.reviewed, p.taken_at AS t, "
        "EXISTS(SELECT 1 FROM persons pn WHERE pn.id = f.person_id AND "
        "       COALESCE(pn.name,'') <> '') AS named "
        "FROM faces f JOIN photos p ON p.id = f.photo_id")
    out = []
    for r in rows:
        v = _vec(r["face_emb"])
        if v is None:
            continue
        blur = r["blur"]
        out.append(dict(
            id=r["id"], photo=r["photo_id"], box=r["box"], vec=v,
            blur=10000.0 if blur is None else float(blur),
            score=float(r["score"]), sideways=bool(r["sideways"]),
            t=float(r["t"] or 0.0), person=r["person_id"],
            reviewed=bool(r["reviewed"]), named=bool(r["named"])))
    return out


def _run(faces: list[dict], seed_clusters: dict[str, int],
         rejected: dict[str, set[int]]):
    emb = np.stack([f["vec"] for f in faces])
    blur = np.array([f["blur"] for f in faces])
    score = np.array([f["score"] for f in faces])
    sideways = np.array([1 if f["sideways"] else 0 for f in faces], dtype=bool)
    photo = np.array([f["photo"] for f in faces])
    times = np.array([f["t"] for f in faces])
    params = EnteParams(config.ENTE_T_GOOD, config.ENTE_T_BAD)
    assigned, n = cluster_faces(emb, blur, score, sideways, photo, times, params,
                                face_keys=[f"{f['photo']}_{f['box']}" for f in faces],
                                rejected=rejected, seed_clusters=seed_clusters,
                                one_per_photo=config.ONE_PER_PHOTO)
    # Fragment recovery. The greedy pass splits one person across clusters
    # whenever no single face pair clears the join bar - blurry and harsh-angle
    # shots land there constantly, which is what the user sees as "two identical
    # faces in different groups". Folding two clusters together ONLY when every
    # cross pair is >= PERSON_SPLIT_SIM cannot mix anyone at that bar: it is the
    # exact inverse of the "Mixed grouping?" breakup, so what that button would
    # keep apart is exactly what this folds back together.
    #
    # Measured on LFW (2339 faces, 311 ids, shipped LVFace-T): split pairs
    # 3091 -> 1670, same-person recall 0.600 -> 0.784, mixed pairs 15 -> 28 out
    # of 2.7M. The bar barely matters between 0.50 and 0.60 (the cross-pair
    # distribution has a gap there), so tying it to the breakup floor is safe.
    assigned = complete_link_merge(emb, assigned, config.PERSON_SPLIT_SIM, photo)
    return assigned, n


def cluster_ente() -> list[list[tuple[str, int]]]:
    """Pure grouping (used by the public-data benches)."""
    faces = _load()
    if not faces:
        return []
    assigned, _ = _run(faces, {}, {})
    if config.PERSON_EJECT_BAR > 0:
        assigned = complete_link_split(
            np.stack([f["vec"] for f in faces]), assigned, config.PERSON_EJECT_BAR)
    groups: dict[int, list] = {}
    for f, c in zip(faces, assigned):
        if int(c) < 0:
            continue  # clusterable-gate failures stay unassigned
        groups.setdefault(int(c), []).append(("face", f["id"]))
    return list(groups.values())


def recluster_ente(stats: dict) -> dict:
    faces = _load()
    c = db.conn()
    if not faces:
        return stats

    seed_clusters: dict[str, int] = {}
    for f in faces:
        if f["person"] and (f["named"] or f["reviewed"]):
            seed_clusters[f"{f['photo']}_{f['box']}"] = int(f["person"])
    rejected: dict[str, set[int]] = {}
    for r in db.query("SELECT face_key, person_id FROM rejected_faces"):
        rejected.setdefault(r["face_key"], set()).add(int(r["person_id"]))

    assigned, _ = _run(faces, seed_clusters, rejected)
    if config.PERSON_EJECT_BAR > 0:
        assigned = complete_link_split(
            np.stack([f["vec"] for f in faces]), assigned, config.PERSON_EJECT_BAR)

    # cluster id -> person row: a seeded cluster keeps its person; new
    # clusters with pinned faces join the pinned face's person. cid < 0 =
    # failed the clusterable gate (blur>10 & score>0.8): stays UNASSIGNED
    # (reviewable fragment), it must never pile into one junk person.
    cluster_person: dict[int, int] = {}
    for f, cid in zip(faces, assigned):
        cid = int(cid)
        if cid < 0 or cid in cluster_person:
            continue
        if f["person"] and (f["named"] or f["reviewed"]):
            cluster_person[cid] = int(f["person"])
    for f, cid in zip(faces, assigned):
        cid = int(cid)
        if cid < 0 or cid in cluster_person:
            continue
        cluster_person[cid] = db.execute(
            "INSERT INTO persons(name, created_at) VALUES (NULL, ?)",
            (db.now(),))
        stats["persons_created"] += 1

    c.execute("UPDATE faces SET person_id=NULL")
    for f, cid in zip(faces, assigned):
        if int(cid) < 0:
            continue
        c.execute("UPDATE faces SET person_id=? WHERE id=?",
                  (cluster_person[int(cid)], f["id"]))
    doomed = [int(r[0]) for r in c.execute(
        "SELECT id FROM persons WHERE COALESCE(name,'')='' AND id NOT IN "
        "(SELECT person_id FROM faces WHERE person_id IS NOT NULL) AND id NOT IN "
        "(SELECT person_id FROM detections WHERE person_id IS NOT NULL "
        " AND reason='manual')")]
    if doomed:
        q = ",".join("?" * len(doomed))
        c.execute(f"DELETE FROM persons WHERE id IN ({q})", doomed)
        # persons.id is a rowid alias: without this the next person to be
        # created would inherit these people's labels and review decisions
        db.purge_person_refs(doomed)
    # The anonymous rebuild above swaps person rows wholesale, so any body
    # claim left pointing at a person that no longer exists is dead. It has
    # to go here, BEFORE fusion.body_attach() runs: that pass copies a
    # detection's person onto its face, which is how a full recluster used
    # to end up with dangling person references on both detections and
    # faces (and body attachments hanging off people that no longer exist).
    # Only DERIVED claims are cleared - reason='manual' rows are user work
    # and the person they name is kept above.
    c.execute("UPDATE detections SET person_id=NULL WHERE person_id IS NOT NULL "
              "AND person_id NOT IN (SELECT id FROM persons)")
    c.execute("UPDATE faces SET person_id=NULL WHERE person_id IS NOT NULL "
              "AND person_id NOT IN (SELECT id FROM persons)")
    # a claim without a person is not a claim: reason/margin/score are shown
    # in the viewer, so they must not outlive the person they refer to
    c.execute("UPDATE detections SET reason=NULL, score=NULL, margin=NULL "
              "WHERE person_id IS NULL")
    c.commit()
    stats["persons"] = c.execute("SELECT COUNT(*) FROM persons").fetchone()[0]
    return stats
