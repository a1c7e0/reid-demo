"""Cluster-policy benchmark on PUBLIC datasets (LFW faces / Market bodies).

Builds a fresh gallery-schema DB from bench_{lfw,market}.npz, runs the
real clustering engines (CLUSTER_ENGINE=ente|immich|ours) over it, and
scores the induced partition against the dataset identity labels.

Usage: python cluster_bench.py /s2/bench_lfw.npz face
       python cluster_bench.py /s2/bench_market.npz body
Prints one JSON line per engine.
"""
import json
import sys

sys.path.insert(0, "/app")

import numpy as np

from app import cluster, config, db

npz_path = sys.argv[1]
kind = sys.argv[2]  # face | body
data = np.load(npz_path, allow_pickle=True)
emb, label = data["emb"], data["label"]
photo, t = data["photo"], data["time"]
print("loaded", emb_path := npz_path, emb.shape, "labels:", len(set(label.tolist())))


def build_db():
    db.init_db()
    c = db.conn()
    c.execute("DELETE FROM faces")
    c.execute("DELETE FROM detections")
    c.execute("DELETE FROM persons")
    c.execute("DELETE FROM photos")
    for i in range(len(emb)):
        c.execute(
            "INSERT INTO photos(id, sha1, path, width, height, taken_at, processed, created_at) "
            "VALUES (?,?,?,?,?,?,1,0)", (int(photo[i]), f"b{i}", f"bench/{i}.jpg", 640, 480, float(t[i])))
        if kind == "face":
            c.execute(
                "INSERT INTO faces(photo_id, box, score, face_emb, person_id, reviewed, quality, yaw) "
                "VALUES (?,?,?,? ,NULL,NULL,?,?)",
                (int(photo[i]), json.dumps([0, 0, 100, 100]), 0.99,
                 emb[i].astype(np.float32).tobytes(), 0.9, 0.0))
        else:
            c.execute(
                "INSERT INTO detections(photo_id, box, conf, face_id, body_emb, person_id, score, "
                "reason, personness, margin, head, track_id) VALUES (?,?,?,NULL,?,NULL,NULL,NULL,1.0,NULL,NULL,NULL)",
                (int(photo[i]), json.dumps([0, 0, 100, 200]), 0.99,
                 emb[i].astype(np.float32).tobytes()))
    c.commit()


def score(engine):
    config.CLUSTER_ENGINE = engine
    build_db()
    for _ in range(4):
        st = cluster.recluster()
    c = db.conn()
    if kind == "face":
        rows = c.execute("SELECT id, photo_id, person_id FROM faces ORDER BY id").fetchall()
    else:
        rows = c.execute("SELECT id, photo_id, person_id FROM detections ORDER BY id").fetchall()
    # person per unit index (unit i = i-th embedding, matched by photo order)
    pid_of = {}
    for r in rows:
        pid_of[int(r["photo_id"])] = r["person_id"]
    got = [pid_of.get(int(p)) for p in photo]
    FM = FS = UN = HIT = same = diff = 0
    for i in range(len(label)):
        for j in range(i + 1, len(label)):
            a, b = got[i], got[j]
            if label[i] == label[j]:
                same += 1
                if a is None or b is None:
                    UN += 1
                elif a == b:
                    HIT += 1
                else:
                    FS += 1
            else:
                diff += 1
                if a is not None and a == b:
                    FM += 1
    persons = len({p for p in got if p is not None})
    sizes = {}
    for p in got:
        if p is not None:
            sizes[p] = sizes.get(p, 0) + 1
    biggest = max(sizes.values()) if sizes else 0
    return {"engine": engine, "kind": kind, "FM": FM, "FS": FS, "UN": UN, "HIT": HIT,
            "same_pairs": same, "diff_pairs": diff, "persons": persons,
            "biggest_person": biggest, "unassigned_units": sum(1 for g in got if g is None)}


for engine in ("ours", "ente", "immich"):
    print(json.dumps(score(engine)), flush=True)
