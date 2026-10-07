"""Clustering trial harness: wipe assignments, recluster from embeddings,
score the induced partition against the human labels (test set).

Metric (pairwise over labeled refs, photo46/47 excluded):
  FM  false-merge: pair labeled different, clustered together  (the evil one)
  FS  false-split: pair labeled same, clustered apart
  UN  pair labeled same, at least one side unassigned
Score = 10*FM + 1*FS + 0.5*UN   (over-split tolerated, over-merge punished)

'Unassign' labels (unknown passers-by) count as different from every
labeled person, so strangers merging into known people are false-merges.
"""
import json
import os
import sys

sys.path.insert(0, "/app")
from app import cluster, config, db  # noqa: E402

ZERO = ("persons_created", "persons_merged", "persons_cooc_split", "faces_evicted",
        "faces_assigned", "dets_assigned", "faces_demoted", "persons_body_dropped",
        "claims_revalidated")


def main() -> None:
    db.init_db()
    c = db.conn()
    c.execute("UPDATE faces SET person_id=NULL, reviewed=NULL")
    c.execute("UPDATE detections SET person_id=NULL, reason=NULL, score=NULL, margin=NULL")
    c.execute("DELETE FROM persons")
    c.commit()

    passes = 0
    for i in range(8):
        st = cluster.recluster()
        passes = i + 1
        if all(st.get(k, 0) == 0 for k in ZERO):
            break

    labels = json.load(open("/s2/labels-20261003.json"))
    refs = []  # (kind, ref, want)  want=None for unassign
    for ph in labels["photos"]:
        if ph["photo_id"] in (46, 47):
            continue
        for l in ph["labels"]:
            key = (l["kind"], l["ref_id"])
            refs.append((key, l["person_id"] if l["action"] != "unassign" else None))
    # keep the latest action per ref
    latest = {}
    for key, want in refs:
        latest[key] = want
    refs = [(k, w) for k, w in latest.items()]

    cur = {}
    for (kind, ref), _ in refs:
        src = "faces" if kind == "face" else "detections"
        row = db.one(f"SELECT person_id FROM {src} WHERE id=?", (ref,))
        cur[(kind, ref)] = row["person_id"] if row else None

    FM = FS = UN = same_pairs = diff_pairs = HIT = 0
    ff_same = ff_hit = ff_un = 0
    for i in range(len(refs)):
        (k1, r1), w1 = refs[i]
        for j in range(i + 1, len(refs)):
            (k2, r2), w2 = refs[j]
            c1, c2 = cur[(k1, r1)], cur[(k2, r2)]
            if w1 is not None and w1 == w2:
                same_pairs += 1
                if c1 is None or c2 is None:
                    UN += 1
                elif c1 != c2:
                    FS += 1
                else:
                    HIT += 1
                if k1 == "face" and k2 == "face":
                    ff_same += 1
                    if c1 is None or c2 is None:
                        ff_un += 1
                    elif c1 == c2:
                        ff_hit += 1
            else:
                diff_pairs += 1
                if c1 is not None and c1 == c2:
                    FM += 1
    score = 10 * FM + FS + 0.5 * UN
    print(json.dumps({
        "fm": FM, "fs": FS, "un": UN, "hit": HIT,
        "same_pairs": same_pairs, "diff_pairs": diff_pairs,
        "ff_same": ff_same, "ff_hit": ff_hit, "ff_un": ff_un,
        "score": score, "passes": passes,
        "cfg": {k: getattr(config, k, None) for k in (
            "FACE_GROUP_SIM", "FACE_EVICT_SIM", "FACE_MERGE_SIM",
            "FACE_QUALITY_PENALTY", "FACE_MAX_JOIN_DAYS",
            "BODY_ASSIGN_T", "BODY_MARGIN", "BODY_MIN_PERSONNESS",
            "BODY_MIN_REL_H", "BODY_MAX_GAP_S", "BODY_W", "TIME_W")},
    }))


if __name__ == "__main__":
    main()
