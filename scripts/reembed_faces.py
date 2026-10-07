"""Re-embed stored faces with the configured recognition model (ops tool).

Switching FACE_REC_MODEL_FILE changes the embedding space, so a model swap
without this step mixes spaces silently. This re-runs the *production*
pipeline (`app.pipeline.analyze`) on every photo and writes the resulting
embedding back onto the matching stored face, so the vectors are exactly
what a fresh scan would have produced - no hand-rolled preprocessing that
can drift from the ingest path (an earlier draft did, and used /128.0 where
the app uses /127.5).

Faces are matched by box IoU, so face ids, person assignments and review
state are all preserved. Run `POST /api/recluster` afterwards.

    docker run --rm --memory=1200m --cpus=1 -v $PWD:/app \\
        -v $PWD/testsets/ui-data:/data reid-gallery:latest \\
        python /app/scripts/reembed_faces.py
"""
from __future__ import annotations

import json
import sys

sys.path.insert(0, "/app")

from app import config, db  # noqa: E402
from app.pipeline import analyze  # noqa: E402


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(area_a + area_b - inter, 1e-9)


def main() -> None:
    photos = db.query("SELECT id, path FROM photos ORDER BY id")
    updated = unmatched = 0
    for photo in photos:
        path = config.PHOTO_DIR / photo["path"]
        if not path.exists():
            print(f"[reembed-face] missing {path}", flush=True)
            continue
        try:
            res = analyze(str(path))
        except Exception as e:            # keep going, report at the end
            print(f"[reembed-face] analyze failed {photo['path']}: {e}",
                  flush=True)
            unmatched += 1
            continue
        stored = db.query("SELECT id, box FROM faces WHERE photo_id=?",
                          (photo["id"],))
        fresh = list(res["faces"])
        used: set[int] = set()
        for s in stored:
            sbox = [float(v) for v in json.loads(s["box"])]
            best, best_iou = None, 0.5          # below this we do not touch it
            for i, f in enumerate(fresh):
                if i in used:
                    continue
                v = iou(sbox, [float(x) for x in f["box"]])
                if v > best_iou:
                    best_iou, best = v, i
            if best is None:
                unmatched += 1
                continue
            used.add(best)
            f = fresh[best]
            db.execute(
                "UPDATE faces SET face_emb=?, score=?, blur=?, sideways=? "
                "WHERE id=?",
                (f["feat"].astype("float32").tobytes(), float(f["score"]),
                 float(f.get("blur") or 0.0), bool(f.get("sideways")),
                 s["id"]))
            updated += 1
        db.conn().commit()
        print(f"[reembed-face] photo {photo['id']}: +{len(used)} "
              f"(total {updated}, unmatched {unmatched})", flush=True)
    print(f"[reembed-face] done: {updated} re-embedded, "
          f"{unmatched} unmatched/failed -> {config.FACE_REC_MODEL.name}")


if __name__ == "__main__":
    main()
