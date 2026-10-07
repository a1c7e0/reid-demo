"""Re-embed existing detections' body vectors (ops tool).

Switching REID_MODEL_FILE changes the embedding space: vectors produced by
the old model and vectors produced by the new one are not comparable, so a
model swap without this step silently breaks every body attach (anchors
come from old-model vectors, queries from new-model ones).

Reuses the production code path exactly (app.pipeline crop + app.ml.reid
embed), one batch per photo. Person/face assignments are untouched - run
`POST /api/recluster` afterwards so the attach pass sees fresh evidence.

    docker run --rm --memory=1200m --cpus=1 -v $PWD/testsets/ui-data:/data \
        -v $PWD:/app reid-gallery:latest python /app/scripts/reembed_body.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, "/app")

from app import config, db  # noqa: E402
from app.ml import common  # noqa: E402
from app.pipeline import engines  # noqa: E402


def main() -> None:
    reid = engines()["reid"]
    rows = db.query(
        "SELECT d.id, d.photo_id, d.box FROM detections d "
        "WHERE d.body_emb IS NOT NULL")
    if not rows:
        print("[reembed] no detections with body embeddings")
        return
    by_photo: dict[int, list] = {}
    for r in rows:
        by_photo.setdefault(r["photo_id"], []).append(r)

    updated = 0
    for photo_id, items in by_photo.items():
        photo = db.one("SELECT path FROM photos WHERE id=?", (photo_id,))
        if photo is None:
            continue
        path = Path(config.PHOTO_DIR) / photo["path"]
        if not path.exists():
            print(f"[reembed] missing photo {path}", flush=True)
            continue
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            rgb = np.asarray(im)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        crops = [common.crop(bgr, json.loads(i["box"])) for i in items]
        embs = reid.embed(crops)
        for item, e in zip(items, embs):
            db.execute("UPDATE detections SET body_emb=? WHERE id=?",
                       (e.astype(np.float32).tobytes(), item["id"]))
            updated += 1
        db.conn().commit()
        if updated % 64 < len(items):
            print(f"[reembed] {updated}/{len(rows)}", flush=True)
    print(f"[reembed] done: {updated} body vectors -> {config.REID_MODEL.name}")


if __name__ == "__main__":
    main()
