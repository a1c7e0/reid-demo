"""FastAPI application: gallery REST API + static web UI."""
import hashlib
import json
import logging
import mimetypes
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from pydantic import BaseModel

from . import cluster, config, db, ingest, worker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("gallery.api")

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(_: FastAPI):
    config.ensure_dirs()
    db.init_db()
    worker.start()
    # Crash recovery: anything that never got analyzed goes back on the queue.
    pending = db.query("SELECT id FROM photos WHERE processed=0")
    if pending:
        log.info("requeueing %d unprocessed photos", len(pending))
        for row in pending:
            worker.submit(row["id"])
    if config.SCAN_PATH:
        try:
            log.info("startup scan of %s", config.SCAN_PATH)
            ingest.scan_dir(config.SCAN_PATH)
        except Exception:
            log.exception("startup scan failed")
    yield
    worker.stop()


app = FastAPI(title="reid-gallery", version="0.1.0", lifespan=lifespan)


class ScanBody(BaseModel):
    path: str


class RenameBody(BaseModel):
    name: str | None = None
    pinned: bool | None = None


def _fields_set(body) -> set:
    """Which fields were explicitly sent (pydantic v1/v2 compatible), so a
    pin-only PATCH never clears the name and vice versa."""
    return set(
        getattr(body, "model_fields_set", None)
        or getattr(body, "__fields_set__", None)
        or ()
    )


def _err(message: str, status: int = 400):
    """Error body with the FastAPI {'detail': ...} shape the UI expects."""
    from fastapi.responses import JSONResponse

    return JSONResponse({"detail": message}, status_code=status)


class MergeBody(BaseModel):
    target_id: int


def _person_seq() -> dict[int, int]:
    """person_id -> stable sequence number (creation order), for labels
    like 'Person 7' on unnamed clusters."""
    rows = db.query("SELECT id FROM persons ORDER BY id")
    return {r["id"]: i + 1 for i, r in enumerate(rows)}


def _display(seq: dict[int, int], pid, name) -> str | None:
    """The ONE display label for a person.

    The 'person #N' label shows the DISPLAY SEQ (creation-order number);
    routes and API calls always carry the DB id. Never mix the two: no code
    may parse the number out of a label and use it as an id — and the DB id
    is never used as a label number either (unknown seq -> no label)."""
    if pid is None:
        return None
    nm = (name or "").strip()
    if nm:
        return nm
    n = seq.get(pid)
    return f"person #{n}" if n is not None else None


def _display_seq(seq: dict[int, int], pid, name) -> str | None:
    if pid is None:
        return None
    nm = (name or "").strip()
    if nm:
        return nm
    return f"Person {seq.get(pid, pid)}"


def _avatar(pid: int) -> str | None:
    """THE avatar rule — the only way any UI gets a person's head image.

    Always a crop that belongs to THIS person, chosen newest first
    (taken_at desc, then id desc — Ente's personFacesSortedNewestFirst):
    a face crop when the person has faces, else that person's own newest
    detection crop for body-only clusters. Never a photo thumbnail (one
    photo can contain several people)."""
    f = db.one(
        "SELECT f.id FROM faces f JOIN photos p ON p.id=f.photo_id "
        "WHERE f.person_id=? ORDER BY COALESCE(p.taken_at,0) DESC, f.id DESC LIMIT 1",
        (pid,),
    )
    if f:
        return f"/face_thumb/{f['id']}.jpg"
    d = db.one(
        "SELECT d.id FROM detections d JOIN photos p ON p.id=d.photo_id "
        "WHERE d.person_id=? ORDER BY COALESCE(p.taken_at,0) DESC, d.id DESC LIMIT 1",
        (pid,),
    )
    if d:
        return f"/det_thumb/{d['id']}.jpg"
    return None


# ---------------------------------------------------------------- status ----
@app.get("/api/config")
def runtime_config():
    """What the app is actually running: models on disk and the numbers that
    decide identity. Display-only - no secrets, no paths outside MODEL_DIR."""

    def model(path: Path):
        try:
            return {"file": path.name, "size_mb": round(path.stat().st_size / 1e6, 2),
                    "present": True}
        except OSError:
            return {"file": str(path.name), "size_mb": None, "present": False}

    return {
        "models": {
            "face_detector": {**model(config.FACE_DET_MODEL),
                              "kind": config.FACE_DET_KIND},
            "face_embedder": {**model(config.FACE_REC_MODEL),
                              "preset": config.FACE_REC_PRESET},
            "person_detector": model(config.PERSON_DET_MODEL),
            "body_embedder": model(config.REID_MODEL),
            "pose_gate": model(config.POSE_MODEL),
        },
        "engine": config.CLUSTER_ENGINE,
        "clustering": {
            "t_good": config.ENTE_T_GOOD,
            "t_bad": config.ENTE_T_BAD,
            "eject_bar": config.PERSON_EJECT_BAR,
            "one_per_photo": config.ONE_PER_PHOTO,
        },
        "body_attach": {
            "enabled": config.BODY_ATTACH_ENABLED,
            "join_t": config.BODY_JOIN_T,
            "margin": config.BODY_MARGIN,
            "max_gap_s": config.BODY_MAX_GAP_S,
        },
        "review": {
            "sim_min": config.REVIEW_SIM_MIN,
            "body_sim_min": config.REVIEW_BODY_SIM_MIN,
            "split_sim": config.PERSON_SPLIT_SIM,
            # MMR ordering (Carbonell & Goldstein, SIGIR 1998)
            "mmr_lambda": config.REVIEW_MMR_LAMBDA,
            "mmr_impact": config.REVIEW_MMR_IMPACT,
            "dup_t": config.REVIEW_DUP_T,
        },
        "gates": {
            "face_min_score": config.FACE_MIN_SCORE,
            "face_min_px": config.FACE_MIN_PX,
            "body_min_rel_h": config.BODY_MIN_REL_H,
            "body_min_personness": config.BODY_MIN_PERSONNESS,
            "person_conf": config.PERSON_CONF,
            "face_conf": config.FACE_CONF,
        },
    }


@app.get("/api/status")
def status():
    photos = db.one("SELECT COUNT(*) FROM photos")[0]
    processed = db.one("SELECT COUNT(*) FROM photos WHERE processed=1")[0]
    persons_visible = db.one(
        "SELECT COUNT(*) FROM ("
        " SELECT person_id, COUNT(DISTINCT photo_id) AS n FROM ("
        "   SELECT person_id, photo_id FROM faces WHERE person_id IS NOT NULL"
        "   UNION SELECT person_id, photo_id FROM detections WHERE person_id IS NOT NULL"
        " ) GROUP BY person_id HAVING n >= ?)",
        (config.MIN_PHOTOS,),
    )[0]
    return {
        "photos": photos,
        "processed": processed,
        "pending": photos - processed,
        "persons": persons_visible,
        "faces": db.one("SELECT COUNT(*) FROM faces")[0],
        "detections": db.one("SELECT COUNT(*) FROM detections")[0],
        "worker": worker.status(),
    }


# --------------------------------------------------------------- ingest -----
@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...)):
    tmp_dir = config.DATA_DIR / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for f in files:
        safe = Path(f.filename or "upload.jpg").name
        dest = tmp_dir / f"{time.time_ns()}-{safe}"
        size = 0
        with open(dest, "wb") as fh:
            while chunk := await f.read(1 << 20):
                size += len(chunk)
                if size > config.MAX_UPLOAD_BYTES:
                    fh.close()
                    dest.unlink(missing_ok=True)
                    raise HTTPException(413, f"{safe} exceeds upload limit")
                fh.write(chunk)
        try:
            results.append({"name": safe, **ingest.import_file(dest)})
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        finally:
            dest.unlink(missing_ok=True)
    return {"results": results}


@app.post("/api/scan")
def scan(body: ScanBody):
    try:
        return ingest.scan_dir(body.path)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/recluster")
def recluster():
    return cluster.recluster()


# --------------------------------------------------------------- photos -----
@app.get("/api/photos")
def list_photos(offset: int = 0, limit: int = 60):
    rows = db.query(
        "SELECT id, sha1, taken_at, width, height FROM photos "
        "ORDER BY COALESCE(taken_at, created_at) DESC, id DESC LIMIT ? OFFSET ?",
        (limit, offset),
    )
    total = db.one("SELECT COUNT(*) FROM photos")[0]
    return {
        "total": total,
        "offset": offset,
        "photos": [
            {
                "id": r["id"],
                "thumb": f"/thumbs/{r['sha1']}.jpg",
                "taken_at": r["taken_at"],
                "width": r["width"],
                "height": r["height"],
            }
            for r in rows
        ],
    }


@app.get("/api/photos/{photo_id}")
def photo_detail(photo_id: int):
    p = db.one(
        "SELECT id, sha1, path, width, height, taken_at, processed FROM photos WHERE id=?",
        (photo_id,),
    )
    if p is None:
        raise HTTPException(404, "photo not found")
    seq = _person_seq()
    dets = db.query(
        "SELECT d.id, d.box, d.conf, d.score, d.reason, d.person_id, d.personness, "
        "       d.margin, d.face_id, d.track_id, pe.name "
        "FROM detections d LEFT JOIN persons pe ON pe.id=d.person_id "
        "WHERE d.photo_id=? ORDER BY d.id",
        (photo_id,),
    )
    faces = db.query(
        "SELECT f.id, f.box, f.score, f.person_id, f.quality, f.blur, f.yaw, "
        "       f.sideways, pe.name "
        "FROM faces f LEFT JOIN persons pe ON pe.id=f.person_id "
        "WHERE f.photo_id=? ORDER BY f.id",
        (photo_id,),
    )
    try:
        nbytes = (config.PHOTO_DIR / p["path"]).stat().st_size
    except OSError:
        nbytes = None
    return {
        "id": p["id"],
        "original": f"/originals/{p['id']}",
        "thumb": f"/thumbs/{p['sha1']}.jpg",
        "width": p["width"],
        "height": p["height"],
        "taken_at": p["taken_at"],
        "processed": bool(p["processed"]),
        # essential metadata: what the file is and where it lives
        "path": p["path"],
        "sha1": p["sha1"],
        "bytes": nbytes,
        "detections": [
            {
                "id": r["id"],
                "box": _loads(r["box"]),
                "conf": r["conf"],
                "score": r["score"],
                "reason": r["reason"],
                "personness": r["personness"],
                "margin": r["margin"],
                "face_id": r["face_id"],
                "track_id": r["track_id"],
                "person": {"id": r["person_id"], "name": r["name"],
                           "seq": seq.get(r["person_id"]) if r["person_id"] else None,
                           "display": _display(seq, r["person_id"], r["name"])},
            }
            for r in dets
        ],
        "faces": [
            {
                "id": r["id"],
                "box": _loads(r["box"]),
                "score": r["score"],
                "quality": r["quality"],
                "blur": r["blur"],
                "yaw": r["yaw"],
                "sideways": r["sideways"],
                "person": {"id": r["person_id"], "name": r["name"],
                           "seq": seq.get(r["person_id"]) if r["person_id"] else None,
                           "display": _display(seq, r["person_id"], r["name"])},
            }
            for r in faces
        ],
    }


@app.delete("/api/photos/{photo_id}")
def delete_photo(photo_id: int):
    p = db.one("SELECT id, sha1, path FROM photos WHERE id=?", (photo_id,))
    if p is None:
        raise HTTPException(404, "photo not found")
    db.execute("DELETE FROM photos WHERE id=?", (photo_id,))
    (config.PHOTO_DIR / p["path"]).unlink(missing_ok=True)
    (config.THUMB_DIR / f"{p['sha1']}.jpg").unlink(missing_ok=True)
    stats = cluster.recluster()
    return {"deleted": photo_id, "recluster": stats}


# -------------------------------------------------------------- persons -----
_PERSON_JOIN = """
FROM persons pe
LEFT JOIN (
    SELECT person_id, photo_id FROM faces WHERE person_id IS NOT NULL
    UNION
    SELECT person_id, photo_id FROM detections WHERE person_id IS NOT NULL
) x ON x.person_id = pe.id
"""


@app.get("/api/persons")
def list_persons():
    rows = db.query(
        "SELECT pe.id, pe.name, pe.pinned, COUNT(x.photo_id) AS n "
        + _PERSON_JOIN
        + " GROUP BY pe.id HAVING n >= ? "
        "ORDER BY COALESCE(pe.pinned,0) DESC, (pe.name IS NULL OR pe.name=''), n DESC, pe.id",
        (config.MIN_PHOTOS,),
    )
    seq = _person_seq()
    out = []
    for r in rows:
        out.append(
            {
                "id": r["id"],
                "name": r["name"],
                "display": _display(seq, r["id"], r["name"]),
                "seq": seq.get(r["id"]),
                "pinned": bool(r["pinned"]),
                "photos": r["n"],
                # same _avatar() rule as /api/persons/{id}: newest-first face
                # of THIS person — the list and the person page can never
                # show different heads for the same person
                "cover": _avatar(r["id"]),
            }
        )
    return {"persons": out}


@app.get("/api/persons/{person_id}")
def person_detail(person_id: int, offset: int = 0, limit: int = 60):
    pe = db.one("SELECT id, name, pinned FROM persons WHERE id=?", (person_id,))
    if pe is None:
        raise HTTPException(404, "person not found")
    rows = db.query(
        "SELECT p.id, p.sha1, p.taken_at, p.width, p.height FROM photos p "
        "WHERE p.id IN (SELECT photo_id FROM faces WHERE person_id=?) "
        "   OR p.id IN (SELECT photo_id FROM detections WHERE person_id=?) "
        "ORDER BY COALESCE(p.taken_at, p.created_at) DESC, p.id DESC "
        "LIMIT ? OFFSET ?",
        (person_id, person_id, limit, offset),
    )
    total = db.one(
        "SELECT COUNT(DISTINCT photo_id) FROM ("
        " SELECT photo_id FROM faces WHERE person_id=?"
        " UNION SELECT photo_id FROM detections WHERE person_id=?)",
        (person_id, person_id),
    )[0]
    face_photos = {r[0] for r in db.query(
        "SELECT DISTINCT photo_id FROM faces WHERE person_id=?", (person_id,))}
    seq = _person_seq()
    det_reasons: dict[int, str] = {}
    det_margins: dict[int, float] = {}
    for r in db.query(
        "SELECT photo_id, reason, margin FROM detections WHERE person_id=? AND reason IS NOT NULL",
        (person_id,),
    ):
        det_reasons.setdefault(r[0], r[1])
        if r[2] is not None:
            det_margins.setdefault(r[0], r[2])
    by_reason = db.query(
        "SELECT reason, COUNT(*) AS n FROM detections WHERE person_id=? AND reason IS NOT NULL "
        "GROUP BY reason",
        (person_id,),
    )
    return {
        "id": pe["id"],
        "name": pe["name"],
        "display": _display(seq, pe["id"], pe["name"]),
        "seq": seq.get(pe["id"]),
        "pinned": bool(pe["pinned"]),
        # same _avatar() rule as the persons list cover: newest-first face
        "avatar": _avatar(pe["id"]),
        "total": total,
        "reasons": {r["reason"]: r["n"] for r in by_reason},
        "photos": [
            {
                "id": r["id"],
                "thumb": f"/thumbs/{r['sha1']}.jpg",
                "taken_at": r["taken_at"],
                "width": r["width"],
                "height": r["height"],
                "reason": ("face" if r["id"] in face_photos
                           else det_reasons.get(r["id"], "body+time")),
                "margin": det_margins.get(r["id"]),
            }
            for r in rows
        ],
    }


@app.patch("/api/persons/{person_id}")
def rename_person(person_id: int, body: RenameBody):
    pe = db.one("SELECT id, name, pinned FROM persons WHERE id=?", (person_id,))
    if pe is None:
        raise HTTPException(404, "person not found")
    fields = _fields_set(body)
    pinned = bool(pe["pinned"])
    if "pinned" in fields:
        pinned = bool(body.pinned)
        db.execute("UPDATE persons SET pinned=? WHERE id=?",
                   (1 if pinned else 0, person_id))
    if "name" not in fields:
        # pin-only update (or empty body): leave the name untouched
        return {"id": person_id, "name": pe["name"], "pinned": pinned}
    name = (body.name or "").strip() or None
    if name:
        other = db.one("SELECT id FROM persons WHERE name=? AND id!=?", (name, person_id))
        if other:
            db.execute("UPDATE faces SET person_id=? WHERE person_id=?", (other["id"], person_id))
            db.execute(
                "UPDATE detections SET person_id=? WHERE person_id=?", (other["id"], person_id)
            )
            db.execute("DELETE FROM persons WHERE id=?", (person_id,))
            db.purge_person_refs([person_id], move_labels_to=other["id"])
            stats = cluster.recluster()
            return {"id": other["id"], "name": name, "merged_from": person_id, "recluster": stats}
    db.execute("UPDATE persons SET name=? WHERE id=?", (name, person_id))
    return {"id": person_id, "name": name, "pinned": pinned}


@app.post("/api/persons/{person_id}/merge")
def merge_person(person_id: int, body: MergeBody):
    """Rough merge: move every face/detection of this person into another one."""
    if person_id == body.target_id:
        raise HTTPException(400, "cannot merge a person into itself")
    src = db.one("SELECT id FROM persons WHERE id=?", (person_id,))
    dst = db.one("SELECT id FROM persons WHERE id=?", (body.target_id,))
    if src is None or dst is None:
        raise HTTPException(404, "person not found")
    db.execute("UPDATE faces SET person_id=? WHERE person_id=?", (body.target_id, person_id))
    db.execute("UPDATE detections SET person_id=? WHERE person_id=?",
               (body.target_id, person_id))
    db.execute("DELETE FROM persons WHERE id=?", (person_id,))
    db.purge_person_refs([person_id], move_labels_to=body.target_id)
    stats = cluster.recluster()
    return {"merged": person_id, "into": body.target_id, "recluster": stats}


@app.post("/api/persons/{person_id}/reset")
def person_reset(person_id: int):
    """Reset one person (Ente 'Reset person' -> deleteCGroup): every face and
    detection goes back to the unassigned pool and the person row disappears."""
    pe = db.one("SELECT id FROM persons WHERE id=?", (person_id,))
    if pe is None:
        raise HTTPException(404, "person not found")
    nf = db.one("SELECT COUNT(*) AS n FROM faces WHERE person_id=?", (person_id,))[0]
    nd = db.one("SELECT COUNT(*) AS n FROM detections WHERE person_id=?", (person_id,))[0]
    db.execute("UPDATE faces SET person_id=NULL, reviewed=NULL WHERE person_id=?",
               (person_id,))
    db.execute(
        "UPDATE detections SET person_id=NULL, reason=NULL, score=NULL, margin=NULL "
        "WHERE person_id=?", (person_id,))
    db.execute("DELETE FROM persons WHERE id=?", (person_id,))
    db.purge_person_refs([person_id])
    db.execute("INSERT INTO review_log(action, payload, created_at) VALUES (?,?,?)",
               ("person_reset", json.dumps({"person_id": person_id}), db.now()))
    return {"reset": person_id, "faces": nf, "detections": nd}


@app.get("/api/persons/{person_id}/faces")
def person_faces(person_id: int, limit: int = 60):
    """The clustered face crops for one person — quick visual cluster check."""
    pe = db.one("SELECT id FROM persons WHERE id=?", (person_id,))
    if pe is None:
        raise HTTPException(404, "person not found")
    rows = db.query(
        "SELECT f.id, f.photo_id, f.score, COALESCE(p.taken_at, 0) AS t "
        "FROM faces f JOIN photos p ON p.id=f.photo_id "
        "WHERE f.person_id=? ORDER BY t DESC, f.id DESC LIMIT ?",
        (person_id, limit),
    )
    return {
        "faces": [
            {
                "id": r["id"],
                "photo_id": r["photo_id"],
                "thumb": f"/face_thumb/{r['id']}.jpg",
                "score": r["score"],
                "taken_at": r["t"] or None,
            }
            for r in rows
        ]
    }


FACE_THUMB_PX = 160


def _cache_path(subdir: str, *parts, suffix: str = "") -> Path:
    """Content-addressed cache file for a derived thumbnail.

    The key is the crop's SOURCE content (photo path + box), never the
    row id: face/detection ids are SQLite rowids and get REUSED after a
    re-scan or full reset, which used to poison this cache with crops of
    a different face that happened to take the same id.
    """
    material = "|".join(str(p) for p in parts)
    digest = hashlib.sha1(material.encode("utf-8")).hexdigest()[:24]
    return config.THUMB_DIR / subdir / f"{digest}{suffix}.jpg"


@app.get("/face_thumb/{face_id}.jpg")
def face_thumb(face_id: int):
    """Square crop of one detected face, generated lazily and cached."""
    row = db.one(
        "SELECT f.box, p.path FROM faces f JOIN photos p ON p.id=f.photo_id WHERE f.id=?",
        (face_id,),
    )
    if row is None:
        raise HTTPException(404, "face not found")
    cache = _cache_path("faces", row["path"], row["box"])
    if cache.is_file():
        return FileResponse(cache, media_type="image/jpeg")
    x1, y1, x2, y2 = _loads(row["box"])
    side = max(x2 - x1, y2 - y1) * 1.4
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    with Image.open(config.PHOTO_DIR / row["path"]) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        left, top = int(cx - side / 2), int(cy - side / 2)
        crop = im.crop((left, top, int(left + side), int(top + side)))
        crop.thumbnail((FACE_THUMB_PX, FACE_THUMB_PX), Image.LANCZOS)
        cache.parent.mkdir(parents=True, exist_ok=True)
        crop.save(cache, "JPEG", quality=85)
    return FileResponse(cache, media_type="image/jpeg")


@app.get("/face_context/{face_id}.jpg")
def face_context(face_id: int):
    """The boxed selection behind a face: the containing person box if any
    (the box as drawn in the viewer), else a wide context crop."""
    row = db.one(
        "SELECT f.box, f.photo_id, p.path FROM faces f JOIN photos p ON p.id=f.photo_id WHERE f.id=?",
        (face_id,),
    )
    if row is None:
        raise HTTPException(404, "face not found")
    cache = _cache_path("faces", row["path"], row["box"], suffix="_ctx")
    if cache.is_file():
        return FileResponse(cache, media_type="image/jpeg")
    x1, y1, x2, y2 = _loads(row["box"])
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    # the person box containing the face center (the visible "selection")
    bx1, by1, bx2, by2 = x1, y1, x2, y2
    for d in db.conn().execute(
            "SELECT box FROM detections WHERE photo_id=?", (row["photo_id"],)):
        dx1, dy1, dx2, dy2 = _loads(d["box"])
        if (dx1 <= cx <= dx2 and dy1 <= cy <= dy2
                and (dx2 - dx1) * (dy2 - dy1) > (x2 - x1) * (y2 - y1)):
            bx1, by1, bx2, by2 = dx1, dy1, dx2, dy2
            break
    else:
        # no containing box: wide context instead of a duplicate face crop
        side = max(x2 - x1, y2 - y1) * 4.0
        bx1, by1 = cx - side / 2, cy - side / 2
        bx2, by2 = cx + side / 2, cy + side / 2
    m = (bx2 - bx1) * 0.05  # small margin around the box
    with Image.open(config.PHOTO_DIR / row["path"]) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        crop = im.crop((int(bx1 - m), int(by1 - m), int(bx2 + m), int(by2 + m)))
        crop.thumbnail((400, 400), Image.LANCZOS)
        cache.parent.mkdir(parents=True, exist_ok=True)
        crop.save(cache, "JPEG", quality=85)
    return FileResponse(cache, media_type="image/jpeg")


@app.get("/det_thumb/{det_id}.jpg")
def det_thumb(det_id: int):
    """Crop of one person detection box (back-view review thumbs)."""
    row = db.one(
        "SELECT d.box, p.path FROM detections d JOIN photos p ON p.id=d.photo_id WHERE d.id=?",
        (det_id,),
    )
    if row is None:
        raise HTTPException(404, "detection not found")
    cache = _cache_path("dets", row["path"], row["box"])
    if cache.is_file():
        return FileResponse(cache, media_type="image/jpeg")
    x1, y1, x2, y2 = _loads(row["box"])
    m = (x2 - x1) * 0.05
    with Image.open(config.PHOTO_DIR / row["path"]) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        crop = im.crop((int(x1 - m), int(y1 - m), int(x2 + m), int(y2 + m)))
        crop.thumbnail((220, 220), Image.LANCZOS)
        cache.parent.mkdir(parents=True, exist_ok=True)
        crop.save(cache, "JPEG", quality=85)
    return FileResponse(cache, media_type="image/jpeg")


@app.middleware("http")
async def _no_cache_static(request, call_next):
    """JS/CSS must always revalidate: browsers caching stale app.js produce
    broken renders against a newer API (empty thumbs, dead click handlers).

    The id-keyed derived thumbs (face/det crops) are also revalidated —
    their URL is stable across a re-scan while the content it points at is
    not, so a browser that keeps them would show a DIFFERENT face under the
    same id. Content-addressed /thumbs/{sha1}.jpg stays freely cacheable."""
    response = await call_next(request)
    path = request.url.path
    if path.endswith((".js", ".css", ".html")) or path.startswith(
            ("/face_thumb/", "/face_context/", "/det_thumb/")):
        response.headers["Cache-Control"] = "no-cache"
    return response


def _means(sel_col: str, sel_tbl: str) -> dict:
    """Person means (same math as clustering: mean of unit embeddings)."""
    import numpy as np

    acc: dict = {}
    for rr in db.conn().execute(f"SELECT person_id, {sel_col} AS e FROM {sel_tbl} "
                                "WHERE person_id IS NOT NULL"):
        uv = _unit(rr["e"])
        if uv is not None:
            acc.setdefault(rr["person_id"], []).append(uv)
    return {pid: np.mean(v, axis=0) for pid, v in acc.items()}


def _face_vec_map() -> dict:
    out: dict = {}
    for rr in db.conn().execute(
            "SELECT person_id, face_emb FROM faces "
            "WHERE person_id IS NOT NULL AND face_emb IS NOT NULL"):
        uv = _unit(rr["face_emb"])
        if uv is not None:
            out.setdefault(rr["person_id"], []).append(uv)
    return out


def _median_cross(a_vecs: list, b_vecs: list):
    """Median of pairwise cosines (Ente borrow).

    The median damps a few misleading face pairs that drag a mean/centroid
    apart — exactly the look-alike contamination problem."""
    import numpy as np

    if not a_vecs or not b_vecs:
        return None
    # one matmul instead of |A|*|B| python-level np.dot calls: the review
    # endpoint scores every person pair and this is its inner loop
    A = np.asarray(a_vecs[:50])
    B = np.asarray(b_vecs[:50])
    sims = np.sort((A @ B.T).ravel())
    return float(sims[len(sims) // 2])


def _unit(b) -> "np.ndarray | None":
    if b is None:
        return None
    import numpy as np
    v = np.frombuffer(b, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else None


@app.get("/api/suggest/{kind}/{ref_id}")
def api_suggest(kind: str, ref_id: int):
    """Person candidates for a face/body, sorted by similarity (best first)."""
    import numpy as np

    if kind == "person":
        # merge picker: other persons sorted by face-mean similarity
        v = None
        acc = [np.frombuffer(r["face_emb"], dtype=np.float32)
               for r in db.conn().execute("SELECT face_emb FROM faces "
                                          "WHERE person_id=? AND face_emb IS NOT NULL", (ref_id,))]
        if acc:
            m = np.mean([a / (np.linalg.norm(a) + 1e-9) for a in acc], axis=0)
            v = m / (np.linalg.norm(m) + 1e-9)
        names = {r["id"]: r["name"] for r in db.conn().execute("SELECT id, name FROM persons")}
        seq = _person_seq()
        ordered = sorted(db.conn().execute("SELECT id FROM persons WHERE id != ? ORDER BY id",
                                           (ref_id,)), key=lambda r: r["id"])
        means = _means("face_emb", "faces")
        fvec = _face_vec_map()
        sims = {}
        if v is not None:
            for r in ordered:
                s = _median_cross([v], fvec.get(r["id"], []))
                if s is None:
                    mv = means.get(r["id"])
                    if mv is not None:
                        s = float(np.dot(v, mv / (np.linalg.norm(mv) + 1e-9)))
                if s is not None:
                    sims[r["id"]] = s
        items = []
        for r in ordered:
            pid = r["id"]
            items.append({
                "person_id": pid,
                "display": _display(seq, pid, names.get(pid)),
                "seq": seq.get(pid),
                "name": names.get(pid),
                "photos": _person_photos(pid),
                "sim": round(sims[pid], 4) if pid in sims else None,
                "thumb": _avatar(pid),
            })
        items.sort(key=lambda x: (x["sim"] is not None, x["sim"] or -1), reverse=True)
        return {"items": items}

    if kind not in ("face", "det"):
        raise HTTPException(400, "kind must be face, det or person")
    if kind == "face":
        row = db.one("SELECT face_emb AS emb, NULL AS face_id FROM faces WHERE id=?", (ref_id,))
    else:
        row = db.one("SELECT body_emb AS emb, face_id FROM detections WHERE id=?", (ref_id,))
    if row is None:
        raise HTTPException(404, "ref not found")
    v = _unit(row["emb"])
    names = {r["id"]: r["name"] for r in db.conn().execute("SELECT id, name FROM persons")}
    ordered = sorted(db.conn().execute("SELECT id FROM persons ORDER BY id"), key=lambda r: r["id"])
    seq = {r["id"]: i + 1 for i, r in enumerate(ordered)}
    # candidate similarity per person: matching cue first, other cue as backup
    sims: dict[int, float] = {}
    if v is not None:
        face_means = _means("face_emb", "faces")
        means = _means("body_emb", "detections") if kind == "det" else face_means
        for r in ordered:
            mv = means.get(r["id"])
            if mv is not None:
                sims[r["id"]] = float(np.dot(v, mv))
        if kind == "det" and row["face_id"] is not None:
            frow = db.one("SELECT face_emb FROM faces WHERE id=?", (row["face_id"],))
            fv = _unit(frow["face_emb"]) if frow else None
            if fv is not None:
                for r in ordered:
                    mv = face_means.get(r["id"])
                    if mv is not None:
                        sims[r["id"]] = max(sims.get(r["id"], -1.0), float(np.dot(fv, mv)))
    items = []
    for r in ordered:
        pid = r["id"]
        n = db.one(
            "SELECT COUNT(DISTINCT photo_id) AS n FROM "
            "((SELECT photo_id FROM faces WHERE person_id=? UNION "
            "  SELECT photo_id FROM detections WHERE person_id=?))",
            (pid, pid))["n"]
        items.append({
            "person_id": pid,
            "display": _display(seq, pid, names.get(pid)),
            "seq": seq.get(pid),
            "name": names.get(pid),
            "photos": n,
            "sim": round(sims[pid], 4) if pid in sims else None,
            "thumb": _avatar(pid),
        })
    items.sort(key=lambda x: (x["sim"] is not None, x["sim"] or -1), reverse=True)
    return {"items": items}


# --------------------------------------------------------------- review ----
class ReviewBody(BaseModel):
    """Review action payload (pairs / back-view dets).

    Kept separate from MergeBody on purpose: both models used to be called
    MergeBody, so FastAPI bound whichever was defined first and /persons/{id}/merge
    silently ignored candidate_id/det_ids."""
    target_id: int
    candidate_id: int | None = None
    det_ids: list[int] | None = None


def _best_face_thumb(pid: int) -> str | None:
    """The person's head image — the shared _avatar() rule (newest-first face
    of this person, else its own newest detection crop)."""
    return _avatar(pid)


def _person_thumbs(pid: int, limit: int = 6) -> list[dict]:
    """Representative thumbs of a person's member photos (face crops first).

    Each entry carries the source photo id so the UI can open the big image."""
    out: list[dict] = []
    seen: set[str] = set()
    for r in db.query(
            "SELECT f.id AS fid, f.photo_id, p.sha1 FROM faces f "
            "JOIN photos p ON p.id=f.photo_id "
            "WHERE f.person_id=? ORDER BY COALESCE(f.quality,0) DESC, f.id DESC", (pid,)):
        if r["sha1"] in seen:
            continue
        seen.add(r["sha1"])
        out.append({"url": f"/face_thumb/{r['fid']}.jpg", "photo_id": r["photo_id"]})
        if len(out) >= limit:
            return out
    for r in db.query(
            "SELECT d.photo_id, p.sha1 FROM detections d JOIN photos p ON p.id=d.photo_id "
            "WHERE d.person_id=? GROUP BY p.sha1 ORDER BY d.id DESC", (pid,)):
        if r["sha1"] in seen:
            continue
        seen.add(r["sha1"])
        out.append({"url": f"/thumbs/{r['sha1']}.jpg", "photo_id": r["photo_id"]})
        if len(out) >= limit:
            break
    return out


def _person_photos(pid: int) -> int:
    return db.one(
        "SELECT COUNT(DISTINCT photo_id) AS n FROM ("
        "  SELECT photo_id FROM faces WHERE person_id=? UNION "
        "  SELECT photo_id FROM detections WHERE person_id=?)",
        (pid, pid))["n"]


@app.get("/api/review")
def review_queue(limit: int = 24, target_id: int | None = None):
    """Merge review (Ente-style): the target person on the left (one
    representative), a similar-but-unconfirmed cluster on the right (several
    of its photos); the user decides whether to merge the candidate in.

    Candidate pairs fall in the uncertain face-similarity band
    [REVIEW_SIM_MIN, FACE_MERGE_SIM): similar enough to ask, not enough to
    merge automatically. Co-occurring pairs are provably different people and
    are never suggested; decided pairs are never re-suggested."""
    import numpy as np

    # Score each pair on a small sample of faces per person, exactly as Ente
    # does (people.ts:289/318 samples <=50). The pairwise loop is
    # O(persons^2 * faces^2) and a photo library has thousands of person pairs;
    # without the cap the endpoint took ~6s and, once the MMR pass ran over the
    # whole pool, hung outright.
    SCORE_FACES = 8
    means: dict[int, np.ndarray] = {}
    acc: dict[int, list] = {}
    for r in db.query("SELECT person_id, face_emb FROM faces WHERE person_id IS NOT NULL"):
        v = db.blob_to_vec(r["face_emb"])
        if v is not None:
            acc.setdefault(r["person_id"], []).append(_unit(r["face_emb"]))
    for pid in list(acc):
        if len(acc[pid]) > SCORE_FACES:
            step = max(1, len(acc[pid]) // SCORE_FACES)
            acc[pid] = acc[pid][::step][:SCORE_FACES]
    face_vecs = {pid: [v for v in vs if v is not None] for pid, vs in acc.items()}
    bacc: dict[int, list] = {}
    for r in db.query("SELECT person_id, body_emb FROM detections "
                      "WHERE person_id IS NOT NULL AND body_emb IS NOT NULL"):
        v = _unit(r["body_emb"])
        if v is not None:
            bacc.setdefault(r["person_id"], []).append(v)
    body_vecs = bacc
    for pid, vs in face_vecs.items():
        if vs:
            m = np.mean(np.stack(vs), axis=0)
            n = float(np.linalg.norm(m))
            if n > 1e-6:
                means[pid] = m / n

    # merged/rejected stick forever; a SKIP is a defer, not a decision -
    # skipped pairs resurface after 24h so the queue can never be drained
    # into a hidden pile of half-seen candidates
    now_ts = time.time()
    decided = set()
    for r in db.query("SELECT a, b, status, updated_at FROM review_pairs"):
        if r["status"] == "skipped":
            if now_ts - float(r["updated_at"] or 0) < 24 * 3600:
                decided.add((r["a"], r["b"]))
        else:
            decided.add((r["a"], r["b"]))
    by_photo: dict[int, set[int]] = {}
    for r in db.query(
            "SELECT photo_id, person_id FROM ("
            "  SELECT photo_id, person_id FROM faces WHERE person_id IS NOT NULL UNION "
            "  SELECT photo_id, person_id FROM detections WHERE person_id IS NOT NULL)"):
        by_photo.setdefault(r["photo_id"], set()).add(r["person_id"])
    cooc_pairs = set()
    for pids in by_photo.values():
        for a in pids:
            for b in pids:
                if a < b:
                    cooc_pairs.add((a, b))

    seq = _person_seq()
    names = {r["id"]: r["name"] for r in db.query("SELECT id, name FROM persons")}
    # the LEFT side (target = absorbs the candidate) must be a NAMED person:
    # unnamed clusters are not yet trusted anchors
    named = {pid for pid, n in names.items() if n and n.strip()}
    photos = {r["id"]: r["n"] for r in db.query(
        "SELECT person_id AS id, COUNT(DISTINCT photo_id) AS n FROM ("
        "  SELECT photo_id, person_id FROM faces WHERE person_id IS NOT NULL UNION "
        "  SELECT photo_id, person_id FROM detections WHERE person_id IS NOT NULL) "
        "GROUP BY person_id")}

    person_span: dict[int, tuple] = {}
    for r in db.query(
            "SELECT x.person_id AS id, MIN(p.taken_at) AS lo, MAX(p.taken_at) AS hi "
            "FROM (SELECT photo_id, person_id FROM faces WHERE person_id IS NOT NULL "
            "UNION SELECT photo_id, person_id FROM detections WHERE person_id IS NOT NULL) x "
            "JOIN photos p ON p.id = x.photo_id GROUP BY x.person_id"):
        person_span[r["id"]] = (r["lo"], r["hi"])

    # "not this person" memory: what was rejected for a target stays rejected,
    # and so does anything that reads as the same person as the rejected
    # cluster - a split person's fragments must not queue up one after another
    rejected_for: dict[int, list] = {}
    for r in db.query("SELECT target_person_id AS t, emb FROM review_rejects"):
        v = _unit(r["emb"])
        if v is not None:
            rejected_for.setdefault(r["t"], []).append(v)

    pairs = []
    # only persons that still exist in the persons table: faces/detections
    # may carry dangling person_ids, and a dangling id has no display seq —
    # surfacing it would put a DB id where a label number belongs
    pids = [pid for pid in sorted(means) if pid in names]
    for i, a in enumerate(pids):
        for b in pids[i + 1:]:
            key = (a, b) if a < b else (b, a)
            if key in decided or key in cooc_pairs:
                continue
            avs, bvs = face_vecs.get(a, []), face_vecs.get(b, [])
            sim = _median_cross(avs, bvs)
            if sim is None and a in means and b in means:
                sim = float(np.dot(means[a], means[b]))
            if sim is None:
                continue
            best = None
            if avs and bvs:
                best = float(np.max(np.asarray(avs) @ np.asarray(bvs).T))
            # smart score: a harsh-angle fragment (weak median) can still
            # carry one clear face pair (best) - composite ranking surfaces
            # it; the DISPLAYED score is the sorted score (no mismatch)
            score = (0.7 * best + 0.3 * sim) if best is not None else sim
            # body channel: same-person bodies at short gaps score 0.75+,
            # different people 0.29-0.46 (Market-1501 calibrated) - a
            # fragment with a garbage face embedding can still be a clear
            # BODY match (harsh-angle faces)
            body_best = None
            ab, bb = body_vecs.get(a, []), body_vecs.get(b, [])
            if ab and bb:
                body_best = float(np.max(np.asarray(ab) @ np.asarray(bb).T))
                bmed = _median_cross(ab, bb)
                body_score = 0.7 * body_best + 0.3 * (bmed or 0.0)
                if body_score > score:
                    score = body_score
            gap_days = None
            ta, tb = person_span.get(a), person_span.get(b)
            if ta and tb:
                gap_days = max(0.0, max(ta[0], tb[0]) - min(ta[1], tb[1])) / 86400.0
            # no upper bound: pairs above the auto-merge bar that did NOT
            # merge (blocked by compatibility/time) are the top review items
            if score >= config.REVIEW_SIM_MIN:
                # target = the NAMED side when only one is named
                if a in named and b not in named:
                    tgt, cand = a, b
                elif b in named and a not in named:
                    tgt, cand = b, a
                else:
                    tgt, cand = (a, b) if photos.get(a, 0) >= photos.get(b, 0) else (b, a)
                # a candidate that reads as the same person as something already
                # rejected is the same answer twice - skip it. Two ways to read
                # "same person": clearly so (>= REVIEW_DUP_T), or the rejected
                # cluster explains this candidate at least as well as the target
                # does - the case where one person is split into fragments and
                # every fragment comes up as its own question on the same target.
                cm = means.get(cand)
                if cm is not None:
                    skip = False
                    for rv in rejected_for.get(tgt, []):
                        s = float(np.dot(cm, rv))
                        if s >= config.REVIEW_DUP_T or s >= float(score) - 1e-6:
                            skip = True
                            break
                    if skip:
                        continue
                # show EVERYTHING ranked; a named side becomes the target
                # when one exists, otherwise the larger side does. Hiding
                # anonymous pairs was the fragmentation black hole.
                pairs.append((score, tgt, cand, sim, best, gap_days, body_best))
    pairs.sort(reverse=True)

    # MMR (Carbonell & Goldstein, SIGIR 1998) picks each next question as the
    # best trade-off between relevance and "not the question you just answered".
    # Relevance is the composite score computed above (a parameterisation, not a
    # new scoring rule); Sim2 is how much a question resembles one already
    # shown, measured on the candidate sides' centroids. Without this, a person
    # split into fragments comes back as the same question several times in a
    # row - which is exactly what Ente does (people.ts:336 sorts on similarity
    # alone and has no diversity pass at all).
    # MMR is O(k^2) in the queue length and the candidate pool is every person
    # pair in the band (tens of thousands), so re-rank only what can actually be
    # shown near the top and leave the tail in score order. Re-ranking beyond
    # the first screen of questions buys nothing and used to hang the endpoint.
    lam = config.REVIEW_MMR_LAMBDA
    head_n = max(64, int(config.REVIEW_MMR_TOP))
    head, tail = pairs[:head_n], pairs[head_n:]
    if not head:
        pairs = tail
    else:
        dim = len(next(iter(means.values())))
        V = np.stack([means.get(p[2], np.zeros(dim, np.float32)) for p in head])
        rel = np.array([float(p[0]) for p in head], dtype=np.float64)
        if config.REVIEW_MMR_IMPACT:
            # expected error reduction (Roy & McCallum, ICML 2001), in its cheap
            # form: merging A and B settles |A|*|B| pairs at once, so a question
            # about two big clusters outranks one about two singletons at the
            # same similarity
            rel = rel * np.log1p([photos.get(p[1], 0) * photos.get(p[2], 0)
                                  for p in head])
        # Sim2(i) = max similarity to anything already picked, maintained as a
        # running vector so each pick is one matvec instead of a rescan of the
        # picked list (the naive form made the endpoint take seconds).
        maxsim = np.zeros(len(head))
        picked = np.zeros(len(head), dtype=bool)
        order = []
        for _ in range(len(head)):
            vals = lam * rel - (1.0 - lam) * maxsim
            vals[picked] = -np.inf
            j = int(np.argmax(vals))
            picked[j] = True
            order.append(head[j])
            np.maximum(maxsim, V @ V[j], out=maxsim)
        pairs = order + tail

    items = []
    for sim, tgt, cand, med_s, best_s, gap_d, body_best in pairs:
        items.append({
            "id": f"{tgt}:{cand}",
            "kind": "pair",
            "target_id": tgt,
            "candidate_id": cand,
            "det_ids": None,
            "sim": round(sim, 3),
            "sim_med": None if med_s is None else round(med_s, 3),
            "sim_best": None if best_s is None else round(best_s, 3),
            "gap_days": None if gap_d is None else round(gap_d, 1),
            "body_best": None if body_best is None else round(body_best, 3),
            "target": {
                "person_id": tgt,
                "display": _display(seq, tgt, names.get(tgt)),
                "seq": seq.get(tgt),
                "name": names.get(tgt),
                "photos": photos.get(tgt, 0),
                "thumb": _best_face_thumb(tgt),
            },
            "candidate": {
                "person_id": cand,
                "display": _display(seq, cand, names.get(cand)),
                "seq": seq.get(cand),
                "name": names.get(cand),
                "photos": photos.get(cand, 0),
                "thumbs": _person_thumbs(cand),
            },
        })

    # back-view (no-face) boxes with high body similarity: allowed into review
    bacc: dict[int, list] = {}
    for r in db.query(
            "SELECT person_id, body_emb FROM detections WHERE person_id IS NOT NULL"):
        v = db.blob_to_vec(r["body_emb"])
        if v is not None:
            bacc.setdefault(r["person_id"], []).append(v)
    body_means = {}
    for pid, vs in bacc.items():
        m = np.mean(np.stack(vs), axis=0)
        n = float(np.linalg.norm(m))
        if n > 1e-6:
            body_means[pid] = m / n
    # once any decision exists for a back view (skip/reject/merge) it is done:
    # never re-offer it, whatever the target person
    decided_dets = {r["det_id"]
                    for r in db.query("SELECT det_id FROM review_dets")}
    # members' capture times per person: back-view suggestions need a member
    # within the same 5h window as the claim rule ("almost claimable")
    member_ts: dict[int, list[float]] = {}
    for r in db.query(
            "SELECT u.person_id AS pid, COALESCE(p.taken_at, 0) AS t FROM ("
            "  SELECT person_id, photo_id FROM faces WHERE person_id IS NOT NULL UNION "
            "  SELECT person_id, photo_id FROM detections WHERE person_id IS NOT NULL) u "
            "JOIN photos p ON p.id=u.photo_id"):
        member_ts.setdefault(r["pid"], []).append(r["t"])
    for r in db.query(
            "SELECT d.id, d.photo_id, d.body_emb, d.personness, d.box, p.width w, p.height h, "
            "       COALESCE(p.taken_at, 0) AS t "
            "FROM detections d JOIN photos p ON p.id=d.photo_id "
            "WHERE d.person_id IS NULL AND d.face_id IS NULL"):
        # same quality gates as the claim path: no hands/mush, no tiny targets
        if (r["personness"] or 0.0) < config.BODY_MIN_PERSONNESS:
            continue
        try:
            x1, y1, x2, y2 = _loads(r["box"])
            if (y2 - y1) / max(r["h"], 1) < config.BODY_MIN_REL_H:
                continue
        except Exception:
            continue
        if r["id"] in decided_dets:
            continue
        v = db.blob_to_vec(r["body_emb"])
        if v is None:
            continue
        best = None
        for pid, m in body_means.items():
            if pid not in names:
                continue  # dangling person_id — no display seq, no identity
            if target_id is None and pid not in named:
                continue  # auto queue: left side must be a named person
            if r["t"] > 0 and not any(
                    mt > 0 and abs(r["t"] - mt) <= config.BODY_MAX_GAP_S
                    for mt in member_ts.get(pid, [])):
                continue  # outside the claim window — not "almost claimable"
            s = float(np.dot(v, m))
            if s >= config.REVIEW_BODY_SIM_MIN and (best is None or s > best[0]):
                best = (s, pid)
        if best is None:
            continue
        sim, tgt = best
        items.append({
            "id": f"det:{r['id']}",
            "kind": "dets",
            "target_id": tgt,
            "candidate_id": None,
            "det_ids": [r["id"]],
            "sim": round(sim, 3),
            "target": {
                "person_id": tgt,
                "display": _display(seq, tgt, names.get(tgt)),
                "seq": seq.get(tgt),
                "name": names.get(tgt),
                "photos": photos.get(tgt, 0),
                "thumb": _best_face_thumb(tgt),
            },
            "candidate": {
                "person_id": None,
                "display": "Back view",
                "photos": 1,
                "thumbs": [{"url": f"/det_thumb/{r['id']}.jpg", "photo_id": r["photo_id"]}],
            },
        })

    # plain similarity order: the visible percentages descend down the queue
    items.sort(key=lambda x: x["sim"] or -1, reverse=True)
    if target_id is not None:
        items = [it for it in items if it["target_id"] == target_id]
    return {"items": items[:limit], "remaining": len(items)}


def _det_containing_face(face_id: int) -> int | None:
    """The single detection whose box contains this face (if unambiguous)."""
    frow = db.one("SELECT box, photo_id FROM faces WHERE id=?", (face_id,))
    if frow is None:
        return None
    try:
        x1, y1, x2, y2 = _loads(frow["box"])
    except Exception:
        return None
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    hits = []
    for r in db.query("SELECT id, box FROM detections WHERE photo_id=?", (frow["photo_id"],)):
        b = _loads(r["box"])
        if b and b[0] <= cx <= b[2] and b[1] <= cy <= b[3]:
            hits.append(r["id"])
    return hits[0] if len(hits) == 1 else None


@app.post("/api/review/merge")
def review_merge(body: ReviewBody):
    """Human decision: the candidate (cluster or back view) IS the target."""
    a = body.target_id
    if db.one("SELECT id FROM persons WHERE id=?", (a,)) is None:
        raise HTTPException(404, "person not found")
    if body.det_ids:
        db.execute("INSERT INTO review_log(action, payload, created_at) VALUES (?,?,?)",
                   ("merge_dets", json.dumps({"target": a, "dets": list(body.det_ids)}),
                    db.now()))
        n = 0
        for did in body.det_ids:
            row = db.one("SELECT id FROM detections WHERE id=? AND person_id IS NULL", (did,))
            if row is None:
                continue
            db.execute(
                "UPDATE detections SET person_id=?, reason='manual', score=1.0, margin=NULL "
                "WHERE id=?", (a, did))
            db.execute(
                "INSERT OR REPLACE INTO review_dets(person_id, det_id, status, updated_at) "
                "VALUES (?,?,?,?)", (a, did, "merged", db.now()))
            n += 1
        return {"merged": True, "detections": n, "target_id": a}
    b = body.candidate_id
    if b is None or a == b or db.one("SELECT id FROM persons WHERE id=?", (b,)) is None:
        raise HTTPException(404, "person not found")
    moved_f = [r["id"] for r in db.query("SELECT id FROM faces WHERE person_id=?", (b,))]
    moved_d = [r["id"] for r in db.query("SELECT id FROM detections WHERE person_id=?", (b,))]
    b_row = db.one("SELECT name FROM persons WHERE id=?", (b,))
    db.execute("INSERT INTO review_log(action, payload, created_at) VALUES (?,?,?)",
               ("merge_pair", json.dumps({"a": a, "b": b,
                                          "name_b": b_row["name"] if b_row else None,
                                          "faces": moved_f, "dets": moved_d}), db.now()))
    nf = len(moved_f)
    nd = len(moved_d)
    # reviewed=1 pins the moved faces: a human merge decision is sticky and
    # clustering can never evict/demote them again
    db.execute("UPDATE faces SET person_id=?, reviewed=1 WHERE person_id=?", (a, b))
    db.execute("UPDATE detections SET person_id=?, reason='manual' "
               "WHERE person_id=? AND COALESCE(reason,'') != 'manual'", (a, b))
    db.execute("UPDATE detections SET person_id=? WHERE person_id=?", (a, b))
    db.execute("UPDATE labels SET person_id=? WHERE person_id=?", (a, b))
    db.execute("DELETE FROM persons WHERE id=?", (b,))
    # b's id is free again (rowid alias) - drop its stale review rows BEFORE
    # the new decision is written below, or they would collide with it
    db.purge_person_refs([b], move_labels_to=a)
    lo, hi = min(a, b), max(a, b)
    db.execute(
        "INSERT OR REPLACE INTO review_pairs(a, b, status, updated_at) VALUES (?,?,?,?)",
        (lo, hi, "merged", db.now()))
    return {"merged": True, "faces": nf, "detections": nd, "target_id": a}


@app.post("/api/review/skip")
def review_skip(body: ReviewBody):
    """Skip for now — recorded so the item does not immediately re-appear."""
    if body.det_ids:
        for did in body.det_ids:
            db.execute(
                "INSERT OR REPLACE INTO review_dets(person_id, det_id, status, updated_at) "
                "VALUES (?,?,?,?)", (body.target_id, did, "skipped", db.now()))
        db.execute("INSERT INTO review_log(action, payload, created_at) VALUES (?,?,?)",
                   ("det_decision", json.dumps({"target": body.target_id,
                                                "dets": list(body.det_ids)}), db.now()))
        return {"skipped": True}
    if body.candidate_id is None:
        raise HTTPException(404, "candidate missing")
    lo, hi = min(body.target_id, body.candidate_id), max(body.target_id, body.candidate_id)
    db.execute(
        "INSERT OR REPLACE INTO review_pairs(a, b, status, updated_at) VALUES (?,?,?,?)",
        (lo, hi, "skipped", db.now()))
    return {"skipped": True}


@app.post("/api/review/undo")
def review_undo():
    """Reverse the most recent review action (merge / reject / skip)."""
    row = db.one("SELECT id, action, payload FROM review_log ORDER BY id DESC LIMIT 1")
    if row is None:
        raise HTTPException(404, "nothing to undo")
    p = json.loads(row["payload"])
    if row["action"] == "merge_pair":
        # resurrect the absorbed person and move its members back
        db.execute("INSERT INTO persons(id, name, created_at) VALUES (?,?,?)",
                   (p["b"], p.get("name_b"), db.now()))
        for fid in p["faces"]:
            db.execute("UPDATE faces SET person_id=?, reviewed=1 WHERE id=?", (p["b"], fid))
        for did in p["dets"]:
            db.execute("UPDATE detections SET person_id=? WHERE id=?", (p["b"], did))
    elif row["action"] == "merge_dets":
        for did in p["dets"]:
            db.execute("UPDATE detections SET person_id=NULL, reason=NULL, "
                       "score=NULL, margin=NULL WHERE id=?", (did,))
    elif row["action"] in ("pair_decision",):
        db.execute("DELETE FROM review_pairs WHERE a=? AND b=?", (p["a"], p["b"]))
    elif row["action"] in ("det_decision",):
        for did in p["dets"]:
            db.execute("DELETE FROM review_dets WHERE person_id=? AND det_id=?",
                       (p["target"], did))
    db.execute("DELETE FROM review_log WHERE id=?", (row["id"],))
    return {"undone": row["action"]}


def _remember_reject(target_id: int, other_id: int | None, vec) -> None:
    """Store what "not this person" looked like. The queue then skips anything
    that reads as the same person as the rejected cluster, so one split
    person does not come back as ten identical questions (Ente's rejected_faces
    memory, kept as a centroid so it survives the cluster merging away)."""
    if vec is None:
        return
    vec = _unit(vec)
    if vec is None:
        return
    db.execute("INSERT INTO review_rejects(target_person_id, other_person_id, emb, ts) "
               "VALUES (?,?,?,?)",
               (target_id, other_id, db.vec_to_blob(vec), db.now()))


@app.post("/api/review/reject")
def review_reject(body: ReviewBody):
    """Human decision: different people — never suggest this again."""
    import numpy as np
    if body.det_ids:
        for did in body.det_ids:
            db.execute(
                "INSERT OR REPLACE INTO review_dets(person_id, det_id, status, updated_at) "
                "VALUES (?,?,?,?)", (body.target_id, did, "rejected", db.now()))
        db.execute("INSERT INTO review_log(action, payload, created_at) VALUES (?,?,?)",
                   ("det_decision", json.dumps({"target": body.target_id,
                                                "dets": list(body.det_ids)}), db.now()))
        vecs = [db.blob_to_vec(r["body_emb"]) for r in db.query(
            "SELECT body_emb FROM detections WHERE id IN (%s) AND body_emb IS NOT NULL"
            % ",".join("?" * len(body.det_ids)), list(body.det_ids))]
        vecs = [v for v in vecs if v is not None]
        if vecs:
            _remember_reject(body.target_id, None, np.mean(np.stack(vecs), axis=0))
        return {"rejected": True}
    if body.candidate_id is None:
        raise HTTPException(404, "candidate missing")
    lo, hi = min(body.target_id, body.candidate_id), max(body.target_id, body.candidate_id)
    db.execute(
        "INSERT OR REPLACE INTO review_pairs(a, b, status, updated_at) VALUES (?,?,?,?)",
        (lo, hi, "rejected", db.now()))
    db.execute("INSERT INTO review_log(action, payload, created_at) VALUES (?,?,?)",
               ("pair_decision", json.dumps({"a": lo, "b": hi}), db.now()))
    # remember what was rejected FROM THE TARGET'S SIDE: the next question must
    # not be "this same person again", so store the candidate's face centroid
    # under the person it was said not to belong to
    vecs = [db.blob_to_vec(r["face_emb"]) for r in db.query(
        "SELECT face_emb FROM faces WHERE person_id=? AND face_emb IS NOT NULL",
        (body.candidate_id,))]
    vecs = [v for v in vecs if v is not None]
    if vecs:
        _remember_reject(body.target_id, body.candidate_id,
                         np.mean(np.stack(vecs), axis=0))
    return {"rejected": True}


# ------------------------------------------------------ manual labeling ----
class LabelBody(BaseModel):
    action: str                     # 'assign' | 'new' | 'unassign'
    person_id: int | None = None


@app.post("/api/label/{kind}/{ref_id}")
def label_ref(kind: str, ref_id: int, body: LabelBody):
    """Manual ground-truth labeling from the photo viewer.

    Actions are recorded in the labels table and exported as a training set
    (/api/labels/export). Manual labels are sticky: clustering never demotes
    a labeled face and never re-validates a labeled detection away."""
    if kind not in ("face", "det"):
        raise HTTPException(400, "kind must be 'face' or 'det'")
    table = "faces" if kind == "face" else "detections"
    row = db.one(f"SELECT id, photo_id FROM {table} WHERE id=?", (ref_id,))
    if row is None:
        raise HTTPException(404, f"{kind} not found")
    person_id = None
    if body.action == "new":
        person_id = db.execute(
            "INSERT INTO persons(name, created_at) VALUES (NULL, ?)", (db.now(),))
    elif body.action == "assign":
        if body.person_id is None or db.one(
                "SELECT id FROM persons WHERE id=?", (body.person_id,)) is None:
            raise HTTPException(404, "person not found")
        person_id = body.person_id
    elif body.action != "unassign":
        raise HTTPException(400, "action must be assign / new / unassign")

    if kind == "face":
        # a face usually sits inside exactly one body box — label that body too
        extra = _det_containing_face(ref_id)
        if person_id is None:
            db.execute("UPDATE faces SET person_id=NULL, reviewed=1 WHERE id=?", (ref_id,))
            db.execute(
                "UPDATE detections SET person_id=NULL, reason=NULL, score=NULL, margin=NULL "
                "WHERE face_id=?", (ref_id,))
        else:
            db.execute("UPDATE faces SET person_id=?, reviewed=1 WHERE id=?",
                       (person_id, ref_id))
            db.execute(
                "UPDATE detections SET person_id=?, reason='manual', score=1.0, margin=NULL "
                "WHERE face_id=?", (person_id, ref_id))
        if extra is not None:
            if person_id is None:
                db.execute(
                    "UPDATE detections SET person_id=NULL, reason=NULL, score=NULL, margin=NULL "
                    "WHERE id=?", (extra,))
            else:
                db.execute(
                    "UPDATE detections SET person_id=?, reason='manual', score=1.0, margin=NULL "
                    "WHERE id=?", (person_id, extra))
            db.execute(
                "INSERT INTO labels(photo_id, kind, ref_id, action, person_id, created_at) "
                "VALUES (?,?,?,?,?,?)",
                (row["photo_id"], "det", extra, body.action, person_id, db.now()))
    else:
        if person_id is None:
            db.execute(
                "UPDATE detections SET person_id=NULL, reason=NULL, score=NULL, margin=NULL "
                "WHERE id=?", (ref_id,))
        else:
            db.execute(
                "UPDATE detections SET person_id=?, reason='manual', score=1.0, margin=NULL "
                "WHERE id=?", (person_id, ref_id))
    db.execute(
        "INSERT INTO labels(photo_id, kind, ref_id, action, person_id, created_at) "
        "VALUES (?,?,?,?,?,?)",
        (row["photo_id"], kind, ref_id, body.action, person_id, db.now()))
    return {"kind": kind, "id": ref_id, "action": body.action, "person_id": person_id}


class BreakupBody(BaseModel):
    apply: bool = False


def _breakup_compute(person_id: int) -> list[list[int]]:
    """Split one person's faces into sub-groups (Ente 'Mixed grouping?').

    Complete-linkage agglomerative cut at PERSON_SPLIT_SIM: a sub-group is
    only formed when its WORST cross-pair still clears the bar, so the
    result errs on splitting rather than mixing."""
    import numpy as np

    rows = db.query(
        "SELECT id, face_emb FROM faces WHERE person_id=? AND face_emb IS NOT NULL",
        (person_id,))
    vecs: dict[int, np.ndarray] = {}
    for r in rows:
        v = db.blob_to_vec(r["face_emb"])
        if v is not None:
            n = float(np.linalg.norm(v))
            if n > 1e-6:
                vecs[r["id"]] = v / n
    groups = [[fid] for fid in vecs]
    if len(groups) < 2:
        return groups

    def min_cross(a: list[int], b: list[int]) -> float:
        return min(float(np.dot(vecs[x], vecs[y])) for x in a for y in b)

    while len(groups) > 1:
        best = None
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                s = min_cross(groups[i], groups[j])
                if best is None or s > best[0]:
                    best = (s, i, j)
        if best is None or best[0] < config.PERSON_SPLIT_SIM:
            break
        _, i, j = best
        groups[i] = groups[i] + groups[j]
        del groups[j]
    groups.sort(key=len, reverse=True)
    return groups


@app.post("/api/persons/{person_id}/breakup")
async def person_breakup(person_id: int, body: BreakupBody):
    pe = db.one("SELECT id FROM persons WHERE id=?", (person_id,))
    if not pe:
        return _err("person not found", 404)
    groups = _breakup_compute(person_id)
    proposal = [[{"face_id": fid, "thumb": f"/face_thumb/{fid}.jpg"} for fid in g]
                for g in groups]
    if not body.apply:
        return {"person_id": person_id, "count": len(groups), "groups": proposal}
    if len(groups) < 2:
        return _err("homogeneous at this threshold - nothing to split", 400)
    new_ids = [person_id]
    for g in groups[1:]:
        npid = db.execute(
            "INSERT INTO persons(name, created_at) VALUES (NULL, ?)", (db.now(),))
        for fid in g:
            db.execute(
                "UPDATE faces SET person_id=?, reviewed=NULL WHERE id=?", (npid, fid))
        new_ids.append(npid)
    return {"person_id": person_id, "new_persons": new_ids, "groups": proposal}


class ReopenBody(BaseModel):
    a: int
    b: int


@app.get("/api/review/decided")
def review_decided(target_id: int | None = None):
    """Decided pairs (Ente 'choices'), with per-row reopen for rejections:
    a mistaken No must be correctable without nuking the whole history."""
    seq = _person_seq()
    names = {r["id"]: r["name"] for r in db.query("SELECT id, name FROM persons")}
    out = []
    for r in db.query(
            "SELECT a, b, status, updated_at FROM review_pairs "
            "ORDER BY updated_at DESC LIMIT 300"):
        if target_id is not None and r["a"] != target_id and r["b"] != target_id:
            continue
        out.append({
            "a": r["a"], "b": r["b"], "status": r["status"],
            "a_display": _display(seq, r["a"], names.get(r["a"])),
            "b_display": _display(seq, r["b"], names.get(r["b"])),
            "a_seq": seq.get(r["a"]), "b_seq": seq.get(r["b"]),
            "a_name": names.get(r["a"]), "b_name": names.get(r["b"]),
            "a_thumb": _best_face_thumb(r["a"]),
            "b_thumb": _best_face_thumb(r["b"]),
        })
    return {"items": out}


@app.post("/api/review/reopen")
async def review_reopen(body: ReopenBody):
    a, b = (body.a, body.b) if body.a < body.b else (body.b, body.a)
    cur = db.conn().execute(
        "DELETE FROM review_pairs WHERE ((a=? AND b=?) OR (a=? AND b=?)) "
        "AND status='rejected'", (a, b, b, a))
    db.conn().commit()
    return {"ok": True, "reopened": cur.rowcount}


class AdminResetBody(BaseModel):
    mode: str = "review"


@app.post("/api/admin/reset")
async def api_admin_reset(body: AdminResetBody):
    """Reset options for the web UI.

    modes: review  - review decisions + undo log
           labels  - user labels
           manual  - ALL manual marks (labels + review + merge pins)
           full    - wipe the whole classification and re-cluster clean
    """
    mode = str(body.mode or "review")
    if mode not in ("review", "labels", "manual", "full"):
        return _err(f"unknown reset mode: {mode}", 400)
    out: dict = {"mode": mode}
    if mode in ("review", "manual", "full"):
        for t in ("review_pairs", "review_dets", "review_log"):
            out[t] = db.execute(f"DELETE FROM {t}")
    if mode in ("labels", "manual", "full"):
        out["labels"] = db.execute("DELETE FROM labels")
    if mode in ("manual", "full"):
        out["pins"] = db.execute("UPDATE faces SET reviewed=NULL WHERE reviewed=1")
    if mode == "full":
        # safety net: snapshot the DB before wiping (names are precious).
        # If the snapshot cannot be written the wipe is REFUSED - a silent
        # no-backup reset would throw away every name and review decision.
        import shutil
        dst = config.DB_PATH.with_name(config.DB_PATH.name + "-pre-fullreset")
        try:
            shutil.copyfile(config.DB_PATH, dst)
            out["backup"] = str(dst)
        except Exception as e:
            return {"ok": False, "error": f"backup failed, nothing wiped: {e}"}
        db.execute("UPDATE faces SET person_id=NULL, reviewed=NULL")
        db.execute("UPDATE detections SET person_id=NULL, reason=NULL, score=NULL, margin=NULL")
        out["persons"] = db.execute("DELETE FROM persons")
        out["recluster"] = recluster()
    return {"ok": True, **out}


@app.get("/api/labels/export")
def labels_export():
    """The manual annotations as a training set (JSON)."""
    seq = _person_seq()
    names = {r["id"]: r["name"] for r in db.query("SELECT id, name FROM persons")}
    photos: dict[int, dict] = {}
    for r in db.query(
        "SELECT l.photo_id, l.kind, l.ref_id, l.action, l.person_id, l.created_at, "
        "p.path, p.sha1 FROM labels l JOIN photos p ON p.id=l.photo_id ORDER BY l.id"
    ):
        src = "faces" if r["kind"] == "face" else "detections"
        box_row = db.one(f"SELECT box FROM {src} WHERE id=?", (r["ref_id"],))
        entry = photos.setdefault(r["photo_id"], {
            "photo_id": r["photo_id"], "path": r["path"], "sha1": r["sha1"],
            "labels": []})
        entry["labels"].append({
            "kind": r["kind"],
            "ref_id": r["ref_id"],
            "action": r["action"],
            "box": _loads(box_row["box"]) if box_row else None,
            "person_id": r["person_id"],
            "person_name": names.get(r["person_id"]),
            "person_display": (_display(seq, r["person_id"], names.get(r["person_id"]))
                               if r["person_id"] else None),
            "created_at": r["created_at"],
        })
    return {
        "persons": [{"id": pid, "name": nm, "display": _display(seq, pid, nm)}
                    for pid, nm in sorted(names.items())],
        "photos": list(photos.values()),
        "count": sum(len(v["labels"]) for v in photos.values()),
    }


# ---------------------------------------------------------- file serving ----
@app.get("/thumbs/{name}")
def thumb(name: str):
    path = config.THUMB_DIR / Path(name).name
    if not path.is_file():
        raise HTTPException(404, "thumb not found")
    return FileResponse(path, media_type="image/jpeg")


def _display_copy(path: Path, sha1: str) -> Path:
    """Re-encode for the screen: plain sRGB JPEG, no metadata.

    Some sources (iPhone / Ultra HDR JPEG) carry a gain map; browsers detect it
    and brighten the picture for HDR, which the user does not want. Dropping
    every metadata block - including the gain map's MP entry, XMP and the ICC
    profile - makes the browser render the pixels exactly as they are. The
    raw file on disk is never touched. Also caps the served size at 2048px,
    which is plenty for a viewer and much faster to send.
    """
    from PIL import Image

    out = config.THUMB_DIR.parent / "display" / f"{sha1}.jpg"
    if out.is_file():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(path)
    img = img.convert("RGB")            # flattens + drops the colour profile
    img.thumbnail((2048, 2048), Image.LANCZOS)
    tmp = out.with_suffix(".tmp")
    img.save(tmp, "JPEG", quality=90)   # no exif / icc / xmp = nothing to boost
    tmp.replace(out)
    return out


@app.get("/originals/{photo_id}")
def original(photo_id: int):
    p = db.one("SELECT path, sha1 FROM photos WHERE id=?", (photo_id,))
    if p is None:
        raise HTTPException(404, "photo not found")
    path = config.PHOTO_DIR / p["path"]
    if not path.is_file():
        raise HTTPException(404, "file missing on disk")
    try:
        return FileResponse(_display_copy(path, p["sha1"]), media_type="image/jpeg")
    except Exception:
        # un-decodable source: serve the raw bytes rather than nothing
        media = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        return FileResponse(path, media_type=media)


def _loads(s):
    import json

    return json.loads(s)


app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
