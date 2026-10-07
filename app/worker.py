"""Background worker: ML analysis of new photos + incremental re-clustering."""
import logging
import queue
import threading

from . import cluster, config, db

log = logging.getLogger("gallery.worker")

_q: "queue.Queue[int]" = queue.Queue()
_STOP = object()
_lock = threading.Lock()
_state = {"processed": 0, "failed": 0, "last_error": None}


def submit(photo_id: int) -> None:
    _q.put(photo_id)


def status() -> dict:
    with _lock:
        return {
            "queue": _q.qsize(),
            "processed": _state["processed"],
            "failed": _state["failed"],
            "last_error": _state["last_error"],
        }


def _process(photo_id: int) -> None:
    row = db.one("SELECT id, path, processed FROM photos WHERE id=?", (photo_id,))
    if row is None or row["processed"]:
        return
    from . import pipeline  # local import: engine loading is heavy

    full = str(config.PHOTO_DIR / row["path"])
    res = pipeline.analyze(full)

    # Clear any partial records from an earlier failed attempt, then insert.
    db.execute("DELETE FROM detections WHERE photo_id=?", (photo_id,))
    db.execute("DELETE FROM faces WHERE photo_id=?", (photo_id,))
    face_ids: dict[int, int] = {}
    for i, f in enumerate(res["faces"]):
        face_ids[i] = db.execute(
            "INSERT INTO faces(photo_id, box, score, face_emb, quality, yaw, blur, sideways) VALUES (?,?,?,?,?,?,?,?)",
            (photo_id, _box(f["box"]), f["score"], db.vec_to_blob(f["feat"]),
             f.get("quality"), f.get("yaw"), f.get("blur"), f.get("sideways")),
        )
    for d in res["detections"]:
        face_id = face_ids.get(d["face_idx"]) if d["face_idx"] is not None else None
        db.execute(
            "INSERT INTO detections(photo_id, box, conf, face_id, body_emb, personness, head) "
            "VALUES (?,?,?,?,?,?,?)",
            (photo_id, _box(d["box"]), d["conf"], face_id, db.vec_to_blob(d["body"]),
             d.get("personness"), d.get("head")),
        )
    db.execute("UPDATE photos SET processed=1 WHERE id=?", (photo_id,))


def _box(box) -> str:
    import json

    return json.dumps([round(float(v), 1) for v in box])


def _loop() -> None:
    while True:
        item = _q.get()
        if item is _STOP:
            break
        try:
            _process(item)
            with _lock:
                _state["processed"] += 1
        except Exception as exc:  # keep the worker alive on bad files
            with _lock:
                _state["failed"] += 1
                _state["last_error"] = f"photo {item}: {exc}"
            log.exception("analysis failed for photo %s", item)
        if _q.empty():
            try:
                stats = cluster.recluster()
                log.info("recluster: %s", stats)
            except Exception:
                log.exception("recluster failed")


def start() -> None:
    threading.Thread(target=_loop, daemon=True, name="gallery-worker").start()
    log.info("worker thread started")


def stop() -> None:
    _q.put(_STOP)
