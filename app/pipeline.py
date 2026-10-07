"""Per-photo ML analysis: thumbnail, person boxes, faces, ReID embeddings."""
import json
import logging
import threading
import time

import cv2
import numpy as np
from PIL import Image, ImageOps

from . import config
from .ml import common

log = logging.getLogger("gallery.pipeline")

try:  # optional: HEIC/HEIC iPhone photos
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:  # pragma: no cover
    pass

_engines = None
_engines_lock = threading.Lock()


def engines() -> dict:
    """Lazily construct the ONNX/OpenCV engines (heavy, shared per process)."""
    global _engines
    with _engines_lock:
        if _engines is None:
            from .ml.face import FaceEngine
            from .ml.person import PersonDetector
            from .ml.pose import PoseValidator
            from .ml.reid import ReId

            log.info("loading models from %s", config.MODEL_DIR)
            _engines = {
                "person": PersonDetector(),
                "face": FaceEngine(),
                "reid": ReId(),
                "pose": PoseValidator(),
            }
            log.info("models ready")
    return _engines


def read_taken_at(path: str) -> float | None:
    """EXIF capture time (DateTimeOriginal/DateTimeDigitized/DateTime) as epoch."""
    try:
        with Image.open(path) as im:
            exif = im.getexif()
            for tag in (36867, 36868, 306):
                raw = exif.get(tag)
                if not raw:
                    continue
                s = str(raw).strip()
                for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y:%m:%d"):
                    try:
                        return time.mktime(time.strptime(s, fmt))
                    except ValueError:
                        continue
    except Exception:
        log.warning("failed to read EXIF from %s", path)
    return None


def analyze(path: str) -> dict:
    """Run the full detection pipeline on one image.

    Returns original-coordinate records; persistence happens in the worker.
    """
    eng = engines()
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        width, height = im.size
        rgb = np.asarray(im)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    frame, tf = common.letterbox(bgr, 640)

    people = eng["person"].detect(frame)
    faces = eng["face"].detect(frame)

    detections = [
        {"box": tf.to_original_box(p["box"]), "conf": p["conf"], "face_idx": None}
        for p in people
    ]
    face_out = [
        {"box": tf.to_original_box(f["box"]), "score": f["score"], "feat": f["feat"],
         "quality": f.get("quality"), "yaw": f.get("yaw"),
         "blur": f.get("blur"), "sideways": f.get("sideways")}
        for f in faces
    ]

    # Attach at most one face to each person box (best containment/IoU first).
    candidates = []
    for fi, f in enumerate(face_out):
        cx = (f["box"][0] + f["box"][2]) / 2
        cy = (f["box"][1] + f["box"][3]) / 2
        for di, d in enumerate(detections):
            inside = d["box"][0] <= cx <= d["box"][2] and d["box"][1] <= cy <= d["box"][3]
            overlap = common.iou(f["box"], d["box"])
            if inside or overlap > 0.05:
                candidates.append(((1.0 if inside else 0.0) + overlap, fi, di))
    candidates.sort(reverse=True)
    taken_faces: set[int] = set()
    for _, fi, di in candidates:
        if fi in taken_faces or detections[di]["face_idx"] is not None:
            continue
        detections[di]["face_idx"] = fi
        taken_faces.add(fi)

    crops = [common.crop(bgr, d["box"]) for d in detections]
    body_embs = eng["reid"].embed(crops) if crops else np.zeros((0, 512), dtype=np.float32)
    for i, d in enumerate(detections):
        d["body"] = body_embs[i]
        pn, head, limbs = eng["pose"].inspect(crops[i])
        d["personness"] = pn
        bx1, by1 = d["box"][0], d["box"][1]
        d["head"] = json.dumps({
            "h": ([head[0] + bx1, head[1] + by1, head[2]] if head else None),
            "l": [[x + bx1, y + by1] for x, y in limbs],
        })

    return {"width": width, "height": height, "faces": face_out, "detections": detections}
