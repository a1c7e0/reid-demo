"""Face detection + embedding via OpenCV's built-in wrappers.

YuNet detects faces (with 5 landmarks), SFace aligns and embeds them.
Both consume the shared 640x640 letterboxed frame.

OpenCV 4.x exposes these under ``cv2.dnn``; OpenCV 5.x moved them to the
top level, so we resolve either spelling at import time.
"""
import cv2
import numpy as np

from .. import config


def _resolve():
    ns = cv2.dnn if hasattr(cv2.dnn, "FaceDetectorYN") else cv2
    return ns.FaceDetectorYN


# Embedding-model presets: (scale, mean, std on raw BGR, swap_rb, input size)
_PRESETS = {
    # InsightFace family (glint R100 / w600k_r50 / MobileFaceNet): RGB,
    # (x-127.5)/127.5 -> [-1, 1]
    "arcface": dict(scale=1.0, mean=127.5, std=127.5, swap_rb=True, size=112),
    # OpenCV SFace MobileFaceNet: raw RGB [0,255]
    "sface": dict(scale=1.0, mean=0.0, std=1.0, swap_rb=True, size=112),
    # AdaFace / LVFace convention: RGB, (x/255-0.5)/0.5 -> [-1, 1]
    "adaface": dict(scale=1 / 255.0, mean=0.5, std=0.5, swap_rb=True, size=112),
    # Ente stock MobileFaceNet portable: NHWC 112x112x3, (x-127.5)/127.5 RGB
    "ente-mfn": dict(scale=1.0, mean=127.5, std=127.5, swap_rb=True, size=112, nhwc=True),
}

# ArcFace 5-point alignment template (112x112 output)
_ALIGN_DST = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


def align5(frame: np.ndarray, row: np.ndarray) -> np.ndarray:
    """Similarity-transform the 5 YuNet landmarks onto the ArcFace template.

    Replaces FaceRecognizerSF.alignCrop: OpenCV's dnn cannot even PARSE
    quantized recognition models (dequantizelinear assert), and the embedding
    path already runs through onnxruntime."""
    src = np.array([[float(row[4]), float(row[5])], [float(row[6]), float(row[7])],
                    [float(row[8]), float(row[9])], [float(row[10]), float(row[11])],
                    [float(row[12]), float(row[13])]], dtype=np.float32)
    m, _ = cv2.estimateAffinePartial2D(src, _ALIGN_DST, method=cv2.LMEDS)
    if m is None:
        return cv2.resize(frame, (112, 112))
    return cv2.warpAffine(frame, m, (112, 112), borderValue=0.0)


class FaceEngine:
    def __init__(
        self,
        det_model=config.FACE_DET_MODEL,
        rec_model=config.FACE_REC_MODEL,
        score=config.FACE_CONF,
    ):
        self._det_kind = config.FACE_DET_KIND
        if self._det_kind == "yolo5face":
            # Ente stock detector (YOLOv5s-face 640x640 static b1).
            # Output (25200, 16): cx cy w h obj, 5x(x,y), face-score.
            import onnxruntime as _ort
            self.detector = None
            self._yolo = _ort.InferenceSession(str(det_model),
                                               providers=["CPUExecutionProvider"])
            self._yolo_in = self._yolo.get_inputs()[0].name
            self._det_score = score
        else:
            detector_cls = _resolve()
            self.detector = detector_cls.create(
                str(det_model), "", (640, 640), score, 0.3, 5000
            )
        # embeddings run through onnxruntime — OpenCV's dnn forward silently
        # produces garbage for larger recognition models (verified with
        # w600k_r50: cross-person similarities collapsed to ~0.9), and it
        # cannot parse quantized models at all.
        import onnxruntime as ort

        self._sess = ort.InferenceSession(str(rec_model), providers=["CPUExecutionProvider"])
        self._inp = self._sess.get_inputs()[0].name

    def _detect_yolo(self, frame: np.ndarray):
        """YOLOv5s-face decode -> YuNet-style rows (x, y, w, h, 5 lmk, score)."""
        h0, w0 = frame.shape[:2]
        s = 640.0 / max(h0, w0)
        nh, nw = int(round(h0 * s)), int(round(w0 * s))
        img = cv2.resize(frame, (nw, nh))
        canvas = np.zeros((640, 640, 3), dtype=np.uint8)
        canvas[:nh, :nw] = img
        blob = (canvas.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        out = self._yolo.run(None, {self._yolo_in: blob})[0][0]
        cands = []
        for r in out:
            score = float(r[4]) * float(r[15])
            if score < self._det_score:
                continue
            cx, cy, w, h = (float(v) / s for v in r[:4])
            lmk = [(float(r[5 + 2 * k]) / s, float(r[6 + 2 * k]) / s) for k in range(5)]
            row = [cx - w / 2, cy - h / 2, w, h] + [c for p in lmk for c in p] + [score]
            cands.append((score, row))
        # NMS on face boxes (YOLO raw outputs are heavily duplicated)
        cands.sort(key=lambda x: -x[0])
        rows = []
        for score, row in cands:
            x1, y1, w, h = row[0], row[1], row[2], row[3]
            ok = True
            for kept in rows:
                kx, ky, kw, kh = kept[0], kept[1], kept[2], kept[3]
                ix = max(0.0, min(x1 + w, kx + kw) - max(x1, kx))
                iy = max(0.0, min(y1 + h, ky + kh) - max(y1, ky))
                inter = ix * iy
                uni = w * h + kw * kh - inter
                if uni > 0 and inter / uni > 0.45:
                    ok = False
                    break
            if ok:
                rows.append(np.array(row, dtype=np.float32))
            if len(rows) >= 50:
                break
        return rows

    def detect(self, frame: np.ndarray) -> list[dict]:
        """Detect faces and compute embeddings in letterboxed frame space."""
        if self._det_kind == "yolo5face":
            faces = self._detect_yolo(frame)
            if not faces:
                return []
        else:
            self.detector.setInputSize((frame.shape[1], frame.shape[0]))
            _, faces = self.detector.detect(frame)
        results = []
        if faces is None:
            return results
        for row in faces:
            # YuNet rows are (x, y, w, h, 5 landmarks..., score): convert to
            # x1/y1/x2/y2 for storage/UI, but alignCrop needs the raw row.
            x, y, w, h = (float(v) for v in row[:4])
            box = [x, y, x + w, y + h]
            score = float(row[14])
            aligned = align5(frame, row)
            ps = _PRESETS[config.FACE_REC_PRESET]
            im = aligned
            if ps["size"] != 112:
                im = cv2.resize(im, (ps["size"], ps["size"]))
            x = im[:, :, ::-1].copy() if ps["swap_rb"] else im.copy()
            x = (x.astype(np.float32) * ps["scale"] - ps["mean"]) / ps["std"]
            blob = x[None] if ps.get("nhwc") else x.transpose(2, 0, 1)[None]
            feat = self._sess.run(None, {self._inp: blob})[0].reshape(-1).astype(np.float32)
            norm = float(np.linalg.norm(feat))
            if norm > 0:
                feat = feat / norm
            # quality score: detection confidence x sharpness x frontal-ness
            # (borrowed from the prototypes survey) — weak faces must clear a
            # higher bar to drive identity
            x1, y1, x2, y2 = (int(v) for v in box)
            crop = frame[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)]
            sharp = 0.0
            if crop.size:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                sharp = min(1.0, float(cv2.Laplacian(gray, cv2.CV_64F).var()) / 120.0)
            re_x, le_x, nose_x = float(row[4]), float(row[6]), float(row[8])
            nose_off = abs(nose_x - (re_x + le_x) / 2) / max(w, 1e-3)
            eye_gap = abs(le_x - re_x) / max(w, 1e-3)
            # profile proxy: nose drift OR eye collapse (in a true profile the
            # two eyes project onto each other -> small eye gap; calibrated on
            # real data: frontal 0.42, full profile 0.05-0.12)
            yaw = max(nose_off, 0.30 - eye_gap)
            frontal = max(0.0, 1.0 - yaw * 2.0)
            quality = float(min(1.0, score / 0.9)) * sharp * (0.5 + 0.5 * frontal)
            # Ente quality primitives (docs/ENTE-PORT-SPEC.md 2.6): blurValue
            # = Laplacian variance on the aligned crop; sideways = faceDirection
            # from the 5 landmarks != "straight"
            from ..ente_core import face_direction
            gray_a = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
            blur = float(cv2.Laplacian(gray_a, cv2.CV_64F).var())
            lm = np.array([[float(row[4]), float(row[5])],
                           [float(row[6]), float(row[7])],
                           [float(row[8]), float(row[9])],
                           [float(row[10]), float(row[11])],
                           [float(row[12]), float(row[13])]], dtype=np.float32)
            sideways = 1 if face_direction(lm) != "straight" else 0
            results.append({"box": box, "score": score, "feat": feat,
                            "quality": round(quality, 3), "yaw": round(yaw, 3),
                            "blur": round(blur, 2), "sideways": sideways})
        return results
