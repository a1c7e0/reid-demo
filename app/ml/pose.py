"""Pose-based personness validation (YOLO11n-pose, ~5 MB).

Why: the person detector happily calls hands, held objects and giant
crowd-covering boxes "person" (COCO's class includes body parts). Those
crops then get body-claimed into real people's clusters. A *coherent pose
skeleton* is what separates "an actual human standing in front of us"
from "a hand holding a card":

* hand / object crops        -> no instance, or a scattered handful of keypoints
* a box spanning N people    -> N instances (reject: ambiguous which one)
* one real person crop       -> exactly 1 instance with a sensible skeleton

Returns 1.0 (is a person) or 0.0. Fail-open when the model file is absent
so deployments without the model don't lose every body claim.
"""
import logging

import numpy as np

from .. import config
from . import common

log = logging.getLogger("gallery.pose")

MIN_INSTANCE_CONF = 0.5
MIN_KPT_CONF = 0.3
MIN_VALID_KPTS = 6
MIN_SKEL_SPAN = 0.35  # skeleton must span >=35% of the crop box


class PoseValidator:
    def __init__(self) -> None:
        self.ok = False
        self.size = 320
        try:
            import onnxruntime as ort

            self.session = ort.InferenceSession(
                str(config.POSE_MODEL), providers=["CPUExecutionProvider"])
            shape = self.session.get_inputs()[0].shape
            self.size = int(shape[2]) if isinstance(shape[2], int) else 320
            self.ok = True
            log.info("pose model ready (%s, %dpx)", config.POSE_MODEL.name, self.size)
        except Exception:
            log.warning("pose model unavailable at %s — personness disabled (fail-open)",
                        config.POSE_MODEL)

    def inspect(self, crop_bgr: np.ndarray) -> tuple[float, tuple | None, list]:
        """(personness, head) where head = (x, y, r) in ORIGINAL crop pixels.

        head = facial keypoints of the skeleton (informational), limbs =
        wrist/ankle endpoints in crop pixels. Face validation uses limbs as
        an asymmetric veto: a "face" whose center sits ON a wrist/ankle
        (e.g. YuNet firing on hands gripping a steering wheel) is a false
        positive — while absence of a head estimate never disqualifies
        anything (profiles have unreliable facial keypoints)."""
        if not self.ok:
            return 1.0, None, []
        if crop_bgr is None or crop_bgr.size == 0:
            return 0.0, None, []
        frame, tf = common.letterbox(crop_bgr, self.size)
        img = frame[:, :, ::-1].transpose(2, 0, 1)  # BGR -> RGB CHW
        blob = np.ascontiguousarray(img, dtype=np.float32)[None] / 255.0
        out = self.session.run(None, {self.session.get_inputs()[0].name: blob})[0]
        instances = self._decode(out)
        instances = [k for k in instances if k[0] >= MIN_INSTANCE_CONF]
        if len(instances) != 1:
            return 0.0, None, []  # zero (hand/card) or several people (mush)
        _, kpts = instances[0]
        visible = kpts[kpts[:, 2] >= MIN_KPT_CONF]
        if len(visible) < MIN_VALID_KPTS:
            return 0.0, None, []
        span = float(visible[:, 1].max() - visible[:, 1].min())
        if span < MIN_SKEL_SPAN * self.size:
            return 0.0, None, []
        # facial keypoints 0..4 = nose, eyes, ears (COCO order)
        face_k = kpts[:5]
        fk = face_k[face_k[:, 2] >= MIN_KPT_CONF]
        head = None
        if len(fk) >= 2:
            cx = float(fk[:, 0].mean())
            cy = float(fk[:, 1].mean())
            r = float(max(fk[:, 0].max() - fk[:, 0].min(),
                          fk[:, 1].max() - fk[:, 1].min(), 8.0))
            # map letterboxed coords back to crop pixels
            head = ((cx - tf.dx) / tf.scale, (cy - tf.dy) / tf.scale, r / tf.scale)
        # limbs whose endpoints host false "faces" (hands on wheels/cards):
        # wrists (9,10) and ankles (15,16)
        limbs = []
        for i in (9, 10, 15, 16):
            if kpts[i, 2] >= MIN_KPT_CONF:
                limbs.append(((float(kpts[i, 0]) - tf.dx) / tf.scale,
                              (float(kpts[i, 1]) - tf.dy) / tf.scale))
        return 1.0, head, limbs

    def personness(self, crop_bgr: np.ndarray) -> float:
        return self.inspect(crop_bgr)[0]

    def _decode(self, out: np.ndarray) -> list[tuple[float, np.ndarray]]:
        """Normalize ONNX output to [(instance_conf, (17,3) kpts)]."""
        arr = np.asarray(out)
        if arr.ndim == 3:  # raw YOLO head: (1, C, N) -> (N, C)
            arr = arr[0].transpose(1, 0)
        if arr.size == 0:
            return []
        n_kpt = (arr.shape[1] - 5) // 3
        kpt_cols = arr.shape[1] - n_kpt * 3
        rows = arr
        # manual NMS on xywh boxes (raw head) — harmless for already-NMS'd rows
        boxes, scores, keep_rows = [], [], []
        xywh = rows[:, 0:4].copy()
        xywh[:, 2] = np.maximum(xywh[:, 2], 1e-3)
        xywh[:, 3] = np.maximum(xywh[:, 3], 1e-3)
        import cv2
        idx = cv2.dnn.NMSBoxes(xywh.tolist(), rows[:, 4].tolist(),
                               MIN_INSTANCE_CONF, 0.5)
        for i in (idx.flatten() if len(idx) else []):
            keep_rows.append(rows[int(i)])
        out_list = []
        for r in keep_rows:
            conf = float(r[4])
            kpts = r[kpt_cols:kpt_cols + n_kpt * 3].reshape(n_kpt, 3)
            out_list.append((conf, kpts.astype(np.float32)))
        return out_list
