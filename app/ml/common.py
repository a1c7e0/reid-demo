"""Letterbox resize shared by the detector and the face detector.

Both ONNX models consume a fixed 640x640 input, so we letterbox the frame once
and reuse it. Boxes are mapped back to original image coordinates through
:class:`LetterboxTransform`.
"""
from dataclasses import dataclass

import cv2
import numpy as np

PAD_COLOR = (114, 114, 114)


@dataclass(frozen=True)
class LetterboxTransform:
    scale: float
    dx: int
    dy: int
    orig_w: int
    orig_h: int

    def to_original_box(self, box) -> list:
        """Map [x1, y1, x2, y2] from letterboxed space to original space."""
        x1, y1, x2, y2 = box
        ox1 = (x1 - self.dx) / self.scale
        oy1 = (y1 - self.dy) / self.scale
        ox2 = (x2 - self.dx) / self.scale
        oy2 = (y2 - self.dy) / self.scale
        ox1 = float(min(max(ox1, 0.0), self.orig_w - 1))
        oy1 = float(min(max(oy1, 0.0), self.orig_h - 1))
        ox2 = float(min(max(ox2, 0.0), self.orig_w - 1))
        oy2 = float(min(max(oy2, 0.0), self.orig_h - 1))
        return [ox1, oy1, ox2, oy2]


def letterbox(img: np.ndarray, size: int = 640):
    """Aspect-preserving resize + padding into a square ``size`` canvas."""
    h, w = img.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), PAD_COLOR, dtype=img.dtype)
    dx = (size - nw) // 2
    dy = (size - nh) // 2
    canvas[dy : dy + nh, dx : dx + nw] = resized
    tf = LetterboxTransform(scale=scale, dx=dx, dy=dy, orig_w=w, orig_h=h)
    return canvas, tf


def box_area(box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def clip_box(box, w: int, h: int) -> list:
    return [
        float(min(max(box[0], 0), w - 1)),
        float(min(max(box[1], 0), h - 1)),
        float(min(max(box[2], 0), w - 1)),
        float(min(max(box[3], 0), h - 1)),
    ]


def crop(img: np.ndarray, box) -> np.ndarray:
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 0, 3), dtype=img.dtype)
    return img[y1:y2, x1:x2]
