"""Person re-identification embedding: OSNet x0.25 (ONNX, ~0.9 MB).

The exported model has a fixed batch dimension (16), so batches are padded
with the last crop and sliced back down after inference.
"""
import cv2
import numpy as np
import onnxruntime as ort

from .. import config

INPUT_W, INPUT_H = 128, 256
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ReId:
    def __init__(self, model_path=config.REID_MODEL):
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        shape = self.session.get_inputs()[0].shape
        self.batch = int(shape[0]) if isinstance(shape[0], int) and shape[0] > 0 else 16

    def _prep(self, crop_img: np.ndarray) -> np.ndarray:
        if crop_img.size == 0:
            arr = np.zeros((INPUT_H, INPUT_W, 3), dtype=np.uint8)
        else:
            arr = cv2.resize(crop_img, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
        arr = arr[:, :, ::-1].astype(np.float32) / 255.0  # BGR -> RGB
        arr = (arr - MEAN) / STD
        return arr.transpose(2, 0, 1)

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        """Embed person crops (BGR). Returns L2-normalized (N, 512) floats.

        Chunks into fixed-size batches: the fixed-batch export only accepts
        exactly ``self.batch`` rows per call (a single padded batch would
        overflow it once there are more than ``batch`` crops, e.g. a crowd
        photo with 17+ persons).
        """
        if not crops:
            return np.zeros((0, 512), dtype=np.float32)
        items = [self._prep(c) for c in crops]
        b = self.batch
        outs = []
        for start in range(0, len(items), b):
            chunk = items[start:start + b]
            pad = (b - len(chunk) % b) % b
            batch = np.stack(chunk + [chunk[-1]] * pad).astype(np.float32)
            outs.append(self.session.run(None, {self.input_name: batch})[0][:len(chunk)])
        out = np.concatenate(outs, axis=0)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms < 1e-6] = 1.0
        return (out / norms).astype(np.float32)
