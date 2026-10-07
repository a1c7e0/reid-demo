"""Person detection: YOLO11n exported to ONNX with NMS embedded.

The model is exported once at Docker build time (see
``scripts/export_detector.py``) so the runtime only needs onnxruntime.
"""
import numpy as np
import onnxruntime as ort

from .. import config


class PersonDetector:
    def __init__(self, model_path=config.PERSON_DET_MODEL, conf=config.PERSON_CONF):
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.conf = conf

    def detect(self, frame: np.ndarray) -> list[dict]:
        """Detect persons in a BGR 640x640 letterboxed frame.

        Returns boxes in *letterboxed* coordinates; the caller maps them back.
        """
        blob = frame[:, :, ::-1].astype(np.float32) / 255.0  # BGR -> RGB
        blob = blob.transpose(2, 0, 1)[None]                  # CHW + batch
        out = self.session.run(None, {self.input_name: blob})[0]
        if out.shape[-1] != 6:
            raise RuntimeError(
                f"unexpected detector output shape {out.shape}; "
                "expected an NMS export of (1, N, 6)"
            )
        rows = out[0]
        results = []
        for x1, y1, x2, y2, conf, cls_id in rows:
            if conf < self.conf:
                continue  # rows are NMS-sorted: first miss means we are done
            if int(cls_id) != 0:  # COCO class 0 = person
                continue
            results.append(
                {"box": [float(x1), float(y1), float(x2), float(y2)], "conf": float(conf)}
            )
        return results
