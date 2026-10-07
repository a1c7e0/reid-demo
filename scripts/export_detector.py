"""Export YOLO11n and YOLO11s to ONNX with NMS embedded (docker build).

Ultralytics does the heavy lifting: weights are auto-downloaded and the
export embeds a standard ONNX NonMaxSuppression subgraph, so at runtime we
only feed a 640x640 letterboxed frame and read (1, N, 6) rows of
[x1, y1, x2, y2, confidence, class_id]. Both sizes are baked in so the
detector can be swapped via PERSON_DET_MODEL_FILE without rebuilding.
"""
import shutil
from pathlib import Path


def export_one(arch: str) -> Path:
    from ultralytics import YOLO

    model = YOLO(f"{arch}.pt")
    exported = Path(model.export(format="onnx", imgsz=640, nms=True, simplify=True,
                                 opset=17, device="cpu"))
    print(f"[export] {arch}: ultralytics produced {exported} "
          f"({exported.stat().st_size / 1e6:.2f} MB)")

    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(str(exported), providers=["CPUExecutionProvider"])
    inp = session.get_inputs()[0]
    shape = [d if isinstance(d, int) else 1 for d in inp.shape]
    out = session.run(None, {inp.name: np.zeros(shape, dtype=np.float32)})[0]
    if out.shape[-1] != 6:
        raise SystemExit(f"[export] {arch}: unexpected output shape {out.shape}; "
                         "NMS export failed")

    dest = Path(f"/models/{arch}-person.onnx")
    shutil.copy(exported, dest)
    print(f"[export] wrote {dest} ({dest.stat().st_size / 1e6:.2f} MB), "
          f"smoke output shape {tuple(out.shape)}")
    return dest


def export_pose() -> Path:
    """YOLO11n-pose for personness validation (raw head, no embedded NMS;
    decoded + NMS'd in app/ml/pose.py)."""
    from ultralytics import YOLO

    model = YOLO("yolo11n-pose.pt")
    exported = Path(model.export(format="onnx", imgsz=320, simplify=True,
                                 opset=17, device="cpu"))
    print(f"[export] yolo11n-pose: ultralytics produced {exported} "
          f"({exported.stat().st_size / 1e6:.2f} MB)")

    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(str(exported), providers=["CPUExecutionProvider"])
    inp = session.get_inputs()[0]
    shape = [d if isinstance(d, int) else 1 for d in inp.shape]
    out = session.run(None, {inp.name: np.zeros(shape, dtype=np.float32)})[0]
    if out.ndim != 3 or out.shape[1] not in (55, 56, 57):
        raise SystemExit(f"[export] yolo11n-pose: unexpected output shape {out.shape}")

    dest = Path("/models/yolo11n-pose.onnx")
    shutil.copy(exported, dest)
    print(f"[export] wrote {dest} ({dest.stat().st_size / 1e6:.2f} MB), "
          f"smoke output shape {tuple(out.shape)}")
    return dest


def main() -> None:
    for arch in ("yolo11n", "yolo11s"):
        export_one(arch)
    export_pose()


if __name__ == "__main__":
    main()
