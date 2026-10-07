"""Load every runtime model once at build time so a bad download fails early."""
from pathlib import Path

import onnxruntime as ort


def main() -> None:
    models = sorted(Path("/models").glob("*.onnx"))
    if not models:
        raise SystemExit("[verify] no models found in /models")
    for path in models:
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        ins = [(i.name, i.shape) for i in session.get_inputs()]
        outs = [(o.name, o.shape) for o in session.get_outputs()]
        print(f"[verify] {path.name} ({path.stat().st_size / 1e6:.2f} MB) in={ins} out={outs}")
    print(f"[verify] {len(models)} models OK")


if __name__ == "__main__":
    main()
