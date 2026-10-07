"""Export OSNet x1.0 (official MSMT17 checkpoint) to ONNX, dynamic batch.

The torchreid *package* needs Cython/GCC to build, but the model definition
itself (torchreid/models/osnet.py) is pure PyTorch, so we clone the repo and
load that single file standalone. In eval mode OSNet's forward() returns the
512-d embedding (the classifier head is skipped), which is exactly what the
gallery consumes.
"""
import importlib.util
import sys
import urllib.request
from pathlib import Path

REPO = Path("/build/deep-person-reid")
OSNET_PY = REPO / "torchreid" / "models" / "osnet.py"
CKPT = Path("/build/osnet_x1_0_msmt17.pth")
OUT = Path("/models/osnet_x1_0_msmt17.onnx")
CKPT_URL = (
    "https://huggingface.co/datasets/aveocr/Market-1501-v15.09.15.zip/resolve/main/"
    "osnet_x1_0_msmt17_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth"
)


def load_osnet_module():
    spec = importlib.util.spec_from_file_location("osnet_standalone", OSNET_PY)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["osnet_standalone"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    import torch

    if not OSNET_PY.exists():
        raise SystemExit(f"[reid] {OSNET_PY} missing; clone the repo in the Dockerfile")
    if not CKPT.exists():
        print(f"[reid] downloading {CKPT_URL}")
        urllib.request.urlretrieve(CKPT_URL, CKPT)

    mod = load_osnet_module()
    model = mod.osnet_x1_0(pretrained=False, num_classes=1000)
    ckpt = torch.load(CKPT, map_location="cpu")
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    # The classifier head is trained for MSMT17's class count and is unused in
    # eval mode; drop it so the remaining (backbone + bottleneck) keys match.
    state = {k: v for k, v in state.items()
             if not k.startswith("classifier.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    real_missing = [k for k in missing if not k.startswith("classifier.")]
    if real_missing:
        raise SystemExit(f"[reid] checkpoint missing backbone keys: {real_missing[:5]}")
    if unexpected:
        print(f"[reid] warning: ignored unexpected keys: {unexpected[:5]}")
    model.eval().requires_grad_(False)

    params = sum(p.numel() for p in model.parameters())
    print(f"[reid] osnet_x1_0 params={params / 1e6:.2f}M")

    dummy = torch.randn(1, 3, 256, 128)
    # dynamo=False: the legacy exporter embeds weights in the .onnx file.
    # The dynamo exporter writes external data (a .onnx.data sidecar) which
    # left a truncated 770 KB graph-only file when only the .onnx was kept.
    torch.onnx.export(
        model,
        dummy,
        str(OUT),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"[reid] wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB)")

    # Smoke inference at two batch sizes proves the dynamic dim works and the
    # output really is the 512-d embedding (not classifier logits).
    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(str(OUT), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    for b in (1, 8):
        out = session.run(None, {name: np.zeros((b, 3, 256, 128), np.float32)})[0]
        if out.shape != (b, 512):
            raise SystemExit(f"[reid] unexpected embedding shape {out.shape} for batch {b}")
        print(f"[reid] smoke batch={b} -> {out.shape}")
    print("[reid] OSNet x1.0 export OK")


if __name__ == "__main__":
    main()
