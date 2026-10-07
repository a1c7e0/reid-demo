"""Export person-ReID zoo checkpoints to ONNX (feature-only, dynamic batch).

Models:
  1. osnet_ain_x1_0_msmt17.pth        (torchreid state dict, `module.` prefix,
                                       classifier 4101 = MSMT17 ids)
  2. osnet_ain_x0_75_multisource_MS_D_C.pt / osnet_ain_x1_0_...  (LibreYOLO
                                       plain state dicts, classifier stripped)
  3. fastreid_msmt_bot_R50.pth        (FastReID v0.1.1 training state; only the
                                       `model` part is used, optimizer/scheduler
                                       stripped; BNNeck eval feature exported)

All exported graphs take the app's normalized input (RGB, /255, ImageNet
mean/std - see app/ml/reid.py) and return the raw embedding; the app L2-normalizes.

OSNet arch comes from deep-person-reid torchreid/models/osnet_ain.py loaded
standalone (the torchreid package itself needs Cython to build).
The BoT arch is hand-rolled from fast-reid v0.1.1 sources (ResNet-50, last
stride 1 + BNNeck over 2048-d pooled features; eval returns post-BN feature).
"""
import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Build-time paths (the original /zoo /dpr mounts were a manual workflow);
# env overrides let the image build run this from its own workspace.
ZOO = Path(__import__("os").getenv("ZOO", "/zoo"))
WORK = Path(__import__("os").getenv("WORK", "/work"))
OSNET_AIN_PY = Path(__import__("os").getenv(
    "OSNET_AIN_PY", "/dpr/torchreid/models/osnet_ain.py"))
OUT_DIR = ZOO  # exported ONNX saved next to the checkpoints

EXPECTED_INPUT = (3, 256, 128)
EXPECTED_DIM = {"osnet": 512, "bot": 2048}


def load_osnet_ain_module():
    spec = importlib.util.spec_from_file_location("osnet_ain_standalone", OSNET_AIN_PY)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["osnet_ain_standalone"] = mod
    spec.loader.exec_module(mod)
    return mod


def export_onnx(model, out_path: Path, tag: str) -> None:
    model.eval().requires_grad_(False)
    dummy = torch.randn(1, *EXPECTED_INPUT)
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"[export] {tag}: wrote {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")


def smoke_and_parity(model, out_path: Path, dim: int, tag: str) -> None:
    import numpy as np
    import onnxruntime as ort

    model.eval()
    x = torch.randn(3, *EXPECTED_INPUT)
    with torch.no_grad():
        ref = model(x).numpy()
    assert ref.shape == (3, dim), f"{tag}: torch output shape {ref.shape}"

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    assert sess.get_inputs()[0].shape[1:] == list(EXPECTED_INPUT), sess.get_inputs()[0].shape
    out = sess.run(None, {iname: x.numpy()})[0]
    assert out.shape == (3, dim), f"{tag}: onnx output shape {out.shape}"
    max_diff = float(np.abs(out - ref).max())
    assert max_diff < 1e-4, f"{tag}: torch/onnx mismatch {max_diff}"
    out1 = sess.run(None, {iname: x.numpy()[:1]})[0]
    assert out1.shape == (1, dim), f"{tag}: batch-1 shape {out1.shape}"
    print(f"[export] {tag}: smoke+parity OK (max|Δ|={max_diff:.2e}, dim={dim})")


# --------------------------------------------------------------------------
# OSNet / OSNet-AIN (torchreid)
# --------------------------------------------------------------------------
def export_osnet(factory: str, ckpt_name: str, out_name: str) -> Path:
    mod = load_osnet_ain_module()
    model = getattr(mod, factory)(num_classes=1000, pretrained=False)

    ckpt = torch.load(ZOO / ckpt_name, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        ckpt = ckpt["state_dict"]
    state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in ckpt.items()}
    dropped = sorted(k for k in state if k.startswith("classifier."))
    state = {k: v for k, v in state.items() if not k.startswith("classifier.")}

    missing, unexpected = model.load_state_dict(state, strict=False)
    real_missing = [k for k in missing if not k.startswith("classifier.")]
    assert not real_missing, f"{out_name}: backbone keys missing: {real_missing[:5]}"
    assert not unexpected, f"{out_name}: unexpected keys: {unexpected[:5]}"
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[export] {out_name}: loaded {len(state)} keys (dropped {dropped}), "
          f"params={n_params / 1e6:.2f}M")

    out_path = OUT_DIR / out_name
    export_onnx(model, out_path, out_name)
    smoke_and_parity(model, out_path, EXPECTED_DIM["osnet"], out_name)
    return out_path


# --------------------------------------------------------------------------
# FastReID BoT (bagtricks R50 + BNNeck), hand-rolled from v0.1.1 sources
# --------------------------------------------------------------------------
class BotR50Feature(nn.Module):
    """fast-reid v0.1.1 Baseline eval feature: BNNeck(2048) over ResNet-50 GAP.

    torchvision ResNet-50 is structurally identical to fast-reid's ResNet
    (same Bottleneck: 1x1/3x3/1x1, stride on the 3x3, downsample=Conv+BN);
    with last_stride=1 the graphs match. Input is pre-normalized by the caller.
    """

    def __init__(self):
        super().__init__()
        from torchvision.models import resnet50

        r = resnet50(weights=None)
        r.layer4[0].conv2.stride = (1, 1)          # LAST_STRIDE=1
        r.layer4[0].downsample[0].stride = (1, 1)
        self.conv1, self.bn1, self.relu, self.maxpool = r.conv1, r.bn1, r.relu, r.maxpool
        self.layer1, self.layer2 = r.layer1, r.layer2
        self.layer3, self.layer4 = r.layer3, r.layer4
        self.bnneck = nn.BatchNorm1d(2048)          # bias_freeze=True in fast-reid

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.bnneck(x)


def export_fastreid(ckpt_name: str, out_name: str) -> Path:
    ckpt = torch.load(ZOO / ckpt_name, map_location="cpu", weights_only=True)
    state = ckpt["model"] if "model" in ckpt else ckpt

    model = BotR50Feature()
    mapped = {}
    skipped = []
    for k, v in state.items():
        if k.startswith("backbone."):
            mapped[k[len("backbone."):]] = v
        elif k.startswith("heads.bnneck."):
            mapped["bnneck." + k[len("heads.bnneck."):]] = v
        elif k in ("pixel_mean", "pixel_std") or k.startswith("heads.classifier."):
            skipped.append(k)
        else:
            raise AssertionError(f"{out_name}: unmapped checkpoint key {k}")
    missing, unexpected = model.load_state_dict(mapped, strict=True)
    assert not missing and not unexpected
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[export] {out_name}: loaded {len(mapped)} keys "
          f"(skipped {skipped}), params={n_params / 1e6:.2f}M")

    out_path = OUT_DIR / out_name
    export_onnx(model, out_path, out_name)
    smoke_and_parity(model, out_path, EXPECTED_DIM["bot"], out_name)
    return out_path


# --------------------------------------------------------------------------
# GEFF / CAL (Gu et al. CVPR 2022) - clothes-changing ReID, RGB only
# Arch hand-rolled from bar371/GEFF ReIDModules/CAL (img_resnet.ResNet50):
# torchvision ResNet-50 children[:-2] (RES4_STRIDE=1) -> MaxAvgPooling
# (max+avg concat -> 4096) -> BatchNorm1d(4096); eval returns the BN feature.
# Note: CAL's own eval resizes to 384x192 (DATA.HEIGHT/WIDTH); our bench feeds
# the app's 256x128 preprocessing for cross-model comparability.
# --------------------------------------------------------------------------
class CalResNet50Feature(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import resnet50

        r = resnet50(weights=None)
        r.layer4[0].conv2.stride = (1, 1)          # RES4_STRIDE=1
        r.layer4[0].downsample[0].stride = (1, 1)
        self.base = nn.Sequential(*list(r.children())[:-2])
        self.bn = nn.BatchNorm1d(4096)             # FEATURE_DIM=4096

    def forward(self, x):
        f = self.base(x)
        max_f = F.adaptive_max_pool2d(f, 1)
        avg_f = F.adaptive_avg_pool2d(f, 1)
        f = torch.cat((max_f, avg_f), 1).flatten(1)  # MaxAvgPooling
        return self.bn(f)


def export_cal(ckpt_rel: str, out_rel: str) -> Path:
    ckpt = torch.load(ZOO / ckpt_rel, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]
    model = CalResNet50Feature()
    missing, unexpected = model.load_state_dict(state, strict=True)
    assert not missing and not unexpected
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[export] {out_rel}: loaded {len(state)} keys, params={n_params / 1e6:.2f}M")

    out_path = OUT_DIR / out_rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    export_onnx(model, out_path, out_rel)
    smoke_and_parity(model, out_path, 4096, out_rel)
    return out_path


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which == "default":
        # the model the app actually runs (config.REID_MODEL default):
        # OSNet-AIN x1.0 trained on MSMT17 - wins the pairwise benchmark
        # in docs/REID-MODEL-ZOO.md and the attach A/B in fusion_bench.py
        export_osnet("osnet_ain_x1_0", "osnet_ain_x1_0_msmt17.pth",
                     "osnet_ain_x1_0_msmt17.onnx")
        print("EXPORT ALL DONE")
        return
    if which in ("all", "osnet"):
        export_osnet("osnet_ain_x1_0", "osnet_ain_x1_0_msmt17.pth",
                     "osnet_ain_x1_0_msmt17.onnx")
        export_osnet("osnet_ain_x0_75", "osnet_ain_x0_75_multisource_MS_D_C.pt",
                     "osnet_ain_x0_75_multisource_MS_D_C.onnx")
        export_osnet("osnet_ain_x1_0", "osnet_ain_x1_0_multisource_MS_D_C.pt",
                     "osnet_ain_x1_0_multisource_MS_D_C.onnx")
    if which in ("all", "fastreid"):
        export_fastreid("fastreid_msmt_bot_R50.pth", "fastreid_msmt_bot_R50.onnx")
    if which in ("all", "cal"):
        export_cal("geff_ccreid_weights/CAL/prcc-checkpoint.pth.tar",
                   "geff_ccreid_weights/CAL/geff_cal_prcc.onnx")
    print("EXPORT ALL DONE")


if __name__ == "__main__":
    main()
