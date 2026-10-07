"""Person re-identification benchmark on Market-1501 (standard protocol).

query  = query/               (3368 probes)
gallery = bounding_box_test/  (15913 candidates, junk/distractors excluded)

Reports rank-1 / rank-5 accuracy and mAP for each ONNX model given, using
the same preprocessing the gallery uses (app.ml.reid.ReId).

Usage (inside the app container):
    PYTHONPATH=/app python /eval/market_reid.py \
        --market /eval/Market-1501-v15.09.15 \
        --model /models/osnet_x0_25_msmt17.onnx \
        --model /models/osnet_x1_0_msmt17.onnx \
        --out /eval/out/market_reid.json
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

from app.ml.reid import ReId

PAT = re.compile(r"^(-?\d+)_c\d+")  # e.g. 0007_c3s3_061350_00.jpg -> pid 7


def list_dir(d: Path) -> tuple[np.ndarray, list[str]]:
    """Return (pids, paths) for one Market-1501 split; junk pid<0 and pid=0 dropped
    from evaluation, but pid=0 (distractors) are kept out entirely per convention."""
    pids, paths = [], []
    for p in sorted(d.glob("*.jpg")):
        m = PAT.search(p.name)
        if not m:
            continue
        pid = int(m.group(1))
        if pid <= 0:
            continue  # junk (-1) and distractor (0) images are excluded from eval
        pids.append(pid)
        paths.append(str(p))
    return np.asarray(pids), paths


def make_embedder(model_path: str, arch: str):
    """Return embed_fn(list[bgr]) -> (n, d) for the chosen architecture."""
    import numpy as _np
    from PIL import Image
    if arch == "osnet":
        reid = ReId(model_path)
        return lambda buf: reid.embed(buf)
    import onnxruntime as ort
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    hidden = sess.get_outputs()[0].name
    mean = _np.array([0.485, 0.456, 0.406], dtype=_np.float32)
    std = _np.array([0.229, 0.224, 0.225], dtype=_np.float32)

    def fn(buf):
        xs = []
        for bgr in buf:
            rgb = bgr[:, :, ::-1]
            h, w = rgb.shape[:2]
            s = 256 / min(w, h)
            im = _np.asarray(Image.fromarray(rgb).resize(
                (int(w * s + 0.5), int(h * s + 0.5)), Image.BICUBIC), dtype=_np.float32)
            h, w = im.shape[:2]
            l, t = (w - 224) // 2, (h - 224) // 2
            im = im[t:t + 224, l:l + 224] / 255.0
            im = (im - mean) / std
            xs.append(im.transpose(2, 0, 1))
        out = sess.run([hidden], {inp.name: _np.stack(xs).astype(_np.float32)})[0]
        # Copy the CLS row: a bare `out[:, 0]` is a view that keeps the whole
        # (batch, 257, 384) hidden-state buffer alive for every batch parked in
        # the caller's embedding list (~3.15 MB/batch), which OOM-killed the
        # 1200 MB Market run after ~4000 gallery images.
        return out[:, 0].copy()
    return fn


def embed_all(model_path: str, paths: list[str], batch_note: str = "",
              arch: str = "osnet") -> np.ndarray:
    from PIL import Image, ImageOps
    import cv2

    embed_fn = make_embedder(model_path, arch)
    batch = 16 if arch == "osnet" else 8
    embs = []
    t0 = time.time()
    buf = []
    for i, p in enumerate(paths):
        with Image.open(p) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            bgr = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
        buf.append(bgr)
        if len(buf) == batch:
            embs.append(embed_fn(buf))
            buf = []
            if (i + 1) % 2048 == 0:
                print(f"  [market] {i + 1}/{len(paths)} {batch_note}", flush=True)
    if buf:
        real = len(buf)
        if arch == "osnet":  # fixed-batch exports need a full batch
            while len(buf) < 16:
                buf.append(buf[-1])
            embs.append(embed_fn(buf)[:real])
        else:
            embs.append(embed_fn(buf))
    out = np.concatenate(embs) if embs else np.zeros((0, 512), np.float32)
    print(f"  [market] embedded {len(out)} images in {time.time() - t0:.1f}s",
          flush=True)
    return out


def evaluate(q_emb, q_pid, g_emb, g_pid) -> dict:
    """Chunked cosine ranking -> CMC rank-k + mAP."""
    ranks = {1: 0, 5: 0}
    aps = []
    chunk = 256
    for s in range(0, len(q_emb), chunk):
        sims = q_emb[s:s + chunk] @ g_emb.T  # (c, G)
        q_chunk = q_pid[s:s + chunk]
        for row in range(sims.shape[0]):
            order = np.argsort(-sims[row])
            g = g_pid[order]
            q = q_chunk[row]
            hits = (g == q).astype(np.int32)
            if hits.sum() == 0:
                continue
            first = int(np.argmax(hits))
            ranks[1] += first == 0
            ranks[5] += first < 5
            # AP over all relevant gallery items
            precisions = np.cumsum(hits) / (np.arange(len(hits)) + 1)
            aps.append(float((precisions * hits).sum() / hits.sum()))
    n = len(q_emb)
    return {
        "rank1": round(ranks[1] / n, 4),
        "rank5": round(ranks[5] / n, 4),
        "mAP": round(float(np.mean(aps)) if aps else 0.0, 4),
        "queries": n,
        "gallery": len(g_pid),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", required=True, help="unzipped Market-1501 root")
    ap.add_argument("--model", action="append", required=True,
                    help="ONNX ReID model path (repeatable)")
    ap.add_argument("--arch", default="osnet", choices=["osnet", "dinov2"])
    ap.add_argument("--out", default="/eval/out/market_reid.json")
    args = ap.parse_args()

    root = Path(args.market)
    q_dir = root / "query"
    g_dir = root / "bounding_box_test"
    if not q_dir.is_dir() or not g_dir.is_dir():
        raise SystemExit(f"[market] expected splits not found under {root}")

    q_pid, q_paths = list_dir(q_dir)
    g_pid, g_paths = list_dir(g_dir)
    print(f"[market] query={len(q_paths)} gallery={len(g_paths)}", flush=True)

    results = {}
    for model in args.model:
        name = Path(model).name
        print(f"[market] model={name}", flush=True)
        q_emb = embed_all(model, q_paths, name, args.arch)
        g_emb = embed_all(model, g_paths, name, args.arch)
        q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-9)
        g_emb = g_emb / (np.linalg.norm(g_emb, axis=1, keepdims=True) + 1e-9)
        res = evaluate(q_emb, q_pid, g_emb, g_pid)
        res["model"] = name
        res["model_bytes"] = Path(model).stat().st_size
        results[name] = res
        print(f"[market] {name}: rank1={res['rank1']} rank5={res['rank5']} "
              f"mAP={res['mAP']}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"[market] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
