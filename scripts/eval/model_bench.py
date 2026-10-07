"""Face-model benchmark under the Ente clustering algorithm on LFW.

Pipeline per model: YuNet detect + 5-point align -> crop -> model-specific
normalization -> embedding -> Ente greedy single-linkage clustering
(newest-first, backward-only 1-NN, quality-adaptive 0.76/0.84, Ente's
isBadFace and clusterable gates) -> pairwise scores vs LFW identity labels.

Stages:
  crop   - detect+align once, cache aligned crops to /s2/model_bench_crops.npz
  run    - embed + cluster + score each model, print one JSON line each

Usage: python model_bench.py crop [n_images]
       python model_bench.py run [model ...]
"""
import glob
import json
import random
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, "/app")

from app.ml.face import FaceEngine, align5

CROPS = "/s2/model_bench_crops.npz"

# name -> (onnx path, input size, scale, mean, std, swap_rb, l2)
MODELS = {
    # (onnx, input size, scale, mean, std, swap_rb, l2): embedding input is
    # (x*scale - mean)/std on raw RGB; w600k family means (x-127.5)/127.5
    "sface_fp32": ("/zoo/face_recognition_sface_2021dec.onnx", 112, 1.0, 0.0, 1.0, True, True),
    "sface_int8": ("/zoo/face_recognition_sface_2021dec_int8.onnx", 112, 1.0, 0.0, 1.0, True, True),
    "mobilefacenet_w600k": ("/zoo/w600k_mbf.onnx", 112, 1.0, 127.5, 127.5, True, True),
    "adaface_irv1": ("/zoo/adaface_ronai_model.onnx", 160, 1 / 255.0, 0.5, 0.5, True, True),
    "lvface_t": ("/zoo/LVFace-T_Glint360K.onnx", 112, 1 / 255.0, 0.5, 0.5, True, True),
    "arcface_r50": ("/zoo/w600k_r50.onnx", 112, 1.0, 127.5, 127.5, True, True),
    "glintr100": ("/zoo/glintr100.onnx", 112, 1.0, 127.5, 127.5, True, True),
    "lvface_b": ("/zoo/LVFace-B_Glint360K.onnx", 112, 1 / 255.0, 0.5, 0.5, True, True),
}


def do_crop():
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 1200
    files = sorted(glob.glob("/lfw/*.jpg"))
    random.seed(7)
    random.shuffle(files)
    files = files[:n]
    eng = FaceEngine()
    crops, labels, meta = [], [], []
    for k, p in enumerate(files):
        try:
            lab = open(p[:-4] + ".cls").read().strip()
        except Exception:
            continue
        im = cv2.imread(p)
        eng.detector.setInputSize((im.shape[1], im.shape[0]))
        _, faces = eng.detector.detect(im)
        if faces is None or len(faces) == 0:
            continue
        row = max(faces, key=lambda r: float(r[14]))
        a = align5(im, row)
        lm = [[float(row[4]), float(row[5])], [float(row[6]), float(row[7])],
              [float(row[8]), float(row[9])], [float(row[10]), float(row[11])],
              [float(row[12]), float(row[13])]]
        crops.append(a)
        labels.append(lab)
        meta.append((float(row[14]), lm))
        if (k + 1) % 200 == 0:
            print("cropped", k + 1, flush=True)
    np.savez_compressed(CROPS, crops=np.stack(crops), label=np.array(labels),
                        meta=json.dumps(meta))
    print("crops saved:", len(crops))


def quality_flags(crops, meta):
    """Ente gates: clusterable = blur>10 && score>0.8; isBadFace = blur<50
    || (blur<200 && score<0.85) || sideways (faceDirection from landmarks)."""
    ok, bad = [], []
    for i, (score, lm) in enumerate(meta):
        g = cv2.cvtColor(crops[i], cv2.COLOR_RGB2GRAY)
        blur = cv2.Laplacian(g, cv2.CV_64F).var()
        lm = np.array(lm, dtype=np.float32)
        eye_mid = (lm[0] + lm[1]) / 2
        mouth_mid = (lm[3] + lm[4]) / 2
        dist = float(np.linalg.norm(eye_mid - mouth_mid))
        eye_dist = float(np.linalg.norm(lm[0] - lm[1]))
        sideways = (abs(float(lm[0][0] - lm[1][0])) < 0.2 * eye_dist or
                    abs(float(eye_mid[1] - mouth_mid[1])) < 0.5 * dist)
        clusterable = blur > 10 and score > 0.8
        is_bad = blur < 50 or (blur < 200 and score < 0.85) or sideways
        ok.append(clusterable)
        bad.append(is_bad)
    return np.array(ok), np.array(bad)


def ente_cluster(emb, clusterable, is_bad, photo, t, t_good=0.76, t_bad=0.84):
    """Faithful Ente greedy: newest-first, backward-only 1-NN join."""
    idx = [i for i in range(len(emb)) if clusterable[i]]
    idx.sort(key=lambda i: (-t[i], i))
    cids = np.full(len(emb), -1, dtype=np.int64)
    n = 0
    for r, i in enumerate(idx):
        best, best_s = -1, t_good
        for j in idx[:r]:
            if photo[j] == photo[i]:
                continue
            th = t_bad if is_bad[j] else t_good
            s = float(np.dot(emb[i], emb[j]))
            if s >= th and s >= best_s:
                best, best_s = cids[j], s
        if best >= 0:
            cids[i] = best
        else:
            cids[i] = n
            n += 1
    return cids, n


def score(cids, labels, photo):
    FM = FS = UN = HIT = same = 0
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = cids[i], cids[j]
            if labels[i] == labels[j]:
                same += 1
                if a < 0 or b < 0:
                    UN += 1
                elif a == b:
                    HIT += 1
                else:
                    FS += 1
            else:
                if a >= 0 and a == b:
                    FM += 1
    sizes = {}
    for c in cids:
        if c >= 0:
            sizes[c] = sizes.get(c, 0) + 1
    return dict(FM=FM, FS=FS, UN=UN, HIT=HIT, same_pairs=same,
                persons=len(sizes), biggest=max(sizes.values()) if sizes else 0)


def do_run(names):
    import onnxruntime as ort
    data = np.load(CROPS, allow_pickle=True)
    crops, labels = data["crops"], data["label"]
    meta = json.loads(str(data["meta"]))
    clusterable, is_bad = quality_flags(crops, meta)
    photo = np.arange(len(labels))
    t = photo.astype(np.float64)
    print("units:", len(labels), "clusterable:", int(clusterable.sum()), flush=True)
    for name in names:
        path, size, scale, mean, std, swap, l2 = MODELS[name]
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0].name
        out = sess.get_outputs()[0].name
        embs = []
        t0 = time.time()
        for i in range(len(crops)):
            im = crops[i]
            if size != 112:
                im = cv2.resize(im, (size, size))
            x = im[:, :, ::-1].copy() if swap else im.copy()
            x = (x.astype(np.float32) * scale - mean) / std
            x = x.transpose(2, 0, 1)[None]
            v = sess.run([out], {inp: x})[0][0].astype(np.float32)
            if l2:
                v = v / max(float(np.linalg.norm(v)), 1e-6)
            embs.append(v)
        dt = time.time() - t0
        emb = np.stack(embs)
        np.save(f"/s2/memb_{name}.npy", emb)
        cids, _ = ente_cluster(emb, clusterable, is_bad, photo, t)
        res = score(cids, labels, photo)
        res.update(model=name, embed_ms_per_img=round(dt / len(crops) * 1000, 1),
                   dim=int(emb.shape[1]))
        print(json.dumps(res), flush=True)
        # threshold sweep on cached embeddings (Ente's good/bad offset 0.08)
        for tg in [round(x, 2) for x in np.arange(0.30, 0.86, 0.05)]:
            cids, _ = ente_cluster(emb, clusterable, is_bad, photo, t,
                                   t_good=tg, t_bad=round(tg + 0.08, 2))
            r2 = score(cids, labels, photo)
            r2.update(model=name, t_good=tg)
            print(json.dumps(r2), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "crop":
        do_crop()
    else:
        do_run(sys.argv[2:] or list(MODELS))
