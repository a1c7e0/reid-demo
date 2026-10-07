"""Hard-sample face benchmark (pairwise verification on hard regimes).

Datasets under /h (testsets/hard):
  kinfacew   KinFaceW-I/II kin pairs = look-alike IMPOSTORS (different ids)
  cfp        CFP FF / FP protocols (folds subsampled)
  calfw      cross-age pairs (consecutive-line pair file)
  cplfw      cross-pose pairs
  tinyface   native low-res, same-id vs diff-id pairs sampled from
             Testing_Set/Gallery_{Match,Distractor}

Stages (same crop cache as model_bench):
  crop - detect+align every unique image once -> /s2/hard_crops.npz
  run  - embed per model -> pair scores -> per-(model,set) JSON line

Usage: python hard_bench.py crop
       python hard_bench.py run [model ...]
"""
import glob
import json
import os
import random
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, "/app")

from app.ml.face import FaceEngine, align5

H = "/h"
CROPS = "/s2/hard_crops.npz"
SEED = 42

MODELS = {
    "sface_fp32": ("/zoo/face_recognition_sface_2021dec.onnx", 112, 1.0, 0.0, 1.0, True, True),
    "sface_int8": ("/zoo/face_recognition_sface_2021dec_int8.onnx", 112, 1.0, 0.0, 1.0, True, True),
    "mobilefacenet_w600k": ("/zoo/w600k_mbf.onnx", 112, 1.0, 127.5, 127.5, True, True),
    "adaface_irv1": ("/zoo/adaface_ronai_model.onnx", 160, 1 / 255.0, 0.5, 0.5, True, True),
    "lvface_t": ("/zoo/LVFace-T_Glint360K.onnx", 112, 1 / 255.0, 0.5, 0.5, True, True),
    "arcface_r50": ("/zoo/w600k_r50.onnx", 112, 1.0, 127.5, 127.5, True, True),
}
# operating points chosen on LFW (best HIT with FM<=5)
T_OP = {"sface_fp32": 0.50, "sface_int8": 0.50, "mobilefacenet_w600k": 0.40,
        "adaface_irv1": 0.80, "lvface_t": 0.50, "arcface_r50": 0.35}


def kin_pairs():
    from scipy.io import loadmat
    out = []
    for f in sorted(glob.glob(f"{H}/kinfacew/KinFaceW-*/meta_data/*_pairs.mat")):
        d = loadmat(f)
        imgdir = os.path.join(os.path.dirname(os.path.dirname(f)), "images")
        rel_dir = {"fd": "father-dau", "fs": "father-son",
                   "md": "mother-dau", "ms": "mother-son"}

        def resolve(name):
            sub = rel_dir.get(name.split("_")[0], "")
            p1 = os.path.join(imgdir, sub, name)
            return p1 if os.path.exists(p1) else os.path.join(imgdir, name)

        for row in d["pairs"]:
            a = str(np.ravel(row[2])[0]).strip("[]'\" ")
            b = str(np.ravel(row[3])[0]).strip("[]'\" ")
            out.append((resolve(a), resolve(b), 0))
    return out  # all impostors


def cfp_pairs(n_per=400):
    out = []
    prot = f"{H}/cfp/cfp-dataset/Protocol"
    for kind in ("FF", "FP"):
        fdir = f"{prot}/Split/{kind}"
        same, diff = [], []
        for fold in sorted(os.listdir(fdir))[:3]:
            same += [l.strip() for l in open(f"{fdir}/{fold}/same.txt") if l.strip()]
            diff += [l.strip() for l in open(f"{fdir}/{fold}/diff.txt") if l.strip()]
        rnd = random.Random(SEED)
        rnd.shuffle(same)
        rnd.shuffle(diff)
        for lst, lab in ((same[:n_per], 1), (diff[:n_per], 0)):
            for line in lst:
                ia, ib = line.split(",")
                la = f"{prot}/Pair_list_F.txt"
                lb = f"{prot}/Pair_list_P.txt" if kind == "FP" else la
                pa = [l.split()[1] for l in open(la)][int(ia) - 1]
                pb = [l.split()[1] for l in open(lb)][int(ib) - 1]
                out.append((os.path.normpath(f"{prot}/{pa}"),
                            os.path.normpath(f"{prot}/{pb}"), lab))
    return out


def line_pair_file(path, imgdir, n_same, n_diff):
    lines = [l.strip().split() for l in open(path) if l.strip()]
    pairs = []
    for i in range(0, len(lines) - 1, 2):
        a, b = lines[i], lines[i + 1]
        lab = int(a[1])
        pairs.append((os.path.join(imgdir, a[0]), os.path.join(imgdir, b[0]), lab))
    same = [p for p in pairs if p[2] == 1]
    diff = [p for p in pairs if p[2] == 0]
    rnd = random.Random(SEED)
    rnd.shuffle(same)
    rnd.shuffle(diff)
    return same[:n_same] + diff[:n_diff]


def tinyface_pairs(n=400):
    by_id = {}
    for f in glob.glob(f"{H}/tinyface/tinyface/Testing_Set/Gallery_Match/*.jpg"):
        pid = os.path.basename(f).split("_")[0]
        by_id.setdefault(pid, []).append(f)
    for f in glob.glob(f"{H}/tinyface/tinyface/Testing_Set/Gallery_Distractor/*.jpg"):
        pid = os.path.basename(f).split("_")[0]
        by_id.setdefault(pid, []).append(f)
    rnd = random.Random(SEED)
    multi = [p for p, v in by_id.items() if len(v) > 1]
    rnd.shuffle(multi)
    same = []
    for pid in multi[: n * 2]:
        v = by_id[pid]
        same.append((v[0], v[1], 1))
        if len(same) >= n:
            break
    ids = list(by_id)
    diff = []
    while len(diff) < n:
        a, b = rnd.sample(ids, 2)
        diff.append((by_id[a][0], by_id[b][0], 0))
    return same + diff


def build_pairs():
    sets = {
        "kinfacew": kin_pairs(),
        "cfp": cfp_pairs(),
        "calfw": line_pair_file(f"{H}/calfw/calfw/pairs_CALFW.txt",
                                f"{H}/calfw/calfw/aligned images", 800, 800),
        "cplfw": line_pair_file(f"{H}/cplfw/cplfw/pairs_CPLFW.txt",
                                f"{H}/cplfw/cplfw/aligned images", 600, 600),
        "tinyface": tinyface_pairs(),
    }
    return sets


def do_crop():
    sets = build_pairs()
    paths = sorted({p for s in sets.values() for pair in s for p in pair[:2]})
    eng = FaceEngine()
    crops, kept = [], []
    for k, p in enumerate(paths):
        im = cv2.imread(p)
        if im is None:
            continue
        crop = None
        eng.detector.setInputSize((im.shape[1], im.shape[0]))
        try:
            _, faces = eng.detector.detect(im)
        except Exception:
            faces = None
        if faces is not None and len(faces) > 0:
            row = max(faces, key=lambda r: float(r[14]))
            crop = align5(im, row)
        else:
            crop = cv2.resize(im, (112, 112))
        crops.append(crop)
        kept.append(p)
        if (k + 1) % 500 == 0:
            print("crop", k + 1, flush=True)
    np.savez_compressed(CROPS, crops=np.stack(crops), paths=np.array(kept))
    json.dump({k: v for k, v in sets.items()}, open("/s2/hard_pairs.json", "w"))
    print("crops:", len(kept), "pairs:", {k: len(v) for k, v in sets.items()})


def percentile(x, q):
    return round(float(np.percentile(x, q)), 4)


def do_run(names):
    data = np.load(CROPS, allow_pickle=True)
    crops, paths = data["crops"], list(data["paths"])
    pidx = {p: i for i, p in enumerate(paths)}
    sets = json.load(open("/s2/hard_pairs.json"))
    import onnxruntime as ort
    for name in names:
        path, size, scale, mean, std, swap, l2 = MODELS[name]
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        inp, out = sess.get_inputs()[0].name, sess.get_outputs()[0].name
        embs = []
        t0 = time.time()
        for im in crops:
            if size != 112:
                im = cv2.resize(im, (size, size))
            x = im[:, :, ::-1].copy() if swap else im.copy()
            x = (x.astype(np.float32) * scale - mean) / std
            v = sess.run([out], {inp: x.transpose(2, 0, 1)[None]})[0][0].astype(np.float32)
            if l2:
                v = v / max(float(np.linalg.norm(v)), 1e-6)
            embs.append(v)
        emb = np.stack(embs)
        np.save(f"/s2/hemb_{name}.npy", emb)
        ms = round((time.time() - t0) / len(crops) * 1000, 1)
        for sname, pairs in sets.items():
            sims = np.array([float(np.dot(emb[pidx[a]], emb[pidx[b]]))
                             for a, b, lab in pairs if a in pidx and b in pidx])
            labs = np.array([lab for a, b, lab in pairs if a in pidx and b in pidx])
            same, diff = sims[labs == 1], sims[labs == 0]
            t = T_OP[name]
            fmr = float((diff >= t).mean()) if len(diff) else -1
            tpr = float((same >= t).mean()) if len(same) else -1
            res = dict(model=name, set=sname, n_same=int(len(same)), n_diff=int(len(diff)),
                       same_p50=percentile(same, 50) if len(same) else None,
                       diff_p50=percentile(diff, 50), diff_p90=percentile(diff, 90),
                       diff_p99=percentile(diff, 99), diff_max=round(float(diff.max()), 4),
                       fmr_at_t=fmr, tpr_at_t=tpr, t_op=t, embed_ms=ms)
            print(json.dumps(res), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "crop":
        do_crop()
    else:
        do_run(sys.argv[2:] or list(MODELS))
