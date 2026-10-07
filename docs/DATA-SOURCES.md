# Data Sources — datasets and models, with download links

Everything under `testsets/` is **untracked and re-downloadable**. Nothing in
this repository ships data or weights; the Docker build fetches the models it
needs and verifies them by hash.

Datasets are split into two roles and must not be blurred:

- **Tuning sets** — allowed to influence thresholds and parameter choices.
- **Evaluation sets** — used only to report numbers. Never tune on them.

---

## 1. Face datasets

| Set | Role | Size | Regime | Source |
|---|---|---|---|---|
| **LFW** | tuning | 13,233 imgs / 5,749 ids | clean, front-facing | <http://vis-www.cs.umass.edu/lfw/> (`lfw.tgz`, `lfw-funneled.tgz`) |
| **KinFaceW-I / II** | evaluation | 533 / 1002 pairs | **look-alike impostors** (parent–child) | <http://www.kinfacew.com/> (direct zips) |
| **CFP** | evaluation | 7,000 imgs, 500 ids | frontal–profile pose | <http://www.cfpw.io/cfp-dataset.zip> |
| **CALFW** | evaluation | ~12k imgs, 3,000 positive pairs | cross-age | <http://www.whdeng.cn/CALFW> (Google Drive) |
| **CPLFW** | evaluation | ~12k imgs | cross-pose | <http://www.whdeng.cn/CPLFW> (Google Drive) |
| **TinyFace** | evaluation | 169k imgs, 5,139 ids | native low resolution (avg 20×16 px) | <http://qmul-tinyface.github.io/> (Google Drive) |
| **AgeDB-30** | — | — | cross-age | not used (Kaggle-auth mirror only; CALFW covers the regime) |

### Expected on-disk layout

```
testsets/
  lfw_flat/                LFW, flattened: <Name>_<NNNN>.jpg + <Name>_<NNNN>.cls
                           (.cls = one line, the identity label)
  hard/
    kinfacew/KinFaceW-{I,II}/images/     pairs in meta_data/*_pairs.mat
    cfp/cfp-dataset/{Data,Protocol}/     Protocol/Pair_list_{F,P}.txt
    calfw/calfw/{aligned images,pairs_CALFW.txt}
    cplfw/cplfw/{images,aligned images}
    tinyface/tinyface/{Testing_Set,Training_Set}
```

`lfw_flat` is LFW re-laid-out so the benchmark can stream it without
extracting the per-person folders. To rebuild it from the official tarball:
rename `lfw/<Person>/<Person>_<NNNN>.jpg` to `<Person>_<NNNN>.jpg` and write
the person name into a sibling `.cls` file.

**Why these specific sets.** LFW is saturated for modern face models, so it
drives threshold calibration while the hard sets carry the signal: KinFaceW is
genuine different-identity look-alikes (parent–child), which is the exact
"similar boys" failure mode this project must not merge; CFP/CALFW/CPLFW break
the pose and age axes; TinyFace is the ceiling for the quality gates (blur /
score / sideways), which decide what is clustered at all.

---

## 2. Person re-ID datasets (body channel)

| Set | Role | Size | Source |
|---|---|---|---|
| **Market-1501** | tuning | 1,501 ids / 32k imgs | <https://huggingface.co/datasets/aveocr/Market-1501-v15.09.15.zip> |
| **MSMT17_V1** | tuning | 4,101 ids / 126k imgs, multi-scene multi-time | <https://huggingface.co/datasets/xianpeijie/MSMT17_V1> |
| **PRCC / VC-Clothes** | tuning | cross-clothes re-ID | Google Drive bundle from the `Clothes_Change_Person_ReID` repo (torch-serialized `archive/{data.pkl,data,version}`, 277 MB) |

Market-1501 is the model-comparison protocol (test/gallery splits). MSMT17 is
much harder (multiple scenes and capture times) and is what the time-window and
multi-scene rules were fitted on. The cross-clothes bundle arrives
torch-serialized; load it with `torch.load` when needed.

**Note:** DukeMTMC-reID's official host (vision.cs.duke.edu) is dead and no
public mirror was reachable. MSMT17 covers the same multi-scene regime, harder.

### Expected on-disk layout

```
testsets/
  Market-1501-v15.09.15/{bounding_box_train,query,bounding_box_test}/
  reid-train/MSMT17_V1/
  reid-train/archive/       cross-clothes bundle (torch-serialized)
```

---

## 3. Real-photo test data

| Set | Role | Size | Source |
|---|---|---|---|
| `testsets/ente/` | evaluation (manual inspection) | 203 photos | Ente's public sample photography set |
| `testsets/ui-data/` | demo gallery | the app's own store | built by running the app over `testsets/ente` |
| `testsets/ui-data-ente/` | reference run | same photos | a stock-Ente-parameter run, for A/B comparison |

These are for eyeballing behaviour, never for tuning.

---

## 4. Models

All downloaded by `scripts/download_models.sh` at build time and verified.
See `docs/MODEL-ZOO.md` (faces) and `docs/REID-MODEL-ZOO.md` (bodies) for the
full A/B tables and the reasons each was kept or rejected.

### Shipped (≈108 MB total)

| Role | File | Size | Source |
|---|---|---|---|
| Face detector | `face_detection_yunet_2023mar.onnx` | 232 KB | <https://huggingface.co/opencv/face_detection_yunet> |
| Face embedder | `LVFace-T_Glint360K.onnx` | 76.7 MB | <https://huggingface.co/bytedance-research/LVFace> |
| Person detector | `yolo11n-person.onnx` (exported, NMS embedded) | 10.8 MB | Ultralytics YOLO11n |
| Body embedder | `osnet_ain_x1_0_msmt17.onnx` | 8.7 MB | torchreid checkpoint → ONNX (see below) |
| Personness gate | `yolo11n-pose.onnx` | 11.7 MB | Ultralytics YOLO11n-pose |

The body embedder is exported at build time from the torch checkpoint
`osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth`
(downloaded from Hugging Face, sha256
`8a07e8da38946f7cee37f4561617bf8b6d2fe8f3a4027852893ea092e46d919f`,
exported through `scripts/export_reid_zoo.py` with a torch↔ONNX parity assert).

### Benchmarked and rejected (available, not shipped)

| File | Source | Why not |
|---|---|---|
| `w600k_mbf.onnx` | <https://github.com/deepinsight/insightface/releases> | 13.6 MB and fast, but doubles look-alike merges vs LVFace-T |
| `w600k_r50.onnx` | <https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/> | accuracy ceiling, but 174 MB / 656 ms — too heavy for edge |
| `glintr100.onnx` / `_int8` | <https://huggingface.co/HammadNaseer/arcface-glintr100-onnx> | ~105 MB int8, still over budget |
| `adaface_ronai_model.onnx` | <https://huggingface.co/AlKal-3/Ronai-Vision> | catastrophic mixing on the look-alike set |
| `face_recognition_sface_2021dec.onnx` | <https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface> | 38.7 MB fp32; worst look-alike mixer of all (31 KinFaceW pairs) |
| `osnet_x0_25_msmt17.onnx` | <https://huggingface.co/anriha/osnet_x0_25_msmt17> | 0.9 MB, kept as a tiny fallback body model |
| `market_bot_R50.pth`, `msmt_bot_R50.pth` | <https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/> | body candidates, lost the A/B |

### Ente's own models

Ente distributes its model bundle at <https://models.ente.com/> (SHA-256
verified). `docs/ENTE-PORT-SPEC.md` records which of those files this port
uses and exactly which thresholds were carried over.
