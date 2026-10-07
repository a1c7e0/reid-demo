# Face-Recognition Model Zoo for LFW Clustering Benchmarks

Candidate embedding models for benchmarking our photo-gallery people pipeline. All
candidates run in **onnxruntime** (verified on 1.30.0) and fit our existing input
pipeline: **YuNet 5-point landmarks → similarity warp to 112×112 (ArcFace template)
→ per-model normalization**. No benchmark numbers are included here — download and
document only.

Downloaded weights live in `/home/a1c7e0/.hermes/cache/scratch/models-zoo/`
(8 ONNX files, 1123.41 MB total). Every file was smoke-tested: loads in onnxruntime,
accepts its declared input shape, produces a finite embedding.

---

## 1. Downloaded candidates

| Model | Source URL | License | Size (MB) | Architecture | Input size | Preprocessing (scale / mean / std) | Embedding dim | Suitability |
|---|---|---|---|---|---|---|---|---|
| SFace fp32 (`face_recognition_sface_2021dec.onnx`) | https://huggingface.co/opencv/face_recognition_sface (mirror of https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) | Apache-2.0 (OpenCV Zoo dir license) | 38.70 | MobileFaceNet trained with SFace loss (9.67 M params, conv + PReLU-style depthwise stack) | 3×112×112 | scale 1.0, mean 0, std 1 — **raw RGB [0,255]**, BGR→RGB (`swapRB`) | 128 | **Edge.** Permissive license, small, pairs natively with YuNet; accuracy ceiling ~99.4 % LFW |
| SFace int8 (`face_recognition_sface_2021dec_int8.onnx`) | same repo as above | Apache-2.0 | 9.90 | same as SFace fp32, static int8 weights | 3×112×112 | identical to SFace fp32 (input stays float32) | 128 | **Edge.** Quantized SFace for on-device latency/memory studies |
| MobileFaceNet (`w600k_mbf.onnx`) | https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip (buffalo_sc pack) | InsightFace weights: non-commercial research (code MIT) | 13.62 | MobileFaceNet (3.4 M params, depthwise convs + PReLU) trained on WebFace600K | 3×112×112 | RGB, scale 1/127.5, mean 127.5, std 127.5 on [0,255] → [-1,1] | 512 | **Edge.** The canonical InsightFace MobileFaceNet; best tiny model with the standard 512-d InsightFace normalization |
| AdaFace-labeled InceptionResNetV1 (`adaface_ronai_model.onnx`) | https://huggingface.co/AlKal-3/Ronai-Vision/resolve/main/face-recognition-adaface/1/model.onnx | Repo tagged Apache-2.0; training provenance unverified | 93.96 | InceptionResNetV1 / FaceNet-style (facenet-pytorch graph naming: `repeat_*`, `last_linear`, `last_bn`; 23.47 M params), output L2-normalized in-graph | 3×**160**×160 | scale 1/255, mean 0.5, std 0.5 on [0,1] (AdaFace convention). **Channel order and exact recipe unverified** — AdaFace README specifies BGR; test both orders against reference pairs before benchmarking | 512 | **Experimental / mid-tier.** Mismatched input size (160 vs our 112) forces a re-alignment path; treat as a wildcard, not a headline candidate |
| LVFace-T Glint360K (`LVFace-T_Glint360K.onnx`) | https://huggingface.co/bytedance-research/LVFace (arXiv 2501.13420, ICCV 2025) | MIT | 76.65 | Vision-transformer (LVFace-T, 19.1 M params), multi-stage cluster-optimized training | 3×112×112 | RGB, scale 1/255 then (x−0.5)/0.5, i.e. mean 0.5, std 0.5 on [0,1] → [-1,1] | 512 | **Edge→mid.** Newest-gen accuracy at small size (IJB-C 96.67 % @ FAR 1e-4); the modern small-model candidate |
| ArcFace R50 WebFace600K (`w600k_r50.onnx`) | https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/w600k_r50.onnx (buffalo_l pack: https://github.com/deepinsight/insightface/releases) | InsightFace weights: non-commercial research (code MIT) | 174.38 | IResNet-50 (ArcFace, 43.6 M params) trained on WebFace600K | 3×112×112 | RGB, scale 1/127.5, mean 127.5, std 127.5 on [0,255] → [-1,1] | 512 | **Server baseline.** The de-facto reference embedding model (buffalo_l); use it as the "known-good" anchor of the benchmark |
| ArcFace R100 Glint360K (`glintr100.onnx`) | https://huggingface.co/HammadNaseer/arcface-glintr100-onnx/resolve/main/glintr100.onnx (mirror of InsightFace antelopev2 `glintr100.onnx`) | InsightFace weights: non-commercial research (code MIT) | 260.67 | IResNet-100 (ArcFace, 65.2 M params) trained on Glint360K | 3×112×112 | RGB, scale 1/127.5, mean 127.5, std 127.5 on [0,255] → [-1,1] | 512 | **Server / heavy.** The requested Glint360K R100 checkpoint; strong accuracy on low-quality gallery faces |
| LVFace-B Glint360K (`LVFace-B_Glint360K.onnx`) | https://huggingface.co/bytedance-research/LVFace (arXiv 2501.13420) | MIT | 455.53 | Vision-transformer (LVFace-B, 113.8 M params) | 3×112×112 | RGB, scale 1/255 then (x−0.5)/0.5 (mean 0.5, std 0.5 on [0,1]) | 512 | **Server / heavy ceiling.** 2025 SOTA-class accuracy (IJB-C 97.70 % @ FAR 1e-4) — the quality-vs-cost upper bound for clustering |

Verified input/output tensors (onnxruntime):

| File | Input tensor | Output tensor | Smoke-run |
|---|---|---|---|
| `face_recognition_sface_2021dec.onnx` | `data` [1,3,112,112] f32 | `fc1` [1,128] | ok |
| `face_recognition_sface_2021dec_int8.onnx` | `data` [1,3,112,112] f32 | `fc1` [1,128] | ok |
| `w600k_mbf.onnx` | `input.1` [N,3,112,112] f32 | [1,512] f32 | ok |
| `adaface_ronai_model.onnx` | `input` [N,3,160,160] f32 | `embedding` [N,512] f32 (‖e‖=1) | ok |
| `LVFace-T_Glint360K.onnx` | `data` [N,3,112,112] f32 | [N,512] f32 | ok |
| `LVFace-B_Glint360K.onnx` | `data` [N,3,112,112] f32 | [N,512] f32 | ok |
| `w600k_r50.onnx` | `input.1` [N,3,112,112] f32 | [1,512] f32 | ok |
| `glintr100.onnx` | `input.1` [N,3,112,112] f32 | [1,512] f32 | ok |

SHA-256 (first 16 hex): `LVFace-B` 9d834ed8e927fd35 · `LVFace-T` bf8da0e1e93c432d ·
`adaface_ronai` 681739465934b5c7 · `sface fp32` 0ba9fbfa01b5270c ·
`sface int8` 2b0e941e6f16cc04 · `glintr100` 4ab1d6435d639628 (matches the mirror's
published full hash) · `w600k_mbf` 9cc6e4a75f0e2bf0 · `w600k_r50` 4c06341c33c2ca1f.

---

## 2. Exact preprocessing reference

Alignment for all models is our YuNet 5-point → similarity warp to 112×112 with the
ArcFace reference template (`arcface_dst = [(38.2946,51.6963), (73.5318,51.5014),
(56.0252,71.7366), (41.5493,92.3655), (70.7299,92.2041)]`, the template used by
InsightFace `face_align.norm_crop`). Only the pixel normalization differs per model:

| Family | Color order | Formula (input 0–255) | Equivalent on [0,1] |
|---|---|---|---|
| InsightFace (`w600k_r50`, `glintr100`, `w600k_mbf`) | RGB | `(x − 127.5) / 127.5` | mean 0.5, std 0.5 |
| LVFace (`LVFace-T`, `LVFace-B`) | RGB | `(x − 127.5) / 127.5` | mean 0.5, std 0.5 |
| SFace (fp32 + int8) | RGB | `x` unchanged (scale 1.0, mean 0, std 1) | raw [0,255] |
| AdaFace convention (`adaface_ronai_model`) | BGR per AdaFace README (unverified for this export) | `(x − 127.5) / 127.5` | mean 0.5, std 0.5 |

Notes and pitfalls:

- **InsightFace normalization**: the official `ArcFaceONNX` wrapper feeds
  `blobFromImages(imgs, 1/127.5, (112,112), (127.5,127.5,127.5), swapRB=True)` for the
  PyTorch-exported ONNX models. The legacy MXNet exports baked `(x−127.5)/128` into the
  graph. The `/127.5` vs `/128` difference is small but real — pick one and never mix
  enrolled and queried vectors across variants.
- **SFace is NOT [-1,1]-normalized.** OpenCV's `FaceRecognizerSF::feature` does
  `blobFromImage(aligned, 1.0, (112,112), (0,0,0), swapRB=true)` — raw RGB bytes in
  0–255. Feeding it [-1,1] tensors (a common third-party mistake) silently destroys
  accuracy.
- **AdaFace color order**: the official README specifies a **BGR** 112×112 tensor
  normalized with mean 0.5 / std 0.5. The downloaded community export is unverified;
  validate channel order empirically on LFW pairs before use.
- **Never mix embedding spaces.** Cosine comparisons are only meaningful within one
  model; re-embed the whole gallery when switching (we already record the model per
  vector in `faces.spec`).

## 3. Documented but not downloaded

Policy: only ready-to-run ONNX with no authentication wall is downloaded. The
candidates below ship PyTorch `.pth`/`.ckpt`, `.safetensors`, `.h5`/`.tflite`, or
vendor-binary formats only; converting them is out of scope and marked accordingly.

| Model | Source URL | License | Why not downloaded | Note |
|---|---|---|---|---|
| AdaFace IR-50 (MS1MV2 / WebFace4M / CASIA / VGGFace2) | https://github.com/mk-minchul/AdaFace (ckpt); mirrors: https://huggingface.co/VishalMishraTss/AdaFace, https://huggingface.co/minchul/cvlface_adaface_ir50_ms1mv2 | MIT (code + released ckpts) | **No upstream ONNX** — PyTorch `.ckpt` / HF safetensors only; export requires rebuilding the IR-50 backbone | The most important missing candidate: strong on low-resolution faces. If we convert one model ourselves, convert this one (WebFace4M variant has the cleanest license) |
| AdaFace IR-101 (MS1MV2 / MS1MV3 / WebFace12M) | https://github.com/mk-minchul/AdaFace; https://huggingface.co/minchul/cvlface_adaface_ir101_webface12m | MIT | Same — `.ckpt`/safetensors only | Heavy sibling of IR-50; ~2× compute for ~0.1 pt gain (per third-party evaluation) |
| AdaFace ViT-tiny (`vit_tiny_patch8_112.adaface_ms1mv3`) | https://huggingface.co/gaunernst/vit_tiny_patch8_112.adaface_ms1mv3 | not stated | safetensors only (22 MB) | Interesting 2024-era edge ViT trained with AdaFace on MS1MV3; would need export |
| AdaFace ViT-base KPRPE (WebFace4M/12M) | https://huggingface.co/minchul/cvlface_adaface_vit_base_kprpe_webface12m | MIT (CVLFace collection) | safetensors only | Transformer AdaFace alternative to LVFace; export only if LVFace results disappoint |
| Glint360K R200 (`glint360k` cosface R200) | https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch (checkpoints on Google Drive / Baidu) | InsightFace non-commercial | `.pth` only, download behind Google Drive | The requested R200 is not published in ONNX anywhere; R100 (`glintr100.onnx`) covers the Glint360K CNN slot |
| ElasticFace-Cos+ (R50/R100) | https://github.com/fdbtrs/ElasticFace | none stated (research code) | `backbone.pth` only, no ONNX release anywhere | Elastic-margin loss variant; conversion = torch load + export |
| MagFace (R50/R100) | https://github.com/IrvingMeng/MagFace | research code | `.pth` on Google Drive; official repo has no ONNX (issue #32 is a DIY conversion thread) | Adds face-quality magnitude signal — useful later for gallery QC, not just identity |
| CurricularFace (R50/R100) | https://github.com/HuangYG123/CurricularFace | research code | `.pth` on Baidu/Google Drive, no ONNX | Curriculum-margin loss; older baseline, low priority vs AdaFace |
| GhostFaceNet V1/V2 (W1.3) | https://github.com/HamadYA/GhostFaceNets | Apache-2.0 code (weights: MS1MV2-derived) | `.h5`/`.tflite` only; third-party ONNX conversions exist but are unofficial | Ghost-module lightweight edge model; take only if MobileFaceNet quality is insufficient |
| ArcFace LResNet100E-IR (`arcfaceresnet100-8.onnx`) | https://huggingface.co/onnxmodelzoo/arcfaceresnet100-8 | Apache-2.0 | Not downloaded — redundant with `glintr100.onnx` (same IResNet-100 class, older MS1M-v2 training data) | Notable as the only heavy model with a permissive license; trained on Refined MS-Celeb-1M (dataset has GDPR/consent baggage) |
| IResNet-50/100 Glint360K (Sophon) | https://huggingface.co/Blaize-AI/IResnet50_Glint360K, .../IResnet100_Glint360K | other | `.bm` (Sophon TPU binary) only | Vendor-binary Glint360K checkpoints; no ONNX path |

## 3b. Measured comparison (public data, product objective)

Three views of the same question — "which embedder buys the most correct
grouping per unit of mixing" — because no single number answers it. The
objective throughout is the asymmetric one: **FM (two identities sharing a
cluster) is expensive, splitting is free**. Caches: `memb_*.npy` /
`hemb_*.npy` under `~/.hermes/cache/scratch`; scripts in `scripts/eval/`.

### A. All models at the SHIPPED threshold (0.65 / 0.73, no eject)

`model_compare.py`. `hard rec` = same-identity pairs clustered together on
CFP/CALFW/CPLFW/TinyFace; `hold FM` = mixed pairs on KinFaceW (pure
look-alike impostors); `worst imp` = most faces sitting in a cluster that
holds two identities, worst set.

| model | MB | ms/img | LFW rec | LFW FM | hard rec | hard FM | hold FM | worst imp |
|---|---|---|---|---|---|---|---|---|
| adaface_irv1 | 94.0 | 97.1 | 0.9565 | **101** | 0.3947 | **1403** | **1008** | 202 |
| arcface_r50 | 174.4 | 656.1 | 0.8437 | 2 | 0.2147 | 6 | 3 | 4 |
| lvface_t | 76.7 | 115.9 | 0.6068 | 2 | 0.2162 | 0 | 2 | 0 |
| mobilefacenet_w600k | 13.6 | 56.2 | 0.5587 | 0 | 0.1129 | 1 | 2 | 2 |
| sface_fp32 | 38.7 | 95.6 | 0.8611 | 2 | 0.1781 | 0 | 3 | 0 |
| sface_int8 | 9.9 | 244.0 | 0.7910 | 1 | 0.1588 | 2 | 2 | 4 |

A fixed absolute threshold is not equally strict across models (score
distributions differ), so this table is diagnostic, not decisive — it is
why `adaface_irv1` looks great on recall and is in fact unusable.

### B. At MATCHED purity (the fair comparison)

`model_matched.py --budget 5 --hard-budget 20`: for each model, the
threshold that maximises recall while keeping LFW false merges ≤ 5 and
hard-set false merges ≤ 20.

| model | MB | ms/img | t* | LFW rec | hard rec | LFW FM | hard FM | hold FM (KinFaceW) |
|---|---|---|---|---|---|---|---|---|
| arcface_r50 | 174.4 | 656.1 | 0.450 | **0.9904** | **0.5372** | 2 | 15 | 5 |
| lvface_t | 76.7 | 115.9 | 0.500 | 0.9812 | 0.4365 | 2 | **2** | 4 |
| mobilefacenet_w600k | 13.6 | 56.2 | 0.450 | 0.9569 | 0.3969 | 2 | 9 | 10 |
| sface_fp32 | 38.7 | 95.6 | 0.525 | 0.9688 | 0.3713 | 2 | 7 | 31 |
| sface_int8 | 9.9 | 244.0 | 0.550 | 0.9528 | 0.3077 | 2 | 6 | 10 |
| adaface_irv1 | 94.0 | 97.1 | 0.850 | 0.2172 | 0.0537 | 0 | 15 | 3 |

### C. Per-model operating point (hard_bench)

`hard_bench.py`: each model gets its own threshold chosen on LFW (best HIT
with FM ≤ 5), then is tested on the hard sets — pair-level TPR/FMR rather
than cluster-level.

| model | TPR mean | TPR min | worst FMR | worst diff-p99 |
|---|---|---|---|---|
| arcface_r50 | 0.8006 | 0.6250 | 0.00196 | 0.3088 |
| mobilefacenet_w600k | 0.5772 | 0.4850 | 0.00130 | 0.3365 |
| sface_fp32 | 0.4883 | 0.3775 | 0.00098 | 0.4123 |
| lvface_t | 0.4639 | 0.2700 | 0.00125 | 0.2536 |
| sface_int8 | 0.4566 | 0.3350 | 0.00163 | 0.4260 |
| adaface_irv1 | 0.1106 | 0.0517 | 0.00098 | 0.6831 |

### Verdict as shipped

**`lvface_t` is the default** (76.7 MB, 116 ms/img). The threshold was
recalibrated for it (`eject_bench.py --model lvface_t`): `ENTE_T_GOOD=0.64`
is the highest recall while the look-alike impurity (KinFaceW) stays
exactly where the old `w600k_mbf` line sat (0.57%), and it buys **+35%
grouping (recall 0.3121 → 0.4221)**. At the stricter 0.66 the two models
are equal (0.3121) — the gain lives just below the old floor.

What the tables say about each candidate:

- **`lvface_t` — shipped.** Best grouping per unit of mixing: 5× fewer
  look-alike mixes than `w600k_mbf` at matched purity (KinFaceW 4 vs 10
  pairs), impostor tail 0.254 vs 0.337 (diff-p99).
- **`w600k_mbf`** — the previous default (13.6 MB, 56 ms/img). Fine and
  cheap, but it mixes look-alikes more; kept in the image as the tiny
  fallback.
- **`arcface_r50` is the accuracy ceiling** (0.99 / 0.54 at matched
  purity) at 174 MB and 656 ms/img — too heavy for the edge target.
- `adaface_irv1` is rejected outright: its preprocessing is unverified and
  it mixes catastrophically (101/1403 pairs) at any permissive threshold.
- `sface_fp32` mixes look-alikes the most (31 KinFaceW pairs) despite good
  LFW numbers — rejected on the objective.
- `sface_int8` (9.9 MB) is the size floor option if 77 MB is ever too
  much; it gives up ~25% grouping.

## 4. Licensing summary (matters for a shipped product)

- **Permissive, product-safe:** SFace (Apache-2.0), LVFace-T/B (MIT), AdaFace code and
  its released checkpoints (MIT — dataset terms of MS1MV2/WebFace still apply),
  `arcfaceresnet100-8` (Apache-2.0).
- **Non-commercial only:** all InsightFace model packages (`w600k_r50`, `glintr100`,
  `w600k_mbf`, buffalo/antelope packs) — research/benchmarking is fine, shipping a
  photo-gallery product requires a commercial license from insightface.ai. Prefer
  LVFace (MIT) as the shipping model and keep the InsightFace models as benchmark
  references.

## 5. Suggested benchmark coverage (edge → heavy)

`SFace int8 (9.9 MB)` → `MobileFaceNet w600k_mbf (13.6 MB)` → `SFace fp32 (38.7 MB)` →
`LVFace-T (76.7 MB)` → `w600k_r50 (174 MB)` → `glintr100 (261 MB)` →
`LVFace-B (455 MB)`, with `adaface_ronai_model` as an experimental side entry
(requires the 160×160 alignment variant).
