# Person Re-Identification (Body) Model Zoo — Market-1501 Benchmark Candidates

Candidate body-embedding models for the photo-gallery people pipeline
(person detection → body embedding → clustering; embedding is auxiliary
grouping evidence only, never identity-anchoring). Benchmark protocol:
pairwise cosine distributions (same-person vs different-person) on
Market-1501 (`~/workspace/reid-gallery/testsets/Market-1501-v15.09.15`).

Downloaded weights live in `/home/a1c7e0/.hermes/cache/scratch/models-zoo-reid/`.
No benchmark numbers are included here — download and document only.

Priority coverage: **(1)** OSNet-AIN x0.75/x1.0 variants, **(2)** FastReID
BoT (Bag-of-Tricks, ResNet-50), **(3)** clothes-change-robust / long-span
ReID. Ready ONNX preferred; PyTorch checkpoints downloaded when ONNX does
not exist (we export ourselves).

---

## 1. Baseline (current production model)

| Name | Source URL | License | Size MB | Architecture | Input (norm) | Training set | Expected Market-1501 | ONNX-ready | Suitability |
|---|---|---|---|---|---|---|---|---|---|
| **OSNet x0.25 MSMT17 (baseline)** `osnet_x0_25_msmt17.onnx` | https://huggingface.co/kornia/osnet (byte-identical to deployed `/models/osnet_x0_25_msmt17.onnx`, sha256 `e78604f4…`); upstream torchreid: https://github.com/KaiyangZhou/deep-person-reid | MIT (torchreid); weights trained on MSMT17 (research dataset) | 0.91 | OSNet x0.25 (0.2 M params), 512-d output | 256×128 (W×H), RGB, `/255` then ImageNet mean `[0.485,0.456,0.406]`, std `[0.229,0.224,0.225]` (matches `app/ml/reid.py`) | MSMT17 (combineall) | **59.9 R1 / 31.0 mAP** cross-domain (MSMT17→Market, torchreid MODEL_ZOO). Same arch trained on Market reaches 91.2/75.0 | **Yes** (deployed) — static batch `[16,3,256,128]` → `[16,512]`, raw embedding (L2-norm applied in app) | **Edge.** The production baseline row; all candidates compared pairwise against it |

Also present in the deployed `/models` (not re-downloaded): `osnet_x1_0_msmt17.onnx`
(8.73 MB, sha256 `ac0ff0c4…`) — presumed ONNX export of OSNet x1.0 MSMT17
(non-AIN); provenance unverified.

## 2. Priority 1 — OSNet-AIN variants (torchreid pretrained)

torchreid ships **no MSMT17-trained OSNet-AIN x0.75** — only x1.0 is MSMT17-trained.
The x0.75 gap is covered by (a) the non-AIN OSNet x0.75 MSMT17 checkpoint and
(b) the multi-source OSNet-AIN x0.75 (MSMT17+Duke+CUHK03).

| Name | Source URL | License | Size MB | Architecture | Input (norm) | Training set | Expected Market-1501 | ONNX-ready | Suitability |
|---|---|---|---|---|---|---|---|---|---|
| `osnet_ain_x1_0_msmt17.pth` | https://huggingface.co/kaiyangzhou/osnet (official torchreid mirror) | MIT (deep-person-reid); MSMT17 research-only terms | 17.29 | OSNet-AIN x1.0 (2.2 M params), classifier head included (state dict, `module.` prefix) | 256×128, RGB, ImageNet mean/std on [0,1]; test distance **cosine** | MSMT17 (combineall), softmax + label-smoothing | **70.1 R1 / 43.3 mAP** cross-domain (MSMT17→Market, torchreid MODEL_ZOO: 70.1 (43.3)) | Needs export (torchreid `osnet_ain_x1_0` + `torch.onnx.export`, cf. deep-person-reid issue #302) | **Edge.** The headline AIN candidate; ~2× baseline size, +10 R1 cross-domain over baseline |
| `osnet_ain_x0_75_multisource_MS_D_C.pt` | https://huggingface.co/LibreYOLO/LibreReID-osnet (`osnet_ain_x0_75.pt`; sha256 matches published `57b31d7f…`) | MIT (LibreYOLO conversion; upstream torchreid MIT) | 5.38 | OSNet-AIN x0.75 (1.3 M params), classifier stripped — plain state dict, strict-loads into torchreid `osnet_ain_x0_75` | 256×128, RGB, ImageNet mean/std | MSMT17 + DukeMTMC-reID + CUHK03 (torchreid multi-source, TPAMI'21) — **Market-1501 not in training, clean for our benchmark** | Not published per-width; same recipe at x1.0 reaches **73.3 R1 / 45.8 mAP** (MS+D+C→M) | Needs export | **Edge.** Best size/quality trade-off candidate (~6 MB class as requested) |
| `osnet_ain_x1_0_multisource_MS_D_C.pt` | https://huggingface.co/LibreYOLO/LibreReID-osnet (`osnet_ain_x1_0.pt`; sha256 matches published `34c24e98…`) | MIT (LibreYOLO; upstream torchreid MIT) | 8.93 | OSNet-AIN x1.0, classifier stripped state dict | 256×128, RGB, ImageNet mean/std | MSMT17 + Duke + CUHK03 (multi-source) | **73.3 R1 / 45.8 mAP** (MS+D+C→M, torchreid MODEL_ZOO) | Needs export | **Edge.** Highest published Market cross-domain number in this zoo; strongest AIN candidate |
| `osnet_ain_x0_75_imagenet.pyth` | https://huggingface.co/kaiyangzhou/osnet | MIT (deep-person-reid) | 7.46 | OSNet-AIN x0.75 + classifier | 256×128, RGB, ImageNet mean/std | ImageNet only (initialization; **no ReID training**) | n/a (not a ReID model yet) | Needs export + MSMT17 fine-tune | **Neither** until fine-tuned; kept as the only official AIN x0.75 weights |
| `osnet_x0_75_msmt17_combineall.pth` | https://huggingface.co/kaiyangzhou/osnet | MIT (deep-person-reid); MSMT17 research terms | 13.72 | OSNet x0.75 (non-AIN, 1.3 M params) + classifier | 256×128, RGB, ImageNet mean/std | MSMT17 (combineall) | **63.6 R1 / 35.5 mAP** cross-domain (MSMT17→Market) | Needs export | **Edge.** x0.75 MSMT17-trained fallback when AIN x0.75 is unavailable |
| `osnet_ain_d_m_c.pth.tar` | https://huggingface.co/Sycarol/osnet-ain-dmc | No license stated (community upload; upstream torchreid MIT) — treat as research-only | 53.55 | OSNet-AIN x1.0 training checkpoint (1535 zip entries; likely optimizer state included) | 256×128, RGB, ImageNet mean/std | Duke + **Market-1501** + CUHK03 (torchreid multi-source "D+M+C" recipe) | D+M+C→MS reported 40.2 R1/16.2 mAP (x1.0); **no valid Market number — Market is in its training set** | Needs export | **Server/aux only.** Market-contaminated for our benchmark; usable as fine-tune warm-start, not as a Market row |

## 3. Priority 2 — BoT (Bag-of-Tricks, ResNet-50) from the FastReID model zoo

FastReID checkpoints are full training states (`model` + `optimizer` +
`scheduler` + `iteration`; verified in file). Deployable `model` state
(ResNet-50 + BNNeck, 512-d) ≈ 95 MB after stripping. Export with
`tools/deploy/onnx_exporter.py` + matching YAML config.

| Name | Source URL | License | Size MB | Architecture | Input (norm) | Training set | Expected Market-1501 | ONNX-ready | Suitability |
|---|---|---|---|---|---|---|---|---|---|
| `fastreid_market_bot_R50.pth` | https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/market_bot_R50.pth (config: `configs/Market1501/bagtricks_R50.yml`) | Apache-2.0 (fast-reid) | 300.96 file / **~95 deployable** | BoT: ResNet-50 + GeM-less GAP + BNNeck + softmax w/ label-smoothing & BNNeck last stride 1 | 256×128 (SIZE_TEST), RGB, ImageNet mean/std (PIXEL_MEAN=`[0.485,0.456,0.406]*255`, PIXEL_STD=`[0.229,0.224,0.225]*255`, `fastreid/config/defaults.py`) | Market-1501 | **94.4 R1 / 86.1 mAP / 59.4 mINP** (FastReID MODEL_ZOO, same-domain) | Needs export (FastReID onnx_exporter) | **Server.** Accuracy reference / server-side ceiling; too heavy for edge |
| `fastreid_msmt_bot_R50.pth` | https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/msmt_bot_R50.pth (config: `configs/MSMT17/bagtricks_R50.yml`) | Apache-2.0 (fast-reid) | 308.08 file / **~95 deployable** | Same BoT R50 | Same as above | MSMT17 | MSMT17 same-domain 74.1/50.2; **Market cross-domain not published** — to be measured in our pairwise benchmark | Needs export | **Server.** MSMT17-trained BoT = the fair "same training set, bigger model" comparison to the OSNet-AIN rows |

Reference (not downloaded — same release v0.1.1, same export path): BoT
`R50-ibn` (94.9/87.6 Market), `S50`, `AGW`, `SBS` families in
https://github.com/JDAI-CV/fast-reid/blob/master/MODEL_ZOO.md.

## 4. Priority 3 — clothes-change-robust / long-span ReID

No public repository or weights could be found for a method named **"Hacil"**
(GitHub + web search, 2026-10) — if it exists, it is not publicly released.
Substituted: the **ReFace/GEFF checkpoint bundle** (https://github.com/bar371/GEFF),
which releases three published clothes-changing ReID methods (CAL, CTL, AIM)
trained by the ReFace authors on the standard clothes-changing benchmarks
(LTCC, PRCC, LaST, CCVID). All are ResNet-50-class checkpoints (~95 MB),
within the requested ≤120 MB size budget. Repo has **no LICENSE file** —
research use only, verify before any product deployment.

| Name | Source URL | License | Size MB | Architecture | Input (norm) | Training set | Expected Market-1501 | ONNX-ready | Suitability |
|---|---|---|---|---|---|---|---|---|---|
| `geff_ccreid_weights/AIM/ltcc-checkpoint.pth.tar` | https://drive.google.com/drive/folders/1qm1D38WzH2Rqv8NKteulTB3bU4W3nBFh (GEFF/ReFace release folder; via `gdown --folder`) | No LICENSE in GEFF repo (research-only) | 94.42 | AIM (Yang et al. 2023) — ResNet-50 + BNNeck (`conv1/bn1/layer*/bn` keys verified in `model_state_dict`) | 256×128, RGB, ImageNet mean/std (ResNet-ReID convention) | LTCC (long-term clothes-changing) | **Not published** — CC-ReID methods report LTCC/PRCC, not Market-1501; must be measured with our pairwise protocol | Needs export | **Edge→server.** Clothes-change robustness for long photo-collection spans; main priority-3 candidate |
| `geff_ccreid_weights/AIM/prcc-checkpoint.pth.tar` | same GEFF folder | research-only | 94.42 | AIM, ResNet-50 + BNNeck | same | PRCC (clothes-changing protocol) | Not published (PRCC: ReFace system reaches 81.9 top-1 / 58.8 mAP clothes-changing) | Needs export | **Edge→server.** PRCC-trained sibling; bench both and keep the better on our crops |
| `geff_ccreid_weights/CAL/ltcc-checkpoint.pth.tar` | same GEFF folder | research-only | 99.83 | CAL (Gu et al., CVPR 2022 — "Clothes-Changing Person Re-identification with RGB Modality Only"; RGB-only body-shape features) | 256×128, RGB, ImageNet mean/std | LTCC | Not published for Market | Needs export | **Edge→server.** RGB-only, no multi-modality dependency — closest fit for back-view/clothes-change gallery use |
| `geff_ccreid_weights/CAL/prcc-checkpoint.pth.tar` | same GEFF folder | research-only | 101.80 | CAL | same | PRCC | Not published for Market | Needs export | **Edge→server.** Same, PRCC-trained |
| `geff_ccreid_weights/CAL/last-checkpoint.pth.tar` | same GEFF folder | research-only | 439.03 | CAL (full training state) | same | LaST (long-term) | Not published for Market | Needs export (strip optimizer first) | **Server only.** Long-span dataset; file includes training state, model ~100 MB |
| `geff_ccreid_weights/CAL/vc-clothes-checkpoint.pth.tar` | same GEFF folder | research-only | 108.76 | CAL | same | VC-Clothes | Not published for Market (ReFace reports 94.9 top-1 on VC-Clothes clothes-changing) | Needs export | **Edge→server.** Smallest-domain CC checkpoint |

Bundle also contains `geff_ccreid_weights/CTL/dukemtmcreid_resnet50_256_128_epoch_120.ckpt`
(305.59 MB, CTL method, Duke-trained) and `CTL/resnet50-19c8e357.pth`
(102.50 MB, ImageNet backbone) — kept for completeness, not benchmark
candidates (Duke-trained CTL has no clothes-change supervision).

## 5. Ready-ONNX edge variants (extra, already export-free)

| Name | Source URL | License | Size MB | Architecture | Input (norm) | Training set | Expected Market-1501 | ONNX-ready | Suitability |
|---|---|---|---|---|---|---|---|---|---|
| Qualcomm AI Hub OSNet float `qaihub_osnet/osnet-onnx-float/osnet.onnx` (+`osnet.data`) | https://huggingface.co/qualcomm/OSNet (release_assets.json v0.63.0) | MIT (HF card); ai-hub-models code BSD-3-Clause | 9.93 (1.33 graph + 8.60 external data) | OSNet x1.0 (`osnet_x1_0_market_…softmax_labelsmooth_flip` torchreid checkpoint), **L2-normalization baked into graph** | `[1,3,256,128]` f32, RGB, ImageNet mean/std (`normalize_image_torchvision`, qai-hub-models osnet model.py) → `embeddings [1,512]` | **Market-1501** | **94.2 R1 / 82.6 mAP** same-domain (torchreid osnet_x1_0 Market-trained) | **Yes** — smoke-tested in onnxruntime (random input, ‖e‖=1.0) | **Edge.** The best ready-made ONNX: drop-in, normalized output; TFLite/QNN-DLC builds also published |
| Qualcomm AI Hub OSNet w8a8 `qaihub_osnet/osnet-onnx-w8a8/osnet.onnx` (+`osnet.data`) | same | MIT / BSD-3-Clause | 9.11 (0.46 + 8.65) | Same, static w8a8 quantization | `[1,3,256,128]` **uint8** → `embeddings [1,512]` **uint8** (quantized I/O) | Market-1501 | ~94 R1 expected (quantized; measure accuracy drop in benchmark) | **Yes** — smoke-tested (uint8 in/out OK) | **Edge.** Quantized for mobile DSP/NPU; re-measure pairwise distributions — quantization shifts cosine margins |
| `osnet_x0_25_msmt17_kornia.onnx` | https://huggingface.co/kornia/osnet | Apache-2.0 (kornia) / MIT upstream | 0.91 | OSNet x0.25 | `[16,3,256,128]` f32 (static batch 16), RGB ImageNet mean/std → `[16,512]` raw | MSMT17 (combineall) | 59.9 R1 / 31.0 mAP cross-domain | **Yes** — smoke-tested; **byte-identical to our baseline** (sha256 `e78604f4…`) | Reference copy of the baseline; use to reproduce baseline rows from clean cache |

## 6. Preprocessing conventions (verified sources)

| Family | Color | Formula | Source |
|---|---|---|---|
| torchreid OSNet/OSNet-AIN | RGB | `/255` then `(x−μ)/σ`, μ=[0.485,0.456,0.406], σ=[0.229,0.224,0.225] | torchreid test transforms; matches our `app/ml/reid.py` |
| FastReID BoT | RGB | identical ImageNet normalization; PIXEL_MEAN/PIXEL_STD are the same stats ×255 applied consistently in their pipeline | `fastreid/config/defaults.py` |
| Qualcomm AI Hub OSNet | RGB | `normalize_image_torchvision` = ImageNet mean/std on [0,1] | qai-hub-models `models/osnet/model.py` |
| GEFF CAL/AIM/CTL | RGB | ImageNet mean/std (ResNet-ReID convention; confirm in GEFF eval script before benching) | convention — unverified |

Test-time distance: use **cosine** for OSNet-AIN MSMT17/multi-source recipes
(their published eval uses cosine) and for BoT BNNeck features; our pipeline
already L2-normalizes and compares with cosine.

## 7. Export notes (when ONNX is required)

- **torchreid OSNet/OSNet-AIN**: `torchreid.models.build_model('osnet_ain_x1_0', num_classes=1000)` +
  `torchreid.utils.load_pretrained_weights(...)` (strip `module.` prefix happens
  automatically) → `torch.onnx.export(model, torch.randn(1,3,256,128), ...)`.
  Export drops the classifier head for feature-only inference (cf.
  deep-person-reid issue #302). Expect ~10.6 MB ONNX for x1.0, ~5.5 MB for x0.75.
- **FastReID**: `tools/deploy/onnx_exporter.py --config-file <yml> --opts MODEL.WEIGHTS <pth>`,
  input `1x3x256x128`. Strip optimizer state first if file size matters
  (`torch.load(...)['model']`).
- **GEFF CAL/AIM/CTL**: load `model_state_dict` in the GEFF repo's model class
  (ResNet-50 + BNNeck), export feature head only.
- After export: keep our pairwise-cosine benchmark protocol; never mix
  embedding spaces across models (re-embed per model).

## 8. Caveats

- **MSMT17/Market/Duke/CUHK03 are research datasets** with research-oriented
  terms; torchreid/FastReID code licenses (MIT/Apache-2.0) do not relicense the
  datasets. Review before production deployment (LibreYOLO's README flags the
  same point).
- **Market-in-training contamination**: `osnet_ain_d_m_c.pth.tar` (D+M+C) and
  Qualcomm `osnet_x1_0_market` are trained on Market-1501 — do not report their
  Market numbers as cross-domain generalization. Qualcomm's is still valid as a
  "same-domain oracle" upper bound.
- **"Hacil" not found** — no public repo/weights located; the priority-3 slot is
  filled by CAL/AIM (RGB-only, ResNet-50, ≤120 MB) from the GEFF bundle. If the
  intended model is a specific paper, provide a link and we can add it.
- FastReID `.pth` files are 300+ MB because they carry optimizer + scheduler
  state; the deployable network is ~95 MB (ResNet-50 + BNNeck).
- SHA-256 (first 16 hex) of downloaded files: `osnet_ain_x1_0_msmt17.pth`
  8a07e8da38946f7c · `osnet_ain_x0_75_imagenet.pyth` 8fcd27a65426f8f9 ·
  `osnet_x0_75_msmt17_combineall.pth` 9dcf5893a39e8a0d ·
  `osnet_ain_x0_75_multisource_MS_D_C.pt` 57b31d7f806edac5 ·
  `osnet_ain_x1_0_multisource_MS_D_C.pt` 34c24e98b6b70c8b ·
  `osnet_ain_d_m_c.pth.tar` 9e610511b54d2f4f · `osnet_x0_25_msmt17_kornia.onnx`
  e78604f4ccda49b8 · `fastreid_market_bot_R50.pth` 764fa8ca18117e2f ·
  `fastreid_msmt_bot_R50.pth` 835eddc0490ad190.

---

## 2. Benchmark results (pairwise on Market-1501, 600 imgs / 179.7k pairs, seed 42)

Protocol: scripts/eval/reid_bench.py, cross-domain rows only (Market-trained
models flagged in-domain and excluded from ranking). Raw: scripts/eval/out/reid_bench.jsonl.

| model | size MB | same_p50 | diff_p50 | diff_max | FMR@0.65 | TPR@0.65 | ms/img | verdict |
|---|---|---|---|---|---|---|---|---|
| osnet_x0_25_msmt17 (baseline) | 0.91 | 0.754 | 0.449 | 0.893 | 0.0399 | 0.828 | 53 | production baseline |
| **osnet_ain_x1_0_msmt17** | 8.74 | 0.765 | 0.453 | 0.882 | **0.0356** | **0.855** | 659 | **DEFAULT body model** (wins every shared threshold) |
| osnet_ain_x0.75/x1.0 multisource MS_D_C | 5-9 | - | - | - | - | - | - | in-domain (Market in training), cosine scale compressed; not comparable |
| fastreid_msmt_bot_R50 | 94 | 0.508 | 0.145 | 0.667 | ~0 | 0.220 | 3200 | thinnest impostor tail but low cosine scale - needs its own threshold sweep; server-side candidate |
| geff_cal_prcc (clothes-change) | 94 | 0.549 | 0.320 | 0.790 | 0.0033 | 0.227 | 800 | clothes-change robustness not measured by this protocol; native input 384x192 |

Notes: thresholds never transfer between embedding spaces (BoT/CAL rows at
0.55/0.65/0.74 are understated); every model needs its own calibrated gate.
Exports pass torch-vs-onnxruntime parity (<=1.4e-5); BoT validated against
fast-reid v0.1.1 Baseline (2.9e-6). Export tooling: scripts/export_reid_zoo.py.
