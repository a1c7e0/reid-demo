# Hard-Sample Public Benchmarks

Downloaded hard-regime face benchmarks under `testsets/hard/` (untracked;
re-downloadable from the sources below). Purpose: separate model quality on
the regimes where LFW is blind — same-look-alike impostors (kin faces),
extreme pose, cross-age, cross-pose, and native low resolution.

| Set | Regime | Size | Source | Layout |
|---|---|---|---|---|
| KinFaceW-I / II | **look-alike impostors** (parent-child pairs: F-D, F-S, M-D, M-S) | 533 / 1002 pairs | kinfacew.com (direct zips) | `kinfacew/KinFaceW-{I,II}/images`, pairs in `meta_data/*_pairs.mat` |
| CFP | **frontal-profile pose** (500 celebrities) | 7000 imgs; 10 folds x 350+350 pairs (F and P) | cfpw.io/cfp-dataset.zip | `cfp/cfp-dataset/{Data,Protocol}` — `Protocol/Pair_list_{F,P}.txt`, `Protocol/Split` |
| CALFW | **cross-age** (3000 positive pairs with age gaps) | ~12k imgs | whdeng.cn/CALFW (Google Drive) | `calfw/calfw/{aligned images, pairs_CALFW.txt}` |
| CPLFW | **cross-pose** | ~12k imgs | whdeng.cn/CPLFW (Google Drive) | `cplfw/cplfw/{images, aligned images}` |
| TinyFace | **native low-res** (avg 20x16 px, 5139 ids) | 169k imgs | qmul-tinyface.github.io (Google Drive) | `tinyface/tinyface/{Testing_Set,Training_Set}` — open-set gallery protocol (`Gallery_Match`, `Gallery_Distractor`) |
| AgeDB-30 | cross-age (curated) | — | not downloaded (Kaggle-auth mirror only; CALFW covers the regime) | — |

## How each is used

- **KinFaceW** = the look-alike collision stress test (the "similar boys"
  failure mode): parent-child pairs are genuine different-identity look-alikes.
  Report impostor pair score distributions (same as a false-merge probe).
- **CFP-FP / CFP-FF** = pose robustness; run the standard 10-fold pair lists.
- **CALFW / CPLFW** = the same 3000-pair protocol style as LFW, harder
  variation axes; direct drop-in for the pairwise scorer.
- **TinyFace** = low-quality ceiling (blur/occlusion/low-res), where our
  quality gates (blur/score/sideways) decide what even gets clustered.
  The open-set gallery protocol also exercises the "unassigned is allowed"
  policy.

All sets are evaluation-only. No tuning on them without recording the split.
