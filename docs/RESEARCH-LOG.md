# Research Log

The running record of what was tried, measured and concluded. Numbers here
come from the benches in `scripts/eval/`; where a claim is not backed by a
run, it says so. Everything is evaluated on **public data only**.

Two rules govern all of it:

1. **No self-invented algorithms.** The clustering core is Ente's, ported.
   Where we go beyond Ente it is a published, citable method.
2. **The objective is asymmetric.** Mixing two people in one cluster is
   heavily penalised (FM); splitting one person across clusters is cheap
   (FS); unassigned is neutral (UN). Tuning sets decide thresholds,
   evaluation sets report them, and the user's own photos are never tuning
   material.

---

## 1. Clustering core — Ente's greedy single-linkage survives the bake-off

**Question.** With the green light to try entirely different algorithms, not
just parameters, is Ente's greedy single-linkage actually the right core?

**Run.** `scripts/eval/algo_bench.py`, 79 configurations: Ente greedy,
complete / average / single linkage, Chinese Whispers, mutual-kNN. LFW for
tuning, the hard sets for validation, KinFaceW as a look-alike blind
stressor. Matched-purity comparison (recall at the same mixing rate), because
raw recall is meaningless when one config is simply allowed to mix more.

**Result.** The shipped Ente max-rule greedy is optimal at matched purity:
recall 0.3121 vs 0.1092 (median), 0.1259 (mean), 0.2519 (knee). Complete and
mutual-kNN are worse; single linkage and Chinese Whispers tie.

**Also found:** turning the auto-eject bar off doubles recall at the same
purity (0.1667 → 0.3121). Shipped: eject off (`PERSON_EJECT_BAR=0`).

**Conclusion.** Ente's core stays. The parameters were recalibrated for our
models, the algorithm was not replaced.

---

## 2. Thresholds recalibrated per face model

**Question.** Ente's thresholds were fitted to their model; ours is different.

**Run.** `scripts/eval/eject_bench.py`, `calib_bench.py`, `model_matched.py`.

| Model | Size | Look-alike merges | Hard recall | Verdict |
|---|---|---|---|---|
| **LVFace-T** | 76.7 MB | **4** | 0.437 | **shipped** |
| w600k_mbf | 13.6 MB | 10 | 0.397 | fast fallback |
| arcface_r50 | 174.4 MB | — | 0.54 | accuracy ceiling, too heavy for edge |
| glintr100 / int8 | ~105 MB | — | — | over budget |
| adaface_irv1 | — | catastrophic | — | rejected |
| sface_fp32 | 38.7 MB | 31 (worst) | — | rejected |

**Result.** Thresholds converge per model and then freeze: **`ENTE_T_GOOD=0.64`
/ `ENTE_T_BAD=0.72`** for LVFace-T (Ente's own lines are 0.76/0.84 — theirs are
higher because their model separates differently).

**Also found.** OpenCV's `FaceRecognizerSF` / dnn silently outputs garbage for
larger ONNX face models (w600k_r50: different people collapse to ~0.9
cosine). Embeddings must go through onnxruntime with `(x-127.5)/128`
normalisation; cv2 is used only for `alignCrop`. This is a trap worth knowing.

---

## 3. Fragmentation and safe fragment recovery

**Question (from the user).** Faces that look identical to the eye — often
blurry ones — land in different clusters. Nothing is mixed, but one person is
scattered over several groups. A small amount of mixing is acceptable *if* the
breakup button can split it again in one click.

**Run.** `scripts/eval/fragment_regen.py` on LFW (2,339 faces, 311 identities,
shipped LVFace-T through the production `FaceEngine` path).

| t_good | merge bar | mixed | split | recall | clusters |
|---|---|---|---|---|---|
| 0.64 (as shipped) | none | 15 | 3091 | 0.600 | 805 |
| **0.64** | **0.53** | **28** | **1670** | **0.784** | **527** |
| 0.58 | 0.53 | 32 | 1298 | 0.832 | 479 |
| 0.52 | 0.53 | 80 | 749 | 0.903 | 407 |

**Result.** Lowering the join bar merges more but starts mixing people
(mixed pairs ×5 at 0.52). The better lever is a **complete-link merge pass**:
two clusters fold together *only when every cross pair clears the bar*, which
cannot mix anyone at that bar. On the shipped threshold that is split pairs
−46% and recall +31% for 15 → 28 mixed pairs out of 2.7M.

The bar barely matters between 0.50 and 0.60 (the cross-pair distribution has
a gap there), so it is tied to `PERSON_SPLIT_SIM` — **what the breakup button
would keep apart is exactly what this folds back together**.

On the demo gallery: 132 persons → 107, stable at 107 on a repeat re-cluster,
zero co-occurrence violations.

---

## 4. Body channel (back views)

**Question.** Does body evidence help, and where does it hurt?

**Run.** `scripts/eval/fusion_bench.py` on Market-1501 in production shape
(300 identities × 6 images, 3 face-anchor images, virtual group photos,
identities deliberately given **overlapping capture times** — an earlier
version separated them by 9 hours and the time window won for free, which was
cheating).

**Result.**

- **The ambiguity margin is the key lever.** `BODY_MARGIN=0.06` cuts
  wrong-attach 8.5% → 0.9% and dirty persons 4.37% → 1.04% at coverage 0.401.
- Best single point: join 0.74 + max rule + margin 0.06 + 3 anchors →
  coverage 0.649, wrong 1.35%, dirty 2.67% (as shipped it was
  0.776 / 8.5% / 14.0%).
- The capture-time window is the biggest structural lever (dirty 24% → 7.7%).
- Body models: **OSNet-AIN x1.0 MSMT17** wins at matched purity
  (`docs/REID-MODEL-ZOO.md`).

**Design consequence.** Bodies only ever *attach* to an existing person — they
never create one. That is the rule that keeps a back view from inventing a new
person every time someone walks away from the camera.

---

## 5. Review ordering — Ente does not solve this

**Question (from the user).** Review is slow and unintelligent: after rejecting
`b`, the next item is `c`, and `c` is really the same person as `b` (a split
person's two fragments). `d` should come first.

**First, what Ente actually does** (vendored sources, `vendor/ente/`):
`people.ts:336` sorts suggestion clusters by median cosine similarity
descending and nothing else. **There is no dedup or diversity pass anywhere in
the review path** — two near-duplicate suggestions are both shown. Review is
per-person cluster-to-person assignment with per-person rejection memory, a
0.48 → 0.46 threshold cascade before singletons are offered, and an 80-item
cap. Scores come from random samples of ≤50 faces, so the order is mildly
stochastic above that size. **The problem is Ente's own design**, not a
porting artefact.

**Then what the literature offers** (citations verified against originals or
Crossref):

- **MMR** — Carbonell & Goldstein, SIGIR 1998:
  `λ·relevance − (1−λ)·max_sim_to_already_shown`. The paper's own guidance is
  λ≈0.3 to survey the space, λ≈0.7 to refocus.
- **Expected error reduction** — Roy & McCallum, ICML 2001.
- **Margin sampling** — Joshi, Porikli & Papanikolopoulos, CVPR 2009. No
  redundancy control at all, i.e. exactly the failure mode reported.
- **Explore/Consolidate** — Basu, Banerjee & Mooney, SDM 2004,
  *Active Semi-Supervision for Pairwise Constrained Clustering*. (Note: the
  title "Active Semi-Supervised Clustering" **does not exist**.)
- **QueryPivot** — Saha & Subramanian, ESA 2019: recover the optimal
  correlation clustering with ≤ 2·C_OPT same-cluster queries.

**Run.** `scripts/eval/review_order_bench.py`, LFW 700 faces / 117 identities,
ground-truth simulated human answering "same person?".

| ordering | duplicate questions | fix 50% | fix 25% | fix 10% |
|---|---|---|---|---|
| relevance only (= Ente) | 44.0% | 71 | 33 | 11 |
| expected error reduction | 44.0% | 65 | 29 | 10 |
| farthest-first (Gonzalez 1985) | 44.4% | 78 | 43 | 17 |
| MMR 0.5 | 38.8% | 78 | 36 | 13 |
| **MMR 0.3 + impact weighting** | **39.6%** | 73 | 30 | 10 |

**Two hard conclusions.**

1. **The total question count is identical (134) across every ordering.**
   Ordering changes the *experience* and the speed of early progress, never
   the total work. Anyone promising otherwise is selling something.
2. **Pure diversity is the worst on both counts.** Farthest-first loses to
   relevance-only everywhere. The correct form is a trade-off.

**Shipped:** MMR at λ=0.3 with the impact term, i.e. published MMR with a
re-parameterised relevance (`sim × log1p(|A|·|B|)`, the cheap form of expected
error reduction). Duplicate questions 44.0% → 39.6% at the same speed.

Alongside it, two older mechanisms that also address the complaint and stay
in: a **rejection memory** (rejecting stores what was rejected; a later
candidate that reads as the same person is skipped — the suppression compares
against the pair score, not the face median, otherwise it hides nearly
everything), and a **diversity demotion** of near-duplicate candidates.

**Engineering note.** The naive MMR re-rank made `/api/review` hang: O(k²)
over tens of thousands of person pairs, plus `max(dot(x,y) for …)` generators
running ~1.3M `np.dot` calls per request (found with `cProfile`). Re-ranking
the top 200 and vectorising the inner loops took it from 5.5 s to 0.33 s. This
is the kind of thing that looks fine in a benchmark and is unusable in a UI.

---

## 6. Quality layer

The blur / score / sideways gates are kept. `scripts/eval/quality_bench.py`
isolates them using real LFW scores and 5-landmark metadata. They decide what
is even clusterable, which matters most exactly where the model is weakest —
low-resolution, occluded and turned-away faces (TinyFace is the ceiling set
for this).

---

## 7. Body attach parameters

`docs/ALGO-TUNING.md` §7 carries the full table. Summary of the shipped
point: join 0.74, max rule, `BODY_MARGIN=0.06`, 3 face anchors,
`BODY_MAX_GAP_S=3h`, co-occurrence invariant on.

---

## Open questions

- **How much mixing is acceptable?** The user has said a small amount is fine
  if the breakup button handles it. The current point (28 mixed pairs on LFW
  at recall 0.784) is the conservative end; `ENTE_T_GOOD=0.58` gives recall
  0.832 for 32 mixed pairs. It is a dial, not a settled answer.
- **Review ordering beyond MMR.** Basu et al.'s Explore/Consolidate consumes
  answers transitively and would cut duplicates further; not implemented
  because it asks many obviously-different pairs, which spends the user's
  attention outside the uncertain band. Worth a bench if review ever becomes
  the bottleneck again.
- **Model-level noise on look-alike children.** LVFace-T halves it versus
  w600k_mbf but cannot remove it. The mitigations are the review queue, the
  mixed-grouping breakup, and the body margin gate.

---

## Reproducing

| Claim | Script | Data |
|---|---|---|
| Clustering bake-off | `scripts/eval/algo_bench.py` | LFW + hard sets |
| Threshold calibration | `scripts/eval/eject_bench.py`, `calib_bench.py` | LFW |
| Model comparison | `scripts/eval/model_matched.py`, `model_compare.py` | LFW + hard sets |
| Fragmentation / recovery | `scripts/eval/fragment_regen.py` | LFW |
| Body attach A/B | `scripts/eval/fusion_bench.py` | Market-1501 |
| Review ordering | `scripts/eval/review_order_bench.py` | LFW |
| Quality layer | `scripts/eval/quality_bench.py` | LFW |

Download links for every dataset: [`DATA-SOURCES.md`](DATA-SOURCES.md).
