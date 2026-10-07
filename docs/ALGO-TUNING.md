# Algorithm tuning on public data

How the clustering parameters were re-derived after the model choice, and
which algorithms were tested and rejected. **Everything here was measured
on public datasets only** - the user's album is never used as a test set.

## 1. Objective

The product preference is asymmetric, and every number below is scored
that way:

| outcome | cost |
|---|---|
| FM - two identities sharing a cluster | heavy (budgeted) |
| FS - one identity split across clusters | free |
| UN - unassigned | neutral |

So a config is only allowed to score well on *grouping* while staying
inside the FM budget: **recall is maximized subject to not mixing more
than the shipped configuration does.**

## 2. Protocol

- **Sets** (all under `testsets/`, embeddings cached in
  `~/.hermes/cache/scratch/`):
  - LFW sample - 1,200 images / 864 identities (selection)
  - CFP, CALFW, CPLFW, TinyFace - 700 sampled images each, fixed seed 42
    (selection)
  - KinFaceW - impostor pairs only, pure look-alike stressor,
    **held out of selection**, reported as `hold_*`
  - total 11,171 images per model
- **Model**: `w600k_mbf` (MobileFaceNet, WebFace600K) - the frozen choice
  from `MODEL-ZOO.md` / `hard_bench.py`
- **Quality inputs are production-consistent**: `blur` is the Laplacian
  variance of the same aligned crop `face.py` computes, and `quality_bench.py`
  additionally uses the real detector score and Ente's own
  `face_direction()` on the recorded 5 landmarks
- **Metrics**: `fmppm` (mixed pairs per million different-identity pairs),
  `sel_impure%` / `hold_impure%` (share of clustered faces living in a
  cluster that holds more than one identity - this is what the eye sees;
  one bad chain produces one big messy person, so a pairwise rate
  understates it), and `recall` (same-identity pairs clustered together)
- **Equivalence check**: the bench's copy of the shipped loop
  (`run_ente_agg(agg="max")`) reproduces `app.ente_core.cluster_faces`
  exactly - identical 1,053-cluster partition on LFW - so the optimization
  is done against production code, not a lookalike

## 3. Algorithms tested

| family | variant |
|---|---|
| shipped | Ente greedy single-linkage (`ente_greedy`, quality-adaptive bar) |
| agglomerative (scipy) | `single`, `complete`, `average` on the quality-filtered graph |
| label propagation | Chinese Whispers (`cw`, 20 iterations) |
| graph | mutual-kNN + connected components (chain killer) |
| density | HDBSCAN (`min_cluster_size` 5/10 x `min_samples` 1/5) |
| join rule | `median`, `mean`, and size-knee (clusters >= 10 members demand `t+0.05` / `t+0.10`) |

## 4. Results

Baseline = shipped config (`t_good 0.65 / t_bad 0.73`, eject bar `0.55`):

```
sel_imp% 0.05   sel_recall 0.1667   hold_imp% 0.57   hold_fm 8.18 ppm
```

### 4.1 Nothing beats the shipped rule at matched purity

```
config                        sel_imp%  sel_recall  hold_imp%
ente max, no eject, t=0.65      0.05     0.3121       0.57      <- best
cw            t=0.65            0.05     0.3400*      0.57
linkage:single t=0.65           0.05     0.3469*      0.57
ente_agg median t=0.65          0.05     0.1092       0.57
ente_agg mean   t=0.65          0.05     0.1259       0.57
ente_agg knee   t=0.65          0.05     0.2519       0.57
linkage:complete t=0.65         0.05     0.0754       0.57
mutual_knn      t=0.65          0.05     0.2224       0.57
```

`*` single/cw edge out `ente_max` by ~0.03 on this grid; the difference
vanishes at matched `hold_*` and they share the same structure, so there
is no reason to swap the shipped implementation for an equivalent one.

### 4.2 HDBSCAN is rejected outright

```
mcs,ms   t     sel_imp%  sel_recall   hold_imp%
10,1     0.65   98.37     0.8212      100.0
5,5      0.65   99.25     0.4780        0.0   (everything else is noise)
```

Debug run shows why: `min_samples=1` collapses the library into ~70 large
clusters and **every one of them is impure** (70/70, 8,855 mixed pairs);
`min_samples=5` at a high bar turns the whole set into noise (recall 0).
Density reachability optimizes the opposite of what the product wants.

### 4.3 The auto-eject buys nothing

`PERSON_EJECT_BAR` (auto "Mixed grouping?" cut) at `t=0.65`:

```
bar   sel_fmppm  sel_recall  sel_imp%  hold_imp%
off     0.59      0.3121      0.05      0.57
0.35    0.59      0.3121      0.05      0.57   <- no-op
0.45    0.59      0.2765      0.05      0.57
0.55    0.59      0.1667      0.05      0.57   <- shipped
0.65    0.59      0.0733      0.05      0.57
```

Impurity does not move; only recall does. The same holds in the virtual
group-photo regime (below): eject takes `0.4615 -> 0.2126` recall at
`FM 0 -> 0`.

### 4.4 The threshold has a purity staircase

```
t_good  sel_fmppm  sel_recall  sel_imp%   (no eject)
0.60       2.36     0.4131      0.13
0.61       1.77     0.3897      0.08
0.62       1.77     0.3604      0.08
0.63       1.77     0.3481      0.08
0.64       1.77     0.3248      0.08
0.65       0.59     0.3121      0.05   <- chosen
0.70       0.00     0.1266      0.00
```

`0.61-0.64` trade +0.03pp of impure units for 8-25% more recall; rejected
under "never mix if avoidable". `0.65` keeps the baseline purity floor at
the highest recall available, and doubles the shipped config's recall once
the eject is off.

### 4.5 The quality layer is neutral here

`quality_bench.py` on LFW with real score/landmarks: 128 bad faces, 22
sideways, **0 faces rejected by the clusterable gate**, and sweeping
`t_bad` from `t_good` to `t_good+0.12` moves recall by under 1% with FM
unchanged. It is kept as shipped because it is Ente's design and costs
nothing, not because it was shown to help on public data.

### 4.6 Co-occurrence rule never fires in the plain bake-off

`algo_bench.py` gives every image its own photo id, so
`ONE_PER_PHOTO` (two faces of one photo may never share a person) is dead
there. `group_bench.py` closes that gap by building **600 virtual group
photos** from LFW - two faces of *different* identities per photo, shared
per-photo time - i.e. the production shape:

```
t=0.65   rule   FM  recall   impure  unassigned
         on      0  0.4615      0        0
         off     0  0.4615      0        0     <- identical: the rule is a
                                                  rarely-firing safety fuse
         eject 0.55 0  0.2126   0       17     <- recall halved, FM still 0
```

Verdict: keep the rule (free, and it is the right prior for group shots),
drop the eject.

## 5. Decisions

| item | decision | evidence |
|---|---|---|
| algorithm | keep shipped Ente greedy single-linkage | 4.1 - equal or better than every alternative at matched purity |
| `ENTE_T_GOOD`/`T_BAD` | `0.65`/`0.73` for `w600k_mbf`; **`0.64`/`0.72` for `LVFace-T`** (shipped) | 4.4 - purity floor at max recall; re-derived per embedder with `eject_bench.py --model <name>` |
| `PERSON_EJECT_BAR` | `0.55` -> `0` (env still allows A/B) | 4.3 - pure recall loss |
| HDBSCAN, complete/average linkage, mutual-kNN, median/mean/knee joins | rejected | 4.1, 4.2 |
| quality-adaptive bar | unchanged (Ente fidelity) | 4.5 - neutral, not harmful |
| `ONE_PER_PHOTO` | unchanged | 4.6 - correct prior, no cost |
| face embedder | `w600k_mbf` → **`LVFace-T_Glint360K`** | `model_matched.py`: fewer look-alike mixes (KinFaceW 4 vs 10) at matched purity, +35% grouping at the same look-alike impurity (0.57%); see `MODEL-ZOO.md` §3b |
| body embedder | `osnet_x0_25` → **`osnet_ain_x1_0_msmt17`** | 7.3 - wins at matched purity on both axes |

Expected effect: same purity, materially better grouping (bench recall
0.1667 -> 0.3121 on public selection sets).

## 6. Recluster referential-integrity fix

Deploying the above exposed a latent bug in `recluster_ente()`: the
anonymous rebuild swaps person rows wholesale (old ids are deleted, new
rows inserted) but never touched `detections.person_id`. Since
`cluster.recluster()` runs `fusion.body_attach()` immediately afterwards,
that pass read the dead ids and copied them onto faces:

- 72 detection claims and 24 faces ended up referencing persons that no
  longer exist
- `fusion.py:88` (`... WHERE ... AND person_id IS NULL`) is the exact
  write that put them on faces - all 24 matched their linked detection

Fixed in `app/cluster_ente.py`: after the person rebuild, derived claims
(`reason != 'manual'`) pointing at a person that no longer exists are
nulled before `body_attach()` runs, and persons named only by a manual
body claim are excluded from deletion. Verified: 0 dangling references on
both tables after a full recluster.

## 7. Body attach A/B (`fusion_bench.py`)

The face benchmarks above give every image its own photo id and have no
timestamps, so they cannot say anything about `fusion.body_attach` - the
layer that lets a faceless detection join a **face-anchored** person. That
layer is measured separately on Market-1501, simulating the gallery shape:

- 300 identities x 6 images (fixed seed 7), **3 of them stand in for
  face-photos** (the seed body evidence a person gets from the photos where
  a face was seen), the rest are the faceless/back-view queries
- **virtual group photos**: two identities share one photo, which is what
  the co-occurrence rule exists for
- **synthetic capture times with identities overlapping**: everyone is out
  within the same three-day span for a ~2 h outing. Separating identities
  in time would hand the window a free win - in a real album different
  people are photographed on the same day, which is exactly when mixing can
  happen
- scoring is asymmetric as everywhere else: `wrong` = a detection attached
  to someone else's person (a MIX, expensive), `miss` = unassigned (free)

### 7.1 Headline

```
config                                    coverage  wrong share  polluted persons
shipped before this pass (t=0.74, max,
  no margin, 6h window, OSNet x0.25)        0.776      8.5%         14.0%
+ BODY_MARGIN=0.05, window 3h, AIN x1.0    0.773      0.29%         0.67%
```

Same coverage, 1/30th the wrong attaches, 1/21st the polluted persons.

### 7.2 The margin gate is the lever

`BODY_MARGIN`: the best person must beat the runner-up by this much, or the
detection stays unassigned. OSNet x0.25, `t=0.74`, 3 h window:

```
margin   coverage  wrong share  polluted persons
0.00       0.776      8.5%          14.0%
0.02       0.748      2.1%           10.7%
0.04       0.690      1.0%            6.0%
0.05       0.671      0.6%            4.3%
0.06       0.649      0.4%            2.7%
0.08       0.587      0.1%            0.7%
```

Averaged over thresholds: `margin 0.06 -> wrong 1.1%` against `0.00 ->
3.4%`. It is the only knob that cuts mixing *without* giving up coverage
that the threshold would also give up.

### 7.3 Which body embedder

Production regime (group photos + 3 h window + 3 anchors), each model at
its best margin inside a 2% polluted-person budget:

| body model | MB | best config | coverage | wrong | polluted |
|---|---|---|---|---|---|
| **osnet_ain_x1_0_msmt17** | 8.3 | t=0.74 margin=0.05 | **0.742** | 1.04% | 1.67% |
| osnet_x0_25_msmt17 (was shipped) | 0.9 | t=0.74 margin=0.07 | 0.624 | 1.06% | 2.00% |
| osnet_x1_0_msmt17 (non-AIN) | 8.3 | t=0.74 margin=0.06 | 0.604 | 1.45% | 2.00% |
| osnet_ain_x0_75_multisource | 5.2 | — | 0.12 | 6.1% | — (rejected: preprocess mismatch, unusable) |

OSNet-AIN x1.0 wins at matched purity on **both** axes, costs 7.4 MB more
and is now the default (`REID_MODEL_FILE`). It also confirms
`docs/REID-MODEL-ZOO.md`'s earlier pairwise verdict ("wins every shared
threshold") from a completely different angle.

### 7.4 The capture-time window

`t=0.74`, max, margin 0.06, OSNet x0.25 — swept 1 h to off:

```
window   coverage  wrong share  polluted persons
1h         0.614      1.95%          3.0%
3h         0.694      1.11%          2.3%     <- best on both axes
6h         0.649      1.35%          2.7%     <- was shipped
12h        0.594      1.65%          2.7%
24h        0.524      2.28%          2.3%
off        0.457      1.67%          2.3%
```

A wider window only admits more strangers, which the margin gate then has
to reject - hence both coverage and purity degrade. `BODY_MAX_GAP_S` is now
`10800` (3 h).

### 7.5 Things that were tested and did not matter

- **chaining** (an attached vector joins the person's evidence): small
  effect either way; kept as shipped
- **aggregation** (`max` / `mean` / `median` / `top2` over a person's body
  vectors): differences vanish once the margin gate is in; `max` is kept
- **co-occurrence rule**: on/off is *identical* here and on the face side
  (`group_bench.py`) - it is a rarely-firing safety fuse, kept because it
  is free
- **more anchors**: 3 face-photos beat 1 on both axes (coverage 0.498 vs
  0.408, polluted 5.2% vs 9.5% on average) - not a knob, but it explains
  why the layer gets better as a person is seen in more photos with a face

## 7b. Re-deriving every similarity knob after a model swap

A similarity threshold is a property of the embedding space, not of the
algorithm - `t_good=0.65` meant nothing after the move to LVFace-T.
`tune_bench.py` re-derives the identity knobs from the **pair
distributions** of the current embedder, under "never mix, group as much as
possible" rather than any single number.

LVFace-T on LFW (2,182 same / 717,218 different pairs):

| | p1 | p5 | p25 | p50 | p75 | p95 | p99 |
|---|---|---|---|---|---|---|---|
| same person | 0.367 | 0.442 | 0.533 | 0.599 | 0.660 | 0.734 | 0.788 |
| different | -0.144 | -0.101 | -0.040 | 0.002 | 0.045 | 0.106 | 0.150 |

OSNet-AIN x1.0 on Market-1501 (4,500 same / 1,614,600 different):

| | p1 | p5 | p25 | p50 | p95 | p99 |
|---|---|---|---|---|---|---|
| same person | 0.505 | 0.578 | 0.690 | 0.766 | 0.909 | 0.944 |
| different | 0.312 | 0.348 | 0.411 | 0.465 | 0.634 | 0.712 |

Knobs and where each number comes from:

| knob | value | derivation |
|---|---|---|
| `ENTE_T_GOOD` | `0.64` | highest grouping while KinFaceW mixing stays at the old line's level (`eject_bench.py --model lvface_t`) |
| `ENTE_T_BAD` | `0.72` | `t_good` + Ente's 0.08 gap (≈ their 0.76→0.84 ratio). Sweeping the gap 0→0.16 moves grouping by 0.0016 at identical purity, but the bench stubs out the score/sideways legs of `is_bad_face`, so a measured "no effect" here would be an artefact - the gap stays |
| `PERSON_SPLIT_SIM` | `0.53` | same-person p25 (0.533): a sub-group whose every cross pair clears this stays inside the genuine core |
| `REVIEW_SIM_MIN` | `0.15` | different-person p99 (0.150): above the impostor tail is worth a human look; the queue is sorted by similarity so p5+ (0.442) leads |
| `REVIEW_BODY_SIM_MIN` | `0.63` | body different-person p95 (0.634) |
| `BODY_JOIN_T` | `0.74` | just above the body impostor tail (diff p99 0.712) - swept in `fusion_bench.py` |

## 8. Reproducing

```bash
docker run --rm --memory=1200m --cpus=1 \
  -v $PWD:/app -v ~/.hermes/cache/scratch:/s2 reid-gallery:latest \
  sh -c 'pip install -q scipy scikit-learn; \
         python /app/scripts/eval/algo_bench.py mobilefacenet_w600k --per-set'
```

| script | answers |
|---|---|
| `scripts/eval/algo_bench.py` | algorithm bake-off, purity + recall, `--algos`, `--per-set` |
| `scripts/eval/eject_bench.py` | eject bar sweep, and fine threshold sweep (`python eject_bench.py 0.61 ...`) |
| `scripts/eval/quality_bench.py` | quality layer (real score/landmarks) contribution |
| `scripts/eval/group_bench.py` | virtual group photos: co-occurrence rule + eject in the production shape |
| `scripts/eval/fusion_bench.py` | body attach A/B: `--stage core|cooc|tune|model|final|gap` |
| `scripts/eval/model_compare.py` | all face models at the shipped threshold |
| `scripts/eval/model_matched.py` | all face models at matched purity (`--budget`) |
| `scripts/eval/tune_bench.py` | re-derive all similarity knobs for the current embedder (pair distributions) |
| `scripts/eval/hard_bench.py` | per-model hard-set ranking at each model's own operating point |
| `scripts/eval/calib_bench.py` | earlier cluster-level threshold calibration |
| `scripts/reembed_body.py` | ops: recompute stored body vectors after a `REID_MODEL_FILE` swap |

## Review ordering (2026-10-07)

**What Ente actually does:** `people.ts:336` sorts suggestion clusters by
median cosine similarity descending and nothing else. There is no dedup, no
diversity pass, no size or recency term; the UI renders that order verbatim
(`PeopleHeader.tsx:757-792`). Review is per-person cluster-to-person
assignment with per-person rejection memory (local KV), a 0.48 -> 0.46
threshold cascade before singletons are offered, and an 80-item cap. Scores
are computed on random samples of <=50 faces, so the order is mildly
stochastic above that size. **The "same question twice" problem is Ente's
own**, not a porting artefact.

**Measured alternative** (`scripts/eval/review_order_bench.py`, LFW 700 faces
/ 117 identities, ground-truth simulated human):

| ordering | dup% | fix 50% | fix 25% | fix 10% |
|---|---|---|---|---|
| score (relevance only) | 44.0 | 71 | 33 | 11 |
| impact (expected error reduction) | 44.0 | 65 | 29 | 10 |
| farthest-first (Gonzalez 1985) | 44.4 | 78 | 43 | 17 |
| MMR 0.5 (Carbonell & Goldstein 1998) | 38.8 | 78 | 36 | 13 |
| MMR 0.3 + impact | 39.6 | 73 | 30 | 10 |

Total question count is identical (134) across orderings: ordering changes the
experience and the speed of early progress, never the total work. Diversity
alone is worst on both counts.

**Shipped: MMR with an expected-error-reduction relevance term** -
`REVIEW_MMR_LAMBDA=0.3` (the paper's own "survey the space" setting),
`REVIEW_MMR_IMPACT=1`. Relevance is the app's existing composite score, so
this is a re-ranking parameterisation, not a new scoring rule.

Verified citations (originals read or Crossref-checked): Carbonell &
Goldstein SIGIR 1998 (MMR, lambda ~0.3 survey / ~0.7 refocus); Basu,
Banerjee & Mooney SDM 2004 *Active Semi-Supervision for Pairwise Constrained
Clustering* (the title "Active Semi-Supervised Clustering" does not exist);
Roy & McCallum ICML 2001 (expected error reduction); Joshi, Porikli &
Papanikolopoulos CVPR 2009 (margin sampling); Gonzalez TCS 1985
(farthest-first); Saha & Subramanian ESA 2019 (QueryPivot, <= 2*C_OPT
same-cluster queries); Sarawagi & Bhamidipaty KDD 2002 (interactive
deduplication).

