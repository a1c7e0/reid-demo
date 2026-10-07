# Open-source survey: face + body person recognition ("back view" ID)

Status: reference notes from a separate survey session (2026-10-03). Four candidate
repositories were cloned, read, and (where cheap) executed. None can be integrated
into Immich/Ente as-is; the value here is the borrowed *patterns*.

Clones: `~/workspace/prototypes/` (4 repos).

## Bottom line

No mature open-source photo gallery combines face + body to identify people from
behind. Immich / PhotoPrism / LibrePhotos are face-only — Immich's own maintainers
state it "entirely rely on faces" (immich-app/immich discussion #9810). The closest
design write-up is an *unimplemented* feature request:
`JaaJSoft/workspace#1060`, whose shape (event grouping by `taken_at` gaps →
whole-body appearance embeddings → per-event clustering → one usable face names the
whole cluster, cross-event linking stays face-based) matches this project's design,
including the open question OSNet vs DINOv2.

## Repo-by-repo

### 1. brezzergit/Query-Based-Image-Retrieval-System — reject
176 LOC scaffold. Not runnable: hard-coded `/Users/...` path, `post_process.py`
uses `np`/`os` without importing them, `save_clusters` assigns a key onto a `list`,
and face embeddings (one per detected face) are `hstack`-ed with clothing features
(one per image) so any face-less image breaks the shapes. Concept only: face +
ResNet50 clothing vector → StandardScaler → DBSCAN(eps=0.5).

### 2. FuaadBashi/cctv-person-reidentification — best engineering, borrow logic
1.3k LOC, CI, tests (**ran: 16 passed in 0.36s** with numpy+pytest only).
Borrow:
- **face-first, body-fallback with a best-vs-second-best margin** — a body match is
  accepted only when it beats the runner-up by `body_margin`; ambiguity creates a
  new identity instead of merging two people. Directly anti-over-merge.
- **Per-assignment reason + distance logged** to `tracks.jsonl`
  (`face_match` / `body_match` / `new`) — the same "explain the grouping in the UI"
  requirement this project has.
Limits: body embedder is an ImageNet ResNet-50 (author documents "NOT
clothing-invariant ReID"); body bank is *not* face-anchored (body-only tracks mint
identities and back-fill body vectors); no capture-time window; video tracker, not
a photo library.

### 3. akippnn/multicue-person-reid — best scoring framework, weakest body cue
4.1k LOC, honest `docs/LIMITATIONS.md`.
Borrow:
- Weighted multi-cue score (face 0.50 / geometry 0.25 / silhouette 0.15 / body 0.10)
  plus **independent gate per cue + open-set rejection** (fail any gate → mark
  unknown instead of forcing the nearest known person) and a rolling-template
  smoothing (mean of last 12/20 vectors) against drift.
- Explicit status taxonomy: face-confirmed / body match / tracked / unknown.
Measured: its hand-crafted `body_fingerprint` (HSV histograms + HOG) on
Market-1501 `gt_bbox`, 150 pids, query = first image per pid, rest = gallery:
**rank-1 45.3%**, mean same-person sim 0.517 vs different-person 0.385 (chance
~0.7%). Note its default `body_cosine_threshold: 0.62` sits *above* the mean
same-person similarity — thresholds were never calibrated on real data, only valid
for its close-range camera enrollment.
Limits: live-camera single-target lock, Apple-Silicon first, no library/event
clustering.

### 4. MalikAnees530/Event-Guest-ReID-Engine — closest architecture, weakest code
1.2k LOC, mostly a 998-line `mainV30.py`.
Architecture worth copying:
- **`event_id` scoping** and per-event clustering (clothing is only a stable signal
  inside an event).
- **Dual named vectors** (face 512 ArcFace / body 768 DINOv2 torso crop cut at
  shoulder-hip keypoints) in Qdrant — embedding spaces never mixed.
- **RRF rank fusion with 2.5x face weight**, switching to body-dominant scoring
  when the face is absent/turned (`face_dominant` from yaw) — the same "yaw decides
  which cue leads" idea.
- **Face quality score** (det conf + blur + face/person area ratio), capped to
  <=0.15 when no reliable face — so a faceless detection can never out-vote a real
  face.
- Adaptive thresholds derived from the candidate score distribution.
Why not to reuse: CI only runs `py_compile`, **zero tests**, `titanium_core/database_engine.py`
is a 16-line stub while everything lives in one monolith, headers say "Generated
automatically ... Do not edit by hand", README is marketing prose with no measured
backing, and deps (buffalo_l + DINOv2 vitb14 + YOLO pose + Qdrant + onnxruntime-gpu)
far exceed this project's weight budget. Its face→body link is also written only
when the detection *has* a reliable face, so faceless points carry no face bank —
linking happens only at query time, weaker than this project's face-anchored rule
(body claims may only target a face-anchored person).

## What this project already does better

Face-anchored identities with body used only as `body+time` evidence inside a
capture-time window, over-split-never-over-merge bias, co-occurrence hard splits,
hysteresis (join bar > evict bar) for cycle-free repair passes, and OSNet x0.25 ONNX
instead of a hand-crafted or ImageNet descriptor. None of the four repositories
combines those.

## Borrow list (actionable)

1. Margin-gated body matching + per-claim reason logging (repo 2).
2. Per-cue independent gates, open-set rejection, explicit status taxonomy (repo 3);
   **re-derive every threshold on real data** — repo 3's defaults do not transfer.
3. Event partitioning, dual named vector spaces, yaw-driven cue weighting,
   face-quality score (repo 4) as *design*, not code.
4. Do not adopt any of their body models: ResNet-50 / HOG-histogram / DINOv2-torso
   are all weaker or heavier than the OSNet choice already made here.

---

# Appendix A — Alternative implementation routes (decision: 2026-10-04)

Owner decisions on the routes below: **(1) accepted, (2) accepted, (3) rejected,
(5) parked for fun, (6) approved as a control experiment.** Route numbers refer
to the list here.

1. **Burst / motion-photo / video track propagation — APPROVED.**
   Carry a face-derived identity across frames of the same physical track
   (ByteTrack / optical flow) instead of re-deriving it from appearance. Much
   stronger evidence than any embedding; requires no ReID model. Covers the
   "face in frame 1, back in frame 5" case natively. Complements, does not
   replace, the `body+time` path for single stills.
2. **Discrete attribute signature — APPROVED.**
   Hair length/colour, glasses, hat, backpack, top colour/type as a symbolic
   vector: feeds ranking *and* gives the UI an explainable reason ("matched
   via ponytail + glasses + black top"). Zero-shot CLIP scoring is the
   no-training implementation. Must stay secondary evidence — never a claim.
3. **VLM pairwise verifier — REJECTED.** Do not design for it.
4. *(graph-only propagation — not selected as a standalone route.)*
5. **Face-teacher distillation into the body embedder — PARKED.**
   Highest ceiling, needs training; violates the "no heavy local compute,
   reuse pretrained wheels" rule for now.
6. **General vision encoder as body embedder — APPROVED as a control.**
   See Appendix B.

## CLIP's role (settled)

CLIP must **never** vote on identity — same outfit ⇒ high similarity across
different people. Its allowed jobs: (a) implement the attribute signature,
(b) recall/candidate generation, (c) scene-similarity help for event
boundaries, (d) user-facing text/image search, (e) an **asymmetric veto only**
— semantic *conflict* can reject a borderline body claim, semantic agreement
earns nothing. Veto-only keeps the over-split-never-over-merge bias intact.

Both target hosts already ship a CLIP-class encoder, so **do not vendor a
second one**: Immich Smart Search = OpenAI CLIP ViT-B-32
(`immich-app/ViT-B-32__openai`, `visual/model.onnx` 351.6 MB,
`textual/model.onnx` 254.2 MB, Postgres/VectorChord index); Ente = on-device
MobileCLIP for "Magic search" (plus MobileFaceNet for faces). Reuse theirs.

---

# Appendix B — Control experiment: DINOv2 vs OSNet

## Question

Does a general-purpose vision encoder match OSNet on *this* project's
face-anchored clustering, or is the ReID-specific model earning its place?

## Candidate

**DINOv2-S, int8 — 24.4 MB** (`onnx-community/dinov2-small`,
`onnx/model_int8.onnx`; fp32 88.5 MB / fp16 44.4 MB). Verified file sizes on
Hugging Face 2026-10-04. Output 384-d; patch tokens also enable torso-only
pooling that a global CLIP vector cannot do.

### Correction to `DESIGN.md` §4

The current text rejects DINOv2 as *"far over the weight budget"*. That is
true of **DINOv2-B fp32 (346.6 MB)**, not of DINOv2-S int8 (**24.4 MB**),
which fits the ≤~400 MB budget with room to spare. The legitimate open
objection is only *"unproven for ReID"* — which is exactly what this control
measures. If DINOv2-S loses, drop it with a number attached, not with the
size claim.

Rejected candidates stay rejected, for the record: CLIP ViT-B/32 vision tower
alone is 351.6 MB (`Marqo/onnx-open_clip-ViT-B-32`, `immich-app` export), so
it cannot serve as a body embedder inside budget.

## Known-broken baseline (must be fixed before any Market number is quoted)

`testsets/out/market_reid.json` (mtime 2026-10-03 20:06) reports, untouched
since:

```
osnet_x0_25_msmt17.onnx : rank1=0.0 rank5=0.0 mAP=0.0  model_bytes=907169
osnet_x1_0_msmt17.onnx  : rank1=0.0 rank5=0.0 mAP=0.0  model_bytes=770055
```

`osnet_x1_0_msmt17.onnx` at 770 KB cannot be a 1.x model (its `.pth` is
10.98 MB) — a truncated / external-data export, and both models scoring
exactly 0.0 means the harness or the export is broken, not that ReID failed.
`scripts/eval/market_reid.py` has not been modified since the initial commit
(16:20). **No Market rank-1 claim is usable until this is fixed.**

## Recommended procedure — use the test-set trial harness, not Market

`scripts/eval/testset_trial.py` (wipe → recluster → score against
`testsets/labels/labels-20261003.json`) already measures what actually
matters here — partition quality (FM/FS/UN/HIT), not single-pair retrieval.

1. Re-embed the 203 labeled photos' person crops with DINOv2-S int8
   (~200 images, CPU, well under a minute); keep OSNet's vectors alongside.
2. Run the identical trial for each embedder: **FM / HIT** is the headline
   comparison, since FM is the stated primary evil.
3. Re-derive the claim threshold per embedder (`BODY_ASSIGN_T = 0.76` is
   calibrated to OSNet's score distribution and will not transfer), then
   compare both at their own tuned operating point.
4. Report `ms/image` and embedding bytes next to the score.
5. Fix `market_reid.py` afterwards for an absolute, cross-project number;
   it is a nice-to-have, not a blocker for this control.

Decision rule: adopt DINOv2-S only if it is competitive on FM at a
re-tuned threshold **and** its latency/size cost is acceptable; otherwise keep
OSNet and record the measurement in `docs/TESTSET-EVAL.md` so the rejection is
auditable.
