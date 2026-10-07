# Design: person grouping without a face

## Problem

Traditional face recognition chains `face → embedding → similarity →
identity`. It fails whenever the face is not visible — the classic back-view
shot. Google Photos solved this by grouping people using clothing/appearance
*and* temporal context instead of refusing to label the photo.

`reid-gallery` implements the same idea with three cheap edge models and a
clustering core ported from Ente.

## Pipeline

```
                      ┌──────────────────────────────┐
                      │ letterbox once → 640×640     │
                      └──────┬──────────────┬────────┘
                             │              │
                   YOLO11n (persons)   YuNet (faces + 5 landmarks)
                             │              │  align 112×112 → MobileFaceNet (512-d)
                             ▼              ▼
              person boxes (orig coords)   face boxes + embeddings + blur/sideways
                             │
              OSNet-AIN x1.0 body embedding (512-d, 128×256 crop)
                             ▼
                      SQLite (photos / detections / faces / persons)
                             ▼
       quality gates → Ente clustering → body attach → review queue → web UI
```

Notes:

- The frame is letterboxed **once** and shared by both detectors; boxes are
  mapped back through the recorded scale/padding.
- Each person box is linked to at most one face (containment first, then
  IoU). The link is stored as `detections.face_id` — and it is *exact*: the
  face's identity and the body's identity are joined through it, never
  through a similarity guess.
- Capture time comes from EXIF (`DateTimeOriginal` → `DateTimeDigitized` →
  `DateTime`), falling back to file mtime.

## Clustering pass

Runs under a global lock after each processed batch (and via
`POST /api/recluster`). The core is a faithful port of Ente's
`cluster.ts` — source of truth and line-level mapping in
[ENTE-PORT-SPEC.md](ENTE-PORT-SPEC.md); parameters and their measurements
are in [ALGO-TUNING.md](ALGO-TUNING.md).

### 1. Quality gates (computed at ingest, persisted)

| signal | formula / source | used for |
|---|---|---|
| `blur` | Laplacian variance of the aligned 112×112 crop | clusterable + bad-face line |
| `score` | detector confidence | clusterable + bad-face line |
| `sideways` | Ente `faceDirection()` on YuNet's 5 landmarks — nose past the eye/mouth line, or upright with the nose within `0.2·eyeDx` of an eye | bad-face line |

- **clusterable** (`ente_core.is_clusterable`): `blur > 10 and score > 0.8`
  (`cluster.ts:143`). Faces that fail stay **unassigned** — a reviewable
  fragment, never piled into a junk person.
- **bad face** (`is_bad_face`): `blur < 50 || (blur < 200 && score < 0.85)
  || sideways` (`cluster.ts:177-183`).

### 2. The clustering core

Greedy single-linkage, newest-first, matches look **backward** only:

```
for each face i (newest first, already-placed first):
    pick the previously-placed face j with the highest cosine sim s(i, j)
    subject to  s(i, j) ≥ threshold(bad[j])        # quality-adaptive bar
                and photo(j) ∉ forbidden(i)         # co-occurrence rule
                and cluster(j) ∉ rejected(i)        # user rejection memory
    if none qualifies → i opens a new person
```

- `threshold(good neighbour) = ENTE_T_GOOD (0.65)`
- `threshold(bad neighbour)   = ENTE_T_BAD (0.73)` — a poor-quality
  neighbour must be much closer before identity is trusted
- `ONE_PER_PHOTO` (co-occurrence, our one deviation from stock Ente): two
  faces of one photo may never share a person. In a group shot those are
  different people; stock Ente has no such rule.
- named / reviewed faces are **immutable seeds**: a rebuild reassigns only
  anonymous faces, so names survive re-clustering.
- `rejected_faces` records per (face × cluster) user rejections and never
  re-proposes them (`cluster.ts:243-252`).

Why greedy 1-NN instead of HDBSCAN/AGNES: measured, not assumed. On 11,171
public images under the product objective (mixing expensive, splitting
free) HDBSCAN either collapses the library into ~70 giant clusters that are
*all* impure or turns everything to noise; complete/average linkage and
mutual-kNN all lose recall at matched purity. The shipped rule is at or
above every alternative — see `docs/ALGO-TUNING.md` §4.

### 3. Anti-mixing invariants

1. **Identity is face-anchored.** Body-only clusters are never created.
2. **Co-occurrence is hard**: no two faces of one photo in one person.
3. **Never split what is provably safe, never merge what is not.** The
   automatic complete-linkage "mixed grouping?" eject is **off**
   (`PERSON_EJECT_BAR=0`): at every bar it was tested (0.35–0.65) it
   reduced no impurity and cost up to 76% of the recall.
4. **The threshold sits at the purity floor** (`0.65 / 0.73`): lowering it
   to 0.61–0.64 buys 8–25% more recall for +0.03pp of faces living in
   impure clusters — rejected under "never mix if avoidable".
5. **Human review is the tie-breaker**, not a lower threshold: similar-but-
   unconfirmed pairs queue at `GET /api/review` with their similarity,
   median similarity, gap in days and body similarity.

### 4. Body attach (re-id fusion)

`app/fusion.py`, run immediately after the face pass. A faceless detection
may join an existing **face-anchored** person when all of:

| gate | knob | value |
|---|---|---|
| body cosine vs the person's best body vector | `BODY_JOIN_T` | `0.74` |
| best must beat the runner-up | `BODY_MARGIN` | `0.05` |
| capture time within the person's mean ± window | `BODY_MAX_GAP_S` | `10800` (3 h) |
| person not already present in this photo | `ONE_PER_PHOTO` | hard |
| enough box pixels / one coherent skeleton | `BODY_MIN_REL_H`, `BODY_MIN_PERSONNESS` | `0.12`, `0.5` |

Failing any gate yields **unassigned** — never a fallback to a weaker cue,
and body evidence never anchors a new person on its own.

Measured on Market-1501 in the production regime (group photos + time
window + 3 seeded anchors per identity, `scripts/eval/fusion_bench.py`):

```
config                                   coverage  wrong attach  polluted persons
shipped before this pass (no margin, 6h)   0.776      8.5%          14.0%
after: AIN x1.0 + margin 0.05 + 3h        0.773      0.29%          0.67%
```

i.e. the same amount of back-view coverage at 1/30th the wrong-attach rate.
The margin gate is the lever: a wider time window only admits more
strangers, which the gate then has to reject — which is why 3 h beat 6 h
and everything wider on *both* axes.

## Failure behavior

- A photo with no detections is still imported (`processed = 1`) and shown in
  the main grid; it simply belongs to nobody.
- Unreadable files are logged (English), counted in `worker.failed`, and do
  not stop the queue.
- Deleting a photo cascades its detections/faces; empty persons are removed
  during the next clustering pass.
- A full recluster swaps person rows wholesale; stale derived body claims
  are cleared between the rebuild and the attach pass so neither detections
  nor faces are ever left pointing at a person that no longer exists.

## Tuning

If back shots fail to attach: lower `BODY_MARGIN` first (it trades purity
for coverage on a straight line), then `BODY_JOIN_T`, then widen
`BODY_MAX_GAP_S` one step at a time. `BODY_ATTACH_ENABLED=0` turns the
layer off entirely for comparison.

If different people get merged: raise `ENTE_T_GOOD` (and keep `ENTE_T_BAD`
8 points above it), or fix the individual case with the review queue /
`POST /api/persons/{id}/breakup`. Do **not** reach for the eject bar — it
was measured as a net loss.

Every number above was derived on public datasets only (LFW + CFP/KinFaceW/
CALFW/CPLFW/TinyFace for faces, Market-1501 for body); methodology and raw
tables: [ALGO-TUNING.md](ALGO-TUNING.md).

## Roadmap: borrowed patterns from the prototypes survey

Source: [prototypes-survey.md](prototypes-survey.md) — four open-source
face+body prototypes were read; none is integrable as-is.

| item | status |
|---|---|
| 1. Margin-gated body matching + claim provenance | **done** — `BODY_MARGIN`, `detections.margin` stores best−runner-up for the UI tooltip |
| 2. Per-cue gates, open-set rejection, status taxonomy | **done** — every gate returns "unknown", never a weaker cue; thresholds re-derived on public data (`ALGO-TUNING.md`, `REID-MODEL-ZOO.md`) |
| 3. Event scoping, dual vector spaces, yaw weighting, face quality | partial — dual spaces and quality score exist; event partitioning superseded by the measured time window; yaw weighting not adopted (measured unnecessary once the quality line was ported) |
| 4. Body model decision | **decided** — OSNet-AIN x1.0 (MSMT17) is the default embedder; ResNet-50/HSV+HOG/DINOv2 rejected on measurement |
