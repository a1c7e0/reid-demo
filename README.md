# reid-demo

A self-hosted photo gallery that groups photos by **person** — faces first,
then **body re-identification**, which is what makes *back-view* recognition
work: a person seen from behind, or with their face hidden, can still be
attached to the right person.

This repository is the **demo** for that capability: a basic gallery, person
pages, person management (review), photo metadata, plus the tools needed to
inspect what the models did — bounding boxes over the photo, re-binding a face
or body to a different person, and similarity numbers.

中文说明见 [README.zh.md](README.zh.md).

---

## Quick start

```bash
docker build -t reid-demo .
docker run -d --name reid-demo \
  -p 8000:8000 \
  -v "$PWD/data:/data" \
  reid-demo
```

Open <http://localhost:8000>.

The first build downloads the ONNX models and exports two of them
(`scripts/download_models.sh`, `scripts/export_reid_zoo.py`); the build fails
loudly if a download does not match its expected hash. Allow a few minutes and
~1 GB of build cache.

`docker-compose.yml` is provided if you prefer that.

---

## Using it

### 1. Get photos in

**Settings → Upload photos** picks files from the browser, or **Settings →
Scan a folder** imports a directory already on the server:

```bash
docker run --rm -v "$PWD/data:/data" -v /path/to/photos:/import:ro \
  reid-demo python -c "from app.ingest import ingest_dir; ingest_dir('/import')"
```

Each photo is detected, embedded and queued. Progress is on the **Settings →
Status** card.

### 2. Photos

Grouped by capture day. Click a photo to open the viewer.

The viewer draws **every box the model produced**: green = face, dashed blue =
body. Each box is tagged with the person it was assigned to (or `face` / `body`
if unassigned) and, for bodies, why it was attached (`body+time`, `body+gap`,
…). The **Info** panel on the right shows the photo's essential metadata —
capture time, pixel size, file size, relative path, sha1 — and one row per
detected unit with a **person picker** to re-bind it.

**Similarity** is shown inside the person picker, sorted descending, so the
most likely person is at the top.

### 3. People

A grid of squircle tiles: avatar, name (or `Person N`), pinned badge.

- **Click** a tile to open that person.
- **Select** (button in the header) enters multi-select; **Done** leaves it.
  Selected tiles show ticks and a bottom action bar: **Edit** (rename),
  **Review**, **Pin / Unpin**, **Reset**, and **Merge** when two or more are
  selected — the merge sheet states the direction explicitly (*"Everyone else
  joins person #12"*), because getting that backwards is not recoverable from
  the UI.
- **Sort**: most/fewest photos, name A–Z / Z–A.

### 4. Person page

The person's photos, their clustered faces, and the reason badges for any
attached bodies. Actions: **Rename**, **Pin / Unpin**, **Review**, and
**Mixed grouping?** — a one-click breakup that splits the cluster wherever two
groups of faces are not mutually similar (complete-link), i.e. exactly where a
human would say "these are two different people".

### 5. Review

The place to fix clustering mistakes. One item at a time: the target person on
the left, a candidate cluster on the right, with the face and body similarity
numbers, the photo gap in days, and a clearly labelled direction.
**Merge into this person** / **Not this person** / **Skip**, plus **Decided
history** with **Reopen**.

Two things make the queue behave:

- **It is ordered by MMR** (Carbonell & Goldstein, SIGIR 1998) with an
  expected-error-reduction relevance term, so the next question is the best
  trade-off between *"likely to matter"* and *"not the question you just
  answered"*. A person split into several clusters does not come back as the
  same question three times in a row.
- **Rejecting remembers**. Saying "not this person" stores what was rejected;
  a later candidate that reads as the same person is skipped outright.

Tunables: `REVIEW_MMR_LAMBDA` (default 0.3), `REVIEW_MMR_IMPACT` (default on),
`REVIEW_DUP_T` (default 0.72).

---

## What the models are doing

Pipeline, in full: **ingest → face detection → face embedding → person
detection → body embedding → clustering → body attachment**.

1. **Face** — YuNet detects, `align5` similarity-transforms onto the ArcFace
   template, **LVFace-T** embeds. Quality gates (blur / score / sideways)
   decide what is even clusterable.
2. **Clustering** — Ente's greedy single-linkage, ported (`app/ente_core.py`):
   newest-first, backward-only 1-NN, quality-adaptive join thresholds
   (`ENTE_T_GOOD=0.64` good neighbour / `ENTE_T_BAD=0.72` bad neighbour), a
   one-face-per-photo guard, and a co-occurrence invariant (two people in one
   photo can never end up in the same cluster).
3. **Fragment recovery** — a complete-link merge pass folds two clusters
   together **only when every cross pair clears `PERSON_SPLIT_SIM`**, which
   cannot mix anyone at that bar. It is the exact inverse of the **Mixed
   grouping?** breakup button.
4. **Body attachment** — OSNet-AIN x1.0 embeds detections; back views are
   attached to a person only when they clear the join bar **and** beat the
   runner-up by `BODY_MARGIN` **and** fall inside a capture-time window
   (`BODY_MAX_GAP_S`). Bodies never create a person on their own.

The full design is in [`docs/DESIGN.md`](docs/DESIGN.md), the Ente port
notes in [`docs/ENTE-PORT-SPEC.md`](docs/ENTE-PORT-SPEC.md), and every
threshold with its evidence in [`docs/ALGO-TUNING.md`](docs/ALGO-TUNING.md).

**Models, sizes and every active threshold are visible in Settings → Model &
configuration**, so you never have to guess what the running build is doing.

---

## Configuration

Environment variables (all optional, defaults shown in
`app/config.py`):

| Variable | Default | Meaning |
|---|---|---|
| `DATA_DIR` | `/data` | photo store, thumbnails, database |
| `MODEL_DIR` | `/models` | ONNX models |
| `ENTE_T_GOOD` | `0.64` | join bar for a good-quality neighbour |
| `ENTE_T_BAD` | `0.72` | join bar for a bad-quality neighbour |
| `PERSON_SPLIT_SIM` | `0.53` | breakup floor; also the fragment-merge bar |
| `PERSON_EJECT_BAR` | `0` | auto-eject bar (off by default) |
| `BODY_JOIN_T` | `0.63` | body attach join bar |
| `BODY_MARGIN` | `0.06` | body attach ambiguity margin |
| `BODY_MAX_GAP_S` | `10800` | capture-time window for body attach (3 h) |
| `REVIEW_SIM_MIN` | `0.15` | lower bound of the review band |
| `REVIEW_MMR_LAMBDA` | `0.3` | MMR relevance/diversity trade-off |
| `REVIEW_MMR_IMPACT` | `1` | weight review relevance by merge impact |
| `REVIEW_MMR_TOP` | `200` | candidates MMR re-ranks |
| `REVIEW_DUP_T` | `0.72` | "reads as the same person" suppression bar |

---

## Frontend

Vue 3 + Vue Router, with **Tom Select** (person pickers), **PhotoSwipe 5**
(viewer) and **Toastify** (toasts) — all vendored, no CDN. UI controls follow
Ente mobile's interaction model (squircle tiles, bottom action bar, confirm
sheets) but are written against these components rather than ported markup.

Above 1000 px the shell switches to a desktop layout: wide content column,
larger thumbnails, and the navigation docks to the top instead of floating
over the photos. All web text is English.

The build runs a **JS syntax gate** (`node --check` over `app/static/js/`) so a
broken ES module fails the image build instead of rendering a blank page.

---

## Development

```bash
docker build -t reid-demo .          # includes the JS syntax gate
docker run --rm -p 8000:8000 -v "$PWD/data:/data" reid-demo
```

Benchmarks live in `scripts/eval/` and run against public data only:

```bash
# fragmentation / fragment recovery on LFW
docker run --rm --cpus=1 --memory=1200m \
  -v "$PWD:/work" -v "$PWD/testsets/lfw_flat:/lfw:ro" \
  reid-demo python /work/scripts/eval/fragment_regen.py

# review-ordering comparison (MMR vs relevance-only vs pure diversity)
python3 scripts/eval/review_order_bench.py
```

Datasets and their download links are in
[`docs/DATA-SOURCES.md`](docs/DATA-SOURCES.md). **No dataset or weight is
committed** — see `.gitignore`.

---

## Research record

The decisions in this build are all measured, not chosen by taste. The record
is kept in `docs/`:

| File | What it holds |
|---|---|
| [`DESIGN.md`](docs/DESIGN.md) | the pipeline end to end |
| [`ALGO-TUNING.md`](docs/ALGO-TUNING.md) | every threshold, with the A/B behind it |
| [`MODEL-ZOO.md`](docs/MODEL-ZOO.md) | face-model comparison (6 models) |
| [`REID-MODEL-ZOO.md`](docs/REID-MODEL-ZOO.md) | body-model comparison |
| [`ENTE-PORT-SPEC.md`](docs/ENTE-PORT-SPEC.md) | what was taken from Ente and what was recalibrated |
| [`HARD-SETS.md`](docs/HARD-SETS.md) | the evaluation sets and why each one |
| [`REID-DATA.md`](docs/REID-DATA.md) | re-ID tuning sets |
| [`DATA-SOURCES.md`](docs/DATA-SOURCES.md) | all download links |
| [`RESEARCH-LOG.md`](docs/RESEARCH-LOG.md) | the running log of experiments and conclusions |
| [`prototypes-survey.md`](docs/prototypes-survey.md) | UI research notes |

Two standing rules: **everything is evaluated on public data only**, and the
clustering objective is deliberately asymmetric — mixing two people in one
cluster is heavily penalised, splitting one person across clusters is cheap.
In a photo library a merge is a lie about who someone is; a split is just a
mess you can fix with one click.
