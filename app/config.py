"""Runtime configuration, driven by environment variables."""
import os
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


# --- paths -----------------------------------------------------------------
DATA_DIR = Path(_env("DATA_DIR", "/data"))
PHOTO_DIR = DATA_DIR / "photos"
THUMB_DIR = DATA_DIR / "thumbs"
DB_PATH = DATA_DIR / "gallery.db"
MODEL_DIR = Path(_env("MODEL_DIR", "/models"))

# Model files can be swapped per-deployment without rebuilding the image
# (several variants are baked in; see docs/EVAL.md for the benchmarks).
PERSON_DET_MODEL = MODEL_DIR / _env("PERSON_DET_MODEL_FILE", "yolo11n-person.onnx")
FACE_DET_MODEL = MODEL_DIR / _env("FACE_DET_MODEL_FILE", "yunet_2023mar.onnx")
# LVFace-T Glint360K (MIT, ICCV 2025): chosen on the matched-purity
# comparison in docs/MODEL-ZOO.md - fewer look-alike mixes (KinFaceW 4 vs
# 10 pairs) and more grouping at the same purity than w600k_mbf. The
# "adaface" preset and "arcface" are numerically identical here,
# (x-127.5)/127.5 == (x/255-0.5)/0.5.
FACE_REC_MODEL = MODEL_DIR / _env("FACE_REC_MODEL_FILE", "LVFace-T_Glint360K.onnx")
FACE_REC_PRESET = _env("FACE_REC_PRESET", "arcface")  # arcface | sface | adaface
FACE_DET_KIND = _env("FACE_DET_KIND", "yunet")  # yunet | yolo5face (Ente stock)
# OSNet-AIN x1.0 (MSMT17): wins the pairwise benchmark (docs/REID-MODEL-ZOO)
# and the attach A/B (scripts/eval/fusion_bench.py) - at the same purity it
# attaches ~19% more back-views than osnet_x0_25. 8.3 MB, edge-fine.
REID_MODEL = MODEL_DIR / _env("REID_MODEL_FILE", "osnet_ain_x1_0_msmt17.onnx")
POSE_MODEL = MODEL_DIR / _env("POSE_MODEL_FILE", "yolo11n-pose.onnx")

# --- detection thresholds --------------------------------------------------
PERSON_CONF = float(_env("PERSON_CONF", "0.40"))          # YOLO person confidence
FACE_CONF = float(_env("FACE_CONF", "0.50"))              # YuNet face score

# --- identity / clustering knobs (cosine similarity, higher = closer) ------
FACE_GROUP_SIM = float(_env("FACE_GROUP_SIM", "0.54"))    # face cosine threshold to join a person (tuned on the labeled test set)
FACE_EVICT_SIM = float(_env("FACE_EVICT_SIM", "0.52"))    # evict/merge-floor; must stay below
                                                          # FACE_GROUP_SIM (hysteresis band) or
                                                          # recluster oscillates. Sweeping the band
                                                          # upward on the test set is non-monotone
                                                          # and never beats (0.52, 0.50) — the
                                                          # look-alike overlap is not separable by
                                                          # thresholds (see docs/TESTSET-EVAL.md)
FACE_MIN_SCORE = float(_env("FACE_MIN_SCORE", "0.70"))    # weaker faces never drive identity
FACE_MIN_PX = int(_env("FACE_MIN_PX", "24"))              # min face box side (original px)
FACE_MIN_FRAME_PX = int(_env("FACE_MIN_FRAME_PX", "16"))   # min side on the 640 detection canvas
FACE_MERGE_SIM = float(_env("FACE_MERGE_SIM", "0.54"))    # merge face clusters (same bar as grouping)
# user-tuned on real data: wrong claims (tiny targets) stop at 0.76
BODY_ASSIGN_T = float(_env("BODY_ASSIGN_T", "0.73"))      # fused score to claim a person
# personness gate: only crops with one coherent pose skeleton may claim
# (hands / held objects / multi-person mush boxes are rejected)
BODY_MIN_PERSONNESS = float(_env("BODY_MIN_PERSONNESS", "0.5"))
# (BODY_MAX_GAP_S and the body attach gate live in the fusion section below -
#  this duplicate used to define 5 h and was silently overridden by 6 h)
BODY_W = float(_env("BODY_W", "0.70"))                    # weight of body similarity
TIME_W = float(_env("TIME_W", "0.30"))                    # weight of temporal proximity
TIME_HALF_LIFE_S = float(_env("TIME_HALF_LIFE_S", "7200")) # 2h decay: time stays meaningful across an outing
MIN_PHOTOS = int(_env("MIN_PHOTOS", "1"))                 # listing floor; 1 shows every person
# event partitioning (borrowed from the prototypes survey): a capture-time
# gap larger than this starts a new event; body/clothing evidence is only
# trusted inside one event — cross-event linking requires a face
EVENT_GAP_S = float(_env("EVENT_GAP_S", "2400"))          # 40 min
# weak faces (blurry / profile / low conf) clear a higher join bar:
# effective threshold = FACE_GROUP_SIM + FACE_QUALITY_PENALTY * (1 - quality)
FACE_QUALITY_PENALTY = float(_env("FACE_QUALITY_PENALTY", "0.15"))
# (removed) FACE_MAX_YAW profile demotion: measured harmful on the labeled
# test set — profiles classify again, guarded by the quality penalty
# cross-time face auto-join limit: beyond this capture-time gap a pure-face
# join is refused (goes to the review queue instead) — verified case: two
# completely different groups of friends half a year apart merged because
# face embeddings cannot separate them (cross-block sim ~= within-block)
FACE_MAX_JOIN_DAYS = float(_env("FACE_MAX_JOIN_DAYS", "7"))
# (removed) FACE_DOMINANCE_MARGIN and FACE_LINK_W: measured harmful/neutral
# on the labeled test set (face-dominance raised false merges)

# burst bridge ("nonlinear absorption"): faces photographed within
# BURST_GAP_S seconds corroborate each other, so the join bar drops to
# BURST_SIM (measured on the label set: 3s window rescues 2 split pairs of
# hard-angle faces at ZERO risk pairs)
BURST_GAP_S = float(_env("BURST_GAP_S", "0"))   # 0 = off; measured negative
BURST_SIM = float(_env("BURST_SIM", "0.32"))   # experimental only
TRACK_ENABLED = _env("TRACK_ENABLED", "0") not in ("0", "false", "no")
# borrowed from Immich density-core clustering: faces only SEED a new
# person when the seed group reaches this size; loners stay unassigned
# (attach-only) instead of creating junk persons that absorb look-alikes
FACE_MIN_SEED = int(_env("FACE_MIN_SEED", "1"))
# "Mixed grouping?" breakup (Ente-style): split a person into sub-groups
# while the MINIMUM pairwise cosine within a sub-group stays above this
# (complete linkage - refuses to chain, per the anti-mix policy)
# LVFace-T: same-person pairs sit at p25=0.533, so a sub-group that keeps
# every cross pair above this stays inside the genuine core instead of
# sliding along a look-alike chain (tune_bench.py)
PERSON_SPLIT_SIM = float(_env("PERSON_SPLIT_SIM", "0.53"))
# corroborated merge: gray-band pairs fold together only with multi-pair
# support (anti-mix: a single misleading pair can never merge persons)
CORR_ENABLED = _env("CORR_ENABLED", "1") not in ("0", "false", "no")
CORR_MED = float(_env("CORR_MED", "0.42"))
CORR_N45 = int(_env("CORR_N45", "2"))
# body+time person merge (calibrated on Market-1501 same/diff body
# distributions: bar 0.74 ~ 1.2% false-merge). Body evidence only
# CONNECTS face-anchored persons; it never anchors identity alone.
BODY_MERGE_ENABLED = _env("BODY_MERGE_ENABLED", "1") not in ("0", "false", "no")
BODY_MERGE_T = float(_env("BODY_MERGE_T", "0.74"))
BODY_MERGE_MAX_GAP_S = float(_env("BODY_MERGE_MAX_GAP_S", "21600"))

# --- Ente clustering core (ported from ente cluster.ts) -------------------
# engine: "ours" (accumulated pipeline) | "ente" (ported greedy 1-NN)
CLUSTER_ENGINE = _env("CLUSTER_ENGINE", "ente")   # "ours" is DEPRECATED (self-developed, kept for reference only)
# evidence mode: "face" (faithful port) | "body" (body-primary, covers
# back views; faces anchor identity afterwards)
ENTE_MODE = _env("ENTE_MODE", "body")
# LVFace-T, calibrated on public sets (LFW + CFP/KinFaceW/CALFW/CPLFW/
# TinyFace) under the product objective (eject_bench.py --model lvface_t):
# 0.64 is the highest recall while the look-alike impurity (KinFaceW) stays
# exactly where the shipped w600k_mbf line sat (0.57%), buying +35% grouping
# (0.3121 -> 0.4221). Stricter 0.66 matches the old purity floor exactly but
# gains nothing over the old model; looser 0.63 starts mixing look-alikes.
ENTE_T_GOOD = float(_env("ENTE_T_GOOD", "0.64"))
ENTE_T_BAD = float(_env("ENTE_T_BAD", "0.72"))
# auto "Mixed grouping?" cut: eject cluster members whose minimum
# similarity to co-members falls below this (0 disables). Default 0:
# measured on public data (79 configs + a virtual-group-photo regime) the
# bar never reduced impurity - 0.35 is a no-op, 0.55 threw away ~47% of
# the recall - so it buys nothing and stays available via env for A/B.
PERSON_EJECT_BAR = float(_env("PERSON_EJECT_BAR", "0"))
# co-occurrence hard rule (two faces of one photo never share a person);
# 0 = pure Ente stock (they have no such rule)
ONE_PER_PHOTO = _env("ONE_PER_PHOTO", "1") != "0"

# --- re-id fusion (body connects face-anchored persons) -------------------
BODY_ATTACH_ENABLED = _env("BODY_ATTACH_ENABLED", "1") != "0"
BODY_JOIN_T = float(_env("BODY_JOIN_T", "0.74"))  # Market-calibrated body bar
# ambiguity margin: the best person must beat the runner-up by this much or
# the detection stays unassigned. Measured on Market-1501 in the production
# regime (fusion_bench.py): without it 14% of persons pick up a stranger's
# body photo; at 0.05 that is 0.7% with no coverage loss (0.776 -> 0.773).
BODY_MARGIN = float(_env("BODY_MARGIN", "0.05"))
# capture-time window. Swept 1h..off in the same benchmark: 3h beat 6h and
# everything wider on BOTH axes (a wider window only admits more strangers,
# which the margin gate then has to reject).
BODY_MAX_GAP_S = int(_env("BODY_MAX_GAP_S", "10800"))  # 3h
ENTE_BAD_QUALITY = float(_env("ENTE_BAD_QUALITY", "0.5"))
ENTE_BAD_YAW = float(_env("ENTE_BAD_YAW", "0.3"))
# time guard: single-linkage chains across seasons without one (measured:
# an April child body chained into an October look-alike group)
ENTE_MAX_GAP_S = float(_env("ENTE_MAX_GAP_S", str(7 * 86400)))
# asymmetric face veto (survey rule: faces only VETO, never boost): a
# face-bearing unit may not join a group whose faces are incompatible
ENTE_FACE_VETO = float(_env("ENTE_FACE_VETO", "0"))  # faces are too noisy on this library to veto

# --- Immich clustering core (ported from person.service.ts) ---------------
# incremental kNN + density-core person creation + attach-only loners
IM_T_JOIN = float(_env("IM_T_JOIN", "0.55"))
IM_MIN_FACES = int(_env("IM_MIN_FACES", "2"))
# borrowed from Ente: process faces newest-first (clustering heuristic)
FACE_NEWEST_FIRST = _env("FACE_NEWEST_FIRST", "0") not in ("0", "false", "no")

# merge review: pairs with face sim in [REVIEW_SIM_MIN, the join bar) are
# "similar but unconfirmed" — queued for a human merge decision. Derived
# from the LVFace-T pair distributions on LFW (tune_bench.py): different
# identities sit at p99=0.150, so anything above that tail is at least
# worth a human look; the queue is sorted by similarity so the dense
# same-person region (p5=0.442 upwards) comes first.
REVIEW_SIM_MIN = float(_env("REVIEW_SIM_MIN", "0.15"))
# two candidates this similar read as the same person to the eye: only the
# better one is worth showing first, and rejecting it suppresses the rest
# (fragments of one split person must not queue up one after another)
REVIEW_DUP_T = float(_env("REVIEW_DUP_T", "0.72"))
# Review ordering is MMR (Carbonell & Goldstein, SIGIR 1998):
#   argmax_i  lambda * relevance(i) - (1-lambda) * max_j sim(i, j)
# where j runs over what the user has already been shown. Ente has NO
# diversity pass at all (people.ts:336 sorts by median similarity alone), so a
# split person comes back as several near-identical questions in a row. The
# paper's own guidance is lambda ~0.3 to survey the space; measuring on LFW
# ground truth put 0.3/0.5 lowest on duplicate questions (39.6%/38.8% vs
# 44.0% for relevance-only).
REVIEW_MMR_LAMBDA = float(_env("REVIEW_MMR_LAMBDA", "0.3"))
# weight relevance by how much a merge would fix (|A|*|B| pairs), the cheap
# form of expected error reduction (Roy & McCallum, ICML 2001). Measured:
# 73 questions to fix half the splits vs 71 without it, at the same duplicate
# rate - the same speed, and it front-loads the merges that matter.
REVIEW_MMR_IMPACT = _env("REVIEW_MMR_IMPACT", "1") not in ("0", "false", "")
# MMR re-ranks only this many of the top candidates (O(k^2)); the tail stays in
# score order. Large enough that every question a person can reach sits in the
# re-ranked head, small enough that the endpoint answers instantly.
REVIEW_MMR_TOP = int(_env("REVIEW_MMR_TOP", "200"))
# back-view boxes with body cos above this enter merge review too. Derived
# the same way for the body space (OSNet-AIN on Market): different-person
# pairs reach p95=0.634, so the review band starts just under that.
REVIEW_BODY_SIM_MIN = float(_env("REVIEW_BODY_SIM_MIN", "0.63"))
# tiny boxes never claim (garbage body embeddings at that size)
BODY_MIN_REL_H = float(_env("BODY_MIN_REL_H", "0.12"))   # box height / photo height


# --- misc ------------------------------------------------------------------
THUMB_MAX_SIDE = int(_env("THUMB_MAX_SIDE", "512"))
SCAN_PATH = _env("SCAN_PATH", "")                         # optional dir scanned at startup
MAX_UPLOAD_BYTES = int(_env("MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))


def ensure_dirs() -> None:
    for d in (PHOTO_DIR, THUMB_DIR):
        d.mkdir(parents=True, exist_ok=True)
