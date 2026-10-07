"""Algorithm bake-off under the PRODUCT objective.

Question: which clustering algorithm groups these identities best when
MIXING is expensive and SPLITTING is free?

Objective (the production preference):
    FM   = different identities sharing a cluster  -> heavy penalty (budget)
    FS   = one identity spread over clusters        -> free
    UN   = unassigned                              -> neutral

Protocol:
    TUNE = LFW sample (used to pick parameters)
    VAL  = the hard sets (CFP / KinFaceW / CALFW / CPLFW / TinyFace),
           sampled fixed-seed, used ONLY to verify generalization
    score = recall (same-identity pairs clustered together) maximized
            subject to FM_rate <= budget on TUNE

Quality gates are production-consistent: blur is the Laplacian variance of
the same 112x112 aligned crop face.py computes, so Ente's isBadFace /
clusterable gates behave here exactly as they do live.

Algorithms:
    ente_greedy    - the shipped core (app.ente_core.cluster_faces),
                     exact production code incl. its asymmetric
                     neighbor-quality threshold
    linkage:complete|average|single - scipy agglomerative on a
                     quality-filtered similarity matrix (edge i-j exists
                     only if sim >= max(th_i, th_j))
    cw             - Chinese Whispers label propagation on the same graph

Usage: python algo_bench.py <model> [--budget PPM] [--per-set]
"""
import json
import sys

import cv2
import numpy as np

sys.path.insert(0, "/app")

from app.ente_core import EnteParams, cluster_faces

MODEL = sys.argv[1] if len(sys.argv) > 1 else "mobilefacenet_w600k"
SHOW_SET = "--per-set" in sys.argv
if "--budget" in sys.argv:
    BUDGET = float(sys.argv[sys.argv.index("--budget") + 1]) / 1e6
else:
    BUDGET = 0.005          # <=0.5% of different-identity pairs may mix

T_GOOD = 0.65               # current production line (calibrated earlier)
T_BAD = 0.73
T_GAP = 0.08                # Ente's own gap
EJECT = 0.55                # current production auto-split bar
VAL_PER_SET = 700           # fixed-seed sample per hard set


# ------------------------------------------------------------------ data ---
def identity_of(path: str) -> str:
    import os
    p = path.replace("\\", "/")
    name = os.path.basename(p)
    if "/cfp/" in p:
        return p.split("/")[-3]
    if "/kinfacew/" in p:
        return name.rsplit("_", 1)[0]
    if "/tinyface/" in p:
        return name.split("_")[0]
    return name.rsplit("_", 2)[0] if name.count("_") >= 2 else name.rsplit("_", 1)[0]


def production_quality(crops: np.ndarray) -> np.ndarray:
    """blur = Laplacian variance on the aligned crop (face.py's exact
    production formula), so is_bad_face() here behaves like it does live."""
    out = np.empty(len(crops), dtype=np.float32)
    for i, c in enumerate(crops):
        g = cv2.cvtColor(c, cv2.COLOR_RGB2GRAY) if c.ndim == 3 and c.shape[2] == 3 \
            else c
        out[i] = cv2.Laplacian(g, cv2.CV_64F).var()
    return out


def load_sets():
    """TUNE = full LFW sample; VAL = sampled hard sets. Returns list of
    (set_name, split, emb, labels, photo, blur, score, sideways)."""
    sets = []
    m = np.load("/s2/model_bench_crops.npz", allow_pickle=True)
    lwf = np.load(f"/s2/memb_{MODEL}.npy")
    sets.append(("lfw", "tune", lwf, np.array([str(x) for x in m["label"]]),
                 np.arange(len(lwf)), production_quality(m["crops"])))

    h = np.load("/s2/hard_crops.npz", allow_pickle=True)
    hemb = np.load(f"/s2/hemb_{MODEL}.npy")
    paths = [str(x) for x in h["paths"]]
    groups = {}
    for i, p in enumerate(paths):
        key = ("kinfacew" if "/kinfacew/" in p else "cfp" if "/cfp/" in p else
               "calfw" if "/calfw/" in p else "cplfw" if "/cplfw/" in p else "tinyface")
        groups.setdefault(key, []).append(i)
    rng = np.random.RandomState(42)
    for key, idx in sorted(groups.items()):
        idx = np.array(idx)
        if len(idx) > VAL_PER_SET:
            idx = np.sort(rng.choice(idx, VAL_PER_SET, replace=False))
        sets.append((key, "val", hemb[idx],
                     np.array([identity_of(paths[i]) for i in idx]),
                     np.arange(len(idx)), production_quality(h["crops"][idx])))
    return sets


# -------------------------------------------------------------- scoring ---
def score(cid: np.ndarray, lab: np.ndarray):
    """O(n) asymmetric scoring. Returns dict of raw counts + rates."""
    codes, inv = np.unique(lab, return_inverse=True)
    K = len(codes)
    valid = cid >= 0
    n = len(cid)

    # same-identity pair totals
    _, allc = np.unique(inv, return_counts=True)
    same_pairs = int((allc * (allc - 1) // 2).sum())
    total_pairs = n * (n - 1) // 2
    diff_pairs = total_pairs - same_pairs

    if valid.any():
        v = inv[valid]
        code = cid[valid].astype(np.int64) * K + v
        _, in_cnt = np.unique(code, return_counts=True)
        same_in_cluster = int((in_cnt * (in_cnt - 1) // 2).sum())
        _, cl_cnt = np.unique(cid[valid], return_counts=True)
        any_in_cluster = int((cl_cnt * (cl_cnt - 1) // 2).sum())
        FM = any_in_cluster - same_in_cluster
        HIT = same_in_cluster
        _, ac = np.unique(v, return_counts=True)
        both_assigned_same = int((ac * (ac - 1) // 2).sum())
    else:
        FM = HIT = both_assigned_same = 0
    FS = both_assigned_same - HIT
    UN = same_pairs - both_assigned_same
    persons = len(np.unique(cid[valid])) if valid.any() else 0

    # visible mixing: units living in an impure cluster (a cluster holding
    # >=2 identities). This is what the eye actually sees - a pairwise FM
    # rate understates it because one chain makes ONE big messy person.
    impure_units = impure_clusters = worst_minority = 0
    if valid.any():
        v = inv[valid]
        code = cid[valid].astype(np.int64) * K + v
        u, cnt = np.unique(code, return_counts=True)          # (cl,label) counts
        cl_of = u // K
        uniq_cl, cl_cnt = np.unique(cid[valid], return_counts=True)
        starts = np.flatnonzero(np.r_[True, cl_of[1:] != cl_of[:-1]])
        distinct = np.diff(np.r_[starts, len(u)])
        max_same = np.maximum.reduceat(cnt, starts)
        impure = distinct > 1
        impure_clusters = int(impure.sum())
        impure_units = int(cl_cnt[impure].sum())
        if impure.any():
            worst_minority = int((cl_cnt - max_same)[impure].max())

    clustered = int(valid.sum())
    return dict(FM=FM, HIT=HIT, FS=FS, UN=UN, same=same_pairs, diff=diff_pairs,
                persons=persons,
                fm_rate=FM / diff_pairs if diff_pairs else 0.0,
                recall=HIT / same_pairs if same_pairs else 0.0,
                impure_units=impure_units, impure_clusters=impure_clusters,
                worst_minority=worst_minority, clustered=clustered)


# ------------------------------------------------------------ algorithms ---
def edge_graph(emb, bad, t_good, t_bad):
    """Quality-filtered similarity: edge exists only if sim >= max(th_i, th_j).
    Symmetric generalization of Ente's neighbor-only rule (order-free)."""
    G = emb @ emb.T
    th = np.where(bad, t_bad, t_good)
    need = np.maximum(th[:, None], th[None, :])
    np.fill_diagonal(need, 0.0)
    return G, G >= need


def run_ente(emb, blur, score_, sideways, photo, times, t_good, t_bad, eject):
    cid, _ = cluster_faces(emb, blur, score_, sideways, photo, times,
                           EnteParams(t_good, t_bad), gram=emb @ emb.T)
    if eject and eject > 0:
        from app.ente_core import complete_link_split
        cid = complete_link_split(emb, cid, eject)
    return cid


def run_linkage(emb, bad, t_good, t_bad, method):
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    G, ok = edge_graph(emb, bad, t_good, t_bad)
    D = 1.0 - np.where(ok, G, -1.0)
    np.fill_diagonal(D, 0.0)
    D = np.clip(D, 0.0, 2.0)
    D = (D + D.T) / 2.0
    condensed = squareform(D, checks=False)
    Z = linkage(condensed, method=method)
    return fcluster(Z, t=1.0 - t_good, criterion="distance").astype(np.int64) - 1


def run_mutual_knn(emb, bad, t_good, t_bad, k=8):
    """Single-linkage on a MUTUAL-kNN graph.

    Chaining happens when a weak one-sided edge bridges two clusters.
    Requiring the edge to be among BOTH endpoints' k nearest neighbours
    (and above the quality bar) removes exactly those weak bridges while
    keeping the recall single-linkage is good at.
    """
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse import csr_matrix
    G, ok = edge_graph(emb, bad, t_good, t_bad)
    n = len(emb)
    order = np.argsort(-G, axis=1)                 # nearest first
    mutual = np.zeros((n, n), dtype=bool)
    knn = order[:, :k]
    for i in range(n):
        for j in knn[i]:
            if i in knn[j] and ok[i, j]:
                mutual[i, j] = True
    np.fill_diagonal(mutual, True)
    _, labels = connected_components(csr_matrix(mutual), directed=False)
    return labels.astype(np.int64)


def run_ente_agg(emb, blur, score_, sideways, photo, times, t_good, t_bad,
                 agg="max", knee=None):
    """Faithful copy of the production loop with one replaceable piece:
    HOW a candidate cluster is scored against a joining face.

    agg="max"    - best single member sim  (== the shipped rule; used as
                   the equivalence check against app.ente_core)
    agg="mean"   - average similarity to the whole cluster
    agg="median" - median similarity to the whole cluster
    knee=(K, d)  - a cluster of >=K members demands t_good + d, so a
                   foreign face cannot slip into an already well-formed
                   group (the exact failure the gallery shows)
    Quality-adaptive per-neighbor threshold is kept for agg="max" only;
    the aggregate variants use one uniform line so the rule is what is
    being compared.
    """
    from app.ente_core import is_bad_face, is_clusterable
    n = len(emb)
    bad = np.array([is_bad_face(blur[i], score_[i], sideways[i]) for i in range(n)])
    keep = np.array([is_clusterable(blur[i], score_[i]) for i in range(n)])
    G = emb @ emb.T
    assigned = np.full(n, -1, dtype=np.int64)
    next_id = 0
    cluster_photos: dict[int, set] = {}
    size: dict[int, int] = {}
    order = [i for i in range(n) if keep[i]]
    order.sort(key=lambda i: (-times[i], i))
    for r, i in enumerate(order):
        if assigned[i] >= 0:
            continue
        cand: dict[int, list[float]] = {}
        for j in order[:r]:
            cid = assigned[j]
            if cid < 0:
                continue
            s = float(G[i, j])
            th = t_bad if bad[j] else t_good
            if agg == "max" and s < th:
                continue
            if photo[i] in cluster_photos.get(cid, ()):
                continue
            cand.setdefault(cid, []).append(s)
        best_cid, best_v = -1, -1.0
        for cid, vs in cand.items():
            v = np.asarray(vs, dtype=np.float32)
            a = float(v.max()) if agg == "max" else (
                float(v.mean()) if agg == "mean" else float(np.median(v)))
            th_eff = t_good
            if agg != "max":
                # uniform line for the aggregate variants
                pass
            if knee is not None and size.get(cid, 0) >= knee[0]:
                th_eff = t_good + knee[1]
            if a >= th_eff and a > best_v:
                best_v, best_cid = a, cid
        if best_cid >= 0:
            assigned[i] = best_cid
            size[best_cid] = size.get(best_cid, 0) + 1
            cluster_photos.setdefault(best_cid, set()).add(photo[i])
        else:
            assigned[i] = next_id
            size[next_id] = 1
            cluster_photos[next_id] = {photo[i]}
            next_id += 1
    return assigned, next_id


def run_hdbscan(emb, bad, t_good, mcs, ms):
    """HDBSCAN - the standard modern density clustering choice.

    Edges below the quality bar are removed first (same graph the other
    density methods get) so the comparison is about the algorithm, not
    about who applies the threshold. Noise points come back unassigned,
    which is the honest treatment for a density method.
    """
    from sklearn.cluster import HDBSCAN
    _, ok = edge_graph(emb, bad, t_good, t_good + T_GAP)
    G = emb @ emb.T
    D = 1.0 - np.where(ok, G, -1.0)
    np.fill_diagonal(D, 0.0)
    D = np.clip((D + D.T) / 2.0, 0.0, 2.0)
    lab = HDBSCAN(min_cluster_size=mcs, min_samples=ms,
                  metric="precomputed").fit_predict(D)
    out = np.full(len(emb), -1, dtype=np.int64)
    for k, c in enumerate(np.unique(lab)):
        if c >= 0:
            out[lab == c] = k
    return out


def run_cw(emb, bad, t_good, t_bad, iters=20, seed=0):
    """Chinese Whispers: seeded iterative label propagation."""
    G, ok = edge_graph(emb, bad, t_good, t_bad)
    np.fill_diagonal(ok, False)
    n = len(emb)
    W = np.where(ok, G, 0.0)
    rng = np.random.RandomState(seed)
    lab = np.arange(n)
    order = np.arange(n)
    for _ in range(iters):
        rng.shuffle(order)
        for i in order:
            w = W[i]
            if not w.any():
                continue
            js = np.nonzero(w)[0]
            wl = w[js]
            # sum weights per candidate label
            cand = lab[js]
            ulab, inv = np.unique(cand, return_inverse=True)
            sums = np.bincount(inv, weights=wl)
            best = ulab[int(np.argmax(sums))]
            if sums.max() <= 0:
                continue
            lab[i] = best
    out = np.full(n, -1, dtype=np.int64)
    for k, c in enumerate(np.unique(lab)):
        out[lab == c] = k
    return out


# ------------------------------------------------------------------ run ---
def build(cfg):
    """cfg = (algo, t_good, t_bad, extra).

    sets tuple layout: (name, split, emb, lab, photo, blur)"""
    algo, tg, tb, extra = cfg
    if algo == "ente_greedy":
        # pseudo-time = image index (each image is its own photo here, so
        # the co-occurrence rule never fires and time order is arbitrary)
        return lambda s: run_ente(s[2], s[5], np.full(len(s[2]), 0.95),
                                  np.zeros(len(s[2]), dtype=bool), s[4],
                                  s[4].astype(np.float64), tg, tb, extra)
    if algo.startswith("linkage:"):
        return lambda s: run_linkage(s[2], _bad(s[5]), tg, tb, algo.split(":")[1])
    if algo == "hdbscan":
        # min_cluster_size / min_samples packed as "mcs,ms"
        mcs, ms = (int(x) for x in extra.split(","))
        return lambda s: run_hdbscan(s[2], _bad(s[5]), tg, mcs, ms)
    if algo == "mutual_knn":
        return lambda s: run_mutual_knn(s[2], _bad(s[5]), tg, tb, k=int(extra))
    if algo == "ente_agg":
        # extra = "agg[:K,delta]"
        agg, _, kn = extra.partition(":")
        knee = (int(kn.split(",")[0]), float(kn.split(",")[1])) if kn else None
        return lambda s: run_ente_agg(s[2], s[5], np.full(len(s[2]), 0.95),
                                      np.zeros(len(s[2]), dtype=bool), s[4],
                                      s[4].astype(np.float64), tg, tb,
                                      agg=agg, knee=knee)[0]
    if algo == "cw":
        return lambda s: run_cw(s[2], _bad(s[5]), tg, tb, iters=int(extra))
    raise ValueError(algo)


def _bad(blur):
    from app.ente_core import is_bad_face
    return np.array([is_bad_face(b, 0.95, False) for b in blur])


def main():
    sets = load_sets()
    print(json.dumps(dict(model=MODEL, budget_ppm=BUDGET * 1e6,
                          sets={f"{n}({sp})": int(len(l))
                                for n, sp, e, l, p, b in sets})), file=sys.stderr)

    GRID = [round(x, 2) for x in np.arange(0.40, 0.81, 0.05)]

    # current production baseline first (for reference), then the eject
    # A/B (same t, bar off), then every algorithm on the same grid
    configs = [("ente_greedy", T_GOOD, T_BAD, EJECT)]
    for tg in (0.55, 0.60, 0.65, 0.70):
        configs.append(("ente_greedy", tg, round(tg + T_GAP, 2), 0))
    for tg in GRID:
        configs.append(("ente_greedy", tg, round(tg + T_GAP, 2), EJECT))
        configs.append(("linkage:complete", tg, round(tg + T_GAP, 2), 0))
        configs.append(("linkage:average", tg, round(tg + T_GAP, 2), 0))
        configs.append(("linkage:single", tg, round(tg + T_GAP, 2), 0))
        configs.append(("cw", tg, round(tg + T_GAP, 2), 20))
        configs.append(("mutual_knn", tg, round(tg + T_GAP, 2), 8))
        for mcs in (5, 10):
            for ms in (1, 5):
                configs.append(("hdbscan", tg, round(tg + T_GAP, 2), f"{mcs},{ms}"))
        if tg in (0.50, 0.55, 0.60, 0.65, 0.70):
            configs.append(("ente_agg", tg, round(tg + T_GAP, 2), "median"))
            configs.append(("ente_agg", tg, round(tg + T_GAP, 2), "mean"))
            configs.append(("ente_agg", tg, round(tg + T_GAP, 2), "max:10,0.05"))
            configs.append(("ente_agg", tg, round(tg + T_GAP, 2), "max:10,0.10"))

    if "--algos" in sys.argv:
        want = set(sys.argv[sys.argv.index("--algos") + 1].split(","))
        configs = [c for c in configs
                   if any(c[0] == w or c[0].startswith(w + ":") for w in want)]

    # "sel" = selection sets (LFW + CFP/CALFW/CPLFW/TinyFace); kinfacew is
    # held OUT of selection as the pure look-alike impostor stressor.
    for cfg in configs:
        fn = build(cfg)
        agg = {"sel": dict(FM=0, HIT=0, same=0, diff=0, iu=0, cl=0),
               "hold": dict(FM=0, HIT=0, same=0, diff=0, iu=0, cl=0)}
        per = {}
        for name, split, emb, lab, photo, blur in sets:
            cid = fn((name, split, emb, lab, photo, blur))
            sc = score(cid, lab)
            grp = "hold" if name == "kinfacew" else "sel"
            a = agg[grp]
            a["FM"] += sc["FM"]; a["HIT"] += sc["HIT"]
            a["same"] += sc["same"]; a["diff"] += sc["diff"]
            a["iu"] += sc["impure_units"]; a["cl"] += sc["clustered"]
            if SHOW_SET:
                per[name] = dict(FM=sc["FM"], recall=round(sc["recall"], 4),
                                 persons=sc["persons"],
                                 impure=sc["impure_units"],
                                 worst=sc["worst_minority"])
        out = dict(algo=cfg[0], t_good=cfg[1], t_bad=cfg[2], extra=cfg[3])
        for grp in ("sel", "hold"):
            a = agg[grp]
            out[f"{grp}_fmppm"] = round(a["FM"] / a["diff"] * 1e6, 2)
            out[f"{grp}_recall"] = round(a["HIT"] / a["same"], 4) if a["same"] else None
            out[f"{grp}_impure%"] = round(100 * a["iu"] / max(a["cl"], 1), 2)
        if SHOW_SET:
            out["per"] = per
        print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
