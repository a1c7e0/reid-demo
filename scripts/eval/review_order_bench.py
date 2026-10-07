"""Which review ordering gets the clustering right with the fewest human answers?

A review queue is a stream of questions of the form "are these two clusters the
same person?". Answering one costs human attention, so the ordering decides how
fast the clustering gets good. The user's complaint is precise: after rejecting
b, the next item is c - and c is essentially b again (one person split into
fragments), so the same question gets asked twice before anything new appears.

This simulates a human on LFW ground truth (311 identities, 2339 faces, the
shipped model through the production FaceEngine path) and compares orderings by
two numbers:

  * questions to fix 90% of the split pairs   (fewer = faster to a good result)
  * redundant questions asked                 (fewer = less "same question twice")

Orderings under test are all published, citable rules - no invented heuristics:

  score      relevance only, composite score descending (what the app does now)
  size       expected error reduction: score weighted by the pairs a merge fixes
  mmr03/05/07 Maximal Marginal Relevance (Carbonell & Goldstein 1998):
             lambda*relevance - (1-lambda)*max_sim_to_already_shown
  farthest   pure diversity: least similar to anything already shown
  mmr_size   MMR with the size-weighted relevance term

    python scripts/eval/review_order_bench.py
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path("/home/a1c7e0/workspace/reid-gallery")
sys.path.insert(0, str(ROOT))
from app import ente_core as core  # noqa: E402

NPZ = ROOT / "testsets/out/fragment_lfw.npz"


def centroid_map(clu, emb):
    out = {}
    for c in np.unique(clu):
        m = emb[clu == c].mean(axis=0)
        n = float(np.linalg.norm(m))
        out[int(c)] = m / max(n, 1e-9)
    return out


def _split_pairs(clu, lab):
    """Same-person pairs still sitting in different clusters."""
    iu = np.triu_indices(len(clu), 1)
    sl = lab[iu[0]] == lab[iu[1]]
    sc = clu[iu[0]] == clu[iu[1]]
    return int((sl & ~sc).sum())


def build_queue(clu, emb, lab, band=(0.53, 0.90)):
    """Candidate cluster pairs in the uncertain band, with the relevance score
    and the size-weighted variant (expected error reduction: merging A and B
    fixes |A|*|B| pairs at once)."""
    cent = centroid_map(clu, emb)
    cs = {}
    for c in np.unique(clu):
        cs[int(c)] = int((clu == c).sum())
    ids = sorted(cent)
    q = []
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            s = float(np.dot(cent[a], cent[b]))
            if s < band[0] or s > band[1]:
                continue
            q.append({
                "a": a, "b": b, "sim": s,
                "impact": cs[a] * cs[b],      # pairs fixed if they are one person
                "va": cent[a], "vb": cent[b],
            })
    for it in q:
        it["size_score"] = it["sim"] * np.log1p(it["impact"])
    return q


def order_queue(q, rule, shown_vecs):
    if rule == "score":
        return sorted(q, key=lambda x: -x["sim"])
    if rule == "size":
        return sorted(q, key=lambda x: -x["size_score"])
    if rule == "farthest":
        return sorted(q, key=lambda x: _diversity(x, shown_vecs))
    if rule.startswith("mmr") or rule.startswith("smmr"):
        # mmrNN  = lambda*sim            - (1-lambda)*max_sim_to_shown
        # smmrNN = lambda*sim*log1p(impact) - (1-lambda)*max_sim_to_shown
        use_size = rule.startswith("smmr")
        lam = float(rule[4 if use_size else 3:]) / 10.0
        key = "size_score" if use_size else "sim"

        def mmr(x):
            return lam * x[key] - (1 - lam) * _diversity_pen(x, shown_vecs)

        return sorted(q, key=lambda x: -mmr(x))
    raise ValueError(rule)


def _shown_max(x, shown):
    if not shown:
        return 0.0
    best = -1.0
    for v in shown:
        s = max(float(np.dot(x["va"], v)), float(np.dot(x["vb"], v)))
        best = max(best, s)
    return best


def _diversity(x, shown):
    return -_shown_max(x, shown)        # most diverse first


def _diversity_pen(x, shown):
    return _shown_max(x, shown)


def simulate(clu0, emb, lab, rule, cap=400):
    clu = clu0.copy()
    sizes = {}
    for c in np.unique(clu):
        sizes[int(c)] = int((clu == c).sum())
    shown_vecs = []
    asked = 0
    redundant = 0
    fixed = 0
    base = _split_pairs(clu, lab)
    curve = []
    answered = []                     # rejected centroids, for the suppression
    while asked < cap:
        q = build_queue(clu, emb, lab)
        q = [x for x in q if not _is_redundant(x, answered)]
        if not q:
            break
        ordered = order_queue(q, rule, shown_vecs)
        pick = ordered[0]
        # ground truth: one person iff every face in the two clusters shares it
        ids = np.unique(np.concatenate([lab[clu == pick["a"]], lab[clu == pick["b"]]]))
        same = len(ids) == 1
        shown_vecs.append((pick["va"] + pick["vb"]) / 2.0)
        if _shown_max(pick, shown_vecs[:-1]) >= 0.72:
            redundant += 1
        asked += 1
        if same:
            before = _split_pairs(clu, lab)
            clu[clu == pick["b"]] = pick["a"]
            fixed += before - _split_pairs(clu, lab)
        else:
            answered.append(pick["vb"])
        curve.append(_split_pairs(clu, lab))
        if curve[-1] <= 0.10 * base:
            break
    return asked, redundant, base, curve


def _is_redundant(x, answered):
    for v in answered:
        if float(np.dot(x["va"], v)) >= 0.72 or float(np.dot(x["vb"], v)) >= 0.72:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=300)
    ap.add_argument("--n", type=int, default=700)
    args = ap.parse_args()
    z = np.load(NPZ, allow_pickle=True)
    emb = z["emb"].astype(np.float32)
    lab = z["lab"].astype(object)
    if args.n and args.n < len(emb):
        # keep whole identities so the ground truth stays meaningful
        want, keep = args.n, []
        seen = {}
        for i, l in enumerate(lab):
            if seen.get(l, 0) < 6:
                seen[l] = seen.get(l, 0) + 1
                keep.append(i)
            if len(keep) >= want:
                break
        emb, lab = emb[keep], lab[keep]
    print(f"# {len(emb)} faces, {len(set(lab))} identities")
    print("# 'questions' to reach 90% of split pairs fixed; 'dup' = questions that")
    print("# repeated a near-duplicate of something already answered")
    print("#")
    print(f"{'rule':>12} {'questions':>10} {'dup':>6} {'dup%':>7} "
          f"{'fix50%':>7} {'fix25%':>7} {'fix10%':>7}")
    for rule in ("score", "size", "farthest", "mmr03", "mmr05", "mmr07",
                 "smmr03", "smmr05", "smmr07"):
        clu0, _ = core.cluster_faces(
            emb, np.full(len(emb), 150.0, np.float32), np.full(len(emb), 0.99, np.float32),
            np.zeros(len(emb), np.float32), np.array([f"s{i}" for i in range(len(emb))]),
            np.zeros(len(emb), np.float32), core.EnteParams(0.64, 0.72),
            one_per_photo=True, gram=emb @ emb.T)
        asked, dup, base, curve = simulate(np.asarray(clu0), emb, lab, rule, args.cap)

        def q_at(frac):
            tgt = frac * base
            for i, v in enumerate(curve):
                if v <= tgt:
                    return i + 1
            return len(curve)

        pct = 100.0 * dup / max(asked, 1)
        print(f"{rule:>12} {asked:>10} {dup:>6} {pct:>6.1f}% "
              f"{q_at(0.50):>6} {q_at(0.75):>6} {q_at(0.90):>6}")


if __name__ == "__main__":
    main()
