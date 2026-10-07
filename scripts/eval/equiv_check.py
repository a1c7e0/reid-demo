"""Equivalence check: run_ente_agg(agg='max', knee=None) must reproduce
app.ente_core.cluster_faces exactly (same clusters, same ids)."""
import sys
import numpy as np

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/scripts/eval")
from app.ente_core import EnteParams, cluster_faces
from algo_bench import run_ente_agg, load_sets, T_GOOD, T_BAD

sets = load_sets()
emb, blur = sets[0][2], sets[0][5]
n = len(emb)
score_ = np.full(n, 0.95)
sideways = np.zeros(n, dtype=bool)
photo = np.arange(n)
times = photo.astype(np.float64)

ref, _ = cluster_faces(emb, blur, score_, sideways, photo, times,
                       EnteParams(T_GOOD, T_BAD), gram=emb @ emb.T)
mine, _ = run_ente_agg(emb, blur, score_, sideways, photo, times,
                       T_GOOD, T_BAD, agg="max", knee=None)

# compare as partitions (ids may differ in numbering)
def part(c):
    d = {}
    for i, x in enumerate(c):
        if x >= 0:
            d.setdefault(int(x), set()).add(i)
    return sorted(tuple(sorted(v)) for v in d.values())

p_ref, p_mine = part(ref), part(mine)
same = p_ref == p_mine
print(f"ref clusters={len(p_ref)} mine={len(p_mine)} identical_partition={same}")
if not same:
    only_ref = [c for c in p_ref if c not in p_mine][:3]
    only_mine = [c for c in p_mine if c not in p_ref][:3]
    print("only in ref (first 3):", only_ref)
    print("only in mine (first 3):", only_mine)
print("OK" if same else "MISMATCH")
