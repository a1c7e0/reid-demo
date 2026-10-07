"""Debug HDBSCAN label handling: cluster sizes vs the metrics score()."""
import sys

import numpy as np

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/scripts/eval")

from sklearn.cluster import HDBSCAN
from algo_bench import load_sets, score, edge_graph, T_GAP

lfw = next(s for s in load_sets() if s[0] == "lfw")
emb, lab, blur = lfw[2], lfw[3], lfw[5]
bad = np.array([False] * len(emb))
_, ok = edge_graph(emb, bad, 0.65, 0.73)
G = emb @ emb.T
D = 1.0 - np.where(ok, G, -1.0)
np.fill_diagonal(D, 0.0)
D = np.clip((D + D.T) / 2.0, 0.0, 2.0)

raw = HDBSCAN(min_cluster_size=10, min_samples=5,
              metric="precomputed").fit_predict(D)
print("dtype:", raw.dtype, "unique:", np.unique(raw)[:12],
      "n_unique:", len(np.unique(raw)))
noise = int((raw < 0).sum())
sizes = {c: int((raw == c).sum()) for c in np.unique(raw) if c >= 0}
big = sorted(sizes.items(), key=lambda kv: -kv[1])[:6]
print(f"noise={noise} clusters={len(sizes)} biggest={big}")

out = np.full(len(emb), -1, dtype=np.int64)
for k, c in enumerate(np.unique(raw)):
    if c >= 0:
        out[raw == c] = k
sc = score(out, lab)
print("score():", sc)
print("sizes sum:", sum(sizes.values()), "len emb:", len(emb))
