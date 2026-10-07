"""Virtual group photos: the production regime the plain bake-off missed.

algo_bench gives every image its own photo id, so the co-occurrence hard
rule (two faces of one photo may never land in one cluster) never fires -
yet in a real album most photos are group shots and that rule is the main
guard against forced mixing. This bench builds virtual group photos from
public data:

    600 virtual photos, each holding two faces of DIFFERENT identities,
    pseudo-time shared per photo, identities spread across photos so
    single-linkage still has to hop chains to mix anyone.

Ground truth stays exact (identity labels), so the question "did the
co-occurrence rule / the auto-eject actually reduce mixing?" is answerable
in the same non-symmetric terms as everything else: FM is expensive,
splitting is free.

Usage: python group_bench.py [t_good ...]
"""
import json
import sys

import numpy as np

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/scripts/eval")

T_GOODS = [float(x) for x in sys.argv[1:]] or [0.65]
sys.argv = sys.argv[:1]

from app.ente_core import EnteParams, cluster_faces, complete_link_split
from algo_bench import load_sets, score

FACES_PER_PHOTO = 2


def build_photos(rng, n, lab):
    """Pair up images into virtual photos, always across identities."""
    idx = np.arange(n)
    rng.shuffle(idx)
    pairs, i = [], 0
    while i + 1 < len(idx):
        a, b = int(idx[i]), int(idx[i + 1])
        if lab[a] == lab[b]:                     # same identity in one photo
            swap = next((j for j in range(i + 2, len(idx))
                         if lab[int(idx[j])] != lab[a]), None)
            if swap is None:
                i += 2
                continue
            b = int(idx[swap])
            idx[i + 1], idx[swap] = idx[swap], idx[i + 1]
        pairs.append((a, b))
        i += 2
    photo = np.empty(n, dtype=np.int64)
    for k, (a, b) in enumerate(pairs):
        photo[a] = photo[b] = k
    return photo, len(pairs)


def main():
    sets = load_sets()
    lfw = next(s for s in sets if s[0] == "lfw")
    emb, lab, blur = lfw[2], lfw[3], lfw[5]
    n = len(emb)
    score_ = np.full(n, 0.95)
    sideways = np.zeros(n, dtype=bool)
    G = emb @ emb.T

    photo, nphotos = build_photos(np.random.RandomState(7), n, lab)
    times = photo.astype(np.float64)
    print(json.dumps(dict(n=n, photos=nphotos,
                          identities=int(len(np.unique(lab))))), file=sys.stderr)

    for tg in T_GOODS:
        tb = round(tg + 0.08, 2)
        for one_per, label in ((True, "rule_on"), (False, "rule_off")):
            for bar in (0.0, 0.55):
                cid, _ = cluster_faces(
                    emb, blur, score_, sideways, photo, times,
                    EnteParams(tg, tb), one_per_photo=one_per, gram=G)
                if bar > 0:
                    cid = complete_link_split(emb, cid, bar)
                sc = score(cid, lab)
                print(json.dumps(dict(
                    t_good=tg, rule=label, bar=bar,
                    FM=sc["FM"], recall=round(sc["recall"], 4),
                    impure_units=sc["impure_units"], persons=sc["persons"],
                    unassigned=int((cid < 0).sum()),
                    fmppm=round(sc["fm_rate"] * 1e6, 2))), flush=True)


if __name__ == "__main__":
    main()
