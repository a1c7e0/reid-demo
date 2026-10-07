"""Eject bar sweep (public data only).

PERSON_EJECT_BAR auto-splits a cluster member whose minimum similarity to
its co-members falls below the bar - the anti-chain layer. The bake-off
showed the shipped 0.55 costs about half the recall at unchanged purity;
this sweep asks whether a GENTLER bar keeps any of that value.

Usage: python eject_bench.py [t_good ...]
"""
import json
import sys

import numpy as np

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/scripts/eval")

# parse our own args first and reset argv: algo_bench takes sys.argv[1]
# as the MODEL name, and these are thresholds, not a model
#   python eject_bench.py [--model NAME] [t_good ...]
MODEL = "mobilefacenet_w600k"
if "--model" in sys.argv:
    i = sys.argv.index("--model")
    MODEL = sys.argv[i + 1]
    del sys.argv[i:i + 2]
T_GOODS = [float(x) for x in sys.argv[1:]] or [0.60, 0.65]
sys.argv = ["eject_bench.py", MODEL]

from app.ente_core import complete_link_split
from algo_bench import load_sets, score, run_ente, T_GAP
BARS = [0.0, 0.35, 0.45, 0.55, 0.65]

sets = load_sets()
sel = [s for s in sets if s[0] != "kinfacew"]
hold = [s for s in sets if s[0] == "kinfacew"]


for tg in T_GOODS:
    tb = round(tg + T_GAP, 2)
    for bar in BARS:
        a = dict(FM=0, HIT=0, same=0, diff=0, iu=0, cl=0)
        h = dict(FM=0, HIT=0, same=0, diff=0, iu=0, cl=0)
        for s in sets:
            cid = run_ente(s[2], s[5], np.full(len(s[2]), 0.95),
                           np.zeros(len(s[2]), dtype=bool), s[4],
                           s[4].astype(np.float64), tg, tb, 0)
            if bar > 0:
                cid = complete_link_split(s[2], cid, bar)
            sc = score(cid, s[3])
            t = h if s[0] == "kinfacew" else a
            t["FM"] += sc["FM"]; t["HIT"] += sc["HIT"]
            t["same"] += sc["same"]; t["diff"] += sc["diff"]
            t["iu"] += sc["impure_units"]; t["cl"] += sc["clustered"]
        print(json.dumps(dict(
            model=MODEL, t_good=tg, t_bad=tb, bar=bar,
            sel_fmppm=round(a["FM"] / a["diff"] * 1e6, 2),
            sel_recall=round(a["HIT"] / a["same"], 4),
            sel_impure=round(100 * a["iu"] / max(a["cl"], 1), 2),
            hold_fmppm=round(h["FM"] / h["diff"] * 1e6, 2),
            hold_impure=round(100 * h["iu"] / max(h["cl"], 1), 2))), flush=True)
