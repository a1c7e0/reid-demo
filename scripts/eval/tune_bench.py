"""Re-derive every similarity knob for the CURRENT embedder.

The face model moved to LVFace-T, and a similarity threshold does not
transfer between embedding spaces. This sweeps the parameters that decide
identity, against the product objective - **never mix, group as much as
possible** - not a proxy metric:

  * t_good   the join bar for a good-quality neighbour
  * gap      t_bad - t_good, the extra bar demanded next to a bad-quality
             neighbour (blur / low score / sideways). Ente's own 0.08 came
             from THEIR embedder and has never been re-derived here.
  * the review band (REVIEW_SIM_MIN): which similarity region is genuinely
     uncertain and therefore worth showing a human. Derived from where the
     same-person and different-person pair distributions overlap.

Selection rule, in this order:
  1. look-alike mixing (KinFaceW, impostor-only) must not exceed what the
     shipped line produced
  2. impure units on the selection sets must not exceed the shipped line
  3. among what is left, maximise grouping (same-person pairs clustered)

Usage: python tune_bench.py [model]
"""
import json
import sys

import numpy as np

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/scripts/eval")

import algo_bench as ab  # noqa: E402
from algo_bench import run_ente, score  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else "lvface_t"
ab.MODEL = MODEL

T_GOODS = [0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67]
GAPS = [0.0, 0.04, 0.08, 0.12, 0.16]


def main() -> None:
    sets = ab.load_sets()
    prepared = []
    for name, _split, emb, lab, photo, blur in sets:
        prepared.append((name, emb, lab, photo, blur))

    rows = []
    for tg in T_GOODS:
        for gap in GAPS:
            sel = dict(FM=0, HIT=0, same=0, diff=0, iu=0, cl=0)
            hold = dict(FM=0, HIT=0, same=0, diff=0, iu=0, cl=0)
            for name, emb, lab, photo, blur in prepared:
                cid = run_ente(emb, blur, np.full(len(emb), 0.95),
                               np.zeros(len(emb), dtype=bool), photo,
                               photo.astype(float), tg, round(tg + gap, 2), 0)
                sc = score(cid, lab)
                d = hold if name == "kinfacew" else sel
                d["FM"] += sc["FM"]; d["HIT"] += sc["HIT"]
                d["same"] += sc["same"]; d["diff"] += sc["diff"]
                d["iu"] += sc["impure_units"]; d["cl"] += sc["clustered"]
            rows.append(dict(
                model=MODEL, t_good=tg, gap=gap, t_bad=round(tg + gap, 2),
                sel_imp=round(100 * sel["iu"] / max(sel["cl"], 1), 3),
                sel_rec=round(sel["HIT"] / max(sel["same"], 1), 4),
                sel_fmppm=round(sel["FM"] / max(sel["diff"], 1) * 1e6, 2),
                hold_imp=round(100 * hold["iu"] / max(hold["cl"], 1), 3),
                hold_fm=hold["FM"]))
            print(json.dumps(rows[-1]), flush=True)

    # ---- selection ----------------------------------------------------
    # reference = the line the app runs right now
    ref = next((r for r in rows if r["t_good"] == 0.64 and r["gap"] == 0.08),
               None)
    print("\n== reference (shipped now) ==", json.dumps(ref))
    feas = [r for r in rows
            if r["hold_imp"] <= ref["hold_imp"] and r["sel_imp"] <= ref["sel_imp"]]
    feas.sort(key=lambda r: -r["sel_rec"])
    print(f"\n== inside the shipped purity (hold_imp<={ref['hold_imp']}%, "
          f"sel_imp<={ref['sel_imp']}%) ==  n={len(feas)}")
    for r in feas[:8]:
        print(f"   t_good={r['t_good']} gap={r['gap']:<5} sel_rec={r['sel_rec']:<7} "
              f"sel_imp={r['sel_imp']}% hold_imp={r['hold_imp']}% hold_fm={r['hold_fm']}")

    # ---- review band --------------------------------------------------
    lfw = next(s for s in sets if s[0] == "lfw")
    emb, lab = lfw[2], lfw[3]
    G = emb @ emb.T
    codes, inv = np.unique(lab, return_inverse=True)
    iu = np.triu_indices(len(emb), k=1)
    same_mask = (inv[iu[0]] == inv[iu[1]])
    sims = G[iu]
    same, diff = sims[same_mask], sims[~same_mask]
    print(f"\n== pair similarity distribution (LFW, {len(same)} same / "
          f"{len(diff)} different) ==")
    for name, arr in (("same", same), ("different", diff)):
        q = np.percentile(arr, [1, 5, 25, 50, 75, 95, 99])
        print(f"   {name:10} p1={q[0]:.3f} p5={q[1]:.3f} p25={q[2]:.3f} "
              f"p50={q[3]:.3f} p75={q[4]:.3f} p95={q[5]:.3f} p99={q[6]:.3f}")
    overlap_lo = float(np.percentile(diff, 99))
    overlap_hi = float(np.percentile(same, 25))
    print(f"   overlap band (diff p99 .. same p25) = "
          f"[{overlap_lo:.3f}, {overlap_hi:.3f}]  -> REVIEW_SIM_MIN candidate")


if __name__ == "__main__":
    main()
