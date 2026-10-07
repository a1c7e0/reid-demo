"""Model comparison at MATCHED purity (the fair question).

A single absolute threshold is not equally strict for models with different
score distributions - at 0.65 one model may be far more permissive than
another. This sweeps each model's threshold and reports the best recall it
reaches while staying inside the same false-merge budget, so the comparison
is about how much correct grouping each model buys per unit of mixing.

Budget: FM (different identities sharing a cluster) <= budget_lfw on LFW
and <= budget_hard across the four non-hold hard sets; KinFaceW (impostor
only) is reported separately as the look-alike stressor.

Usage: python model_matched.py [--budget 5] [--hard-budget 20] [model ...]
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/scripts/eval")

import algo_bench as ab  # noqa: E402
from algo_bench import run_ente, score  # noqa: E402

SIZES_MB = {"adaface_irv1": 93.96, "arcface_r50": 174.38, "lvface_t": 76.65,
            "mobilefacenet_w600k": 13.62, "sface_fp32": 38.70,
            "sface_int8": 9.90}
MS_PER_IMG = {"adaface_irv1": 97.1, "arcface_r50": 656.1, "lvface_t": 115.9,
              "mobilefacenet_w600k": 56.2, "sface_fp32": 95.6,
              "sface_int8": 244.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=5,
                    help="max LFW false-merge pairs")
    ap.add_argument("--hard-budget", type=int, default=20,
                    help="max false-merge pairs across the four hard sets")
    ap.add_argument("models", nargs="*")
    args = ap.parse_args()
    models = args.models or ["adaface_irv1", "arcface_r50", "lvface_t",
                             "mobilefacenet_w600k", "sface_fp32",
                             "sface_int8"]

    hdr = (f"{'model':22}{'MB':>7}{'ms/img':>8}{'t*':>6}{'LFW rec':>9}"
           f"{'hard rec':>10}{'LFW FM':>8}{'hard FM':>9}{'hold FM':>9}")
    print(hdr)
    print("-" * len(hdr))
    for model in models:
        ab.MODEL = model
        try:
            sets = ab.load_sets()
        except FileNotFoundError as e:
            print(f"{model:22}  missing cache: {e}")
            continue
        # precompute the cosine matrix per set once (the sweep is then
        # pure thresholding)
        prepared = []
        for name, _split, emb, lab, photo, blur in sets:
            prepared.append((name, emb, lab, photo, blur, emb @ emb.T))

        best = None
        for t in np.arange(0.45, 0.86, 0.025):
            t = round(float(t), 3)
            lfw = hard_hit = hard_same = hard_fm = hold_fm = lfw_fm = 0
            lfw_rec = 0.0
            ok = True
            for name, emb, lab, photo, blur, G in prepared:
                cid = run_ente(emb, blur, np.full(len(emb), 0.95),
                               np.zeros(len(emb), dtype=bool), photo,
                               photo.astype(float), t, round(t + 0.08, 2), 0,
                               )
                sc = score(cid, lab)
                if name == "lfw":
                    lfw_rec, lfw_fm = sc["recall"], sc["FM"]
                elif name == "kinfacew":
                    hold_fm = sc["FM"]
                else:
                    hard_hit += sc["HIT"]
                    hard_same += sc["same"]
                    hard_fm += sc["FM"]
            if lfw_fm <= args.budget and hard_fm <= args.hard_budget:
                cand = (lfw_rec + hard_hit / max(hard_same, 1), t,
                        lfw_rec, hard_hit / max(hard_same, 1),
                        lfw_fm, hard_fm, hold_fm)
                if best is None or cand[0] > best[0]:
                    best = cand
        if best is None:
            print(f"{model:22}{SIZES_MB[model]:>7.1f}{MS_PER_IMG[model]:>8.1f}"
                  f"  no threshold inside the budget")
            continue
        _, t, lfw_rec, hard_rec, lfw_fm, hard_fm, hold_fm = best
        print(f"{model:22}{SIZES_MB[model]:>7.1f}{MS_PER_IMG[model]:>8.1f}"
              f"{t:>6.3f}{lfw_rec:>9.4f}{hard_rec:>10.4f}{lfw_fm:>8}"
              f"{hard_fm:>9}{hold_fm:>9}")


if __name__ == "__main__":
    main()
