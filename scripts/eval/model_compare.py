"""Multi-model comparison at the PRODUCTION threshold (ENTE_T_GOOD/T_BAD).

`model_bench.py`'s first row per model is measured at Ente's own line
(0.76/0.84), which is not the line this deployment runs - comparing there
answers the wrong question. This runs every model with cached embeddings
through the SAME shipped configuration (0.65/0.73, no eject) on LFW plus
the five hard sets and reports the product metrics.

Usage: python model_compare.py [model ...]
"""
import sys

import numpy as np

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/scripts/eval")

import algo_bench as ab  # noqa: E402
from algo_bench import T_GOOD, T_BAD, run_ente, score  # noqa: E402

MODELS = (["adaface_irv1", "arcface_r50", "lvface_t",
           "mobilefacenet_w600k", "sface_fp32", "sface_int8"]
          if len(sys.argv) < 2 else sys.argv[1:])

SIZES_MB = {"adaface_irv1": 93.96, "arcface_r50": 174.38, "lvface_t": 76.65,
            "mobilefacenet_w600k": 13.62, "sface_fp32": 38.70,
            "sface_int8": 9.90}
MS_PER_IMG = {"adaface_irv1": 97.1, "arcface_r50": 656.1, "lvface_t": 115.9,
              "mobilefacenet_w600k": 56.2, "sface_fp32": 95.6,
              "sface_int8": 244.0}


def main() -> None:
    hdr = (f"{'model':22}{'MB':>7}{'ms/img':>8}{'LFW rec':>9}{'LFW FM':>8}"
           f"{'hard rec':>10}{'hard FM':>9}{'hold FM':>9}{'worst imp':>10}")
    print(hdr)
    print("-" * len(hdr))
    for model in MODELS:
        ab.MODEL = model
        try:
            sets = ab.load_sets()      # (name, split, emb, lab, photo, blur)
        except FileNotFoundError as e:
            print(f"{model:22}  missing cache: {e}")
            continue
        lfw_rec = lfw_fm = hard_hit = hard_fm = hold_fm = 0
        hard_same = worst_impure = 0
        for name, _split, emb, lab, photo, blur in sets:
            cid = run_ente(emb, blur, np.full(len(emb), 0.95),
                           np.zeros(len(emb), dtype=bool),
                           photo, photo.astype(float), T_GOOD, T_BAD, 0)
            sc = score(cid, lab)
            if name == "lfw":
                lfw_rec, lfw_fm = sc["recall"], sc["FM"]
            elif name == "kinfacew":
                hold_fm = sc["FM"]
            else:
                hard_hit += sc["HIT"]
                hard_fm += sc["FM"]
                hard_same += sc["same"]
                worst_impure = max(worst_impure, sc["impure_units"])
        print(f"{model:22}{SIZES_MB[model]:>7.1f}{MS_PER_IMG[model]:>8.1f}"
              f"{lfw_rec:>9.4f}{lfw_fm:>8}"
              f"{hard_hit / max(hard_same, 1):>10.4f}{hard_fm:>9}"
              f"{hold_fm:>9}{worst_impure:>10}")


if __name__ == "__main__":
    main()
