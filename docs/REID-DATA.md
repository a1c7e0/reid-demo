# Re-ID Training / Tuning Datasets

Under `testsets/reid-train/` (untracked; re-downloadable). Purpose: fit
BODY-channel parameters (join thresholds, time-window rules) — these are
TUNING sets; the hard sets (`testsets/hard/`) stay evaluation-only.

| Set | Size | What it trains | Source |
|---|---|---|---|
| Market-1501 (train split) | 751 ids / 12.9k imgs | base same/diff body separation, join threshold | already in `testsets/Market-1501-v15.09.15` |
| MSMT17_V1 (train split) | 4,101 ids / 126k imgs, multi-scene multi-time | hard multi-scene separation, time-window tuning | huggingface.co/datasets/xianpeijie/MSMT17_V1 |
| PRCC / VC-Clothes bundle | cross-clothes re-id | long-span / clothes-change connection rules | Google Drive (Clothes_Change_Person_ReID repo) |

The cross-clothes bundle arrives as a torch-serialized dataset
(`archive/{data.pkl,data,version}`, 279 MB); extract with `torch.load`
when needed.

Protocol: sample fixed-seed identity subsets per run; fit thresholds on
train splits; never tune on `testsets/hard/` evaluation pairs.

Notes:
- DukeMTMC-reID official host (vision.cs.duke.edu) is dead and no public
  mirror was reachable; MSMT17 covers the same multi-scene regime harder.
- Market test/gallery splits remain the model-comparison protocol.
