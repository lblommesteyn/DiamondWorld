"""Pool the replicas of several simulator variants into one arrays file (a deep ensemble).

The game-level results pool every leak-free variant on the corrected pipeline rather than
picking one, so the reported numbers average over model uncertainty instead of selecting on
the metric being reported. Pooling replicas is the same as averaging the variants' forecast
distributions with equal weight per replica.

Two outputs:
  <out>            raw pool. Use for win probability and the within-series validation, where
                   a variant's constant run-level offset largely cancels.
  <out, aligned>   each variant's sim_total shifted to the pooled mean before pooling. Use
                   for run-distribution scores: the variants' run levels differ (8.4 to 10.0
                   per game), and pooling unaligned levels widens the mixture artificially.

  python -m diamondworldjax.scripts.pool_sims --glob "data/eval2/calib_v2*-pregame-leakfree_arrays.npz" \
      --out data/eval2/calib_ens_allpost_arrays.npz
"""
from __future__ import annotations

import argparse
import glob

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="data/eval2/calib_v2*-pregame-leakfree_arrays.npz")
    ap.add_argument("--out", default="data/eval2/calib_ens_allpost_arrays.npz")
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    if not files:
        raise SystemExit(f"no arrays match {args.glob}")
    ds = [np.load(f) for f in files]
    for f, x in zip(files, ds):
        if not (x["game_pk"] == ds[0]["game_pk"]).all():
            raise SystemExit(f"{f} is on a different game slate; refusing to pool")
        print(f"  {f}  R={x['sim_total'].shape[1]}  mean total {x['sim_total'].mean():.2f}")

    shared = {k: ds[0][k] for k in ("real_home", "real_away", "real_total", "game_pk")}
    pooled = {k: np.concatenate([x[k] for x in ds], axis=1)
              for k in ("sim_home", "sim_away", "sim_total")}
    np.savez(args.out, **pooled, **shared)

    level = np.mean([x["sim_total"].mean() for x in ds])
    aligned = dict(pooled)
    aligned["sim_total"] = np.concatenate(
        [x["sim_total"] - x["sim_total"].mean() + level for x in ds], axis=1)
    out_aligned = args.out.replace("_arrays.npz", "_aligned_arrays.npz")
    np.savez(out_aligned, **aligned, **shared)
    print(f"{len(files)} variants, R={pooled['sim_total'].shape[1]} -> {args.out}, {out_aligned}")


if __name__ == "__main__":
    main()
