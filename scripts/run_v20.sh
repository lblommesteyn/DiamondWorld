#!/usr/bin/env bash
# v20: per-stat shrinkage of the rate FEATURES. The first variant on the axis the
# evidence actually points to.
#
# WHY THIS AXIS. Everything tried so far has been architecture, prior, or
# objective, and all of it failed: bilinear +0.003, nested -0.006, aggregation
# loss -0.001 / -0.015, learned prior -0.025, LKJ -0.013. Across five variants and
# twenty-five stat cells there is no positive result anywhere. What HAS moved this
# model is features: v16's contact-quality columns bought +0.017 with the interval
# excluding zero. So features are where the remaining headroom is, and the gap to
# close is real and measured, not hypothetical: Steamer scores 0.671 against
# v16's 0.611, and a Marcel-style estimator with contact quality reaches Hit 0.497
# against v16's 0.422. A simple regularised average beats the full hierarchical
# model on that stat.
#
# THE INTERVENTION. Columns 0..3 of the player table are RAW observed rates, so a
# 150-PA batter's hit rate is mostly noise while his strikeout rate is already
# informative. Both are fed in raw, leaving the model to infer how far to trust
# each from the PA count in column 4. It can in principle; the whole v17/v18/v19
# series is evidence that this model does not learn what it could learn. This
# shrinks each rate toward the league rate by its OWN measured stabilisation
# constant (K ~200 PA, BB ~400, hit and HR ~2200), which projection_levers.py
# established while flagging the single shipped REG=1200 as "badly wrong at both
# ends".
#
# PREREGISTERED PREDICTION: a small win, concentrated in Hit and HR (the
# high-constant stats, where raw rates are noisiest and shrinkage helps most), and
# roughly flat in K (already near-stable at typical sample sizes). This is the
# most specific prediction in the series because the mechanism names which cells
# should move.
#
# FALSIFIED IF: AVG interval includes zero, which would mean the model was already
# extracting the sample-size information from column 4 and the explicit shrinkage
# is redundant. A REGRESSION would mean the constants are wrong for this use, most
# likely because they were fitted for a Marcel-style projection rather than as
# inputs to a learned model.
#
# GATE: paired bootstrap vs v16 must exclude zero on AVG.
set -u
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
source ~/dwjax-venv/bin/activate
cd ~/DiamondWorld

RC=data/eval2/v13_cal_params.npz
echo "START $(date -u +%FT%TZ)" > data/run_v20_done.txt

# Sanity-check the feature transform before spending GPU: shrinkage must pull
# low-PA players toward the league rate and leave high-PA players alone.
python - >> data/run_v20_done.txt 2>&1 <<'PY'
import numpy as np
reg = {"hit": 2200.0, "bb": 400.0, "k": 200.0, "hr": 2200.0}
league, raw = 0.20, 0.60
for n in (150.0, 3000.0):
    p = {k: (n * raw + c * league) / (n + c) for k, c in reg.items()}
    print(f"  n={n:6.0f} PA raw={raw}: " +
          " ".join(f"{k}={p[k]:.3f}" for k in ("k", "bb", "hit", "hr")))
print("  (K should retain most raw signal at low PA; hit/HR should sit near league)")
PY

python -m diamondworldjax.scripts.train_pa --steps 50000 \
  --outcome-only --fatigue --recency-halflife 2.0 \
  --train-end 2023 --contact-quality --per-stat-shrink \
  --tag v20 > data/train_v20.log 2>&1
echo "[v20] train rc=$? $(date)" >> data/run_v20_done.txt

python -m diamondworldjax.scripts.prod_playercorr \
  --ckpt checkpoints/dwjax_pa_v20/dwjax_step_0050000.pkl \
  --recal "$RC" --recency-halflife 2.0 --skill-mode mean \
  --train-end 2023 --test-seasons 2024 --contact-quality --per-stat-shrink \
  --tag v20 > data/eval2/v20_eval.log 2>&1
echo "[v20] eval rc=$? $(date)" >> data/run_v20_done.txt
tail -1 data/eval2/prod_playercorr_v20.txt >> data/run_v20_done.txt

python -m diamondworldjax.scripts.bootstrap_playercorr \
  --rates v16=data/eval2/prod_rates_v16.npz \
  --rates v20=data/eval2/prod_rates_v20.npz \
  --baseline v16 --reps 20000 \
  --out data/eval2/bootstrap_v20.txt --json-out data/eval2/bootstrap_v20.json \
  >> data/run_v20_done.txt 2>&1

echo "DONE $(date -u +%FT%TZ)" >> data/run_v20_done.txt
