#!/usr/bin/env bash
# v21: v19w (per-season random-walk skill) + v20 (per-stat feature shrinkage),
# COMBINED. The first variant in this series justified by convergent evidence
# rather than by a fresh hypothesis.
#
# WHY. Both components failed their gates individually, and both failed in the
# same direction with the same signature:
#
#   model                       AVG     Hit     AVG delta vs v16
#   v16 (incumbent)             0.611   0.422   -
#   v19w (temporal structure)   0.620   0.444   +0.009 [-0.011, +0.029]
#   v20 (feature shrinkage)     0.622   0.444   +0.011 [-0.002, +0.023]
#
# Two INDEPENDENT mechanisms, a per-season random walk on the latent versus
# per-stat shrinkage of static input features, produced the same +0.022 on hit
# rate and nearly the same AVG. That matters because hit rate had previously
# resisted every lever in this project: v13, v14, v16 and v17a null, v17b and
# v17c negative. Two unrelated changes moving the one immovable stat by an
# identical amount is either coincidence or a real effect that neither run had
# the power to confirm alone.
#
# THE TEST. The mechanisms are orthogonal (one changes latent structure, one
# changes input features), so if both effects are real they should be roughly
# additive: about +0.020 AVG, which clears the gate comfortably. This is a real
# test rather than a fishing expedition, because the prediction is quantitative
# and made before running.
#
# PREREGISTERED OUTCOMES:
#   * ~+0.020 with the interval excluding zero: both effects real and additive.
#     v21 becomes the new incumbent and the features/structure axis is the live
#     one after eight failures on architecture, prior and objective.
#   * ~+0.010, still spanning zero: the two are capturing the SAME underlying
#     signal by different routes rather than two separate gains. Informative in
#     its own right, since it would say temporal information and sample-size
#     shrinkage are one lever wearing two hats.
#   * ~0 or negative: the individual point estimates were noise, and eight nulls
#     becomes nine.
#
# CAVEAT, stated because this is the first result in the series I could plausibly
# over-read: two near-misses pointing the same way is suggestive, not evidence.
# Even a +0.020 landing with p just under 0.05 would want a second seed or a 2025
# test season before it went into a paper.
#
# GATE: paired bootstrap vs v16 must exclude zero on AVG.
set -u
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
source ~/dwjax-venv/bin/activate
cd ~/DiamondWorld

RC=data/eval2/v13_cal_params.npz
echo "START $(date -u +%FT%TZ)" > data/run_v21_done.txt

# Sanity-check the feature transform before spending GPU: shrinkage must pull
# low-PA players toward the league rate and leave high-PA players alone.
python - >> data/run_v21_done.txt 2>&1 <<'PY'
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
  --skill-prior walk --tag v21 > data/train_v21.log 2>&1
echo "[v20] train rc=$? $(date)" >> data/run_v21_done.txt

python -m diamondworldjax.scripts.prod_playercorr \
  --ckpt checkpoints/dwjax_pa_v21/dwjax_step_0050000.pkl \
  --recal "$RC" --recency-halflife 2.0 --skill-mode mean \
  --train-end 2023 --test-seasons 2024 --contact-quality --per-stat-shrink \
  --skill-prior walk --tag v21 > data/eval2/v21_eval.log 2>&1
echo "[v20] eval rc=$? $(date)" >> data/run_v21_done.txt
tail -1 data/eval2/prod_playercorr_v21.txt >> data/run_v21_done.txt

python -m diamondworldjax.scripts.bootstrap_playercorr \
  --rates v16=data/eval2/prod_rates_v16.npz \
  --rates v21=data/eval2/prod_rates_v21.npz \
  --baseline v16 --reps 20000 \
  --out data/eval2/bootstrap_v21.txt --json-out data/eval2/bootstrap_v21.json \
  >> data/run_v21_done.txt 2>&1

echo "DONE $(date -u +%FT%TZ)" >> data/run_v21_done.txt
