#!/usr/bin/env bash
# v19w: the per-season RANDOM-WALK skill prior. The last unbuilt piece of the
# original skill-prior proposal, and the only variant in this whole series that
# adds STRUCTURE rather than just freedom.
#
#   z[p, 0] ~ N(0, 1),   z[p, s] ~ N(z[p, s-1], sigma_walk)
#
# WHY IT IS DIFFERENT FROM v19c/v19d. Those free the prior's scale and correlation
# but keep one static skill per player. This makes skill a TRAJECTORY across
# seasons, which is the principled version of a lever the project already relies
# on: recency weighting (v12+) hand-builds a "current form" prior by exponentially
# discounting older seasons in the FEATURES. A random walk instead lets the model
# infer how fast talent actually drifts, and it is the same structure age curves
# would need. At eval the test season (2024) is beyond the trained range and is
# clamped to the last trained season's skill, which is exactly the "most recent
# form" quantity recency weighting approximates by hand.
#
# WHY IT WAS NOT RUN EARLIER, stated plainly: it was descoped without being
# flagged, on the grounds that it needed a per-season player table. That was only
# half true. The stat table stays one row per player (the deterministic encoding
# is season-invariant and is broadcast); only the stochastic latent gains a season
# axis, plus a `season` column threaded through pa_batching.
#
# PREDICTION, recorded before the fact: genuinely uncertain, and this is the one
# variant where I would not bet on a null. It multiplies the latent count by
# n_seasons (3520 x 9 x 32), so it could easily overfit, but unlike every other
# variant it encodes real temporal structure the model currently lacks. Both a win
# and a regression are plausible; a regression would say the recency FEATURE
# already captures the drift and the extra latents only add variance.
#
# GATE: paired bootstrap vs v16 must exclude zero on AVG.
set -u
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
source ~/dwjax-venv/bin/activate
cd ~/DiamondWorld

RC=data/eval2/v13_cal_params.npz
echo "START $(date -u +%FT%TZ)" > data/run_v19w_done.txt

python scripts/_check_guide_coverage.py > data/eval2/guide_coverage_w.txt 2>&1
if grep -q UNCOVERED data/eval2/guide_coverage_w.txt; then
  echo "ABORT: guide coverage failed" >> data/run_v19w_done.txt
  cat data/eval2/guide_coverage_w.txt >> data/run_v19w_done.txt
  exit 1
fi

# Second pre-flight: the walk latent's site is `player_skill_eps`, so eval has to
# special-case the --skill-mode mean substitution. If it misses, the latent is
# sampled from the prior at eval and the scoreboard is meaningless. Checked
# against the 6-step smoke checkpoint, which costs seconds.
if [ -f checkpoints/dwjax_pa_smoke_v19w/dwjax_step_0000006.pkl ]; then
  if ! python scripts/_check_walk_eval.py >> data/run_v19w_done.txt 2>&1; then
    echo "ABORT: walk eval path broken" >> data/run_v19w_done.txt
    exit 1
  fi
fi

python -m diamondworldjax.scripts.train_pa --steps 50000 \
  --outcome-only --fatigue --recency-halflife 2.0 \
  --train-end 2023 --contact-quality \
  --skill-prior walk --tag v19w > data/train_v19w.log 2>&1
echo "[v19w] train rc=$? $(date)" >> data/run_v19w_done.txt

# The walk latent must be (P, n_seasons, SKILL_DIM) and the step size must have
# been fitted; anything else means the season axis silently collapsed.
python - >> data/run_v19w_done.txt 2>&1 <<'PY'
import pickle, numpy as np
p = pickle.load(open("checkpoints/dwjax_pa_v19w/dwjax_step_0050000.pkl","rb"))["params"]
mu = np.asarray(p["player_mu"])
print(f"  player_mu shape {mu.shape} (expect (P, 9, 32))")
print(f"  skill_walk_sigma_loc = {float(np.asarray(p['skill_walk_sigma_loc'])):.4f}")
# Does skill actually drift across seasons, or did it collapse to a constant path?
drift = np.abs(np.diff(mu, axis=1)).mean()
print(f"  mean |season-to-season skill change| = {drift:.4f} (0 => walk collapsed)")
PY

python -m diamondworldjax.scripts.prod_playercorr \
  --ckpt checkpoints/dwjax_pa_v19w/dwjax_step_0050000.pkl \
  --recal "$RC" --recency-halflife 2.0 --skill-mode mean \
  --train-end 2023 --test-seasons 2024 --contact-quality \
  --skill-prior walk --tag v19w > data/eval2/v19w_eval.log 2>&1
echo "[v19w] eval rc=$? $(date)" >> data/run_v19w_done.txt
tail -1 data/eval2/prod_playercorr_v19w.txt >> data/run_v19w_done.txt

python -m diamondworldjax.scripts.bootstrap_playercorr \
  --rates v16=data/eval2/prod_rates_v16.npz \
  --rates v19w=data/eval2/prod_rates_v19w.npz \
  --baseline v16 --reps 20000 \
  --out data/eval2/bootstrap_v19w.txt --json-out data/eval2/bootstrap_v19w.json \
  >> data/run_v19w_done.txt 2>&1

echo "DONE $(date -u +%FT%TZ)" >> data/run_v19w_done.txt
