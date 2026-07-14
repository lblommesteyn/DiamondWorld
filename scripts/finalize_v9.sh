#!/usr/bin/env bash
# Canonical v9 evaluation with v9's OWN recalibration (fatigue + park_idx model).
# Supersedes the auto-eval in watch_v9.sh, which reused v6's recal vector (too weak
# for v9) and read park_idx=0 from the test parquet (v9 collapses to all-K on the
# unknown-park index; simulate_games --use-park sidesteps it, diag/eval scripts now
# rebuild real park indices via apply_park_idx).
set -e
source ~/dwjax-venv/bin/activate
cd ~/DiamondWorld
CKPT=checkpoints/dwjax_pa_v9/dwjax_step_0030000.pkl

echo "=== v9 true sim, own recal @0.5, 512 games ==="
python -m diamondworldjax.scripts.simulate_games --ckpt "$CKPT" \
  --outcome-only --fatigue --use-park --recal --recal-version v9 --recal-scale 0.5 \
  --limit-games 512 --dump-runs data/v9_runs_ownrecal.npy --player-stats \
  > data/eval_v9_ownrecal.log 2>&1

echo "=== apples-to-apples vs baselines ==="
python -m diamondworldjax.scripts.compare_baselines \
  --v6-runs data/v9_runs_ownrecal.npy --model-label v9 --player-corr "$PLAYER_CORR" \
  > data/compare_v9_ownrecal.log 2>&1

echo DONE
