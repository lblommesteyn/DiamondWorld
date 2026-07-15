#!/usr/bin/env bash
# Mean-matched full-N sim at scale 0.55 (interpolated from 0.35->8.46 and
# 0.65->9.10, real 8.86 lands at ~0.54). GPU load eased, so a larger memory cap
# runs faster. Produces the definitive mean-matched scoreboard row + extras.
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
source ~/dwjax-venv/bin/activate
cd ~/DiamondWorld
V9=checkpoints/dwjax_pa_v9/dwjax_step_0030000.pkl
OUT=data/eval2

echo "=== v9 @0.55 (mean-matched), N=4859 ==="
python -m diamondworldjax.scripts.simulate_games --ckpt "$V9" \
  --outcome-only --fatigue --use-park --recal --recal-version v9 --recal-scale 0.55 \
  --limit-games 4859 --dump-runs "$OUT/final_v9_s55.npy" \
  --dump-scores "$OUT/final_v9_s55_scores.npz" > "$OUT/final_v9_s55.log" 2>&1

echo "=== FINAL SCOREBOARD ==="
python -m diamondworldjax.scripts.score_runs --runs \
  "v6-final@0.40:$OUT/final_v6.npy" "v9@0.35:$OUT/final_v9_s35.npy" \
  "v9@0.55:$OUT/final_v9_s55.npy" "v9@0.65:$OUT/final_v9_s65.npy" | tee "$OUT/scoreboard_final.txt"

echo "=== EXTRAS @0.55 ==="
python -m diamondworldjax.scripts.analyze_extras --scores "$OUT/final_v9_s55_scores.npz" \
  | tee "$OUT/extras_final.txt"

echo SHARE2_DONE
