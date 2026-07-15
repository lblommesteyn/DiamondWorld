#!/usr/bin/env bash
# Finish the eval by SHARING the contended GPU (a persistent ~4GB load ran for
# 11h). A small memory cap lets the sim coexist without OOM (verified on a 64-game
# test). One full-N sim at the mean-matched recal (0.65, confirmed on the test)
# gives both the mean-matched scoreboard row and the extras score-dump.
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.35
source ~/dwjax-venv/bin/activate
cd ~/DiamondWorld
V9=checkpoints/dwjax_pa_v9/dwjax_step_0030000.pkl
OUT=data/eval2
rm -f data/_share_test.npy data/_share_test.log

echo "=== v9 @0.65 (mean-matched), N=4859, shared GPU ==="
python -m diamondworldjax.scripts.simulate_games --ckpt "$V9" \
  --outcome-only --fatigue --use-park --recal --recal-version v9 --recal-scale 0.65 \
  --limit-games 4859 --dump-runs "$OUT/final_v9_s65.npy" \
  --dump-scores "$OUT/final_v9_s65_scores.npz" > "$OUT/final_v9_s65.log" 2>&1

echo "=== SCOREBOARD ==="
python -m diamondworldjax.scripts.score_runs --runs \
  "v6-final@0.40:$OUT/final_v6.npy" "v9@0.35:$OUT/final_v9_s35.npy" \
  "v9@0.65:$OUT/final_v9_s65.npy" | tee "$OUT/scoreboard.txt"

echo "=== EXTRAS DIAGNOSIS ==="
python -m diamondworldjax.scripts.analyze_extras --scores "$OUT/final_v9_s65_scores.npz" \
  | tee "$OUT/extras.txt"

echo SHARE_DONE
