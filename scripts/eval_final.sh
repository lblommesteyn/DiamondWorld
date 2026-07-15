#!/usr/bin/env bash
# Final large-sample scoreboard at the best recal scale per model (from the
# sweep). Large N so tail metrics (p8+, wasserstein) have low sampling error.
# Also dumps v9 home/away scores for the extras/tie diagnostic.
set -e
source ~/dwjax-venv/bin/activate
cd ~/DiamondWorld
V6=checkpoints/dwjax_pa_BEST/v6_final_step50000.pkl
V9=checkpoints/dwjax_pa_v9/dwjax_step_0030000.pkl
# Best recal scales from the N=1200 sweep (scored vs full real test):
#   v6-final: 0.40 (KL 0.0198, p8+ 0.0038); v9: 0.35 (KL 0.0104, p8+ 0.0004).
N=${N:-4859}
V6S=${V6S:-0.40}
V9S=${V9S:-0.35}
OUT=data/eval2
mkdir -p "$OUT"

echo "=== v6-final @${V6S}, N=$N ==="
python -m diamondworldjax.scripts.simulate_games --ckpt "$V6" \
  --outcome-only --recal --recal-version v6 --recal-scale "$V6S" \
  --limit-games "$N" --dump-runs "$OUT/final_v6.npy" > "$OUT/final_v6.log" 2>&1

echo "=== v9 @0.35 (best KL/tail), N=$N ==="
python -m diamondworldjax.scripts.simulate_games --ckpt "$V9" \
  --outcome-only --fatigue --use-park --recal --recal-version v9 --recal-scale 0.35 \
  --limit-games "$N" --dump-runs "$OUT/final_v9_s35.npy" > "$OUT/final_v9_s35.log" 2>&1

echo "=== v9 @0.55 (mean-matched), N=$N (+ score dump) ==="
python -m diamondworldjax.scripts.simulate_games --ckpt "$V9" \
  --outcome-only --fatigue --use-park --recal --recal-version v9 --recal-scale 0.55 \
  --limit-games "$N" --dump-runs "$OUT/final_v9_s55.npy" \
  --dump-scores "$OUT/final_v9_scores.npz" > "$OUT/final_v9_s55.log" 2>&1

echo "=== SCOREBOARD (models, vs full real test) ==="
python -m diamondworldjax.scripts.score_runs \
  --runs "v6-final@0.40:$OUT/final_v6.npy" "v9@0.35:$OUT/final_v9_s35.npy" \
  "v9@0.55:$OUT/final_v9_s55.npy" | tee "$OUT/scoreboard.txt"

echo "=== BASELINES (B0/B1) on identical metric ==="
python -m diamondworldjax.scripts.compare_baselines \
  --v6-runs "$OUT/final_v9_s55.npy" --model-label "v9@0.55" --player-corr 0.58 \
  --n-games 4859 | tee "$OUT/baselines.txt"

echo "=== EXTRAS DIAGNOSIS (v9@0.55) ==="
python -m diamondworldjax.scripts.analyze_extras --scores "$OUT/final_v9_scores.npz" \
  | tee "$OUT/extras.txt"

echo FINAL_DONE
