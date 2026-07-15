#!/usr/bin/env bash
# Resume the final eval after the day-boundary task cleanup. v6@0.40 and v9@0.35
# full-N sims already completed (data/eval2/final_v6.npy, final_v9_s35.npy). The
# 1200-game sweep was biased high (first games by game_pk are higher-scoring), so
# the full-N mean-matching recal scale is higher: bracket it at 0.60 and 0.70.
# Cap JAX GPU memory so a stray graphics process can't OOM us again.
set -e
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
source ~/dwjax-venv/bin/activate
cd ~/DiamondWorld
V9=checkpoints/dwjax_pa_v9/dwjax_step_0030000.pkl
OUT=data/eval2

for s in 0.60 0.70; do
  tag="final_v9_s${s/./}"
  echo "=== v9 @${s}, N=4859 ==="
  python -m diamondworldjax.scripts.simulate_games --ckpt "$V9" \
    --outcome-only --fatigue --use-park --recal --recal-version v9 --recal-scale "$s" \
    --limit-games 4859 --dump-runs "$OUT/${tag}.npy" \
    --dump-scores "$OUT/${tag}_scores.npz" > "$OUT/${tag}.log" 2>&1
done

echo "=== SCOREBOARD (models, vs full real test) ==="
python -m diamondworldjax.scripts.score_runs \
  --runs "v6-final@0.40:$OUT/final_v6.npy" "v9@0.35:$OUT/final_v9_s35.npy" \
  "v9@0.60:$OUT/final_v9_s60.npy" "v9@0.70:$OUT/final_v9_s70.npy" | tee "$OUT/scoreboard.txt"

echo "=== BASELINES (B0/B1) ==="
python -m diamondworldjax.scripts.compare_baselines \
  --v6-runs "$OUT/final_v9_s70.npy" --model-label "v9@0.70" --player-corr 0.58 \
  --n-games 4859 | tee "$OUT/baselines.txt"

echo "=== EXTRAS DIAGNOSIS (v9@0.70) ==="
python -m diamondworldjax.scripts.analyze_extras --scores "$OUT/final_v9_s70_scores.npz" \
  | tee "$OUT/extras.txt"

echo FINAL_DONE
