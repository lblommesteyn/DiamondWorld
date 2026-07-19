#!/usr/bin/env bash
# Full-N (4,859-game) evaluation of v10 (50K-step park+fatigue model).
# Runs three recal scales to bracket the mean-matched scale, scores each vs real
# alongside v9 and the baselines, then runs the conditioned player-stat eval.
set -e
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
source ~/dwjax-venv/bin/activate
cd ~/DiamondWorld
V10=checkpoints/dwjax_pa_v10/dwjax_step_0050000.pkl
OUT=data/eval2
mkdir -p "$OUT"
echo "V10_EVAL_START $(date -u +%FT%TZ)" > "$OUT/v10_eval_done.txt"

for S in 0.35 0.55 0.75; do
  TAG=${S/./}
  echo "=== v10 @$S, N=4859 ==="
  python -m diamondworldjax.scripts.simulate_games --ckpt "$V10" \
    --outcome-only --fatigue --use-park --recal --recal-version v10 --recal-scale "$S" \
    --limit-games 4859 --dump-runs "$OUT/v10_s${TAG}.npy" \
    --dump-scores "$OUT/v10_s${TAG}_scores.npz" > "$OUT/v10_s${TAG}.log" 2>&1
  echo "  done $S -> $OUT/v10_s${TAG}.npy"
done

echo "=== V10 SCOREBOARD ===" | tee "$OUT/v10_scoreboard.txt"
python -m diamondworldjax.scripts.score_runs --runs \
  "v9@0.55:$OUT/final_v9_s55.npy" \
  "v10@0.35:$OUT/v10_s035.npy" \
  "v10@0.55:$OUT/v10_s055.npy" \
  "v10@0.75:$OUT/v10_s075.npy" | tee -a "$OUT/v10_scoreboard.txt"

echo "=== V10 PLAYER STATS ===" | tee "$OUT/v10_players.txt"
python -m diamondworldjax.scripts.eval_players --ckpt "$V10" \
  --outcome-only --fatigue --use-park --min-pa 150 2>&1 | tee -a "$OUT/v10_players.txt"

echo "V10_EVAL_DONE $(date -u +%FT%TZ)" >> "$OUT/v10_eval_done.txt"
