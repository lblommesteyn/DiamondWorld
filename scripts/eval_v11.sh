#!/usr/bin/env bash
# Full evaluation of v11 (v10 + platoon). Derives v11's own calibration (heuristic
# per-class recal AND learned bias+temperature), runs the full-N run-distribution
# scoreboard vs v10, the conditioned player-stat eval, and the conditional-
# calibration audit vs v10. Fully automated via --recal-file (no hardcoded const).
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
source ~/dwjax-venv/bin/activate
cd ~/DiamondWorld
V11=checkpoints/dwjax_pa_v11/dwjax_step_0050000.pkl
OUT=data/eval2
mkdir -p "$OUT"
echo "V11_EVAL_START $(date -u +%FT%TZ)" > "$OUT/v11_eval_done.txt"

echo "=== 1. derive v11 calibration (marginals + logit dump) ==="
python -m diamondworldjax.scripts.diag_outcomes --ckpt "$V11" \
  --outcome-only --fatigue --platoon --use-park \
  --dump-logits "$OUT/v11_logits.npz" > "$OUT/v11_diag.txt" 2>&1

echo "=== 2. fit learned calibration (bias + temperature) ==="
python -m diamondworldjax.scripts.fit_calibration --logits "$OUT/v11_logits.npz" \
  --out "$OUT/v11_cal.txt" > "$OUT/v11_fitcal.log" 2>&1

echo "=== 3. full-N scoreboard: heuristic recal at 3 scales ==="
for S in 0.35 0.55 0.75; do
  TAG=${S/./}
  python -m diamondworldjax.scripts.simulate_games --ckpt "$V11" \
    --outcome-only --fatigue --platoon --use-park \
    --recal --recal-file "$OUT/v11_cal_params.npz" --recal-key b_heur --recal-scale "$S" \
    --limit-games 4859 --dump-runs "$OUT/v11_s${TAG}.npy" \
    --dump-scores "$OUT/v11_s${TAG}_scores.npz" > "$OUT/v11_s${TAG}.log" 2>&1
  echo "  heuristic @$S done"
done

echo "=== 4. full-N: LEARNED calibration (bias+temp, scale 1.0) ==="
python -m diamondworldjax.scripts.simulate_games --ckpt "$V11" \
  --outcome-only --fatigue --platoon --use-park \
  --recal --recal-file "$OUT/v11_cal_params.npz" --recal-key b --recal-scale 1.0 \
  --recal-temp "$(python -c "import numpy as np;print(float(np.load('$OUT/v11_cal_params.npz')['T']))")" \
  --limit-games 4859 --dump-runs "$OUT/v11_learned.npy" \
  --dump-scores "$OUT/v11_learned_scores.npz" > "$OUT/v11_learned.log" 2>&1

echo "=== 5. SCOREBOARD (v11 variants vs v10, v9) ==="
python -m diamondworldjax.scripts.score_runs --runs \
  "v9@0.55:$OUT/final_v9_s55.npy" "v10@0.35:$OUT/v10_s035.npy" \
  "v11h@0.35:$OUT/v11_s035.npy" "v11h@0.55:$OUT/v11_s055.npy" \
  "v11h@0.75:$OUT/v11_s075.npy" "v11learned:$OUT/v11_learned.npy" \
  | tee "$OUT/v11_scoreboard.txt"

echo "=== 6. player stats (platoon-conditioned) ==="
python -m diamondworldjax.scripts.eval_players --ckpt "$V11" \
  --outcome-only --fatigue --platoon --use-park --min-pa 150 2>&1 | tee "$OUT/v11_players.txt"

echo "=== 7. conditional-calibration audit: v10 vs v11 (lean, matched config) ==="
G=600; R=60; CH=300
python -m diamondworldjax.scripts.calib_audit \
  --ckpt checkpoints/dwjax_pa_v10/dwjax_step_0050000.pkl \
  --outcome-only --fatigue --use-park --recal --recal-version v10 --recal-scale 0.35 \
  --limit-games $G --replicas $R --chunk-games $CH --out "$OUT/calib_v10.txt" \
  > "$OUT/calib_v10.log" 2>&1
echo "  v10 calib done"
# v11 at its mean-matched heuristic scale (retune below if scoreboard says otherwise)
python -m diamondworldjax.scripts.calib_audit --ckpt "$V11" \
  --outcome-only --fatigue --platoon --use-park \
  --recal --recal-file "$OUT/v11_cal_params.npz" --recal-key b_heur --recal-scale 0.35 \
  --limit-games $G --replicas $R --chunk-games $CH --out "$OUT/calib_v11.txt" \
  > "$OUT/calib_v11.log" 2>&1
echo "  v11 calib done"

echo "V11_EVAL_DONE $(date -u +%FT%TZ)" >> "$OUT/v11_eval_done.txt"
