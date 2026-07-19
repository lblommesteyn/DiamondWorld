#!/usr/bin/env bash
# Full evaluation of v12 (v10 + recency-weighted stats). Same pipeline as v11 but
# with --recency-halflife 2.0 threaded everywhere so the eval player table matches
# training. Derives calibration, full-N scoreboard vs v10, player stats, calib audit.
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
source ~/dwjax-venv/bin/activate
cd ~/DiamondWorld
V12=checkpoints/dwjax_pa_v12/dwjax_step_0050000.pkl
OUT=data/eval2
HL=2.0
mkdir -p "$OUT"
echo "V12_EVAL_START $(date -u +%FT%TZ)" > "$OUT/v12_eval_done.txt"

echo "=== 1. derive v12 calibration + logit dump ==="
python -m diamondworldjax.scripts.diag_outcomes --ckpt "$V12" \
  --outcome-only --fatigue --use-park --recency-halflife $HL \
  --dump-logits "$OUT/v12_logits.npz" > "$OUT/v12_diag.txt" 2>&1

echo "=== 2. fit learned calibration ==="
python -m diamondworldjax.scripts.fit_calibration --logits "$OUT/v12_logits.npz" \
  --out "$OUT/v12_cal.txt" > "$OUT/v12_fitcal.log" 2>&1

echo "=== 3. full-N scoreboard: heuristic recal at 3 scales ==="
for S in 0.35 0.55 0.75; do
  TAG=${S/./}
  python -m diamondworldjax.scripts.simulate_games --ckpt "$V12" \
    --outcome-only --fatigue --use-park --recency-halflife $HL \
    --recal --recal-file "$OUT/v12_cal_params.npz" --recal-key b_heur --recal-scale "$S" \
    --limit-games 4859 --dump-runs "$OUT/v12_s${TAG}.npy" \
    --dump-scores "$OUT/v12_s${TAG}_scores.npz" > "$OUT/v12_s${TAG}.log" 2>&1
  echo "  heuristic @$S done"
done

echo "=== 4. SCOREBOARD (v12 vs v10, v11) ==="
python -m diamondworldjax.scripts.score_runs --runs \
  "v10@0.35:$OUT/v10_s035.npy" "v11h@0.75:$OUT/v11_s075.npy" \
  "v12@0.35:$OUT/v12_s035.npy" "v12@0.55:$OUT/v12_s055.npy" \
  "v12@0.75:$OUT/v12_s075.npy" | tee "$OUT/v12_scoreboard.txt"

echo "=== 5. player stats ==="
python -m diamondworldjax.scripts.eval_players --ckpt "$V12" \
  --outcome-only --fatigue --use-park --recency-halflife $HL --min-pa 150 2>&1 | tee "$OUT/v12_players.txt"

echo "=== 6. conditional-calibration audit (find mean-matched scale from scoreboard first) ==="
python -m diamondworldjax.scripts.calib_audit --ckpt "$V12" \
  --outcome-only --fatigue --use-park --recency-halflife $HL \
  --recal --recal-file "$OUT/v12_cal_params.npz" --recal-key b_heur --recal-scale 0.35 \
  --limit-games 600 --replicas 60 --chunk-games 300 --out "$OUT/calib_v12.txt" \
  > "$OUT/calib_v12.log" 2>&1

echo "V12_EVAL_DONE $(date -u +%FT%TZ)" >> "$OUT/v12_eval_done.txt"
