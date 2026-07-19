#!/usr/bin/env bash
# Conditional-calibration audit for v10 (primary) and v9 (comparison): does the
# extra training help or hurt PER-MATCHUP calibration, not just marginal fit?
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
source ~/dwjax-venv/bin/activate
cd ~/DiamondWorld
OUT=data/eval2
G=800; R=100; CH=400
echo "CALIB_START $(date -u +%FT%TZ)" > "$OUT/calib_done.txt"

echo "=== v10 @0.35 calibration ==="
python -m diamondworldjax.scripts.calib_audit \
  --ckpt checkpoints/dwjax_pa_v10/dwjax_step_0050000.pkl \
  --outcome-only --fatigue --use-park --recal --recal-version v10 --recal-scale 0.35 \
  --limit-games $G --replicas $R --chunk-games $CH --out "$OUT/calib_v10.txt" \
  > "$OUT/calib_v10.log" 2>&1

echo "=== v9 @0.55 calibration ==="
python -m diamondworldjax.scripts.calib_audit \
  --ckpt checkpoints/dwjax_pa_v9/dwjax_step_0030000.pkl \
  --outcome-only --fatigue --use-park --recal --recal-version v9 --recal-scale 0.55 \
  --limit-games $G --replicas $R --chunk-games $CH --out "$OUT/calib_v9.txt" \
  > "$OUT/calib_v9.log" 2>&1

echo "CALIB_DONE $(date -u +%FT%TZ)" >> "$OUT/calib_done.txt"
