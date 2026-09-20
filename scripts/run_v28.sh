#!/usr/bin/env bash
# v28 = v22's exact recipe plus ONE lever: --shrink-contact-quality.
#
# Single change per rung, per LADDER.md. The lever regularises player-table
# columns 5:7 (expected hit / expected HR from contact quality), which until now
# were the only rate features fed in completely raw while columns 0..3 were
# shrunk. Constants tuned on 2023 with the table built from 2015-2022, so the
# 2024 test season never entered the choice:
#
#   expected hit  reg 700   2024 feature corr 0.409 -> 0.496
#   expected HR   reg 100   2024 feature corr 0.630 -> 0.643
#
# The question this run answers is whether the model converts a better input into
# a better output, which the v17-v21 series gives real reason to doubt.
#
# Seed 42 to pair against prod_rates_v22_s42.npz.
#
# Detach discipline (learned the hard way, v21b): the real work is setsid'd so it
# survives the launching shell, and this script traps TERM/INT/EXIT and kills the
# whole process GROUP so a scancel cannot orphan a run that keeps holding the GPU.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source ~/dwjax-venv/bin/activate
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
export PYTHONPATH="$PWD"

TAG=v28_s42
SEED=42
STEPS=50000
RC=data/eval2/v13_cal_params.npz
DONE=data/run_${TAG}_done.txt
LOG=data/train_${TAG}.log

CHILD_PGID=""
cleanup() {
  if [ -n "$CHILD_PGID" ]; then
    kill -- "-$CHILD_PGID" 2>/dev/null || true
  fi
}
trap cleanup TERM INT EXIT

echo "START $(date -u +%FT%TZ)" > "$DONE"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader >> "$DONE"

# Guard before spending five hours: the flag must move columns 5:7 and nothing
# else, and must move them toward the league rate.
python - >> "$DONE" 2>&1 <<'PY'
import numpy as np
from diamondworldjax.scripts.train_pa import shrink_toward_league, CQ_SHRINK_REG
print("  CQ_SHRINK_REG =", CQ_SHRINK_REG)
rates = np.array([[0.30, 0.05], [0.30, 0.05]])
n = np.array([[150.0], [3000.0]])
anchor = np.array([[0.22, 0.03]])
r = np.vstack([rates, anchor]); nn = np.vstack([n, np.array([[50000.0]])])
out = shrink_toward_league(r, nn, np.array(CQ_SHRINK_REG))
print("  low-PA  row ->", np.round(out[0], 4))
print("  high-PA row ->", np.round(out[1], 4))
assert abs(out[0,0]-0.22) < abs(out[1,0]-0.22), "low PA must shrink further"
print("  guard OK: shrinkage scales with sample size")
PY
if ! grep -q "guard OK" "$DONE"; then
  echo "GUARD FAILED, not training" >> "$DONE"; exit 1
fi

setsid nohup python -m diamondworldjax.scripts.train_pa --steps "$STEPS" \
  --outcome-only --fatigue --recency-halflife 2.0 \
  --train-end 2023 --contact-quality --per-stat-shrink --shrink-contact-quality \
  --skill-prior walk --seed "$SEED" --tag "$TAG" > "$LOG" 2>&1 &
CHILD_PID=$!
CHILD_PGID=$(ps -o pgid= -p "$CHILD_PID" | tr -d ' ')
echo "train pid $CHILD_PID pgid $CHILD_PGID" >> "$DONE"

# Hold the allocation by polling for the checkpoint, not by pgrep (which races
# under load and false-positives "died").
CKPT="checkpoints/dwjax_pa_${TAG}/dwjax_step_$(printf %07d "$STEPS").pkl"
while kill -0 "$CHILD_PID" 2>/dev/null; do
  sleep 60
done
wait "$CHILD_PID"; TRC=$?
echo "[${TAG}] train rc=$TRC $(date)" >> "$DONE"
trap - EXIT
CHILD_PGID=""

if [ ! -f "$CKPT" ]; then
  echo "no checkpoint at $CKPT, stopping" >> "$DONE"; exit 1
fi

# Score on 2024 (pairs against prod_rates_v22_s42.npz) and on 2025 (the fresh
# independent-batter slate ingested 2026-09-20).
for TS in 2024 2025; do
  python -m diamondworldjax.scripts.prod_playercorr \
    --ckpt "$CKPT" --recal "$RC" --recency-halflife 2.0 --skill-mode mean \
    --train-end 2023 --test-seasons "$TS" \
    --contact-quality --per-stat-shrink --shrink-contact-quality \
    --skill-prior walk --tag "${TAG}_t${TS}" > "data/eval2/${TAG}_t${TS}_eval.log" 2>&1
  echo "[${TAG}] eval ${TS} rc=$? $(date)" >> "$DONE"
  tail -1 "data/eval2/prod_playercorr_${TAG}_t${TS}.txt" >> "$DONE" 2>/dev/null
done

# The decisive comparison is against v22, not v16: v16-vs-v22 is the bug fixes,
# and this run is asking what the feature lever adds on top of them.
python -m diamondworldjax.scripts.bootstrap_playercorr \
  --rates v16=data/eval2/prod_rates_v16.npz \
  --rates v22_s42=data/eval2/prod_rates_v22_s42.npz \
  --rates v27a_pa=data/eval2/prod_rates_v27a_pa_abcd_s42_pa.npz \
  --rates ${TAG}=data/eval2/prod_rates_${TAG}_t2024.npz \
  --baseline v22_s42 --reps 20000 \
  --out data/eval2/bootstrap_${TAG}.txt \
  --json-out data/eval2/bootstrap_${TAG}.json >> "$DONE" 2>&1
echo "[${TAG}] bootstrap rc=$? $(date)" >> "$DONE"

# And the blend, since that is where the hit-rate gain has to show up to matter.
python -m diamondworldjax.scripts.blend_projections \
  --rates data/eval2/prod_rates_${TAG}_t2024.npz --dw-name "DW ${TAG}" \
  --out data/eval2/blend_${TAG}.txt >> "$DONE" 2>&1
echo "[${TAG}] blend rc=$? $(date)" >> "$DONE"

echo "DONE $(date -u +%FT%TZ)" >> "$DONE"
