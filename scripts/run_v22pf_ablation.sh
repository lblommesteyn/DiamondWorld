#!/usr/bin/env bash
# Item 2, unblocked by training our own pitchformer instead of waiting for Jaden's.
#
# The question is NOT "what does Jaden's v22pf score". It is "does the PA-level
# pitchformer's headline gain come from the player representation, or from
# within-game context that a projection system like Steamer cannot see?" That is
# answered by scoring ONE checkpoint twice, with and without the history, so any
# checkpoint trained on the v22pf recipe will do. We do not need his weights.
#
# Recipe copied verbatim from scripts/run_v22pf.sh (seed 1) so this is the same
# model class that produced 0.681/0.676, then:
#   1. score it normally            -> expect a K/BB-heavy gain over v22
#   2. score it history-ablated     -> the decisive read
#   3. paired bootstrap of all three against v22_s42
#
# If the K and BB gain survives the ablation it is the player representation and
# the number is real. If it collapses toward v22 it is in-game context, and the
# Steamer comparison has to be withdrawn.
#
# Caveat carried into the report: positional encoding still applies under the
# ablation, so lineup slot is NOT removed, only within-game history.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source ~/dwjax-venv/bin/activate
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
export PYTHONPATH="$PWD"

TAG=v22pf_audit
SEED=1
# 20k, not 50k, and that is a deliberate and defensible choice. The mechanism
# question is answered by an INTERNAL contrast: the SAME checkpoint scored with and
# without within-game history. That contrast is valid at any step count, so the only
# requirement is that the K/BB gain has emerged enough to be ablatable. 20k cuts the
# wait from ~8h to ~3h on a single 3080 that also owes time to the v28 run.
#
# The cost is explicit and is stated in the report: v22pf_audit's ABSOLUTE level is
# NOT comparable to v22_s42 at 50k, because step count differs. The against-v22 rows
# below are therefore context, not verdicts. The verdict is the with-vs-without-history
# difference, and the script scores BOTH 10k and 20k so that contrast can be checked
# for stability rather than read off a single point.
STEPS=20000
EVAL_STEPS="10000 20000"
RC=data/eval2/v13_cal_params.npz
CKPT="checkpoints/dwjax_pa_${TAG}/dwjax_step_$(printf %07d "$STEPS").pkl"
DONE=data/run_${TAG}_done.txt
LOG=data/train_${TAG}.log

CHILD_PGID=""
cleanup() { [ -n "$CHILD_PGID" ] && kill -- "-$CHILD_PGID" 2>/dev/null || true; }
trap cleanup TERM INT EXIT

echo "START $(date -u +%FT%TZ)" > "$DONE"

if [ -f "$CKPT" ]; then
  echo "[${TAG}] reusing $CKPT" >> "$DONE"
else
  setsid nohup python -m diamondworldjax.scripts.train_pa --steps "$STEPS" \
    --outcome-only --fatigue --recency-halflife 2.0 \
    --train-end 2023 --contact-quality --per-stat-shrink \
    --skill-prior walk --seed "$SEED" --pitchformer --tag "$TAG" > "$LOG" 2>&1 &
  CHILD_PID=$!
  CHILD_PGID=$(ps -o pgid= -p "$CHILD_PID" | tr -d ' ')
  echo "train pid $CHILD_PID pgid $CHILD_PGID" >> "$DONE"
  while kill -0 "$CHILD_PID" 2>/dev/null; do sleep 60; done
  wait "$CHILD_PID"; echo "[${TAG}] train rc=$? $(date)" >> "$DONE"
  trap - EXIT; CHILD_PGID=""
fi

[ -f "$CKPT" ] || { echo "no checkpoint at $CKPT" >> "$DONE"; exit 1; }

# Score every requested step BOTH ways. The pair at each step is the experiment;
# two steps let the contrast be checked for stability instead of trusted once.
RATE_ARGS=""
for S in $EVAL_STEPS; do
  C="checkpoints/dwjax_pa_${TAG}/dwjax_step_$(printf %07d "$S").pkl"
  [ -f "$C" ] || { echo "[${TAG}] missing $C, skipping" >> "$DONE"; continue; }
  for MODE in hist nohist; do
    if [ "$MODE" = "nohist" ]; then ABL="--pitchformer-ablate-history"; else ABL=""; fi
    T="${TAG}_s${S}_${MODE}"
    python -m diamondworldjax.scripts.prod_playercorr \
      --ckpt "$C" --recal "$RC" --recency-halflife 2.0 --skill-mode mean \
      --train-end 2023 --test-seasons 2024 --contact-quality --per-stat-shrink \
      --skill-prior walk --pitchformer $ABL \
      --tag "$T" > "data/eval2/${T}_eval.log" 2>&1
    echo "[${TAG}] eval $T rc=$? $(date)" >> "$DONE"
    tail -1 "data/eval2/prod_playercorr_${T}.txt" >> "$DONE" 2>/dev/null
    RATE_ARGS="$RATE_ARGS --rates ${T}=data/eval2/prod_rates_${T}.npz"
  done
done

# The decisive table. v16 and v22_s42 are context only: they are 50k-step runs and
# these are 20k, so the WITH-vs-WITHOUT-history difference is the verdict, not the
# absolute levels.
python -m diamondworldjax.scripts.bootstrap_playercorr \
  --rates v16=data/eval2/prod_rates_v16.npz \
  --rates v22_s42=data/eval2/prod_rates_v22_s42.npz \
  --rates v22pf_s42_jaden=data/eval2/prod_rates_v22pf_s42.npz \
  $RATE_ARGS \
  --baseline v22_s42 --reps 20000 \
  --out data/eval2/bootstrap_${TAG}_ablation.txt \
  --json-out data/eval2/bootstrap_${TAG}_ablation.json >> "$DONE" 2>&1
echo "[${TAG}] bootstrap rc=$? $(date)" >> "$DONE"

# And the contrast stated directly, baselined on the ablated run at the final step,
# so the number that answers the item is in the log without needing a reader to
# subtract two rows.
FINAL=$(echo $EVAL_STEPS | awk "{print \$NF}")
python -m diamondworldjax.scripts.bootstrap_playercorr \
  --rates ${TAG}_s${FINAL}_nohist=data/eval2/prod_rates_${TAG}_s${FINAL}_nohist.npz \
  --rates ${TAG}_s${FINAL}_hist=data/eval2/prod_rates_${TAG}_s${FINAL}_hist.npz \
  --baseline ${TAG}_s${FINAL}_nohist --reps 20000 \
  --out data/eval2/bootstrap_${TAG}_contrast.txt \
  --json-out data/eval2/bootstrap_${TAG}_contrast.json >> "$DONE" 2>&1
echo "[${TAG}] contrast rc=$? $(date)" >> "$DONE"

echo "DONE $(date -u +%FT%TZ)" >> "$DONE"
