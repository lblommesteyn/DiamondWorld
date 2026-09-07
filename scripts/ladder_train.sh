#!/usr/bin/env bash
# Train one ladder rung with the production recipe (scripts/run_train_v16.sh), so a rung
# differs from the one below it by exactly one thing. See LADDER.md.
#
#   bash scripts/ladder_train.sh v22   <seed>     # R1: bug fixes only
#   bash scripts/ladder_train.sh v22pf <seed>     # R2: R1 + PA attention
#   bash scripts/ladder_train.sh v23   <seed>     # R3: R2 + shared player latent
set -euo pipefail
rung=${1:?rung: v22 | v22pf | v23}
seed=${2:-0}
RECIPE="--outcome-only --fatigue --recency-halflife 2.0 --train-end 2023 --contact-quality"
case "$rung" in
  v22)   python -m diamondworldjax.scripts.train_pa --steps 50000 $RECIPE \
           --seed "$seed" --tag "v22_s$seed" ;;
  v22pf) python -m diamondworldjax.scripts.train_pa --steps 50000 $RECIPE \
           --pitchformer --seed "$seed" --tag "v22pf_s$seed" ;;
  v23)   python -m diamondworldjax.scripts.train_shared_skills --steps 50000 $RECIPE \
           --pa-pitchformer --seed "$seed" --tag "v23_s$seed" ;;
  *) echo "unknown rung $rung" >&2; exit 2 ;;
esac
