#!/usr/bin/env bash
# Train one ladder rung with the production recipe (scripts/run_train_v16.sh), so a rung
# differs from the one below it by exactly one thing. See LADDER.md.
#
#   bash scripts/ladder_train.sh v22   <seed>     # R1: bug fixes (+ V22 extras, see below)
#   bash scripts/ladder_train.sh v22pf <seed>     # R2: R1 + PA attention
#   bash scripts/ladder_train.sh v23   <seed>     # R3: R2 + shared player latent
set -euo pipefail
rung=${1:?rung: v22 | v22pf | v23}
seed=${2:-0}
RECIPE="--outcome-only --fatigue --recency-halflife 2.0 --train-end 2023 --contact-quality"
# v22 as Jaden trained it also carries these two (LADDER.md, "What v22 actually is").
V22="--per-stat-shrink --skill-prior walk"
case "$rung" in
  v22)   python -m diamondworldjax.scripts.train_pa --steps 50000 $RECIPE $V22 \
           --seed "$seed" --tag "v22_s$seed" ;;
  v22pf) python -m diamondworldjax.scripts.train_pa --steps 50000 $RECIPE $V22 \
           --pitchformer --seed "$seed" --tag "v22pf_s$seed" ;;
  # multitask has no walk prior, so R3 cannot carry --skill-prior walk: a second
  # difference from R2 until it does.
  v23)   python -m diamondworldjax.scripts.train_shared_skills --steps 50000 $RECIPE \
           --per-stat-shrink --pa-pitchformer --seed "$seed" --tag "v23_s$seed" ;;
  *) echo "unknown rung $rung" >&2; exit 2 ;;
esac
