#!/usr/bin/env bash
# v16 = the v15 recipe with exactly one change: --contact-quality.
#
# v15 = v13 (outcome-only + fatigue + park + recency half-life 2, KL-scale fix)
#       + --train-end 2023 (fold in the previous season).
# v16 adds xBA-style expected hit/HR columns to the player table, so the model's
# player prior is built from the quality of contact a batter made rather than
# from what happened to fall in.
#
# The hypothesis, and what would falsify it: projection_levers.py measured this
# construction lifting hit-rate correlation 0.418 -> 0.497 and HR 0.608 -> 0.641
# on a Marcel-style projection. If that information is usable by the SVI model,
# v16 should beat v15's 0.594 average player-corr, with the gain concentrated in
# hit and HR. If v16 lands at 0.594 the information is real but the model cannot
# exploit it, which is a different and equally publishable finding.
#
# Single lever on purpose: per-stat regression tuning was a wash on its own
# (+0.001) and is deliberately NOT bundled here, so any change is attributable.
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
source ~/dwjax-venv/bin/activate
cd ~/DiamondWorld
echo "START $(date -u +%FT%TZ)" > data/train_v16_done.txt
python -m diamondworldjax.scripts.train_pa --steps 50000 \
  --outcome-only --fatigue --recency-halflife 2.0 \
  --train-end 2023 --contact-quality --tag v16 \
  > data/train_v16.log 2>&1
echo "DONE $(date -u +%FT%TZ) rc=$?" >> data/train_v16_done.txt
