#!/usr/bin/env bash
# v14 = v13 recipe (outcome-only + fatigue + park + recency 2 + KL-scale fix) PLUS
# the likelihood-mask fix: padded PAs (~16% of positions, previously labeled class
# 0 = K) are now masked out of the likelihood, so the model is no longer trained on
# fake strikeouts. Should reduce the chronic K over-prediction and the recal needed.
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
source ~/dwjax-venv/bin/activate
cd ~/DiamondWorld
echo "START $(date -u +%FT%TZ)" > data/train_v14_done.txt
python -m diamondworldjax.scripts.train_pa --steps 50000 \
  --outcome-only --fatigue --recency-halflife 2.0 --tag v14 \
  > data/train_v14.log 2>&1
echo "DONE $(date -u +%FT%TZ)" >> data/train_v14_done.txt
