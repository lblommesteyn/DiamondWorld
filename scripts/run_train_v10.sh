#!/usr/bin/env bash
# v10 = v9 recipe (outcome-only + fatigue + park_idx fix) trained to 50K steps,
# matching v6-final's budget. v9 was only 30K; this gives the park-aware/fatigue
# architecture a full-length run to see if it can beat v6-final on a rigorous,
# large-sample eval (not the noisy 512-game numbers).
source ~/dwjax-venv/bin/activate
cd ~/DiamondWorld
python -m diamondworldjax.scripts.train_pa --steps 50000 --outcome-only --fatigue --tag v10 \
  > data/train_v10.log 2>&1
echo DONE > data/train_v10_done.txt
