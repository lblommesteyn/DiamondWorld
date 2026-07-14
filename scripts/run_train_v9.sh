#!/usr/bin/env bash
# v9 = v8 recipe (outcome-only + fatigue) + park_idx fix (park embedding now
# sees real park indices instead of all-zeros). 30k steps: v8's ELBO plateaued
# by ~10k, 50k added nothing.
source ~/dwjax-venv/bin/activate
cd ~/DiamondWorld
python -m diamondworldjax.scripts.train_pa --steps 30000 --outcome-only --fatigue --tag v9 > data/train_v9.log 2>&1
echo DONE > data/train_v9_done.txt
