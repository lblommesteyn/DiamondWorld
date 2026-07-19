#!/usr/bin/env bash
# v10 = v9 recipe (outcome-only + fatigue + park_idx fix) trained to 50K steps,
# matching v6-final's budget. v9 was only 30K; this gives the park-aware/fatigue
# architecture a full-length run to see if it can beat v6-final on a rigorous,
# large-sample eval (not the noisy 512-game numbers).
# Cap the JAX allocator so this coexists with the other session's ~2.7GB on the
# shared 10GB GPU (the PA model is lightweight and fits well under this).
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
source ~/dwjax-venv/bin/activate
cd ~/DiamondWorld
echo "START $(date -u +%FT%TZ)" > data/train_v10_done.txt
python -m diamondworldjax.scripts.train_pa --steps 50000 --outcome-only --fatigue --tag v10 \
  > data/train_v10.log 2>&1
echo "DONE $(date -u +%FT%TZ)" >> data/train_v10_done.txt
