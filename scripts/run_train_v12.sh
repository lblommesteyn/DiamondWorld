#!/usr/bin/env bash
# v12 = v10 recipe (outcome-only + fatigue + park) PLUS recency-weighted player
# stats (half-life 2 seasons): a leakage-free current-form prior that weights the
# most recent training season (2022) highest as the best proxy for 2023-24 talent.
# Isolates recency (NOT bundled with platoon) for a clean comparison to v10.
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
source ~/dwjax-venv/bin/activate
cd ~/DiamondWorld
echo "START $(date -u +%FT%TZ)" > data/train_v12_done.txt
python -m diamondworldjax.scripts.train_pa --steps 50000 \
  --outcome-only --fatigue --recency-halflife 2.0 --tag v12 \
  > data/train_v12.log 2>&1
echo "DONE $(date -u +%FT%TZ)" >> data/train_v12_done.txt
