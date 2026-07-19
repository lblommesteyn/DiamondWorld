#!/usr/bin/env bash
# v11 = v10 recipe (outcome-only + fatigue + park) PLUS platoon: real per-PA
# batter side + pitcher throw hand added to the context (+2 dims, correct for
# switch hitters). Also populates the previously all-zero hand embedding. Fresh
# 50K-step train (context dim changes). Tests whether the platoon interaction,
# previously only extractable indirectly from two 32-dim player embeddings,
# improves per-matchup calibration and player-stat reproduction.
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
source ~/dwjax-venv/bin/activate
cd ~/DiamondWorld
echo "START $(date -u +%FT%TZ)" > data/train_v11_done.txt
python -m diamondworldjax.scripts.train_pa --steps 50000 \
  --outcome-only --fatigue --platoon --tag v11 \
  > data/train_v11.log 2>&1
echo "DONE $(date -u +%FT%TZ)" >> data/train_v11_done.txt
