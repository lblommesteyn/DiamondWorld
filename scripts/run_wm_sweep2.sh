#!/usr/bin/env bash
# Phase 2: drill into the MLP (Phase-1 winner on player differentiation). Test the
# hypothesis that the learned player embedding dilutes the generalizable rate
# features: --no-player-emb uses rate stats only. Plus ensembles and capacity.
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
source ~/dwjax-venv/bin/activate
cd ~/DiamondWorld
S="--steps 6000 --patience 1500 --arch mlp"
run(){ echo ">>> $*"; python -m diamondworldjax.scripts.wm_sweep $S "$@" 2>>data/eval2/wm_sweep_err.log; }

run --no-player-emb
run --no-player-emb --ensemble 3
run --embed-dropout 0.5
run --embed-dropout 0.3 --ensemble 3
run --layers 4 --dm 256 --embed-dropout 0.3
run --embed-dropout 0.3 --label-smooth 0.02
echo "WM_SWEEP2_DONE $(date -u +%FT%TZ)"
