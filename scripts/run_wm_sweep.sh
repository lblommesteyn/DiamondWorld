#!/usr/bin/env bash
# World-model technique sweep, scored by player-stat reproduction (cross-player
# rate correlation) + calibration. The hypothesis: the discriminative model
# under-uses batter identity (esp. for K); regularizing the learned embedding
# toward the generalizable rate features (embed-dropout) and other techniques
# should recover player differentiation. Each run ~2 min.
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
source ~/dwjax-venv/bin/activate
cd ~/DiamondWorld
S="--steps 6000 --patience 1500"
run(){ echo ">>> $*"; python -m diamondworldjax.scripts.wm_sweep $S "$@" 2>>data/eval2/wm_sweep_err.log; }

run --arch mlp
run --arch mlp --embed-dropout 0.3
run --arch transformer
run --arch transformer --embed-dropout 0.3
run --arch transformer --embed-dropout 0.5
run --arch transformer --dropout 0.1 --wd 1e-3
run --arch transformer --embed-dropout 0.3 --label-smooth 0.05
run --arch transformer --layers 6 --dm 192
run --arch transformer --embed-dropout 0.3 --dropout 0.1 --wd 1e-3
run --arch gru
run --arch lstm --embed-dropout 0.3
run --arch transformer --embed-dropout 0.3 --ensemble 3
echo "WM_SWEEP_DONE $(date -u +%FT%TZ)"
