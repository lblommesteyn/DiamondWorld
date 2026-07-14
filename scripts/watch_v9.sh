#!/usr/bin/env bash
source ~/dwjax-venv/bin/activate
cd ~/DiamondWorld
while [ ! -f data/train_v9_done.txt ]; do sleep 60; done
CKPT=$(ls -t checkpoints/dwjax_pa_v9/*.pkl | head -1)
echo "evaluating $CKPT" > data/eval_v9_status.txt
python -m diamondworldjax.scripts.simulate_games --ckpt $CKPT --outcome-only --fatigue --use-park --limit-games 512 > data/eval_v9_sim_raw.log 2>&1
python -m diamondworldjax.scripts.simulate_games --ckpt $CKPT --outcome-only --fatigue --use-park --recal --recal-scale 0.4 --limit-games 512 --dump-runs data/v9_runs.npy --player-stats > data/eval_v9_sim_recal.log 2>&1
python -m diamondworldjax.scripts.compare_baselines --v6-runs data/v9_runs.npy > data/compare_v9.log 2>&1
echo DONE > data/eval_v9_done.txt
