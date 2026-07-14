#!/usr/bin/env bash
source ~/dwjax-venv/bin/activate
cd ~/DiamondWorld
CKPT=checkpoints/dwjax_pa_v8/dwjax_step_0050000.pkl
python -m diamondworldjax.scripts.simulate_games --ckpt $CKPT --outcome-only --fatigue --recal --recal-scale 0.4 --limit-games 512 --fixed-nine --no-bullpen > data/simv2_ab_legacy.log 2>&1
python -m diamondworldjax.scripts.simulate_games --ckpt $CKPT --outcome-only --fatigue --recal --recal-scale 0.4 --limit-games 512 > data/simv2_ab_new.log 2>&1
echo DONE > data/simv2_ab_done.txt
