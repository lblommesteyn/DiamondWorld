#!/usr/bin/env bash
# v13 = v12 recipe (outcome-only + fatigue + park + recency half-life 2) WITH the
# minibatch player_skills KL-scale fix (now default). The fix scales the global
# player-skill KL to the minibatch fraction (~0.0036), so the latent is no longer
# ~280x over-regularized and can actually learn instead of collapsing to the prior.
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
source ~/dwjax-venv/bin/activate
cd ~/DiamondWorld
echo "START $(date -u +%FT%TZ)" > data/train_v13_done.txt
python -m diamondworldjax.scripts.train_pa --steps 50000 \
  --outcome-only --fatigue --recency-halflife 2.0 --tag v13 \
  > data/train_v13.log 2>&1
echo "DONE $(date -u +%FT%TZ)" >> data/train_v13_done.txt
