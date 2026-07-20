#!/usr/bin/env bash
# Architecture comparison for per-PA outcome prediction, identical inputs+training:
# MLP baseline vs causal transformer (sequence attention) vs JEPA (latent-predictive
# pretrain + linear probe). Tests whether sequence structure or representation
# pretraining beats the hand-crafted context the production model already uses.
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
source ~/dwjax-venv/bin/activate
cd ~/DiamondWorld
echo "SEQ_START $(date -u +%FT%TZ)" > data/eval2/seq_done.txt
for A in mlp transformer jepa; do
  echo "=== $A ==="
  python -m diamondworldjax.scripts.seq_models --arch $A --steps 4000 --batch 64 \
    > data/eval2/seq_$A.log 2>&1
  echo "  $A done: $(cat data/eval2/seq_$A.txt 2>/dev/null)"
done
echo "SEQ_DONE $(date -u +%FT%TZ)" >> data/eval2/seq_done.txt
