#!/usr/bin/env bash
# JEPA collapse investigation: does fine-tuning (or dropping the frozen probe)
# recover the player differentiation that the frozen SSL probe loses (0.10)?
# Three modes on the SAME encoder and the +prev-season features the arch sweep used
# (recency 2.0, train through 2023, test 2024), so the numbers sit next to the
# transformer/GRU/MLP (~0.577) and v16 SVI (0.611) already on record:
#   frozen   = SSL pretrain + frozen linear probe   (the collapse baseline)
#   finetune = SSL pretrain, then fine-tune the encoder end-to-end
#   scratch  = no SSL pretrain, same encoder trained supervised (isolates SSL's value)
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
source ~/dwjax-venv/bin/activate
cd ~/DiamondWorld
echo "START $(date -u +%FT%TZ)" > data/jepa_modes_done.txt
for mode in frozen finetune scratch; do
  echo "[jepa] $mode $(date -u +%H:%M:%S)" >> data/jepa_modes_done.txt
  python -m diamondworldjax.scripts.seq_models --arch jepa --jepa-mode "$mode" \
    --steps 4000 --batch 64 --train-end 2023 --recency-halflife 2.0 \
    --test-seasons 2024 --tag "jepa-$mode" >> "data/jepa_$mode.log" 2>&1
  echo "[jepa] $mode rc=$? $(date -u +%H:%M:%S)" >> data/jepa_modes_done.txt
done
echo "DONE $(date -u +%FT%TZ)" >> data/jepa_modes_done.txt
