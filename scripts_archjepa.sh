#!/bin/bash
# Corrected architecture x new-features sweep (KIDX fixed to K=0) + JEPA on player-corr.
cd ~/DiamondWorld
source ~/dwjax-venv/bin/activate
OUT=data/eval2/arch_corrected.txt
LOG=/tmp/archjepa.log
echo "CORRECTED ARCH x NEW-FEATURES SWEEP (K=0 fixed; recency 2.0; test 2024; player-corr AVG)" > $OUT
echo "baseline=train<=2022, +prev=train<=2023" >> $OUT
echo "[archjepa] start $(date)" > $LOG

SW="python -m diamondworldjax.scripts.wm_sweep --recency-halflife 2.0 --test-seasons 2024"
runsw(){ label="$1"; shift; echo "### $label" >> $OUT; $SW "$@" 2>>/tmp/archjepa.err | tail -1 >> $OUT; echo "[archjepa] done $label $(date)" >> $LOG; }
runsw "mlp ens3      baseline" --arch mlp --layers 2 --dm 128 --embed-dropout 0.1 --ensemble 3 --wd 1e-4 --train-end 2022
runsw "mlp ens3      +prev"    --arch mlp --layers 2 --dm 128 --embed-dropout 0.1 --ensemble 3 --wd 1e-4 --train-end 2023
runsw "transformer   baseline" --arch transformer --layers 3 --dm 128 --heads 4 --train-end 2022
runsw "transformer   +prev"    --arch transformer --layers 3 --dm 128 --heads 4 --train-end 2023
runsw "gru           baseline" --arch gru --layers 2 --dm 128 --train-end 2022
runsw "gru           +prev"    --arch gru --layers 2 --dm 128 --train-end 2023

# JEPA (SSL pretrain + frozen linear probe) — the world-model architecture, now on player-corr
JEPA="python -m diamondworldjax.scripts.seq_models --arch jepa --recency-halflife 2.0 --test-seasons 2024 --steps 4000"
runjepa(){ label="$1"; shift; echo "### $label" >> $OUT; $JEPA "$@" > /tmp/jepa_run.log 2>>/tmp/archjepa.err
           grep "corr K" /tmp/jepa_run.log | tail -1 >> $OUT
           grep "per-PA NLL" /tmp/jepa_run.log | tail -1 >> $OUT
           echo "[archjepa] done $label $(date)" >> $LOG; }
runjepa "jepa         baseline" --train-end 2022 --tag "jepa baseline (train<=2022)"
runjepa "jepa         +prev"    --train-end 2023 --tag "jepa +prev   (train<=2023)"

echo "=== DONE ===" >> $OUT
echo "[archjepa] ALL DONE $(date)" >> $LOG
