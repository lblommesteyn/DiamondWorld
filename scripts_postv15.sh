#!/bin/bash
# Detached post-v15 pipeline: survives harness/context-compaction kills (nohup).
# 1) wait for v15 training to finish, 2) v15 player-corr eval, 3) architecture x new-features sweep.
cd ~/DiamondWorld
source ~/dwjax-venv/bin/activate
CKPT=checkpoints/dwjax_pa_v15/dwjax_step_0035000.pkl   # resumed run: 15k + 35k = 50k-equiv

echo "[postv15] waiting for $CKPT ..." > /tmp/postv15.log
until [ -f "$CKPT" ]; do sleep 120; done
# wait for the training process to fully exit so the GPU is free
while ps aux | grep -q "[t]rain_pa"; do sleep 60; done
sleep 20
echo "[postv15] v15 done; starting eval $(date)" >> /tmp/postv15.log

# ---- v15 player-corr on 2024 (recal is a per-class bias; corr is insensitive to it) ----
RC=data/eval2/v13_cal_params.npz
COM="python -m diamondworldjax.scripts.prod_playercorr --ckpt $CKPT --recal $RC --recency-halflife 2.0 --skill-mode mean --train-end 2023 --test-seasons 2024"
{
  echo "BASELINE v13 (train<=2022, test 2024): AVG 0.503 (K .641 BB .570 Hit .280 HR .522, np=332)"
  echo "### v15 (train<=2023, test 2024): previous season folded into features"
  $COM --tag v15_2024 2>>/tmp/v15eval.err | tail -1
  echo "### v15 + MLE rookie fills (train<=2023, test 2024)"
  $COM --mle data/eval2/mle_rates.npz --tag v15_2024_mle 2>>/tmp/v15eval.err | tail -1
} | tee data/eval2/v15_playercorr.txt
echo "[postv15] v15 eval done $(date)" >> /tmp/postv15.log

# ---- architecture x new-features sweep (recency 2.0, test 2024) ----
OUT=data/eval2/arch_newfeatures.txt
echo "ARCHITECTURE x NEW-FEATURES SWEEP (recency 2.0, test 2024; player-corr AVG)" > $OUT
echo "baseline=train<=2022, +prev=train<=2023, +mle adds rookie fills" >> $OUT
SW="python -m diamondworldjax.scripts.wm_sweep --recency-halflife 2.0 --test-seasons 2024"
runcfg(){ label="$1"; shift; echo "### $label" | tee -a $OUT; $SW "$@" 2>>/tmp/arch.err | tail -1 | tee -a $OUT; }
runcfg "mlp ens3 ed0.1  baseline"   --arch mlp --layers 2 --dm 128 --embed-dropout 0.1 --ensemble 3 --wd 1e-4 --train-end 2022
runcfg "mlp ens3 ed0.1  +prev"      --arch mlp --layers 2 --dm 128 --embed-dropout 0.1 --ensemble 3 --wd 1e-4 --train-end 2023
runcfg "mlp ens3 ed0.1  +prev +mle" --arch mlp --layers 2 --dm 128 --embed-dropout 0.1 --ensemble 3 --wd 1e-4 --train-end 2023 --mle data/eval2/mle_rates.npz
runcfg "transformer L3  baseline"   --arch transformer --layers 3 --dm 128 --heads 4 --train-end 2022
runcfg "transformer L3  +prev"      --arch transformer --layers 3 --dm 128 --heads 4 --train-end 2023
runcfg "gru L2          baseline"   --arch gru --layers 2 --dm 128 --train-end 2022
runcfg "gru L2          +prev"      --arch gru --layers 2 --dm 128 --train-end 2023
echo "=== arch sweep DONE ===" | tee -a $OUT
echo "[postv15] ALL DONE $(date)" >> /tmp/postv15.log
