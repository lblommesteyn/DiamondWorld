#!/bin/bash
# v15 betting backtest: pre-game-only (no-bullpen), skill=mean, 2024 real closing lines.
cd ~/DiamondWorld
source ~/dwjax-venv/bin/activate
LOG=/tmp/v15bet.log
echo "[v15bet] start $(date)" > $LOG

CK=checkpoints/dwjax_pa_v15/dwjax_step_0035000.pkl
ARR=data/eval2/calib_v15_bt_arrays.npz

# 1) per-game predictive sim (2024 only; player table over 2015-2023 to match v15)
python -m diamondworldjax.scripts.calib_audit \
  --ckpt $CK --outcome-only --fatigue --use-park --recency-halflife 2.0 \
  --skill-mode mean --no-bullpen --train-end 2023 --test-seasons 2024 \
  --recal --recal-version v10 --recal-scale 0.18 \
  --limit-games 4000 --replicas 40 --chunk-games 300 \
  --out $ARR >> $LOG 2>&1
echo "[v15bet] calib_audit done $(date)" >> $LOG

# 2) backtest vs real closing lines (moneyline + totals + runline), edge sweep
for M in moneyline totals runline; do
  echo "===== v15 $M (pre-game/no-bullpen, 2024) =====" >> data/eval2/backtest_v15.txt
  for E in 0.0 0.02 0.04 0.06 0.10; do
    python -m diamondworldjax.scripts.backtest --arrays $ARR \
      --odds data/eval2/odds_2023_2024.csv --market $M --edge $E 2>>/tmp/v15bet.err \
      | grep -iE "ROI|bets|matched|units" | sed "s/^/  edge $E: /" >> data/eval2/backtest_v15.txt
  done
done
echo "[v15bet] ALL DONE $(date)" >> $LOG
