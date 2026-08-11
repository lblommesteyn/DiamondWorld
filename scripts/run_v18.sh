#!/usr/bin/env bash
# v18: attack the OBJECTIVE, not the architecture.
#
# WHY THIS AND NOT ANOTHER ARCHITECTURE. Every architecture lever this project
# has tried lands on zero: transformer / GRU / LSTM / MLP all cluster at ~0.577,
# and v17a (bilinear matchup) came back +0.003 with CI [-0.007, +0.012]. The
# feature levers, by contrast, move the number (v16 contact quality, +0.017 with
# CI excluding zero). That pattern has a single explanation.
#
# THE DIAGNOSIS. We train on per-PA likelihood and evaluate on cross-player rate
# correlation. Marginal outcome entropy is 1.495 nats; the best model reaches
# ~1.49. So the player-attributable share of the training signal is ~0.005 nats,
# about 0.3% of the loss. Roughly 99.7% of every gradient step goes into the
# league-average plate appearance, while 100% of the metric is player
# differentiation. Architecture changes add capacity to fit the saturated 99.7%,
# which is exactly why they do nothing.
#
# THE INTERVENTION. --player-agg-weight adds a squared-error term on per-BATTER
# aggregated rates (K, BB, Hit, HR) within each minibatch. This is not new
# information: the likelihood has the same optimum, and the aggregation gradient
# is unbiased for the same target. It is a REWEIGHTING. Cross-entropy weights
# every plate appearance equally, so high-PA batters dominate; aggregation
# weights every BATTER equally, which is the axis the metric measures.
#
# EVIDENCE THE SIGNAL IS STILL THERE (i.e. this is not chasing a ceiling):
# Steamer scores 0.671 against our 0.611, and more damningly, a Marcel-style
# estimator with contact quality reaches Hit 0.497 where v16 manages 0.422. A
# simple regularised average beats the full hierarchical model on that stat, so
# the information is present and extractable and our machinery is not using it.
#
# PREREGISTERED GATE: paired bootstrap vs v16 must exclude zero on AVG.
#
# PREREGISTERED FAILURE MODES, both real:
#   1. Null. The reweighting is too weak at this lambda, or player parameters
#      were not actually starved of gradient and the diagnosis is wrong.
#   2. REGRESSION. Pushed hard, this degenerates toward reproducing each
#      batter's own historical rate, and a rate-features-only model scores 0.438
#      against v16's 0.611. If v18 lands well below v16, that is evidence lambda
#      is too high, not that the idea is dead; the follow-up is a lower lambda,
#      not abandonment.
# lambda = 1.0 puts the aux term at roughly 10% of the total objective. One probe
# only: at ~6h/run a sweep is not affordable, so this is deliberately a single
# informative point rather than a curve.
set -u
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
source ~/dwjax-venv/bin/activate
cd ~/DiamondWorld

RC=data/eval2/v13_cal_params.npz
LAMBDA=1.0

echo "START $(date -u +%FT%TZ) lambda=$LAMBDA" > data/run_v18_done.txt

python -m diamondworldjax.scripts.train_pa --steps 50000 \
  --outcome-only --fatigue --recency-halflife 2.0 \
  --train-end 2023 --contact-quality \
  --player-agg-weight "$LAMBDA" --tag v18 > data/train_v18.log 2>&1
echo "[v18] train rc=$? $(date)" >> data/run_v18_done.txt

# NOTE: no matching flag at eval. The aggregation term is a numpyro.factor used
# only under teacher_force, so it adds no parameters and is skipped at eval; the
# checkpoint is scored exactly like v16.
python -m diamondworldjax.scripts.prod_playercorr \
  --ckpt checkpoints/dwjax_pa_v18/dwjax_step_0050000.pkl \
  --recal "$RC" --recency-halflife 2.0 --skill-mode mean \
  --train-end 2023 --test-seasons 2024 --contact-quality \
  --tag v18 > data/eval2/v18_eval.log 2>&1
echo "[v18] eval rc=$? $(date)" >> data/run_v18_done.txt
tail -1 data/eval2/prod_playercorr_v18.txt >> data/run_v18_done.txt

python -m diamondworldjax.scripts.bootstrap_playercorr \
  --rates v16=data/eval2/prod_rates_v16.npz \
  --rates v18=data/eval2/prod_rates_v18.npz \
  --baseline v16 --reps 20000 \
  --out data/eval2/bootstrap_v18.txt --json-out data/eval2/bootstrap_v18.json \
  >> data/run_v18_done.txt 2>&1

echo "DONE $(date -u +%FT%TZ)" >> data/run_v18_done.txt
