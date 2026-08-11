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
# ---------------------------------------------------------------------------
# v18b: lambda = 4.0. This run exists to resolve the one ambiguity v18 left open.
#
# v18 at lambda = 1.0 returned a PURE NULL (-0.001 AVG, CI [-0.014, +0.010]), not
# the regression that would have indicated lambda was too high. A pure null is
# consistent with two different worlds, and one run cannot separate them:
#   (a) the diagnosis is wrong, and player parameters were never gradient-starved;
#   (b) lambda = 1.0 was simply too small to bite.
# The obvious sanity check does not settle it either: the aux term's share of the
# objective VALUE is dominated by irreducible minibatch noise, so it does not
# bound its share of the informative GRADIENT.
#
# 4x the weight is the discriminating test, and every outcome is informative:
#   * NULL again -> the term does not move the solution even at 4x, which makes
#     (a) much more likely and lets the objective-mismatch hypothesis be reported
#     as refuted rather than merely unsupported.
#   * REGRESSION toward rate-copying (a rate-features-only model scores 0.438 vs
#     v16's 0.611) -> the term demonstrably bites, so lambda = 1.0 sat in a
#     sensible range and v18's null measured the real effect rather than a
#     too-weak knob.
#   * WIN -> the interior optimum sits above lambda = 1.0, which would overturn
#     the v17/v18 conclusion and reopen the objective axis.
set -u
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
source ~/dwjax-venv/bin/activate
cd ~/DiamondWorld

RC=data/eval2/v13_cal_params.npz
LAMBDA=4.0

echo "START $(date -u +%FT%TZ) lambda=$LAMBDA" > data/run_v18b_done.txt

python -m diamondworldjax.scripts.train_pa --steps 50000 \
  --outcome-only --fatigue --recency-halflife 2.0 \
  --train-end 2023 --contact-quality \
  --player-agg-weight "$LAMBDA" --tag v18b > data/train_v18b.log 2>&1
echo "[v18] train rc=$? $(date)" >> data/run_v18b_done.txt

# NOTE: no matching flag at eval. The aggregation term is a numpyro.factor used
# only under teacher_force, so it adds no parameters and is skipped at eval; the
# checkpoint is scored exactly like v16.
python -m diamondworldjax.scripts.prod_playercorr \
  --ckpt checkpoints/dwjax_pa_v18b/dwjax_step_0050000.pkl \
  --recal "$RC" --recency-halflife 2.0 --skill-mode mean \
  --train-end 2023 --test-seasons 2024 --contact-quality \
  --tag v18b > data/eval2/v18b_eval.log 2>&1
echo "[v18] eval rc=$? $(date)" >> data/run_v18b_done.txt
tail -1 data/eval2/prod_playercorr_v18b.txt >> data/run_v18b_done.txt

python -m diamondworldjax.scripts.bootstrap_playercorr \
  --rates v16=data/eval2/prod_rates_v16.npz \
  --rates v18b=data/eval2/prod_rates_v18b.npz \
  --baseline v16 --reps 20000 \
  --out data/eval2/bootstrap_v18b.txt --json-out data/eval2/bootstrap_v18b.json \
  >> data/run_v18b_done.txt 2>&1

echo "DONE $(date -u +%FT%TZ)" >> data/run_v18b_done.txt
