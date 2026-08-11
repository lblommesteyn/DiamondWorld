#!/usr/bin/env bash
# v17d: --skill-prior lkj. The last variant that was BUILT but never RUN.
#
# I argued this was largely redundant and skipped it: player_skills already
# passes through SkillFusionLayer, a Dense map, and a linear map of an isotropic
# Gaussian is already a correlated Gaussian, so LKJ adds correlation to the PRIOR
# (hence to the KL) rather than new expressive power. That was an assertion, not
# a measurement, and it is the only piece of the skill-prior proposal left
# untested. The GPU is idle, so it costs nothing but wall-clock to settle.
#
# PRIOR EXPECTATION, recorded before the fact: null or a regression. v17c, the
# strictly simpler version of this idea (learned per-dimension scale, no
# correlation), REGRESSED at -0.032 with its CI excluding zero, because the fixed
# unit prior was doing real regularisation work. LKJ frees strictly more of the
# prior than v17c did, so if that diagnosis is right this should be at least as
# bad. A WIN would mean the correlation structure buys back more than the extra
# freedom costs, which would reframe v17c's regression as a story about the scale
# parameter specifically rather than about freeing the prior in general.
#
# Smoke-tested clean at 300 steps (rc=0, ELBO -10000, no NaN), though it
# converged visibly slower than the other variants, consistent with the extra
# LKJ and tau latents.
#
# GATE: paired bootstrap vs v16 must exclude zero on AVG, same as every other
# variant in this series.
set -u
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
source ~/dwjax-venv/bin/activate
cd ~/DiamondWorld

RC=data/eval2/v13_cal_params.npz
echo "START $(date -u +%FT%TZ)" > data/run_v17d_done.txt

python -m diamondworldjax.scripts.train_pa --steps 50000 \
  --outcome-only --fatigue --recency-halflife 2.0 \
  --train-end 2023 --contact-quality \
  --skill-prior lkj --tag v17d > data/train_v17d.log 2>&1
echo "[v17d] train rc=$? $(date)" >> data/run_v17d_done.txt

# Unlike the aggregation loss, this one DOES need a matching eval flag: the prior
# changes the sample sites, so the checkpoint must be scored under the same model.
python -m diamondworldjax.scripts.prod_playercorr \
  --ckpt checkpoints/dwjax_pa_v17d/dwjax_step_0050000.pkl \
  --recal "$RC" --recency-halflife 2.0 --skill-mode mean \
  --train-end 2023 --test-seasons 2024 --contact-quality \
  --skill-prior lkj --tag v17d > data/eval2/v17d_eval.log 2>&1
echo "[v17d] eval rc=$? $(date)" >> data/run_v17d_done.txt
tail -1 data/eval2/prod_playercorr_v17d.txt >> data/run_v17d_done.txt

python -m diamondworldjax.scripts.bootstrap_playercorr \
  --rates v16=data/eval2/prod_rates_v16.npz \
  --rates v17c=data/eval2/prod_rates_v17c.npz \
  --rates v17d=data/eval2/prod_rates_v17d.npz \
  --baseline v16 --reps 20000 \
  --out data/eval2/bootstrap_v17d.txt --json-out data/eval2/bootstrap_v17d.json \
  >> data/run_v17d_done.txt 2>&1

echo "DONE $(date -u +%FT%TZ)" >> data/run_v17d_done.txt
