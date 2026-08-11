#!/usr/bin/env bash
# v19: re-run the two skill-prior variants against the CORRECTED guide.
#
# v17c and v17d were invalidated by a guide-coverage bug. The hand-written SVI
# guide covered only `player_skills`, and a NumPyro guide that omits a latent does
# not raise: under Trace_ELBO the missing site is drawn from its prior every step
# and never learned. So v17c's skill_tau, and v17d's skill_tau plus skill_L, were
# resampled as fresh noise on every step. Both runs measured noise injected into
# the prior rather than a learned prior, and their conclusion (that freeing the
# prior monotonically hurts) was withdrawn.
#
# Evidence it was real, not a guess: the v17c/v17d checkpoints contain exactly
# v16's parameter set, with no entry for skill_tau or skill_L.
#
# The guide now dispatches per prior and scripts/_check_guide_coverage.py asserts
# every latent is covered. These are the honest versions of the two experiments.
#
#   v19c  --skill-prior learned   per-dimension learned prior scale
#   v19d  --skill-prior lkj       learned scale plus LKJ correlation
#
# PREDICTION, recorded before the fact and deliberately DIFFERENT from last time.
# The retracted runs are no guide to the outcome, since they measured something
# else. The metric is a shrinkage problem and these variants fit the shrinkage
# strength from data instead of pinning it at 1.0, so the honest prior is now
# "plausible small win, or null". If they regress again with the guide fixed,
# THAT is the finding the retracted runs falsely claimed, and it would then be
# properly earned.
#
# GATE: paired bootstrap vs v16 must exclude zero on AVG.
set -u
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
source ~/dwjax-venv/bin/activate
cd ~/DiamondWorld

RC=data/eval2/v13_cal_params.npz
COMMON="--outcome-only --fatigue --recency-halflife 2.0 --train-end 2023 --contact-quality"
EVAL_COMMON="--recal $RC --recency-halflife 2.0 --skill-mode mean --train-end 2023 \
--test-seasons 2024 --contact-quality"

echo "START $(date -u +%FT%TZ)" > data/run_v19_done.txt

# Guard: refuse to burn GPU hours if the guide does not cover every latent.
python scripts/_check_guide_coverage.py > data/eval2/guide_coverage.txt 2>&1
if grep -q UNCOVERED data/eval2/guide_coverage.txt; then
  echo "ABORT: guide coverage check failed" >> data/run_v19_done.txt
  cat data/eval2/guide_coverage.txt >> data/run_v19_done.txt
  exit 1
fi
echo "[v19] guide coverage OK" >> data/run_v19_done.txt

run_variant () {
  local tag="$1"; shift
  local flags="$*"
  echo "[v19] training $tag ($flags) $(date)" >> data/run_v19_done.txt
  python -m diamondworldjax.scripts.train_pa --steps 50000 $COMMON \
    --tag "$tag" $flags > "data/train_${tag}.log" 2>&1
  echo "[v19] $tag train rc=$? $(date)" >> data/run_v19_done.txt

  # Confirm the hyperparameters actually got fitted this time.
  python - "$tag" >> data/run_v19_done.txt 2>&1 <<'PY'
import pickle, sys
p = pickle.load(open(f"checkpoints/dwjax_pa_{sys.argv[1]}/dwjax_step_0050000.pkl","rb"))["params"]
extra = sorted(k for k in p if k not in (
    "pa_outcome_head_v6$params","park_embedding$params",
    "player_encoder$params","player_encoder_skill_fusion$params",
    "player_mu","player_sigma"))
print(f"  [{sys.argv[1]}] learned hyperparams present: {extra if extra else 'NONE (BUG NOT FIXED)'}")
PY

  python -m diamondworldjax.scripts.prod_playercorr \
    --ckpt "checkpoints/dwjax_pa_${tag}/dwjax_step_0050000.pkl" $EVAL_COMMON \
    --tag "$tag" $flags > "data/eval2/${tag}_eval.log" 2>&1
  echo "[v19] $tag eval rc=$? $(date)" >> data/run_v19_done.txt
  tail -1 "data/eval2/prod_playercorr_${tag}.txt" >> data/run_v19_done.txt
}

run_variant v19c --skill-prior learned
run_variant v19d --skill-prior lkj

RATES="--rates v16=data/eval2/prod_rates_v16.npz"
for t in v19c v19d; do
  [ -f "data/eval2/prod_rates_${t}.npz" ] && RATES="$RATES --rates ${t}=data/eval2/prod_rates_${t}.npz"
done
python -m diamondworldjax.scripts.bootstrap_playercorr $RATES \
  --baseline v16 --reps 20000 \
  --out data/eval2/bootstrap_v19.txt --json-out data/eval2/bootstrap_v19.json \
  >> data/run_v19_done.txt 2>&1

echo "DONE $(date -u +%FT%TZ)" >> data/run_v19_done.txt
