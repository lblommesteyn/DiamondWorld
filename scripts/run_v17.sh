#!/usr/bin/env bash
# v17: three structural variants, each a SINGLE lever on top of the v16 recipe.
#
# v16 = v15 + --contact-quality, and it is the current best model at AVG 0.611
# player-corr. The paired bootstrap (bootstrap_playercorr.py) confirms v16 > v15
# is real: +0.017 [+0.004, +0.029], p=0.009, driven by K and HR.
#
# Every variant here keeps v16's recipe EXACTLY and changes one thing, so any
# delta is attributable. All are gated the same way: a paired bootstrap CI on the
# difference from v16 that must exclude zero. A point-estimate improvement is not
# enough; that is the standard this project previously lacked.
#
#   v17a  --bilinear-rank 8    Low-rank batter x pitcher interaction on the
#         logits. HYPOTHESIS: a PA is a matchup, and the baseline forces an MLP
#         to find interactions in a concatenation. The v11 platoon lever is the
#         existing evidence this fails, since it had to feed one interaction in
#         by hand. FALSIFIED IF: AVG delta CI includes zero, which would mean the
#         concatenated MLP already captures whatever matchup structure exists and
#         the platoon result was specific to handedness rather than general.
#
#   v17b  --nested             Two-stage head: {K,BB,HBP,in-play}, then in-play
#         -> {1B,2B,3B,HR,out,E}. HYPOTHESIS: plate discipline and contact
#         quality are different skills driven by different features, and a flat
#         9-way softmax makes one representation serve both. v16's own result
#         points here: contact features moved HR and hit while leaving BB exactly
#         flat. Expect gains concentrated in Hit, the project's worst stat
#         (0.422 vs Steamer's 0.510) and the one it calls BABIP-limited.
#         FALSIFIED IF: Hit delta CI includes zero.
#
#   v17c  --skill-prior learned  Per-dimension learned prior scale on the player
#         latent instead of a fixed N(0,I). HYPOTHESIS: the metric is a shrinkage
#         problem, and shrinkage strength is currently pinned at 1.0 rather than
#         fitted. FALSIFIED IF: AVG delta CI includes zero, which would say the
#         posterior already adapts enough through the likelihood alone.
#
# Deliberately NOT run here: --skill-prior lkj. player_skills is already pushed
# through SkillFusionLayer, a Dense map, and a linear map of an isotropic
# Gaussian is already a correlated Gaussian, so the model can represent
# correlated skills today. LKJ would add correlation to the PRIOR, not new
# expressive power. It is implemented and smoke-tested, but running it would
# spend 5.2h to test a mostly-redundant hypothesis; v17c is the part of that idea
# with a real mechanism behind it.
set -u
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
source ~/dwjax-venv/bin/activate
cd ~/DiamondWorld

RC=data/eval2/v13_cal_params.npz
COMMON="--outcome-only --fatigue --recency-halflife 2.0 --train-end 2023 --contact-quality"
EVAL_COMMON="--recal $RC --recency-halflife 2.0 --skill-mode mean --train-end 2023 \
--test-seasons 2024 --contact-quality"

echo "START $(date -u +%FT%TZ)" > data/run_v17_done.txt

# --- 0. Reproduce v16 with the flag it was TRAINED with -------------------
# The v16 eval was run ad hoc and is not in any committed script, so it is not
# provable from the repo that it passed --contact-quality. If it did not, the
# published 0.611 was measured with a player table that does not match the
# checkpoint, and every comparison below would inherit that error. Cheap to
# settle: re-run it and check we land back on 0.611.
echo "[v17] 0. re-verifying v16 with --contact-quality $(date)" >> data/run_v17_done.txt
python -m diamondworldjax.scripts.prod_playercorr \
  --ckpt checkpoints/dwjax_pa_v16/dwjax_step_0050000.pkl $EVAL_COMMON \
  --tag v16_recheck > data/eval2/v16_recheck.log 2>&1
tail -1 data/eval2/prod_playercorr_v16_recheck.txt >> data/run_v17_done.txt

run_variant () {
  local tag="$1"; shift
  local flags="$*"
  echo "[v17] training $tag ($flags) $(date)" >> data/run_v17_done.txt
  python -m diamondworldjax.scripts.train_pa --steps 50000 $COMMON \
    --tag "$tag" $flags > "data/train_${tag}.log" 2>&1
  local rc=$?
  echo "[v17] $tag train rc=$rc $(date)" >> data/run_v17_done.txt
  [ $rc -ne 0 ] && return $rc

  echo "[v17] evaluating $tag $(date)" >> data/run_v17_done.txt
  python -m diamondworldjax.scripts.prod_playercorr \
    --ckpt "checkpoints/dwjax_pa_${tag}/dwjax_step_0050000.pkl" $EVAL_COMMON \
    --tag "$tag" $flags > "data/eval2/${tag}_eval.log" 2>&1
  echo "[v17] $tag eval rc=$? $(date)" >> data/run_v17_done.txt
  tail -1 "data/eval2/prod_playercorr_${tag}.txt" >> data/run_v17_done.txt
}

run_variant v17a --bilinear-rank 8
run_variant v17b --nested
run_variant v17c --skill-prior learned

# --- 4. Gate every variant against v16 with a paired bootstrap CI ----------
echo "[v17] bootstrap gate $(date)" >> data/run_v17_done.txt
RATES="--rates v16=data/eval2/prod_rates_v16.npz"
for t in v17a v17b v17c; do
  [ -f "data/eval2/prod_rates_${t}.npz" ] && RATES="$RATES --rates ${t}=data/eval2/prod_rates_${t}.npz"
done
python -m diamondworldjax.scripts.bootstrap_playercorr $RATES \
  --baseline v16 --reps 20000 \
  --out data/eval2/bootstrap_v17.txt --json-out data/eval2/bootstrap_v17.json \
  >> data/run_v17_done.txt 2>&1

echo "DONE $(date -u +%FT%TZ)" >> data/run_v17_done.txt
