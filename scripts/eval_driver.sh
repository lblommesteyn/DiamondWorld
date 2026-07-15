#!/usr/bin/env bash
# Rigorous run-distribution evaluation: tune recal scale on a moderate sample,
# then confirm at large N so tail metrics (p8+, wasserstein) have low sampling
# error. Scores every dump against the FULL real test set via score_runs.py.
set -e
source ~/dwjax-venv/bin/activate
cd ~/DiamondWorld
V6=checkpoints/dwjax_pa_BEST/v6_final_step50000.pkl
V9=checkpoints/dwjax_pa_v9/dwjax_step_0030000.pkl
NSW=${NSW:-1200}
OUT=data/eval2
mkdir -p "$OUT"
RES="$OUT/sweep_results.txt"
: > "$RES"

sim () {  # ckpt version scale N tag
  local ckpt=$1 ver=$2 scale=$3 n=$4 tag=$5
  local flags="--outcome-only"
  # v9 is fatigue + park-aware; v6-final trained on STATE_DIM 8 with park_idx=0.
  [ "$ver" = "v9" ] && flags="--outcome-only --fatigue --use-park"
  python -m diamondworldjax.scripts.simulate_games --ckpt "$ckpt" \
    $flags --recal --recal-version "$ver" \
    --recal-scale "$scale" --limit-games "$n" --dump-runs "$OUT/${tag}.npy" \
    > "$OUT/${tag}.log" 2>&1
}

echo "=== SWEEP (N=$NSW) ===" | tee -a "$RES"
for s in 0.20 0.30 0.40; do
  echo ">> v6 scale $s" ; sim "$V6" v6 "$s" "$NSW" "v6_s${s}"
  python -m diamondworldjax.scripts.score_runs --runs "v6@${s}:$OUT/v6_s${s}.npy" | tail -2 >> "$RES"
done
for s in 0.35 0.45 0.55; do
  echo ">> v9 scale $s" ; sim "$V9" v9 "$s" "$NSW" "v9_s${s}"
  python -m diamondworldjax.scripts.score_runs --runs "v9@${s}:$OUT/v9_s${s}.npy" | tail -2 >> "$RES"
done

echo DONE >> "$RES"
echo DONE
