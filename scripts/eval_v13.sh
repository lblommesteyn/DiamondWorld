#!/usr/bin/env bash
# v13 eval: did the KL-scale fix (un-collapsed latent) improve the model? Uses
# --skill-mode mean everywhere so the LEARNED player_mu is actually read.
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
source ~/dwjax-venv/bin/activate
cd ~/DiamondWorld
V12=checkpoints/dwjax_pa_v12/dwjax_step_0050000.pkl
V13=checkpoints/dwjax_pa_v13/dwjax_step_0050000.pkl
OUT=data/eval2
echo "V13_EVAL_START $(date -u +%FT%TZ)" > "$OUT/v13_eval_done.txt"

echo "=== 1. v12 baseline, skill-mode mean (fair comparison) ==="
python -m diamondworldjax.scripts.prod_playercorr --ckpt "$V12" --recal "$OUT/v12_cal_params.npz" \
  --recency-halflife 2.0 --skill-mode mean --tag v12mean > "$OUT/pc_v12mean.log" 2>&1
cat "$OUT/prod_playercorr_v12mean.txt"

echo "=== 2. derive v13 recal (skill-mode mean) ==="
python -m diamondworldjax.scripts.diag_outcomes --ckpt "$V13" --outcome-only --fatigue --use-park \
  --recency-halflife 2.0 --skill-mode mean --dump-logits "$OUT/v13_logits.npz" > "$OUT/v13_diag.txt" 2>&1
python -m diamondworldjax.scripts.fit_calibration --logits "$OUT/v13_logits.npz" \
  --out "$OUT/v13_cal.txt" > "$OUT/v13_fitcal.log" 2>&1

echo "=== 3. v13 player differentiation (skill-mode mean) ==="
python -m diamondworldjax.scripts.prod_playercorr --ckpt "$V13" --recal "$OUT/v13_cal_params.npz" \
  --recency-halflife 2.0 --skill-mode mean --tag v13 > "$OUT/pc_v13.log" 2>&1
cat "$OUT/prod_playercorr_v13.txt"

echo "=== 4. v13 conditioned player stats ==="
python -m diamondworldjax.scripts.eval_players --ckpt "$V13" --outcome-only --fatigue --use-park \
  --recency-halflife 2.0 --skill-mode mean --min-pa 150 2>&1 | sed -n "/Player-level/,/corr =/p" | tee "$OUT/v13_players.txt"

echo "=== 5. hybrid: v13 (SVI) + best discriminative MLP ==="
cp "$OUT/prod_rates_v13.npz" "$OUT/prod_rates.npz"
python -m diamondworldjax.scripts.combine_hybrid 2>&1 | tee "$OUT/hybrid_v13.txt"

echo "V13_EVAL_DONE $(date -u +%FT%TZ)" >> "$OUT/v13_eval_done.txt"
