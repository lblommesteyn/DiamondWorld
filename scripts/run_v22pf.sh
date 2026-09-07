set -u
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
source .venv/bin/activate

RC=data/eval2/v13_cal_params.npz
CKPT=checkpoints/dwjax_pa_v22pf/dwjax_step_0050000.pkl
echo "START $(date -u +%FT%TZ)" > data/run_v22pf_done.txt

# Sanity-check the feature transform before spending GPU: shrinkage must pull
# low-PA players toward the league rate and leave high-PA players alone.
python - >> data/run_v22pf_done.txt 2>&1 <<'PY'
import numpy as np
reg = {"hit": 2200.0, "bb": 400.0, "k": 200.0, "hr": 2200.0}
league, raw = 0.20, 0.60
for n in (150.0, 3000.0):
    p = {k: (n * raw + c * league) / (n + c) for k, c in reg.items()}
    print(f"  n={n:6.0f} PA raw={raw}: " +
          " ".join(f"{k}={p[k]:.3f}" for k in ("k", "bb", "hit", "hr")))
print("  (K should retain most raw signal at low PA; hit/HR should sit near league)")
PY

if [[ -f "$CKPT" ]]; then
  echo "[v22pf] using existing checkpoint $CKPT $(date)" >> data/run_v22pf_done.txt
else
  python -m diamondworldjax.scripts.train_pa --steps 50000 \
    --outcome-only --fatigue --recency-halflife 2.0 \
    --train-end 2023 --contact-quality --per-stat-shrink \
    --skill-prior walk --seed 1 --pitchformer --tag v22pf > data/train_v22pf.log 2>&1
  echo "[v22pf] train rc=$? $(date)" >> data/run_v22pf_done.txt
fi

python -m diamondworldjax.scripts.prod_playercorr \
  --ckpt "$CKPT" \
  --recal "$RC" --recency-halflife 2.0 --skill-mode mean \
  --train-end 2023 --test-seasons 2024 --contact-quality --per-stat-shrink \
  --skill-prior walk --pitchformer --tag v22pf > data/eval2/v22pf_eval.log 2>&1
echo "[v22pf] eval rc=$? $(date)" >> data/run_v22pf_done.txt
tail -1 data/eval2/prod_playercorr_v22pf.txt >> data/run_v22pf_done.txt

python -m diamondworldjax.scripts.bootstrap_playercorr \
  --rates v16=data/eval2/prod_rates_v16.npz \
  --rates v22pf=data/eval2/prod_rates_v22pf.npz \
  --baseline v16 --reps 20000 \
  --out data/eval2/bootstrap_v22pf.txt --json-out data/eval2/bootstrap_v22pf.json \
  >> data/run_v22pf_done.txt 2>&1

echo "DONE $(date -u +%FT%TZ)" >> data/run_v22pf_done.txt
