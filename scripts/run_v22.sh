set -u
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
source .venv/bin/activate

RC=data/eval2/v13_cal_params.npz
echo "START $(date -u +%FT%TZ)" > data/run_v22_done.txt

# Sanity-check the feature transform before spending GPU: shrinkage must pull
# low-PA players toward the league rate and leave high-PA players alone.
python - >> data/run_v22_done.txt 2>&1 <<'PY'
import numpy as np
reg = {"hit": 2200.0, "bb": 400.0, "k": 200.0, "hr": 2200.0}
league, raw = 0.20, 0.60
for n in (150.0, 3000.0):
    p = {k: (n * raw + c * league) / (n + c) for k, c in reg.items()}
    print(f"  n={n:6.0f} PA raw={raw}: " +
          " ".join(f"{k}={p[k]:.3f}" for k in ("k", "bb", "hit", "hr")))
print("  (K should retain most raw signal at low PA; hit/HR should sit near league)")
PY

python -m diamondworldjax.scripts.train_pa --steps 50000 \
  --outcome-only --fatigue --recency-halflife 2.0 \
  --train-end 2023 --contact-quality --per-stat-shrink \
  --skill-prior walk --seed 1 --tag v22 > data/train_v22.log 2>&1
echo "[v22] train rc=$? $(date)" >> data/run_v22_done.txt

python -m diamondworldjax.scripts.prod_playercorr \
  --ckpt checkpoints/dwjax_pa_v22/dwjax_step_0050000.pkl \
  --recal "$RC" --recency-halflife 2.0 --skill-mode mean \
  --train-end 2023 --test-seasons 2024 --contact-quality --per-stat-shrink \
  --skill-prior walk --tag v22 > data/eval2/v22_eval.log 2>&1
echo "[v22] eval rc=$? $(date)" >> data/run_v22_done.txt
tail -1 data/eval2/prod_playercorr_v22.txt >> data/run_v22_done.txt

python -m diamondworldjax.scripts.bootstrap_playercorr \
  --rates v16=data/eval2/prod_rates_v16.npz \
  --rates v22=data/eval2/prod_rates_v22.npz \
  --baseline v16 --reps 20000 \
  --out data/eval2/bootstrap_v22.txt --json-out data/eval2/bootstrap_v22.json \
  >> data/run_v22_done.txt 2>&1

echo "DONE $(date -u +%FT%TZ)" >> data/run_v22_done.txt
