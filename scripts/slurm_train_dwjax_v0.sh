#!/usr/bin/env bash
#SBATCH --job-name=dwjax-train-v0
#SBATCH --account=def-pviswana_gpu
#SBATCH --partition=gpubase_bygpu_b4
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/dwjax_train_v0_%j.out

set -euo pipefail
cd /scratch/lblommes/diamondworld
mkdir -p logs
mkdir -p checkpoints/dwjax_v0 eval/results

source scripts/use_nibi_env.sh

export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export STEPS="${STEPS:-50000}"
export LR="${LR:-1e-3}"
export RANK="${RANK:-20}"
export SEED="${SEED:-0}"
export BATCH="${BATCH:-32}"
export RESUME="${RESUME:-}"

echo "GPU inventory:"
nvidia-smi -L
echo "JAX devices:"
python - <<'PY'
import jax
print(jax.devices())
PY

python -m diamondworldjax.scripts.train_v0 \
    --steps  "${STEPS}"  \
    --lr     "${LR}"     \
    --rank   "${RANK}"   \
    --seed   "${SEED}"   \
    --batch  "${BATCH}"  \
    ${RESUME:+--resume "${RESUME}"}
