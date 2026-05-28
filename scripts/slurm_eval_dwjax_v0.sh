#!/usr/bin/env bash
#SBATCH --job-name=dwjax-eval-v0
#SBATCH --account=def-pviswana_gpu
#SBATCH --partition=gpubase_bygpu_b4
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/dwjax_eval_v0_%j.out

set -euo pipefail
cd /scratch/lblommes/diamondworld
mkdir -p logs eval/results

source scripts/use_nibi_env.sh

export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export CKPT="${CKPT:-checkpoints/dwjax_v0/dwjax_step_0005000.pkl}"
export BATCH="${BATCH:-32}"
export SAMPLES="${SAMPLES:-8}"
export SEED="${SEED:-0}"
export LIMIT_GAMES="${LIMIT_GAMES:-0}"

echo "GPU inventory:"
nvidia-smi -L

python -m diamondworldjax.scripts.eval_v0 \
    --ckpt        "${CKPT}"   \
    --batch       "${BATCH}"  \
    --samples     "${SAMPLES}" \
    --seed        "${SEED}"   \
    --limit-games "${LIMIT_GAMES}"
