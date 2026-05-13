#!/usr/bin/env bash
#SBATCH --job-name=dwjax-train-v0
#SBATCH --account=def-pviswana_gpu
#SBATCH --partition=gpubase_bygpu_b3
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/dwjax_train_v0_%j.out

set -euo pipefail
cd /scratch/lblommes/diamondworld
mkdir -p logs

source scripts/use_nibi_env.sh

python -m diamondworldjax.scripts.train_v0 \
    --steps  50000 \
    --lr     1e-3  \
    --rank   20    \
    --seed   0     \
    --batch  32
