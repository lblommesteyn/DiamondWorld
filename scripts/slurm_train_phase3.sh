#!/usr/bin/env bash
#SBATCH --job-name=dw-phase3-mlp
#SBATCH --account=def-pviswana
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/dw_train_phase3_%j.out

set -euo pipefail

cd /scratch/lblommes/diamondworld
mkdir -p logs checkpoints/no_memory_mlp

source scripts/use_nibi_env.sh

python diamondworld/models/train_no_memory.py \
    --epochs 50 \
    --batch-size 2048 \
    --device cuda
