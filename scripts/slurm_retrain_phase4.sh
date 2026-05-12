#!/usr/bin/env bash
#SBATCH --job-name=dw-train-phase4-v2
#SBATCH --account=def-pviswana
#SBATCH --partition=gpubase_bygpu_b3
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/dw_train_phase4_v2_%j.out

set -euo pipefail
cd /scratch/lblommes/diamondworld
mkdir -p logs

source scripts/use_nibi_env.sh

# Retrain Phase 4 GCT for 40 epochs (original was 9 -- likely undertrained).
# Checkpoint saved to checkpoints/game_context_transformer/final_model.pt
python -m diamondworld.models.train_phase4 \
    --epochs 40 \
    --batch-size 8 \
    --device cuda
