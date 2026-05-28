#!/usr/bin/env bash
#SBATCH --job-name=dw-train-gct-40ep
#SBATCH --account=def-pviswana
#SBATCH --partition=gpubase_bygpu_b4
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/dw_train_gct_40ep_%j.out

set -euo pipefail
cd /scratch/lblommes/diamondworld
mkdir -p logs

source scripts/use_nibi_env.sh

# Retrain Phase 4 GCT for 40 epochs using GameSequenceDataset (train_game_context.py).
# This uses only features the simulator can provide — avoids the train/sim mismatch
# that caused phase4_transformer (trained with GameDataset) to score +3.77 runs/game.
python -m diamondworld.models.train_game_context \
    --epochs 40 \
    --batch-size 8 \
    --device cuda \
    --output-dir checkpoints/gct_40ep
