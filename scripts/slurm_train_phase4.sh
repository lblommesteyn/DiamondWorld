#!/usr/bin/env bash
#SBATCH --job-name=dw-phase4-transformer
#SBATCH --account=def-pviswana
#SBATCH --partition=gpubase_bygpu_b3
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/dw_train_phase4_%j.out

set -euo pipefail
cd /scratch/lblommes/diamondworld
mkdir -p logs checkpoints/phase4_transformer
source scripts/use_nibi_env.sh
python diamondworld/models/train_phase4.py --epochs 50 --batch-size 8 --device cuda
