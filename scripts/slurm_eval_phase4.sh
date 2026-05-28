#!/usr/bin/env bash
#SBATCH --job-name=dw-eval-phase4
#SBATCH --account=def-pviswana
#SBATCH --partition=gpubase_bygpu_b4
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/dw_eval_phase4_%j.out

set -euo pipefail

cd /scratch/lblommes/diamondworld
mkdir -p logs

source scripts/use_nibi_env.sh

python -m diamondworld.scripts.eval_phase4 --device cuda
