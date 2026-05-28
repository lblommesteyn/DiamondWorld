#!/usr/bin/env bash
#SBATCH --job-name=dw-svi-arma
#SBATCH --account=def-pviswana_gpu
#SBATCH --partition=gpubase_bygpu_b3
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/dw_svi_arma_%j.out

set -euo pipefail
cd /scratch/lblommes/diamondworld
mkdir -p logs

source scripts/use_nibi_env.sh

python -m diamondworld.scripts.eval_svi_arma
