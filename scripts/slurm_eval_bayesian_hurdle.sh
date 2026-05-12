#!/usr/bin/env bash
#SBATCH --job-name=dw-eval-bhurdle
#SBATCH --account=def-pviswana_cpu
#SBATCH --partition=cpubase_bycore_b3
#SBATCH --time=0:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/dw_eval_bhurdle_%j.out

set -euo pipefail
cd /scratch/lblommes/diamondworld
mkdir -p logs

source scripts/use_nibi_env.sh

python -m diamondworld.scripts.eval_bayesian_hurdle
