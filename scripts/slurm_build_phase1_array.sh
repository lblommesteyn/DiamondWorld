#!/usr/bin/env bash
#SBATCH --job-name=dw-phase1
#SBATCH --account=def-pviswana
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --array=2015-2023%1
#SBATCH --output=logs/dw_phase1_%A_%a.out

set -euo pipefail

SEASON="${SLURM_ARRAY_TASK_ID}"

cd /scratch/lblommes/diamondworld
mkdir -p logs
source scripts/use_nibi_env.sh

python scripts/build_season.py --season "${SEASON}" --api-sleep-seconds 0.02
