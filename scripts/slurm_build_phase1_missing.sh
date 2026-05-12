#!/usr/bin/env bash
#SBATCH --job-name=dw-phase1-missing
#SBATCH --account=def-pviswana
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --array=2021,2023%1
#SBATCH --output=logs/dw_phase1_missing_%A_%a.out
#SBATCH --error=logs/dw_phase1_missing_%A_%a.err

set -euo pipefail

cd /scratch/lblommes/diamondworld
source scripts/use_nibi_env.sh
python scripts/build_season.py --season "${SLURM_ARRAY_TASK_ID}" --api-sleep-seconds 0.02
