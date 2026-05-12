#!/usr/bin/env bash
#SBATCH --job-name=dw-season
#SBATCH --account=def-pviswana
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=logs/dw_build_%x_%j.out

set -euo pipefail

SEASON="${1:-2024}"

cd /scratch/lblommes/diamondworld
mkdir -p logs
source scripts/use_nibi_env.sh

python scripts/build_season.py --season "${SEASON}" --api-sleep-seconds 0.05
