#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

module --force purge
module load StdEnv/2023
module load python/3.11.5
module load scipy-stack/2026a
module load arrow/23.0.1

python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m ensurepip --upgrade
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-index "polars==1.37.1+computecanada"
python -m pip install -e .

echo "DiamondWorld Nibi environment ready."
echo "Next:"
echo "  cd /scratch/lblommes/diamondworld"
echo "  source scripts/use_nibi_env.sh"
echo "  python scripts/build_season.py --season 2024"
