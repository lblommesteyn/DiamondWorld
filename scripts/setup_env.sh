#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python -m venv .venv
source .venv/bin/activate
python -m ensurepip --upgrade
python -m pip install --upgrade pip
python -m pip install -e .

echo "DiamondWorld environment ready."
echo "Activate it with: source /scratch/lblommes/diamondworld/.venv/bin/activate"
