#!/usr/bin/env bash

module --force purge
module load StdEnv/2023
module load python/3.11.5
module load scipy-stack/2026a
module load arrow/23.0.1

source /scratch/lblommes/diamondworld/.venv/bin/activate
