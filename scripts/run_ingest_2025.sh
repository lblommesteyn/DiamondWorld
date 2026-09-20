#!/usr/bin/env bash
# Ingest the 2025 season: Statcast pitch rows + MLB API runner states/umpires,
# then write data/processed/pitches_2025.parquet.
# Detached-safe: writes a DONE marker so a waiter can poll instead of pgrep.
set -uo pipefail
cd "$(dirname "$0")/.."
source ~/dwjax-venv/bin/activate
MARK=data/ingest_2025.DONE
LOG=data/ingest_2025.log
rm -f "$MARK"
{
  echo "=== ingest 2025 start $(date -Is)"
  python -m diamondworld.data.pipeline --season 2025
  rc=$?
  echo "=== ingest 2025 exit rc=$rc $(date -Is)"
  echo "$rc" > "$MARK"
} >> "$LOG" 2>&1
