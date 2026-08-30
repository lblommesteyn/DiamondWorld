#!/usr/bin/env bash
# Generic launcher for long GPU jobs submitted through pcslurm.
#
# THE PROBLEM. pcslurm dispatches Windows -> powershell -> WSL, so the real work sits
# in the submitting process tree and dies with it. Job 337 (v21b) died at step 5,000 of
# 50,000 this way, and job 393 (the leak-free pre-game re-run) died at 250 of 2,429
# games, both silently: no traceback, no non-zero exit, just a log that stops.
#
# THE FIX, both halves of it. setsid-detach the real work so it has no controlling
# terminal and is outside that tree, then hold the Slurm allocation in the foreground
# by polling for a DONE marker, so the GPU stays correctly reserved for the true
# duration and the queue still serialises.
#
# Detachment cuts both ways: it also survives scancel, which would orphan a run that
# keeps holding the card. The trap kills the whole detached process GROUP on the way
# out so scancel behaves as expected. That mirror bug is not hypothetical; it happened
# with job 341 and had to be cleaned up by hand.
#
# The DONE-marker poll is deliberate. pgrep liveness checks race under load on this box
# and false-positive as "died".
#
# Usage:
#   bash scripts/run_detached.sh <marker-name> <command...>
# Example:
#   bash scripts/run_detached.sh leakfree \
#       ~/dwjax-venv/bin/python -m diamondworldjax.scripts.run_pregame_sim --tag x
set -u

if [ "$#" -lt 2 ]; then
  echo "usage: run_detached.sh <marker-name> <command...>" >&2
  exit 2
fi

NAME="$1"; shift
cd ~/DiamondWorld

MARKER="data/run_${NAME}_done.txt"
OUTER="data/${NAME}_outer.log"
STAMP="data/${NAME}_launch.txt"
rm -f "$MARKER"

# The marker is written by this wrapper, not by the job, so it does not matter whether
# the job knows anything about markers. The exit status is recorded with it.
setsid nohup bash -c '"$@"; echo "DONE rc=$?" > '"$MARKER" _ "$@" > "$OUTER" 2>&1 &
CHILD=$!
echo "detached ${NAME} pgid ${CHILD} at $(date -u +%FT%TZ)" > "$STAMP"

cleanup() {
  kill -TERM -"$CHILD" 2>/dev/null
  sleep 5
  kill -KILL -"$CHILD" 2>/dev/null
  echo "cleaned up pgid ${CHILD} at $(date -u +%FT%TZ)" >> "$STAMP"
}
trap cleanup TERM INT EXIT

until [ -f "$MARKER" ]; do
  sleep 30
done
trap - EXIT

RC_LINE="$(cat "$MARKER")"
echo "holder released at $(date -u +%FT%TZ) (${RC_LINE})" >> "$STAMP"
# Propagate the real exit code so Slurm records a failure as a failure.
case "$RC_LINE" in
  "DONE rc=0") exit 0 ;;
  *) echo "job reported ${RC_LINE}" >&2; exit 1 ;;
esac
