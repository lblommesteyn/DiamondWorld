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

# Poll for the marker, but ALSO notice if the work dies without writing one.
#
# The first version of this loop waited on the marker alone. That is why job 395 held
# the card for an hour and job 399 held it for TWENTY-TWO HOURS after the work had
# already died: a job killed by a signal never gets to write DONE, so the holder waits
# forever on a marker that is never coming, and Slurm keeps the GPU reserved for a
# process group that no longer exists.
#
# The original reason for not liveness-checking was that a single check races under
# load and false-positives as "died". The answer to a racy check is to debounce it,
# not to skip it: the group has to look empty on MISSES consecutive polls, 5 minutes
# apart in total, before we believe it. A real run is never invisible that long, and
# a genuinely dead one costs at most five extra minutes of held GPU instead of a day.
MISSES=0
while [ ! -f "$MARKER" ]; do
  sleep 30
  [ -f "$MARKER" ] && break
  if kill -0 -"$CHILD" 2>/dev/null; then
    MISSES=0
  else
    MISSES=$((MISSES + 1))
    if [ "$MISSES" -ge 10 ]; then
      # Re-check for the marker once more: the work may have finished and exited
      # in the gap between the last liveness poll and this decision.
      sleep 2
      [ -f "$MARKER" ] && break
      echo "process group ${CHILD} gone for 5 minutes with no DONE marker; treating as died" >&2
      echo "died without marker at $(date -u +%FT%TZ)" >> "$STAMP"
      trap - EXIT
      exit 1
    fi
  fi
done
trap - EXIT

RC_LINE="$(cat "$MARKER")"
echo "holder released at $(date -u +%FT%TZ) (${RC_LINE})" >> "$STAMP"
# Propagate the real exit code so Slurm records a failure as a failure.
case "$RC_LINE" in
  "DONE rc=0") exit 0 ;;
  *) echo "job reported ${RC_LINE}" >&2; exit 1 ;;
esac
