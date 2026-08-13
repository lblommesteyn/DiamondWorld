#!/usr/bin/env bash
# Launcher that survives the parent process being torn down.
#
# The first v21b attempt (job 337) died at step 5000 of 50000 when the Claude Code
# process that submitted it exited. pcslurm dispatches through powershell into WSL,
# so the training process sits in the submitting process tree and goes down with it.
#
# Fix: setsid-detach the real work so it has no controlling terminal and is not in
# that tree, then hold the Slurm allocation in the foreground by polling for the
# DONE marker. That keeps the GPU correctly reserved for the real duration (so the
# queue still serialises properly) while making the training itself immune to a
# teardown of whatever launched it.
#
# The DONE-marker poll is deliberate: process-liveness checks (pgrep) race under
# load on this box and false-positive as "died".
set -u
cd ~/DiamondWorld

MARKER=data/run_v21b_done.txt
rm -f "$MARKER"

setsid nohup bash scripts/run_v21b.sh > data/v21b_outer.log 2>&1 &
CHILD=$!
echo "detached v21b pid $CHILD at $(date -u +%FT%TZ)" > data/v21b_launch.txt

# Detachment cuts both ways: it survives a teardown of whatever launched this,
# but it also survives scancel, so cancelling the Slurm job would otherwise
# orphan a training run that keeps holding the GPU. Kill the whole detached
# process GROUP (negative PID) on the way out so scancel behaves as expected.
cleanup() {
  kill -TERM -"$CHILD" 2>/dev/null
  sleep 5
  kill -KILL -"$CHILD" 2>/dev/null
  echo "cleaned up pgid $CHILD at $(date -u +%FT%TZ)" >> data/v21b_launch.txt
}
trap cleanup TERM INT EXIT

# Hold the allocation until the detached run finishes.
until grep -q "DONE" "$MARKER" 2>/dev/null; do
  sleep 60
done
trap - EXIT
echo "holder released at $(date -u +%FT%TZ)" >> data/v21b_launch.txt
