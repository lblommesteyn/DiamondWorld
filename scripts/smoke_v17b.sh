#!/usr/bin/env bash
# Re-smoke the variants not yet cleared: nested (rewritten to avoid the scatter
# form that would not compile), and the two skill priors. bilinear already
# passed (ELBO -12881 -> -7325 over 300 steps, rc=0).
set -u
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4
source ~/dwjax-venv/bin/activate
cd ~/DiamondWorld

LOG=data/smoke_v17b.log
: > "$LOG"

run () {
  local tag="$1"; shift
  echo "=== smoke-train $tag: $* ===" | tee -a "$LOG"
  timeout 900 python -m diamondworldjax.scripts.train_pa --steps 300 \
    --outcome-only --fatigue --recency-halflife 2.0 \
    --train-end 2023 --contact-quality --tag "smoke_$tag" "$@" >> "$LOG" 2>&1
  local r=$?
  # 124 is timeout(1)'s signal that the run wedged, which is exactly the failure
  # the scatter-based nested head produced. Treat it as a hard failure, not a pass.
  echo "--- $tag rc=$r $([ $r -eq 124 ] && echo '(TIMED OUT)')" | tee -a "$LOG"
}

run nested   --nested
run learned  --skill-prior learned
run lkj      --skill-prior lkj

echo "=== summary ===" | tee -a "$LOG"
grep -E "^--- |step +0 |Final ELBO|Traceback" "$LOG" | tee -a "$LOG"
echo "SMOKE2 DONE" | tee -a "$LOG"
