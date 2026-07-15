#!/usr/bin/env bash
# The GPU got contended (a graphics process appeared ~01:00) and CUDA sims started
# getting killed at init. Wait until the GPU is genuinely free, then run the
# remaining full-N eval. Polls quietly so it doesn't touch the GPU while waiting.
cd ~/DiamondWorld
echo "waiting for free GPU..." > data/gpu_wait.log
while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
  echo "$(date +%H:%M) used=${used}MiB util=${util}%" >> data/gpu_wait.log
  if [ -n "$used" ] && [ "$used" -lt 1500 ] && [ "$util" -lt 15 ]; then
    echo "GPU free -> starting eval" >> data/gpu_wait.log
    break
  fi
  sleep 180
done
bash scripts/eval_resume.sh > data/eval2_resume.log 2>&1
echo "ALL_DONE" >> data/gpu_wait.log
