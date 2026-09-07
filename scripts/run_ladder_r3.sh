#!/usr/bin/env bash
# Train the R3 shared hierarchy as R2 plus one change, then score its PA export.
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAG=""
CUDA_DEVICE=""
SEED=0
STEPS=50000
BATCH=64
RESIDUAL_SCALE=0.35
SIM_CHUNK=250
SKIP_TRAIN=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: scripts/run_ladder_r3.sh TAG CUDA_DEVICE [options]

Runs R3 as the R2 recipe plus a shared PA/pitch player-skill hierarchy:
  1. train_shared_skills with R2's data, PA configuration, and walk prior
  2. export the PA component to a standalone checkpoint
  3. run the normal ladder evaluation, bootstrap, pregame simulation, and benchmarks

Options:
  --seed N                 Training seed (default: 0)
  --steps N                Training steps (default: 50000)
  --batch N                Games per minibatch (R2 default: 64)
  --residual-scale X       Task-specific share of skill innovations (default: 0.35)
  --sim-chunk N            Real games per vectorized simulator batch (default: 250)
  --skip-train             Reuse an existing shared checkpoint
  --dry-run                Print commands without running them
  -h, --help

Example:
  scripts/run_ladder_r3.sh v23_shared_s1 0 --seed 1
EOF
}

POSITIONAL=()
while (($#)); do
  case "$1" in
    --seed) SEED="${2:?--seed requires a value}"; shift 2 ;;
    --steps) STEPS="${2:?--steps requires a value}"; shift 2 ;;
    --batch) BATCH="${2:?--batch requires a value}"; shift 2 ;;
    --residual-scale) RESIDUAL_SCALE="${2:?--residual-scale requires a value}"; shift 2 ;;
    --sim-chunk) SIM_CHUNK="${2:?--sim-chunk requires a value}"; shift 2 ;;
    --skip-train) SKIP_TRAIN=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --*) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    *) POSITIONAL+=("$1"); shift ;;
  esac
done

if ((${#POSITIONAL[@]} != 2)); then
  echo "TAG and CUDA_DEVICE are required." >&2
  usage >&2
  exit 2
fi
TAG="${POSITIONAL[0]}"
CUDA_DEVICE="${POSITIONAL[1]}"
if [[ ! "$TAG" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "Invalid tag '$TAG'; use only letters, digits, dot, underscore, and hyphen." >&2
  exit 2
fi
for value_name in SEED STEPS BATCH SIM_CHUNK; do
  value="${!value_name}"
  [[ "$value" =~ ^[0-9]+$ ]] || {
    echo "$value_name must be a non-negative integer, got '$value'." >&2
    exit 2
  }
done
if ((STEPS == 0 || BATCH == 0 || SIM_CHUNK == 0)); then
  echo "STEPS, BATCH, and SIM_CHUNK must be greater than zero." >&2
  exit 2
fi

cd "$ROOT_DIR"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.8}"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON="$PYTHON_BIN"
elif [[ -x .venv/bin/python ]]; then
  PYTHON=".venv/bin/python"
else
  PYTHON="python"
fi

SHARED_CKPT="checkpoints/dwjax_${TAG}/$(printf 'dwjax_step_%07d.pkl' "$STEPS")"
PA_CKPT="checkpoints/dwjax_pa_${TAG}/$(printf 'dwjax_step_%07d.pkl' "$STEPS")"
R2_ARGS=(
  --train-end 2023
  --recency-halflife 2.0
  --contact-quality
  --per-stat-shrink
  --outcome-only
  --fatigue
  --skill-prior walk
  --ss-rate 0.25
  --ss-warmup 20000
  --pa-pitchformer
)

print_command() {
  printf '  '
  printf '%q ' "$@"
  printf '\n'
}

run() {
  print_command "$@"
  if ((!DRY_RUN)); then
    "$@"
  fi
}

if ((SKIP_TRAIN)); then
  if ((!DRY_RUN)) && [[ ! -f "$SHARED_CKPT" ]]; then
    echo "Shared checkpoint not found: $SHARED_CKPT" >&2
    exit 1
  fi
else
  run "$PYTHON" -m diamondworldjax.scripts.train_shared_skills \
    --steps "$STEPS" --batch "$BATCH" --seed "$SEED" \
    --residual-scale "$RESIDUAL_SCALE" --tag "$TAG" "${R2_ARGS[@]}"
fi

run "$PYTHON" -m diamondworldjax.scripts.export_shared_skills_checkpoint \
  --ckpt "$SHARED_CKPT" --task pa --out "$PA_CKPT"

LADDER_ARGS=("$TAG" "$CUDA_DEVICE" --pitchformer --skip-train --steps "$STEPS" --sim-chunk "$SIM_CHUNK")
if ((DRY_RUN)); then
  LADDER_ARGS+=(--dry-run)
fi
export PYTHON_BIN="$PYTHON"
run bash scripts/run_ladder.sh "${LADDER_ARGS[@]}"
