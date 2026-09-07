#!/usr/bin/env bash
# Run one ladder model/seed through training and Steps 1-3 from LADDER.md.
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAG=""
CUDA_DEVICE=""
SEED=0
STEPS=50000
REPS=20000
SIM_REPLICAS=100
SIM_CHUNK=250
RECAL_FILE="data/eval2/v13_cal_params.npz"
RECAL_KEY="b_heur"
RECAL_SCALE=0.18
NO_RECAL=0
PITCHFORMER=0
SKIP_TRAIN=0
DRY_RUN=0
BASELINE_TAG="v16"
BASELINE_RATES="data/eval2/prod_rates_v16.npz"

usage() {
  cat <<'EOF'
Usage:
  scripts/run_ladder.sh TAG CUDA_DEVICE [options]
  scripts/run_ladder.sh --tag TAG --cuda-device DEVICE [options]

Runs one ladder model/seed all the way through:
  train_pa (unless the target checkpoint exists)
  prod_playercorr
  bootstrap_playercorr (paired against v16)
  run_pregame_sim
  simulator_benchmarks

Required:
  TAG                       Checkpoint and output tag, for example v22_s1
  CUDA_DEVICE               CUDA_VISIBLE_DEVICES value, for example 0 or 1

Options:
  --tag TAG
  --cuda-device DEVICE
  --seed N                  Training seed (default: 0)
  --pitchformer             Train/evaluate a Pitchformer model (R2)
  --steps N                 Training/checkpoint step (default: 50000)
  --reps N                  Bootstrap replicates (default: 20000)
  --sim-replicas N          Replicas per simulated game (default: 100)
  --sim-chunk N             Real games per vectorized simulator batch (default: 250;
                            concurrent trajectories = N x --sim-replicas)
  --recal-file PATH         Calibration .npz for pregame simulation (default: v13 heuristic file)
  --recal-key KEY           Calibration-vector key in that file (default: b_heur)
  --recal-scale X           Calibration-vector multiplier (default: 0.18)
  --no-recal                Use raw outcome logits in pregame simulation
  --baseline-tag TAG        Bootstrap baseline label (default: v16)
  --baseline-rates PATH     Baseline .npz (default: data/eval2/prod_rates_v16.npz)
  --skip-train              Require and evaluate an existing checkpoint
  --dry-run                 Print commands without running them
  -h, --help

Examples:
  scripts/run_ladder.sh v22 0 --skip-train
  scripts/run_ladder.sh v22_s1 0 --seed 1
  scripts/run_ladder.sh v22pf_s1 1 --seed 1 --pitchformer

The simulation is restartable: run_pregame_sim reuses completed chunks for the
same tag. On pcslurm, launch this script through scripts/run_detached.sh as required
by LADDER.md.
EOF
}

POSITIONAL=()
while (($#)); do
  case "$1" in
    --tag) TAG="${2:?--tag requires a value}"; shift 2 ;;
    --cuda-device) CUDA_DEVICE="${2:?--cuda-device requires a value}"; shift 2 ;;
    --seed) SEED="${2:?--seed requires a value}"; shift 2 ;;
    --pitchformer) PITCHFORMER=1; shift ;;
    --steps) STEPS="${2:?--steps requires a value}"; shift 2 ;;
    --reps) REPS="${2:?--reps requires a value}"; shift 2 ;;
    --sim-replicas) SIM_REPLICAS="${2:?--sim-replicas requires a value}"; shift 2 ;;
    --sim-chunk) SIM_CHUNK="${2:?--sim-chunk requires a value}"; shift 2 ;;
    --recal-file) RECAL_FILE="${2:?--recal-file requires a value}"; shift 2 ;;
    --recal-key) RECAL_KEY="${2:?--recal-key requires a value}"; shift 2 ;;
    --recal-scale) RECAL_SCALE="${2:?--recal-scale requires a value}"; shift 2 ;;
    --no-recal) NO_RECAL=1; shift ;;
    --baseline-tag) BASELINE_TAG="${2:?--baseline-tag requires a value}"; shift 2 ;;
    --baseline-rates) BASELINE_RATES="${2:?--baseline-rates requires a value}"; shift 2 ;;
    --skip-train) SKIP_TRAIN=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --*) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    *) POSITIONAL+=("$1"); shift ;;
  esac
done

if ((${#POSITIONAL[@]} > 2)); then
  echo "Expected at most two positional arguments: TAG CUDA_DEVICE" >&2
  usage >&2
  exit 2
fi
[[ -n "$TAG" ]] || TAG="${POSITIONAL[0]:-}"
[[ -n "$CUDA_DEVICE" ]] || CUDA_DEVICE="${POSITIONAL[1]:-}"

if [[ -z "$TAG" || -z "$CUDA_DEVICE" ]]; then
  echo "TAG and CUDA_DEVICE are required." >&2
  usage >&2
  exit 2
fi
if [[ ! "$TAG" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "Invalid tag '$TAG'; use only letters, digits, dot, underscore, and hyphen." >&2
  exit 2
fi
for value_name in SEED STEPS REPS SIM_REPLICAS SIM_CHUNK; do
  value="${!value_name}"
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "$value_name must be a non-negative integer, got '$value'." >&2
    exit 2
  fi
done
if ((STEPS == 0 || REPS == 0 || SIM_REPLICAS == 0 || SIM_CHUNK == 0)); then
  echo "STEPS, REPS, SIM_REPLICAS, and SIM_CHUNK must be greater than zero." >&2
  exit 2
fi
if ! [[ "$RECAL_SCALE" =~ ^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]]; then
  echo "RECAL_SCALE must be a finite numeric value, got '$RECAL_SCALE'." >&2
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

RUN_DIR="data/runs/ladder_${TAG}"
CKPT_DIR="checkpoints/dwjax_pa_${TAG}"
EXPECTED_CKPT="$CKPT_DIR/$(printf 'dwjax_step_%07d.pkl' "$STEPS")"
RATES="data/eval2/prod_rates_${TAG}.npz"
SIM_TAG="${TAG}-pregame-leakfree"
ARRAYS="data/eval2/calib_${SIM_TAG}_arrays.npz"
if ((!DRY_RUN)); then
  mkdir -p "$RUN_DIR" data/eval2
fi

TRAIN_MODEL_ARGS=(
  --outcome-only
  --fatigue
  --recency-halflife 2.0
  --contact-quality
  --per-stat-shrink
  --skill-prior walk
)
EVAL_MODEL_ARGS=(
  --recency-halflife 2.0
  --contact-quality
  --per-stat-shrink
  --skill-prior walk
)
SIM_MODEL_ARGS=(
  --contact-quality
  --per-stat-shrink
  --skill-prior walk
)
PREGAME_RECAL_ARGS=(
  --recal-file "$RECAL_FILE"
  --recal-key "$RECAL_KEY"
  --recal-scale "$RECAL_SCALE"
)
if ((NO_RECAL)); then
  PREGAME_RECAL_ARGS+=(--no-recal)
fi
if ((PITCHFORMER)); then
  TRAIN_MODEL_ARGS+=(--pitchformer)
  EVAL_MODEL_ARGS+=(--pitchformer)
  SIM_MODEL_ARGS+=(--pitchformer)
fi

print_command() {
  printf '  '
  printf '%q ' "$@"
  printf '\n'
}

run_logged() {
  local name="$1"
  shift
  echo "[$(date -u +%FT%TZ)] $name"
  print_command "$@"
  if ((DRY_RUN)); then
    return 0
  fi
  "$@" 2>&1 | tee "$RUN_DIR/${name}.log"
}

latest_checkpoint() {
  local matches=()
  if [[ -d "$CKPT_DIR" ]]; then
    mapfile -t matches < <(find "$CKPT_DIR" -maxdepth 1 -type f \
      -name 'dwjax_step_*.pkl' -print | sort)
  fi
  if ((${#matches[@]})); then
    printf '%s\n' "${matches[-1]}"
  fi
}

echo "Ladder run: tag=$TAG seed=$SEED CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Model: $([[ $PITCHFORMER -eq 1 ]] && echo Pitchformer || echo standard-PA)"
echo "Simulation batch: $SIM_CHUNK real games x $SIM_REPLICAS replicas = $((SIM_CHUNK * SIM_REPLICAS)) trajectories"
echo "Logs: $RUN_DIR"

CKPT=""
if [[ -f "$EXPECTED_CKPT" ]]; then
  CKPT="$EXPECTED_CKPT"
  echo "Using existing target checkpoint: $CKPT"
else
  CKPT="$(latest_checkpoint)"
  if ((SKIP_TRAIN)); then
    if [[ -z "$CKPT" ]]; then
      echo "No checkpoint found under $CKPT_DIR." >&2
      exit 1
    fi
    echo "Target step $STEPS not found; using latest checkpoint: $CKPT"
  else
    run_logged train "$PYTHON" -m diamondworldjax.scripts.train_pa \
      --train-end 2023 --seed "$SEED" --steps "$STEPS" --tag "$TAG" \
      "${TRAIN_MODEL_ARGS[@]}"
    if ((DRY_RUN)); then
      CKPT="$EXPECTED_CKPT"
    elif [[ -f "$EXPECTED_CKPT" ]]; then
      CKPT="$EXPECTED_CKPT"
    else
      CKPT="$(latest_checkpoint)"
      if [[ -z "$CKPT" ]]; then
        echo "Training completed but no checkpoint was found under $CKPT_DIR." >&2
        exit 1
      fi
      echo "Expected checkpoint absent; using latest checkpoint: $CKPT"
    fi
  fi
fi

run_logged playercorr "$PYTHON" -m diamondworldjax.scripts.prod_playercorr \
  --ckpt "$CKPT" --recal data/eval2/v13_cal_params.npz --skill-mode mean \
  --train-end 2023 --test-seasons 2024 --tag "$TAG" "${EVAL_MODEL_ARGS[@]}"

if ((!DRY_RUN)) && [[ ! -f "$BASELINE_RATES" ]]; then
  echo "Baseline rates not found: $BASELINE_RATES" >&2
  echo "Fetch the corrected v16 file from the Hugging Face data/eval2 tier." >&2
  exit 1
fi
run_logged bootstrap "$PYTHON" -m diamondworldjax.scripts.bootstrap_playercorr \
  --rates "${BASELINE_TAG}=${BASELINE_RATES}" --rates "${TAG}=${RATES}" \
  --baseline "$BASELINE_TAG" --reps "$REPS" \
  --out "data/eval2/bootstrap_${TAG}.txt" \
  --json-out "data/eval2/bootstrap_${TAG}.json"

run_logged pregame_sim "$PYTHON" -m diamondworldjax.scripts.run_pregame_sim \
  --pregame-staff --r "$SIM_REPLICAS" --chunk "$SIM_CHUNK" --ckpt "$CKPT" \
  --train-end 2023 --tag "$SIM_TAG" "${SIM_MODEL_ARGS[@]}" "${PREGAME_RECAL_ARGS[@]}"

run_logged benchmarks "$PYTHON" -m diamondworldjax.scripts.simulator_benchmarks \
  --arrays "$ARRAYS" --tag "$SIM_TAG"

echo "[$(date -u +%FT%TZ)] Complete: $TAG"
echo "  player rates: $RATES"
echo "  bootstrap:    data/eval2/bootstrap_${TAG}.txt"
echo "  benchmarks:   data/eval2/simulator_benchmarks_${SIM_TAG}.txt"
