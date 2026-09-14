#!/usr/bin/env bash
# Production joint PA + ABCD experiment.  Kept separate from run_six_models.sh
# so the original six independent-model commands and artifact names never vary.
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAG_PREFIX=joint
SEED=0
JOINT_STEPS=50000
JOINT_BATCH=32
MAX_PA=128
LR=3e-4
PA_WEIGHT=1
ABCD_WEIGHT=1
RESIDUAL_SCALE=.35
PA_SS_RATE=0
PA_SS_WARMUP=20000
MISSING_SAMPLES=2
ABCD_WINDOW=32
ABCD_MAX_LEN=160
ABCD_DROPOUT=.1
SIM_REPS=10
LIMIT_GAMES=0
MIN_PA=150
CUDA_DEVICE=""
SKIP_TRAIN=0
DRY_RUN=0
RESUME=""
HYBRID_ROLLOUT=0

usage() {
  cat <<'EOF'
Usage: bash scripts/run_joint_pa_abcd.sh [options]

Trains one production PA + ABCD model with a shared seasonal player-skill
posterior and task-specific PA/ABCD residuals. The existing six-model runner is
not called or modified.

Options:
  --tag-prefix TAG       Artifact tag prefix (default joint)
  --seed N               Training and simulation seed (default 0)
  --joint-steps N        Global joint optimizer updates (default 50000)
  --joint-batch N        Complete games per matched PA/ABCD update (default 32)
  --max-pa N             Fixed PA padding limit (default 128)
  --lr X                 Joint learning rate (default 3e-4)
  --joint-pa-weight X    Normalized PA task weight (default 1)
  --joint-abcd-weight X  Normalized ABCD task weight (default 1)
  --joint-residual-scale X  Task-specific player skill scale (default .35)
  --pa-ss-rate X         PA scheduled-sampling ceiling (default 0)
  --pa-ss-warmup N       PA scheduled-sampling warmup (default 20000)
  --abcd-window N        Strict ABCD pitch-history window (default 32)
  --abcd-max-len N       ABCD sequence length including overlap (default 160)
  --missing-samples N    ABCD current-measurement integration draws (default 2)
  --abcd-dropout X       ABCD dropout (default .1)
  --sim-reps N           Evaluation world draws (default 10)
  --limit-games N        Held-out games per evaluator; 0 = all (default 0)
  --min-pa N             Player eligibility threshold (default 150)
  --cuda-device DEVICE   Optional CUDA_VISIBLE_DEVICES
  --skip-train           Require the finished joint exports; only evaluate
  --resume PATH          Resume a Model 7 checkpoint; --joint-steps is the total target
  --hybrid-rollout       Also run PA-on-ABCD in-play hybrid evaluation
  --dry-run              Print all commands without reading data or writing files
  -h, --help

The fixed split is train 2015-2023 (including 2020), test 2024.  The default is
production scale; use --dry-run first to verify the exact command and artifact
locations.  A true short smoke run is available through train_pa_abcd.py's
--limit-train-rows plus small --steps/--batch settings.

Hybrid evaluation uses the transparent generated-state loop rather than the
cached ABCD decoder, so it is intentionally opt-in and best run after native
PA and ABCD reports are established.
EOF
}

while (($#)); do
  case "$1" in
    --tag-prefix) TAG_PREFIX="${2:?value required}"; shift 2 ;;
    --seed) SEED="${2:?value required}"; shift 2 ;;
    --joint-steps) JOINT_STEPS="${2:?value required}"; shift 2 ;;
    --joint-batch) JOINT_BATCH="${2:?value required}"; shift 2 ;;
    --max-pa) MAX_PA="${2:?value required}"; shift 2 ;;
    --lr) LR="${2:?value required}"; shift 2 ;;
    --joint-pa-weight) PA_WEIGHT="${2:?value required}"; shift 2 ;;
    --joint-abcd-weight) ABCD_WEIGHT="${2:?value required}"; shift 2 ;;
    --joint-residual-scale) RESIDUAL_SCALE="${2:?value required}"; shift 2 ;;
    --pa-ss-rate) PA_SS_RATE="${2:?value required}"; shift 2 ;;
    --pa-ss-warmup) PA_SS_WARMUP="${2:?value required}"; shift 2 ;;
    --abcd-window) ABCD_WINDOW="${2:?value required}"; shift 2 ;;
    --abcd-max-len) ABCD_MAX_LEN="${2:?value required}"; shift 2 ;;
    --missing-samples) MISSING_SAMPLES="${2:?value required}"; shift 2 ;;
    --abcd-dropout) ABCD_DROPOUT="${2:?value required}"; shift 2 ;;
    --sim-reps) SIM_REPS="${2:?value required}"; shift 2 ;;
    --limit-games) LIMIT_GAMES="${2:?value required}"; shift 2 ;;
    --min-pa) MIN_PA="${2:?value required}"; shift 2 ;;
    --cuda-device) CUDA_DEVICE="${2:?value required}"; shift 2 ;;
    --skip-train) SKIP_TRAIN=1; shift ;;
    --resume) RESUME="${2:?value required}"; shift 2 ;;
    --hybrid-rollout) HYBRID_ROLLOUT=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

[[ "$TAG_PREFIX" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo 'Invalid tag prefix' >&2; exit 2; }
if ((SKIP_TRAIN)) && [[ -n "$RESUME" ]]; then
  echo '--resume cannot be combined with --skip-train' >&2
  exit 2
fi
for name in JOINT_STEPS JOINT_BATCH MAX_PA PA_SS_WARMUP ABCD_WINDOW ABCD_MAX_LEN MISSING_SAMPLES SIM_REPS MIN_PA; do
  value="${!name}"
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid positive integer: $name" >&2; exit 2; }
done
[[ "$LIMIT_GAMES" =~ ^(0|[1-9][0-9]*)$ ]] || { echo 'limit-games must be nonnegative' >&2; exit 2; }
[[ "$PA_SS_RATE" =~ ^(0([.][0-9]+)?|1([.]0+)?|[.][0-9]+)$ ]] || { echo 'pa-ss-rate must be in [0,1]' >&2; exit 2; }
[[ "$ABCD_DROPOUT" =~ ^(0([.][0-9]+)?|[.][0-9]+)$ ]] || { echo 'abcd-dropout must be in [0,1)' >&2; exit 2; }
((ABCD_WINDOW < ABCD_MAX_LEN)) || { echo 'abcd-window must be smaller than abcd-max-len' >&2; exit 2; }

cd "$ROOT_DIR"
[[ -z "$CUDA_DEVICE" ]] || export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.8}"
if [[ -n "${PYTHON_BIN:-}" ]]; then PYTHON="$PYTHON_BIN"
elif [[ -x .venv/bin/python ]]; then PYTHON=.venv/bin/python
elif [[ -x .venv/Scripts/python.exe ]]; then PYTHON=.venv/Scripts/python.exe
else PYTHON=python; fi

TAG="${TAG_PREFIX}_pa_abcd_s${SEED}"
OUT="checkpoints/pa_abcd"
RUN_DIR="data/runs/joint_${TAG}"
JOINT_CKPT="$OUT/joint_${TAG}/$(printf 'dwjax_step_%07d.pkl' "$JOINT_STEPS")"
PA_CKPT="$OUT/pa_${TAG}.pkl"
run_logged() {
  local label="$1"; shift
  printf '[%s] ' "$label"; printf '%q ' "$@"; printf '\n'
  if ((DRY_RUN)); then return; fi
  "$@" 2>&1 | tee "$RUN_DIR/$label.log"
}

TRAIN_CMD=("$PYTHON" -m diamondworldjax.scripts.train_pa_abcd
  --tag "$TAG" --out "$OUT" --seed "$SEED" --steps "$JOINT_STEPS"
  --batch "$JOINT_BATCH" --max-pa "$MAX_PA" --lr "$LR"
  --pa-weight "$PA_WEIGHT" --abcd-weight "$ABCD_WEIGHT"
  --residual-scale "$RESIDUAL_SCALE" --pa-ss-rate "$PA_SS_RATE"
  --pa-ss-warmup "$PA_SS_WARMUP" --missing-samples "$MISSING_SAMPLES"
  --window-size "$ABCD_WINDOW" --context-len "$ABCD_WINDOW"
  --max-len "$ABCD_MAX_LEN" --dropout "$ABCD_DROPOUT")
[[ -z "$RESUME" ]] || TRAIN_CMD+=(--resume "$RESUME")

if ((SKIP_TRAIN)); then
  if ((!DRY_RUN)); then
    for path in "$JOINT_CKPT" "$PA_CKPT" "$OUT/${TAG}_metadata.pkl"; do
      [[ -f "$path" ]] || { echo "Missing joint artifact: $path" >&2; exit 1; }
    done
    for head in A B C D; do
      [[ -f "$OUT/${head}_${TAG}_params.pkl" ]] || { echo "Missing $head export" >&2; exit 1; }
    done
  fi
else
  if ((!DRY_RUN)) && [[ -z "$RESUME" ]] && { [[ -e "$JOINT_CKPT" ]] || [[ -e "$PA_CKPT" ]]; }; then
    echo "Joint artifacts already exist; use --skip-train or choose a new tag." >&2
    exit 1
  fi
  if ((!DRY_RUN)); then mkdir -p "$RUN_DIR"; fi
  run_logged train "${TRAIN_CMD[@]}"
fi

if ((!DRY_RUN)); then mkdir -p "$RUN_DIR"; fi
run_logged players "$PYTHON" -m diamondworldjax.scripts.eval_players \
  --ckpt "$PA_CKPT" --min-pa "$MIN_PA" --samples "$SIM_REPS" --skill-mode mean --seed "$SEED"
for ((rep=0; rep<SIM_REPS; rep++)); do
  rep_seed=$((SEED + rep * 1000003))
  run_logged "pa_games_$rep" "$PYTHON" -m diamondworldjax.scripts.simulate_games \
    --ckpt "$PA_CKPT" --player-stats --min-pa "$MIN_PA" --limit-games "$LIMIT_GAMES" \
    --skill-mode mean --seed "$rep_seed" --dump-scores "$RUN_DIR/pa_scores_$rep.npz"
done
run_logged abcd_games "$PYTHON" -m diamondworldjax.scripts.eval_pitchformer_games \
  --params-dir "$OUT" --tag "$TAG" --season 2024 --c-events --player-stats \
  --min-pa "$MIN_PA" --reps "$SIM_REPS" --limit-games "$LIMIT_GAMES" \
  --batch-games "$JOINT_BATCH" --skill-mode mean --seed "$SEED" \
  --out "$RUN_DIR/abcd_games.json"
if ((HYBRID_ROLLOUT)); then
  run_logged hybrid_games "$PYTHON" -m diamondworldjax.scripts.eval_pitchformer_games \
    --params-dir "$OUT" --tag "$TAG" --season 2024 --c-events --player-stats \
    --min-pa "$MIN_PA" --reps "$SIM_REPS" --limit-games "$LIMIT_GAMES" \
    --batch-games "$JOINT_BATCH" --skill-mode mean --seed "$SEED" \
    --hybrid-pa-ckpt "$PA_CKPT" --hybrid-pa-skill-mode mean \
    --out "$RUN_DIR/hybrid_games.json"
fi

if ((!DRY_RUN)); then
  echo "Finished joint run. Compare PA and ABCD reports independently; preserve the joint checkpoint as provenance."
fi
