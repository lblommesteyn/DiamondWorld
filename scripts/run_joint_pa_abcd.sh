#!/usr/bin/env bash
# Production joint PA + ABCD experiment.  Kept separate from run_six_models.sh
# so the original six independent-model commands and artifact names never vary.
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAG_PREFIX=joint
SEED=0
JOINT_STEPS=50000
TRAIN_BATCH=32
EVAL_GAME_BATCH=256
# The longest training game has 152 PA.  Padding to 160 retains every game
# while avoiding the 20% unused PA-token tail from the older 192-token shape.
MAX_PA=160
PREFETCH_DEPTH=1
XLA_AUTOTUNE_LEVEL=""
LR=3e-4
PA_WEIGHT=1
ABCD_WEIGHT=1
RESIDUAL_SCALE=.35
PA_SS_RATE=0
PA_SS_WARMUP=20000
PA_PITCHFORMER=0
PA_ARCH=gru_skip
PA_FEATURE_PROFILE=ladder
MISSING_SAMPLES=2
ABCD_WINDOW=32
ABCD_MAX_LEN=160
ABCD_DROPOUT=.1
ABCD_D_HR_WEIGHT=0
ABCD_HALF_CONTINUATIONS=6
ABCD_SEQUENCES_PER_GAME=1
# PA replicas are independent and cheap; full ABCD rollouts are the expensive
# generated-state path, so their routine default is intentionally lower.
PA_SIM_REPS=100
ABCD_SIM_REPS=5
LIMIT_GAMES=0
MIN_PA=150
CUDA_DEVICE=""
SKIP_TRAIN=0
DRY_RUN=0
RESUME=""
CHECKPOINT=""
HYBRID_ROLLOUT=0
HYBRID_ONLY=0
COMPARE_V16=0
# The PA-on-ABC hybrid supports flat PA, GRU, and GRU-skip exports. Transformer
# PA still needs bounded-cache rollover across very long generated games.
COMPARE_HYBRID=0
BOOTSTRAP_REPS=20000
COMPARISON_REPS=5
PREGAME_REPS=100
PREGAME_CHUNK=250
BASELINE_TAG=v16
BASELINE_RATES=data/eval2/prod_rates_v16.npz
RECAL_FILE=data/eval2/v13_cal_params.npz
RECAL_KEY=b_heur
RECAL_SCALE=0.18
NO_RECAL=0
V16_MAX_PA=90

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
  --train-batch N        Complete games per matched PA/ABCD update (default 32)
  --eval-game-batch N    Active games per ABCD evaluation rollout batch (default 256)
  --joint-batch N        Legacy alias: set both training and evaluation batch sizes
  --max-pa N             Fixed PA padding limit (default 160; current data maximum is 152)
  --prefetch-depth N     Prepared joint batches kept ahead of training (default 1; 0 saves GPU memory)
  --xla-autotune-level N Set XLA GPU autotuning level; use 0 to avoid autotune workspace OOM
  --lr X                 Joint learning rate (default 3e-4)
  --joint-pa-weight X    Normalized PA task weight (default 1)
  --joint-abcd-weight X  Normalized ABCD task weight (default 1)
  --joint-residual-scale X  Task-specific player skill scale (default .35)
  --pa-ss-rate X         PA scheduled-sampling ceiling (default 0)
  --pa-ss-warmup N       PA scheduled-sampling warmup (default 20000)
  --pa-pitchformer       Use a sequential PA model in the joint trainer
  --pa-arch ARCH         PA sequence architecture: gru, gru_skip, transformer;
                         also enables --pa-pitchformer (default gru_skip)
  --pa-feature-profile MODE  PA player features: ladder (default), v16, or basic
  --abcd-window N        Strict ABCD pitch-history window (default 32)
  --abcd-max-len N       ABCD sequence length including overlap (default 160)
  --missing-samples N    ABCD current-measurement integration draws (default 2)
  --abcd-sequences-per-game N  ABCD windows cycled from each matched game per update (default 1)
  --abcd-dropout X       ABCD dropout (default .1)
  --abcd-d-hr-weight X   Auxiliary D binary-HR loss weight (default 0; baseline objective)
  --abcd-half-continuations N  Extra cyclic decodes allowed to finish a half (default 6)
  --pa-sim-reps N        PA evaluation world draws (default 100)
  --abcd-sim-reps N      Native/hybrid ABCD evaluation worlds (default 5)
  --sim-reps N           Legacy alias: set both PA and ABCD world counts
  --limit-games N        Held-out games per evaluator; 0 = all (default 0)
  --min-pa N             Player eligibility threshold (default 150)
  --cuda-device DEVICE   Optional CUDA_VISIBLE_DEVICES
  --skip-train           Require the finished joint exports; only evaluate
  --checkpoint PATH      With --skip-train, export and evaluate this joint checkpoint snapshot
  --resume PATH          Resume a Model 7 checkpoint; --joint-steps is the total target
  --hybrid-rollout       Also run the A/B/C + PA in-play hybrid evaluation (D omitted)
  --hybrid-only          Run only the A/B/C + PA hybrid evaluation after training/export,
                         including its v16 rate/bootstrap/benchmark comparison suite
  --compare-v16          Export PA and native-ABCD rates; paired-bootstrap vs v16
  --compare-v16-hybrid   Also run the A/B/C + PA in-play hybrid (flat, GRU, or GRU-skip PA)
  --compare-v16-native-only  With --compare-v16, skip the hybrid comparison rollout (legacy alias)
  --bootstrap-reps N     Paired-bootstrap replicates with --compare-v16 (default 20000)
  --comparison-reps N    Generated worlds for each ABCD comparison (default 5)
  --pregame-reps N       PA replicas/game for the matched pregame benchmark (default 100)
  --pregame-chunk N      PA pregame games per chunk (default 250)
  --baseline-tag TAG     Baseline label for --compare-v16 (default v16)
  --baseline-rates PATH  Baseline prod_rates NPZ (default data/eval2/prod_rates_v16.npz)
  --recal-file PATH      PA comparison calibration NPZ (default data/eval2/v13_cal_params.npz)
  --recal-key KEY        Calibration-vector key (default b_heur)
  --recal-scale X        Calibration-vector multiplier (default 0.18)
  --no-recal             Use raw PA logits instead of the matched v16 calibration
  --v16-max-pa N         Per-game PA cap for paired rates (default 90; 0 = no cap)
  --dry-run              Print all commands without reading data or writing files
  -h, --help

The fixed split is train 2015-2023 (including 2020), test 2024.  The default is
production scale; use --dry-run first to verify the exact command and artifact
locations.  A true short smoke run is available through train_pa_abcd.py's
--limit-train-rows plus small --steps/--batch settings.

Every joint update contains matched PA and ABCD data from the same complete
games. Each complete batch is visited once per pass, then batch order is
reshuffled; tokens inside a game are always kept chronological.

PA feature profiles: ladder = recency half-life 2, contact-quality inputs, and
per-stat shrinkage; v16 omits only per-stat shrinkage; basic disables these
feature transformations. Checkpoint metadata carries the selected profile into
PA evaluation.

Hybrid evaluation uses the transparent generated-state loop rather than the
cached ABCD decoder, so it is intentionally opt-in and best run after native
PA and ABCD reports are established. It uses A/B/C for pitch generation and
the PA head for every ball-in-play result; D is not loaded. Flat, GRU, and
GRU-skip PA exports are supported; transformer PA remains unsupported.

Its game benchmark is labelled observed-schedule: it uses held-out
lineups/staffs and is not a pre-game roster/staff comparison. Paired rates cap
each game at 90 PAs to match v16.
EOF
}

while (($#)); do
  case "$1" in
    --tag-prefix) TAG_PREFIX="${2:?value required}"; shift 2 ;;
    --seed) SEED="${2:?value required}"; shift 2 ;;
    --joint-steps) JOINT_STEPS="${2:?value required}"; shift 2 ;;
    --train-batch) TRAIN_BATCH="${2:?value required}"; shift 2 ;;
    --eval-game-batch) EVAL_GAME_BATCH="${2:?value required}"; shift 2 ;;
    --joint-batch) TRAIN_BATCH="${2:?value required}"; EVAL_GAME_BATCH="$TRAIN_BATCH"; shift 2 ;;
    --max-pa) MAX_PA="${2:?value required}"; shift 2 ;;
    --prefetch-depth) PREFETCH_DEPTH="${2:?value required}"; shift 2 ;;
    --xla-autotune-level) XLA_AUTOTUNE_LEVEL="${2:?value required}"; shift 2 ;;
    --lr) LR="${2:?value required}"; shift 2 ;;
    --joint-pa-weight) PA_WEIGHT="${2:?value required}"; shift 2 ;;
    --joint-abcd-weight) ABCD_WEIGHT="${2:?value required}"; shift 2 ;;
    --joint-residual-scale) RESIDUAL_SCALE="${2:?value required}"; shift 2 ;;
    --pa-ss-rate) PA_SS_RATE="${2:?value required}"; shift 2 ;;
    --pa-ss-warmup) PA_SS_WARMUP="${2:?value required}"; shift 2 ;;
    --pa-pitchformer) PA_PITCHFORMER=1; shift ;;
    --pa-arch) PA_ARCH="${2:?value required}"; PA_PITCHFORMER=1; shift 2 ;;
    --pa-feature-profile) PA_FEATURE_PROFILE="${2:?value required}"; shift 2 ;;
    --abcd-window) ABCD_WINDOW="${2:?value required}"; shift 2 ;;
    --abcd-max-len) ABCD_MAX_LEN="${2:?value required}"; shift 2 ;;
    --missing-samples) MISSING_SAMPLES="${2:?value required}"; shift 2 ;;
    --abcd-sequences-per-game) ABCD_SEQUENCES_PER_GAME="${2:?value required}"; shift 2 ;;
    --abcd-dropout) ABCD_DROPOUT="${2:?value required}"; shift 2 ;;
    --abcd-d-hr-weight) ABCD_D_HR_WEIGHT="${2:?value required}"; shift 2 ;;
    --abcd-half-continuations) ABCD_HALF_CONTINUATIONS="${2:?value required}"; shift 2 ;;
    --pa-sim-reps) PA_SIM_REPS="${2:?value required}"; shift 2 ;;
    --abcd-sim-reps) ABCD_SIM_REPS="${2:?value required}"; shift 2 ;;
    --sim-reps) PA_SIM_REPS="${2:?value required}"; ABCD_SIM_REPS="$PA_SIM_REPS"; shift 2 ;;
    --limit-games) LIMIT_GAMES="${2:?value required}"; shift 2 ;;
    --min-pa) MIN_PA="${2:?value required}"; shift 2 ;;
    --cuda-device) CUDA_DEVICE="${2:?value required}"; shift 2 ;;
    --skip-train) SKIP_TRAIN=1; shift ;;
    --checkpoint) CHECKPOINT="${2:?value required}"; shift 2 ;;
    --resume) RESUME="${2:?value required}"; shift 2 ;;
    --hybrid-rollout) HYBRID_ROLLOUT=1; shift ;;
    --hybrid-only) HYBRID_ONLY=1; HYBRID_ROLLOUT=1; shift ;;
    --compare-v16) COMPARE_V16=1; shift ;;
    --compare-v16-hybrid) COMPARE_V16=1; COMPARE_HYBRID=1; shift ;;
    --compare-v16-native-only) COMPARE_HYBRID=0; shift ;;
    --bootstrap-reps) BOOTSTRAP_REPS="${2:?value required}"; shift 2 ;;
    --comparison-reps) COMPARISON_REPS="${2:?value required}"; shift 2 ;;
    --pregame-reps) PREGAME_REPS="${2:?value required}"; shift 2 ;;
    --pregame-chunk) PREGAME_CHUNK="${2:?value required}"; shift 2 ;;
    --baseline-tag) BASELINE_TAG="${2:?value required}"; shift 2 ;;
    --baseline-rates) BASELINE_RATES="${2:?value required}"; shift 2 ;;
    --recal-file) RECAL_FILE="${2:?value required}"; shift 2 ;;
    --recal-key) RECAL_KEY="${2:?value required}"; shift 2 ;;
    --recal-scale) RECAL_SCALE="${2:?value required}"; shift 2 ;;
    --no-recal) NO_RECAL=1; shift ;;
    --v16-max-pa) V16_MAX_PA="${2:?value required}"; shift 2 ;;
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
if [[ -n "$CHECKPOINT" ]] && ((!SKIP_TRAIN)); then
  echo '--checkpoint requires --skip-train' >&2
  exit 2
fi
# Hybrid-only comparison is intentionally a third, isolated suite rather than
# a request for PA or native-ABCD results as well.  It always needs the hybrid
# rate and benchmark outputs, even if --compare-v16-native-only appeared first.
if ((HYBRID_ONLY)); then
  # This reads the existing v16 rate artifact; it does not run the standalone
  # v16 simulator.  The resulting metrics are the complete hybrid comparison.
  COMPARE_V16=1
  COMPARE_HYBRID=1
fi
for name in JOINT_STEPS TRAIN_BATCH EVAL_GAME_BATCH MAX_PA PA_SS_WARMUP ABCD_WINDOW ABCD_MAX_LEN ABCD_HALF_CONTINUATIONS ABCD_SEQUENCES_PER_GAME MISSING_SAMPLES PA_SIM_REPS ABCD_SIM_REPS MIN_PA; do
  value="${!name}"
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid positive integer: $name" >&2; exit 2; }
done
[[ "$LIMIT_GAMES" =~ ^(0|[1-9][0-9]*)$ ]] || { echo 'limit-games must be nonnegative' >&2; exit 2; }
[[ "$PREFETCH_DEPTH" =~ ^(0|[1-9][0-9]*)$ ]] || { echo 'prefetch-depth must be nonnegative' >&2; exit 2; }
[[ -z "$XLA_AUTOTUNE_LEVEL" || "$XLA_AUTOTUNE_LEVEL" =~ ^[0-9]+$ ]] || { echo 'xla-autotune-level must be a nonnegative integer' >&2; exit 2; }
if ((COMPARE_V16)); then
  for name in BOOTSTRAP_REPS COMPARISON_REPS PREGAME_REPS PREGAME_CHUNK; do
    value="${!name}"
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "$name must be a positive integer" >&2; exit 2; }
  done
  [[ "$BASELINE_TAG" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo 'Invalid baseline tag' >&2; exit 2; }
  [[ "$V16_MAX_PA" =~ ^(0|[1-9][0-9]*)$ ]] || { echo '--v16-max-pa must be nonnegative' >&2; exit 2; }
  [[ "$RECAL_SCALE" =~ ^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]] || { echo 'Invalid recalibration scale' >&2; exit 2; }
  ((LIMIT_GAMES == 0)) || { echo '--compare-v16 requires --limit-games 0 so the ABCD and v16 rate cohorts match' >&2; exit 2; }
fi
[[ "$PA_SS_RATE" =~ ^(0([.][0-9]+)?|1([.]0+)?|[.][0-9]+)$ ]] || { echo 'pa-ss-rate must be in [0,1]' >&2; exit 2; }
case "$PA_ARCH" in gru|gru_skip|transformer) ;; *) echo 'Invalid PA architecture' >&2; exit 2 ;; esac
if ((PA_PITCHFORMER && (HYBRID_ROLLOUT || (COMPARE_V16 && COMPARE_HYBRID)))) && [[ "$PA_ARCH" == transformer ]]; then
  echo 'PA-on-ABC hybrid evaluation currently supports flat PA, GRU, and GRU-skip exports; transformer PA is not yet supported.' >&2
  exit 2
fi
case "$PA_FEATURE_PROFILE" in
  ladder) PA_FEATURE_ARGS=(--recency-halflife 2.0 --contact-quality --per-stat-shrink) ;;
  v16) PA_FEATURE_ARGS=(--recency-halflife 2.0 --contact-quality) ;;
  basic) PA_FEATURE_ARGS=(--recency-halflife 0) ;;
  *) echo 'Invalid PA feature profile: use ladder, v16, or basic' >&2; exit 2 ;;
esac
[[ "$ABCD_DROPOUT" =~ ^(0([.][0-9]+)?|[.][0-9]+)$ ]] || { echo 'abcd-dropout must be in [0,1)' >&2; exit 2; }
[[ "$ABCD_D_HR_WEIGHT" =~ ^([0-9]+([.][0-9]+)?|[.][0-9]+)$ ]] || { echo 'abcd-d-hr-weight must be nonnegative' >&2; exit 2; }
((ABCD_WINDOW < ABCD_MAX_LEN)) || { echo 'abcd-window must be smaller than abcd-max-len' >&2; exit 2; }

cd "$ROOT_DIR"
[[ -z "$CUDA_DEVICE" ]] || export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
# Prefer JAX's current variable while still respecting either spelling if the
# caller already set one. Preallocation is retained because it is faster and
# less fragmentation-prone for a single training process.
if [[ -z "${XLA_CLIENT_MEM_FRACTION:-}" && -z "${XLA_PYTHON_CLIENT_MEM_FRACTION:-}" ]]; then
  export XLA_CLIENT_MEM_FRACTION=0.8
fi
if [[ -n "$XLA_AUTOTUNE_LEVEL" ]]; then
  export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_autotune_level=$XLA_AUTOTUNE_LEVEL"
fi
if [[ -n "${PYTHON_BIN:-}" ]]; then PYTHON="$PYTHON_BIN"
elif [[ -x .venv/bin/python ]]; then PYTHON=.venv/bin/python
elif [[ -x .venv/Scripts/python.exe ]]; then PYTHON=.venv/Scripts/python.exe
else PYTHON=python; fi
if ((COMPARE_V16 && !DRY_RUN)) && [[ ! -f "$BASELINE_RATES" ]]; then
  echo "Missing baseline rates: $BASELINE_RATES" >&2
  exit 2
fi
if ((COMPARE_V16 && !HYBRID_ONLY && !DRY_RUN && !NO_RECAL)) && [[ ! -f "$RECAL_FILE" ]]; then
  echo "Missing calibration file: $RECAL_FILE" >&2
  exit 2
fi

TAG="${TAG_PREFIX}_pa_abcd_s${SEED}"
OUT="checkpoints/pa_abcd"
RUN_DIR="data/runs/joint_${TAG}"
if [[ -n "$CHECKPOINT" ]]; then
  JOINT_CKPT="$CHECKPOINT"
else
  JOINT_CKPT="$OUT/joint_${TAG}/$(printf 'dwjax_step_%07d.pkl' "$JOINT_STEPS")"
fi
PA_CKPT="$OUT/pa_${TAG}.pkl"
run_logged() {
  local label="$1"; shift
  printf '[%s] ' "$label"; printf '%q ' "$@"; printf '\n'
  if ((DRY_RUN)); then return; fi
  "$@" 2>&1 | tee "$RUN_DIR/$label.log"
}

TRAIN_CMD=("$PYTHON" -m diamondworldjax.scripts.train_pa_abcd
  --tag "$TAG" --out "$OUT" --seed "$SEED" --steps "$JOINT_STEPS"
  --batch "$TRAIN_BATCH" --max-pa "$MAX_PA" --prefetch-depth "$PREFETCH_DEPTH" --lr "$LR"
  --pa-weight "$PA_WEIGHT" --abcd-weight "$ABCD_WEIGHT"
  --residual-scale "$RESIDUAL_SCALE" --pa-ss-rate "$PA_SS_RATE"
  --pa-ss-warmup "$PA_SS_WARMUP" --missing-samples "$MISSING_SAMPLES" --abcd-sequences-per-game "$ABCD_SEQUENCES_PER_GAME"
  --window-size "$ABCD_WINDOW" --context-len "$ABCD_WINDOW"
  --max-len "$ABCD_MAX_LEN" --dropout "$ABCD_DROPOUT" --d-hr-weight "$ABCD_D_HR_WEIGHT"
  "${PA_FEATURE_ARGS[@]}")
if ((PA_PITCHFORMER)); then TRAIN_CMD+=(--pa-pitchformer --pa-arch "$PA_ARCH"); fi
[[ -z "$RESUME" ]] || TRAIN_CMD+=(--resume "$RESUME")

if ((SKIP_TRAIN)); then
  if ((!DRY_RUN)); then
    [[ -f "$JOINT_CKPT" ]] || { echo "Missing joint checkpoint: $JOINT_CKPT" >&2; exit 1; }
    mkdir -p "$RUN_DIR"
  fi
  if [[ -n "$CHECKPOINT" ]]; then
    run_logged export_snapshot "$PYTHON" -m diamondworldjax.scripts.train_pa_abcd \
      --export-checkpoint "$JOINT_CKPT" --tag "$TAG" --out "$OUT"
  fi
  if ((!DRY_RUN)); then
    for path in "$PA_CKPT" "$OUT/${TAG}_metadata.pkl"; do
      [[ -f "$path" ]] || { echo "Missing joint artifact: $path" >&2; exit 1; }
    done
    for head in A B C; do
      [[ -f "$OUT/${head}_${TAG}_params.pkl" ]] || { echo "Missing $head export" >&2; exit 1; }
    done
    if ((!HYBRID_ONLY)); then
      [[ -f "$OUT/D_${TAG}_params.pkl" ]] || { echo "Missing D export" >&2; exit 1; }
    fi
  fi
else
  if ((!DRY_RUN)) && [[ -z "$RESUME" ]] && { [[ -e "$JOINT_CKPT" ]] || [[ -e "$PA_CKPT" ]]; }; then
    echo "Joint artifacts already exist; use --skip-train or choose a new tag." >&2
    exit 1
  fi
  if ((!DRY_RUN)); then mkdir -p "$RUN_DIR"; fi
  run_logged train "${TRAIN_CMD[@]}"
fi

if ((COMPARE_V16)); then
  if ((!HYBRID_ONLY)); then
    PA_TAG="${TAG}_pa"
    PA_RATES="data/eval2/prod_rates_${PA_TAG}.npz"
    PA_SIM_TAG="${PA_TAG}-pregame-leakfree"
    PA_ARRAYS="data/eval2/calib_${PA_SIM_TAG}_arrays.npz"
    PA_RECAL_ARGS=(--recal "$RECAL_FILE")
    PA_PREGAME_RECAL_ARGS=(--recal-file "$RECAL_FILE" --recal-key "$RECAL_KEY" --recal-scale "$RECAL_SCALE")
    if ((NO_RECAL)); then
      PA_RECAL_ARGS=(--no-recal)
      PA_PREGAME_RECAL_ARGS=(--no-recal)
    fi
    run_logged pa_playercorr "$PYTHON" -m diamondworldjax.scripts.prod_playercorr \
      --ckpt "$PA_CKPT" "${PA_RECAL_ARGS[@]}" --skill-mode mean --train-end 2023 --test-seasons 2024 \
      --max-pa-per-game "$V16_MAX_PA" --tag "$PA_TAG" "${PA_FEATURE_ARGS[@]}"
    run_logged pa_bootstrap "$PYTHON" -m diamondworldjax.scripts.bootstrap_playercorr \
      --rates "${BASELINE_TAG}=${BASELINE_RATES}" --rates "${PA_TAG}=${PA_RATES}" \
      --baseline "$BASELINE_TAG" --reps "$BOOTSTRAP_REPS" \
      --out "data/eval2/bootstrap_${PA_TAG}.txt" --json-out "data/eval2/bootstrap_${PA_TAG}.json"
    run_logged pa_pregame "$PYTHON" -m diamondworldjax.scripts.run_pregame_sim \
      --pregame-staff --r "$PREGAME_REPS" --chunk "$PREGAME_CHUNK" --ckpt "$PA_CKPT" --train-end 2023 \
      --tag "$PA_SIM_TAG" "${PA_FEATURE_ARGS[@]}" "${PA_PREGAME_RECAL_ARGS[@]}"
    run_logged pa_benchmarks "$PYTHON" -m diamondworldjax.scripts.simulator_benchmarks \
      --arrays "$PA_ARRAYS" --tag "$PA_SIM_TAG" --season-2024-only

    NATIVE_TAG="${TAG}_abcd"
    NATIVE_RATES="data/eval2/prod_rates_${NATIVE_TAG}.npz"
    NATIVE_SIM_TAG="${NATIVE_TAG}-observed-schedule"
    NATIVE_ARRAYS="data/eval2/calib_${NATIVE_SIM_TAG}_arrays.npz"
    run_logged abcd_playercorr "$PYTHON" -m diamondworldjax.scripts.eval_pitchformer_games \
      --params-dir "$OUT" --tag "$TAG" --season 2024 --c-events --player-stats \
      --min-pa "$MIN_PA" --reps "$COMPARISON_REPS" --limit-games 0 \
      --batch-games "$EVAL_GAME_BATCH" --max-half-continuations "$ABCD_HALF_CONTINUATIONS" --skill-mode mean --seed "$SEED" --rates-max-pa-per-game "$V16_MAX_PA" \
      --rates-out "$NATIVE_RATES" --arrays-out "$NATIVE_ARRAYS" --out "$RUN_DIR/abcd_games.json"
    run_logged abcd_bootstrap "$PYTHON" -m diamondworldjax.scripts.bootstrap_playercorr \
      --rates "${BASELINE_TAG}=${BASELINE_RATES}" --rates "${NATIVE_TAG}=${NATIVE_RATES}" \
      --baseline "$BASELINE_TAG" --reps "$BOOTSTRAP_REPS" \
      --out "data/eval2/bootstrap_${NATIVE_TAG}.txt" --json-out "data/eval2/bootstrap_${NATIVE_TAG}.json"
    run_logged abcd_benchmarks "$PYTHON" -m diamondworldjax.scripts.simulator_benchmarks \
      --arrays "$NATIVE_ARRAYS" --tag "$NATIVE_SIM_TAG" --season-2024-only
  fi

  if ((COMPARE_HYBRID)); then
    HYBRID_TAG="${TAG}_hybrid"
    HYBRID_RATES="data/eval2/prod_rates_${HYBRID_TAG}.npz"
    HYBRID_SIM_TAG="${HYBRID_TAG}-observed-schedule"
    HYBRID_ARRAYS="data/eval2/calib_${HYBRID_SIM_TAG}_arrays.npz"
    run_logged hybrid_playercorr "$PYTHON" -m diamondworldjax.scripts.eval_pitchformer_games \
      --params-dir "$OUT" --tag "$TAG" --season 2024 --c-events --player-stats \
      --min-pa "$MIN_PA" --reps "$COMPARISON_REPS" --limit-games 0 \
      --batch-games "$EVAL_GAME_BATCH" --max-half-continuations "$ABCD_HALF_CONTINUATIONS" --skill-mode mean --seed "$SEED" --rates-max-pa-per-game "$V16_MAX_PA" \
      --hybrid-pa-ckpt "$PA_CKPT" --hybrid-pa-skill-mode mean \
      --rates-out "$HYBRID_RATES" --arrays-out "$HYBRID_ARRAYS" --out "$RUN_DIR/hybrid_games.json"
    run_logged hybrid_bootstrap "$PYTHON" -m diamondworldjax.scripts.bootstrap_playercorr \
      --rates "${BASELINE_TAG}=${BASELINE_RATES}" --rates "${HYBRID_TAG}=${HYBRID_RATES}" \
      --baseline "$BASELINE_TAG" --reps "$BOOTSTRAP_REPS" \
      --out "data/eval2/bootstrap_${HYBRID_TAG}.txt" --json-out "data/eval2/bootstrap_${HYBRID_TAG}.json"
    run_logged hybrid_benchmarks "$PYTHON" -m diamondworldjax.scripts.simulator_benchmarks \
      --arrays "$HYBRID_ARRAYS" --tag "$HYBRID_SIM_TAG" --season-2024-only
  fi
else
  if ((!HYBRID_ONLY)); then
    run_logged players "$PYTHON" -m diamondworldjax.scripts.eval_players \
      --ckpt "$PA_CKPT" --min-pa "$MIN_PA" --samples "$PA_SIM_REPS" --skill-mode mean --seed "$SEED"
    for ((rep=0; rep<PA_SIM_REPS; rep++)); do
      rep_seed=$((SEED + rep * 1000003))
      run_logged "pa_games_$rep" "$PYTHON" -m diamondworldjax.scripts.simulate_games \
        --ckpt "$PA_CKPT" --player-stats --min-pa "$MIN_PA" --limit-games "$LIMIT_GAMES" \
        --skill-mode mean --seed "$rep_seed" --dump-scores "$RUN_DIR/pa_scores_$rep.npz"
    done
    run_logged abcd_games "$PYTHON" -m diamondworldjax.scripts.eval_pitchformer_games \
      --params-dir "$OUT" --tag "$TAG" --season 2024 --c-events --player-stats \
      --min-pa "$MIN_PA" --reps "$ABCD_SIM_REPS" --limit-games "$LIMIT_GAMES" \
      --batch-games "$EVAL_GAME_BATCH" --max-half-continuations "$ABCD_HALF_CONTINUATIONS" --skill-mode mean --seed "$SEED" \
      --out "$RUN_DIR/abcd_games.json"
  fi
  if ((HYBRID_ROLLOUT)); then
    run_logged hybrid_games "$PYTHON" -m diamondworldjax.scripts.eval_pitchformer_games \
      --params-dir "$OUT" --tag "$TAG" --season 2024 --c-events --player-stats \
      --min-pa "$MIN_PA" --reps "$ABCD_SIM_REPS" --limit-games "$LIMIT_GAMES" \
      --batch-games "$EVAL_GAME_BATCH" --max-half-continuations "$ABCD_HALF_CONTINUATIONS" --skill-mode mean --seed "$SEED" \
      --hybrid-pa-ckpt "$PA_CKPT" --hybrid-pa-skill-mode mean \
      --out "$RUN_DIR/hybrid_games.json"
  fi
fi

if ((!DRY_RUN)); then
  echo "Finished joint run. Compare PA and ABCD reports independently; preserve the joint checkpoint as provenance."
fi
