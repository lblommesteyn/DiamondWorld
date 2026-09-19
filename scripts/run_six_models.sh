#!/usr/bin/env bash
# Six standalone model experiments; no joint PA/ABCD integration.
set -Eeuo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS=all
PREFIX=six
SEED=0
PA_STEPS=50000
ABCD_STEPS=4000
PA_BATCH=64
PA_SS_RATE=0
PA_FEATURE_PROFILE=ladder
ABCD_BATCH=32
BAYESIAN_BATCH=8
BAYESIAN_UPDATE_CHUNK_SIZE=1
BAYESIAN_MEM_FRACTION="${BAYESIAN_MEM_FRACTION:-0.8}"
# PA worlds are cheap independent processes; ABCD rollouts compound pitch
# state and are deliberately kept short for routine experiments.
PA_SIM_REPS=100
ABCD_SIM_REPS=5
LIMIT_GAMES=512
MIN_PA=150
RUNS_UPWEIGHT=0
SKILL_MODE=sample
SKILL_FEATURES=""
HISTORY_RESET=game
ABCD_DROPOUT=0.1
ABCD_D_HR_WEIGHT=0
ABCD_WINDOW=32
ABCD_MAX_LEN=160
ABCD_HALF_CONTINUATIONS=6
MISSING_SAMPLES=2
GAME_BATCH=""
SKILL_FEATURE_MODE=pa
CUDA_DEVICE=""
SKIP_TRAIN=0
DRY_RUN=0
COMPARE_V16=0
BOOTSTRAP_REPS=20000
PREGAME_REPS=100
PREGAME_CHUNK=""
BASELINE_TAG=v16
BASELINE_RATES=data/eval2/prod_rates_v16.npz
RECAL_FILE=data/eval2/v13_cal_params.npz
RECAL_KEY=b_heur
RECAL_SCALE=0.18
NO_RECAL=0
V16_MAX_PA=90
usage() {
  cat <<'EOF'
Usage: bash scripts/run_six_models.sh [options]

Trains and evaluates any/all of these independent experiments:
  pa                 PA without a sequence model
  pa_gru             PA with GRU
  pa_gru_skip        PA with GRU plus a raw-context residual path
  pa_transformer     PA with transformer
  abcd_none          ABCD without player identity embeddings
  abcd_bayesian      ABCD shared Bayesian skills + A/B/C/D residual skills
  abcd_id            ABCD learned player-ID embeddings

Options:
  --models LIST          all, or comma-separated names above (default all)
  --tag-prefix TAG       Tags become TAG_MODEL_sSEED (default six)
  --seed N              Training and evaluation seed (default 0)
  --cuda-device DEVICE  Optional CUDA_VISIBLE_DEVICES
  --pa-steps N          PA training/checkpoint step (default 50000)
  --abcd-steps N        ABCD updates per head (default 4000)
  --pa-batch N          PA training batch size (default 64)
  --pa-ss-rate X        PA scheduled-sampling rate, 0..1 (default 0; alias --ss-rate)
  --pa-feature-profile MODE  Shared PA features: ladder (default), v16, or basic
  --abcd-batch N        ABCD training batch size (default 32)
  --bayesian-batch N    Native Bayesian ABCD training batch size (default 8)
  --bayesian-update-chunk-size N  Bayesian updates per compiled chunk (default 1)
  --bayesian-mem-fraction X  JAX GPU fraction for Bayesian ABCD training (default 0.8)
  --game-batch N        ABCD simulation batch size (default: --abcd-batch)
  --abcd-window N       Prior pitch tokens in strict window (default 32)
  --abcd-max-len N      Training sequence length including overlap (default 160)
  --abcd-half-continuations N  Extra cyclic decodes allowed to finish an ABCD half (default 6)
  --missing-samples N   Current-measurement integration samples (default 2)
  --skill-feature-mode MODE  Bayesian ABCD role-aware PA/BF (pa, default) or neutral covariates
  --pa-sim-reps N       PA simulated worlds (default 100)
  --abcd-sim-reps N     ABCD generated worlds (default 5)
  --sim-reps N          Legacy alias: set both PA and ABCD world counts
  --limit-games N       Maximum held-out games; 0 = all (default 512)
  --min-pa N            Player eligibility threshold (default 150)
  --runs-upweight N     PA runs_upweight factor (default 0)
  --skill-mode MODE     sample or mean for Bayesian models (default sample)
  --history-reset MODE  ABCD game (default), half_inning, or batting_side
  --abcd-dropout X      ABCD training dropout (default 0.1)
  --abcd-d-hr-weight X  Auxiliary D binary-HR loss weight (default 0; baseline objective)
  --skill-features PATH Optional ABCD training-only covariate NPZ
  --compare-v16         Run all model types through their available v16 comparison stages
  --bootstrap-reps N    Paired-bootstrap replicates with --compare-v16 (default 20000)
  --pregame-reps N      Simulated replicas/game with --compare-v16 (default 100)
  --pregame-chunk N     Games per PA pregame-simulation chunk (default: 250; 50 for transformer)
  --baseline-tag TAG    Baseline label for --compare-v16 (default v16)
  --baseline-rates PATH Baseline prod_rates NPZ (default data/eval2/prod_rates_v16.npz)
  --recal-file PATH     PA comparison calibration NPZ (default data/eval2/v13_cal_params.npz)
  --recal-key KEY       Calibration-vector key (default b_heur)
  --recal-scale X       Calibration-vector multiplier for pregame simulation (default 0.18)
  --no-recal            Use raw PA outcome logits for PA rate scoring and pregame simulation
  --v16-max-pa N        Per-game PA cap for paired rates (default 90; 0 = no cap)
  --skip-train          Require existing checkpoints; only evaluate
  --dry-run             Print every command without creating files
  -h, --help

Split: train 2015-2023 INCLUDING 2020; test 2024. No recalibration.
PA runs use outcome-only, fatigue, walk skills, and one shared feature profile;
only sequence model varies. ladder = recency 2, contact quality, per-stat shrink;
v16 = recency 2 plus contact quality; basic = no feature transformations.
PA scheduled sampling is off by default. Changing it affects new training only.
ABCD uses sinusoidal positions, strict windows, matching training overlap,
mask-aware history, marginal likelihood, and categorical C event bundles.
All ABCD trainers use dropout, AdamW, warm-up/cosine scheduling, and clipping at 1.
ABCD runs retain pitch history, environment, geometry, and C events.
Game history resets at the game boundary by default, matching PA's boundary.
--skip-train restores saved checkpoint settings; training flags do not retrofit them.
PA game replicas run in separate processes; ABCD uses --reps in one process.
Logs and score artifacts: data/runs/six_TAG_MODEL_sSEED/.
--compare-v16 replaces PA exploratory players/game simulations with playercorr.log,
bootstrap.log, pregame_sim.log, and benchmarks.log. ABCD additionally exports
generated-outcome player rates for the same paired bootstrap and observed-schedule
benchmark arrays; this is not a pre-game roster/staff simulation. Paired rate
artifacts cap each game at 90 PAs to match the historical v16 artifact.
Existing checkpoint artifacts are never silently retrained: use --skip-train or
choose a new tag. Set PYTHON_BIN to override .venv/bin/python or python.

Examples:
  bash scripts/run_six_models.sh --dry-run
  bash scripts/run_six_models.sh --models pa_gru_skip --cuda-device 0 --tag-prefix cmp
  bash scripts/run_six_models.sh --models abcd_none,abcd_bayesian,abcd_id --cuda-device 1 --tag-prefix cmp
  bash scripts/run_six_models.sh --compare-v16 --tag-prefix cmp --cuda-device 0

ABCD 'none' retains contextual matchup features. Native Bayesian ABCD uses PA
statistical features unless --skill-features is supplied.
EOF
}
while (($#)); do
  case "$1" in
    --abcd-window) ABCD_WINDOW="${2:?value required}"; shift 2 ;;
    --abcd-max-len) ABCD_MAX_LEN="${2:?value required}"; shift 2 ;;
    --abcd-half-continuations) ABCD_HALF_CONTINUATIONS="${2:?value required}"; shift 2 ;;
    --missing-samples) MISSING_SAMPLES="${2:?value required}"; shift 2 ;;
    --game-batch) GAME_BATCH="${2:?value required}"; shift 2 ;;
    --skill-feature-mode) SKILL_FEATURE_MODE="${2:?value required}"; shift 2 ;;
    --history-reset) HISTORY_RESET="${2:?value required}"; shift 2 ;;
    --abcd-dropout) ABCD_DROPOUT="${2:?value required}"; shift 2 ;;
    --abcd-d-hr-weight) ABCD_D_HR_WEIGHT="${2:?value required}"; shift 2 ;;
    --models) MODELS="${2:?value required}"; shift 2 ;;
    --tag-prefix) PREFIX="${2:?value required}"; shift 2 ;;
    --seed) SEED="${2:?value required}"; shift 2 ;;
    --cuda-device) CUDA_DEVICE="${2:?value required}"; shift 2 ;;
    --pa-steps) PA_STEPS="${2:?value required}"; shift 2 ;;
    --abcd-steps) ABCD_STEPS="${2:?value required}"; shift 2 ;;
    --pa-batch) PA_BATCH="${2:?value required}"; shift 2 ;;
    --pa-ss-rate|--ss-rate) PA_SS_RATE="${2:?value required}"; shift 2 ;;
    --pa-feature-profile) PA_FEATURE_PROFILE="${2:?value required}"; shift 2 ;;
    --abcd-batch) ABCD_BATCH="${2:?value required}"; shift 2 ;;
    --bayesian-batch) BAYESIAN_BATCH="${2:?value required}"; shift 2 ;;
    --bayesian-update-chunk-size) BAYESIAN_UPDATE_CHUNK_SIZE="${2:?value required}"; shift 2 ;;
    --bayesian-mem-fraction) BAYESIAN_MEM_FRACTION="${2:?value required}"; shift 2 ;;
    --pa-sim-reps) PA_SIM_REPS="${2:?value required}"; shift 2 ;;
    --abcd-sim-reps) ABCD_SIM_REPS="${2:?value required}"; shift 2 ;;
    --sim-reps) PA_SIM_REPS="${2:?value required}"; ABCD_SIM_REPS="$PA_SIM_REPS"; shift 2 ;;
    --limit-games) LIMIT_GAMES="${2:?value required}"; shift 2 ;;
    --min-pa) MIN_PA="${2:?value required}"; shift 2 ;;
    --runs-upweight) RUNS_UPWEIGHT="${2:?value required}"; shift 2 ;;
    --skill-mode) SKILL_MODE="${2:?value required}"; shift 2 ;;
    --skill-features) SKILL_FEATURES="${2:?value required}"; shift 2 ;;
    --compare-v16) COMPARE_V16=1; shift ;;
    --bootstrap-reps) BOOTSTRAP_REPS="${2:?value required}"; shift 2 ;;
    --pregame-reps) PREGAME_REPS="${2:?value required}"; shift 2 ;;
    --pregame-chunk) PREGAME_CHUNK="${2:?value required}"; shift 2 ;;
    --baseline-tag) BASELINE_TAG="${2:?value required}"; shift 2 ;;
    --baseline-rates) BASELINE_RATES="${2:?value required}"; shift 2 ;;
    --recal-file) RECAL_FILE="${2:?value required}"; shift 2 ;;
    --recal-key) RECAL_KEY="${2:?value required}"; shift 2 ;;
    --recal-scale) RECAL_SCALE="${2:?value required}"; shift 2 ;;
    --no-recal) NO_RECAL=1; shift ;;
    --v16-max-pa) V16_MAX_PA="${2:?value required}"; shift 2 ;;
    --skip-train) SKIP_TRAIN=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ "$PREFIX" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo 'Invalid tag prefix' >&2; exit 2; }
[[ "$SKILL_MODE" == mean || "$SKILL_MODE" == sample ]] || { echo 'Invalid skill mode' >&2; exit 2; }
[[ "$PA_SS_RATE" =~ ^(0([.][0-9]+)?|1([.]0+)?|[.][0-9]+)$ ]] || { echo '--pa-ss-rate must be a decimal between 0 and 1' >&2; exit 2; }
case "$PA_FEATURE_PROFILE" in
  ladder) PA_FEATURE_ARGS=(--recency-halflife 2.0 --contact-quality --per-stat-shrink) ;;
  v16) PA_FEATURE_ARGS=(--recency-halflife 2.0 --contact-quality) ;;
  basic) PA_FEATURE_ARGS=(--recency-halflife 0) ;;
  *) echo 'Invalid PA feature profile: use ladder, v16, or basic' >&2; exit 2 ;;
esac
case "$HISTORY_RESET" in game|half_inning|batting_side) ;; *) echo 'Invalid history reset policy' >&2; exit 2 ;; esac
case "$SKILL_FEATURE_MODE" in pa|neutral) ;; *) echo 'Invalid skill feature mode' >&2; exit 2 ;; esac
[[ "$ABCD_DROPOUT" =~ ^(0([.][0-9]+)?|[.][0-9]+)$ ]] || { echo '--abcd-dropout must be >= 0 and < 1' >&2; exit 2; }
[[ "$ABCD_D_HR_WEIGHT" =~ ^([0-9]+([.][0-9]+)?|[.][0-9]+)$ ]] || { echo '--abcd-d-hr-weight must be nonnegative' >&2; exit 2; }
[[ "$RUNS_UPWEIGHT" =~ ^([0-9]+([.][0-9]+)?|[.][0-9]+)$ ]] || { echo '--runs-upweight must be nonnegative' >&2; exit 2; }
GAME_BATCH="${GAME_BATCH:-$ABCD_BATCH}"
for name in ABCD_WINDOW ABCD_MAX_LEN ABCD_HALF_CONTINUATIONS MISSING_SAMPLES GAME_BATCH SEED PA_STEPS ABCD_STEPS PA_BATCH ABCD_BATCH BAYESIAN_BATCH BAYESIAN_UPDATE_CHUNK_SIZE PA_SIM_REPS ABCD_SIM_REPS LIMIT_GAMES MIN_PA; do
  value="${!name}"
  [[ "$value" =~ ^(0|[1-9][0-9]*)$ ]] || { echo "Invalid integer: $name" >&2; exit 2; }
  if [[ "$name" != SEED && "$name" != LIMIT_GAMES ]] && ((value < 1)); then echo "$name must be positive" >&2; exit 2; fi
done
[[ "$BAYESIAN_MEM_FRACTION" =~ ^(0([.][0-9]+)?|[.][0-9]+|1([.]0+)?)$ ]] || { echo '--bayesian-mem-fraction must be in (0, 1]' >&2; exit 2; }
awk -v value="$BAYESIAN_MEM_FRACTION" 'BEGIN { exit !(value > 0 && value <= 1) }' || { echo '--bayesian-mem-fraction must be in (0, 1]' >&2; exit 2; }
if ((COMPARE_V16)); then
  for name in BOOTSTRAP_REPS PREGAME_REPS; do
    value="${!name}"
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "$name must be a positive integer" >&2; exit 2; }
  done
  if [[ -n "$PREGAME_CHUNK" ]]; then
    [[ "$PREGAME_CHUNK" =~ ^[1-9][0-9]*$ ]] || { echo 'PREGAME_CHUNK must be a positive integer' >&2; exit 2; }
  fi
  [[ "$BASELINE_TAG" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo 'Invalid baseline tag' >&2; exit 2; }
  [[ "$RECAL_SCALE" =~ ^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]] || { echo 'Invalid recalibration scale' >&2; exit 2; }
  [[ "$V16_MAX_PA" =~ ^(0|[1-9][0-9]*)$ ]] || { echo '--v16-max-pa must be nonnegative' >&2; exit 2; }
  ((LIMIT_GAMES == 0)) || { echo '--compare-v16 requires --limit-games 0 so ABCD and v16 use the same 2024 batter cohort' >&2; exit 2; }
fi
((ABCD_WINDOW < ABCD_MAX_LEN)) || { echo '--abcd-window must be smaller than --abcd-max-len' >&2; exit 2; }
[[ -z "$SKILL_FEATURES" || "$SKILL_FEATURE_MODE" != neutral ]] || { echo 'Choose neutral features or an NPZ, not both' >&2; exit 2; }
((ABCD_STEPS >= 2)) || { echo 'ABCD steps must be >= 2' >&2; exit 2; }
if [[ "$MODELS" == all ]]; then MODELS=pa,pa_gru,pa_gru_skip,pa_transformer,abcd_none,abcd_bayesian,abcd_id; fi
[[ -n "$MODELS" && "$MODELS" != ,* && "$MODELS" != *, && "$MODELS" != *,,* ]] || { echo 'Empty model name' >&2; exit 2; }
IFS=, read -r -a SELECTED <<< "$MODELS"
SEEN=,
for model in "${SELECTED[@]}"; do
  case "$model" in pa|pa_gru|pa_gru_skip|pa_transformer|abcd_none|abcd_bayesian|abcd_id) ;; *) echo "Invalid model: $model" >&2; exit 2 ;; esac
  [[ "$SEEN" != *",$model,"* ]] || { echo "Duplicate model: $model" >&2; exit 2; }
  SEEN+="$model,"
done
cd "$ROOT_DIR"
[[ -z "$CUDA_DEVICE" ]] || export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.8}"
if [[ -n "${PYTHON_BIN:-}" ]]; then PYTHON="$PYTHON_BIN"
elif [[ -x .venv/bin/python ]]; then PYTHON=.venv/bin/python
elif [[ -x .venv/Scripts/python.exe ]]; then PYTHON=.venv/Scripts/python.exe
else PYTHON=python; fi
if ((!DRY_RUN)) && [[ -n "$SKILL_FEATURES" && ! -f "$SKILL_FEATURES" ]]; then
  echo "Missing skill features: $SKILL_FEATURES" >&2; exit 2
fi
if ((COMPARE_V16 && !DRY_RUN)); then
  [[ -f "$BASELINE_RATES" ]] || { echo "Missing baseline rates: $BASELINE_RATES" >&2; exit 2; }
  [[ -f "$RECAL_FILE" ]] || { echo "Missing calibration file: $RECAL_FILE" >&2; exit 2; }
fi
run_logged() {
  local label="$1"; shift
  printf '[%s] ' "$label"; printf '%q ' "$@"; printf '\n'
  if ((DRY_RUN)); then return; fi
  "$@" 2>&1 | tee "$RUN_DIR/$label.log"
}
for model in "${SELECTED[@]}"; do
  TAG="${PREFIX}_${model}_s${SEED}"
  RUN_DIR="data/runs/six_${TAG}"
  if ((!DRY_RUN)); then mkdir -p "$RUN_DIR"; fi
  printf '\n=== %s ===\n' "$TAG"
  if [[ "$model" == pa || "$model" == pa_* ]]; then
    CKPT="checkpoints/dwjax_pa_${TAG}/$(printf 'dwjax_step_%07d.pkl' "$PA_STEPS")"
    MODEL_FLAGS=(--outcome-only --fatigue --skill-prior walk --ss-rate "$PA_SS_RATE")
    case "$model" in
      pa_gru) MODEL_FLAGS+=(--pitchformer --pa-arch gru) ;;
      pa_gru_skip) MODEL_FLAGS+=(--pitchformer --pa-arch gru_skip) ;;
      pa_transformer) MODEL_FLAGS+=(--pitchformer --pa-arch transformer) ;;
    esac
    if ((SKIP_TRAIN)); then
      if ((!DRY_RUN)) && [[ ! -f "$CKPT" ]]; then echo "Missing $CKPT" >&2; exit 1; fi
    else
      if ((!DRY_RUN)) && [[ -d "checkpoints/dwjax_pa_${TAG}" ]]; then echo "Checkpoint directory exists; use --skip-train or a new tag: $TAG" >&2; exit 1; fi
      run_logged train "$PYTHON" -m diamondworldjax.scripts.train_pa --tag "$TAG" --train-end 2023 --seed "$SEED" --steps "$PA_STEPS" --batch "$PA_BATCH" --runs-upweight "$RUNS_UPWEIGHT" "${PA_FEATURE_ARGS[@]}" "${MODEL_FLAGS[@]}"
    fi
    if ((COMPARE_V16)); then
      COMPARISON_FLAGS=(--skill-prior walk "${PA_FEATURE_ARGS[@]}")
      case "$model" in
        pa_gru) COMPARISON_FLAGS+=(--pitchformer --pa-arch gru) ;;
        pa_gru_skip) COMPARISON_FLAGS+=(--pitchformer --pa-arch gru_skip) ;;
        pa_transformer) COMPARISON_FLAGS+=(--pitchformer --pa-arch transformer) ;;
      esac
      RATES="data/eval2/prod_rates_${TAG}.npz"
      SIM_TAG="${TAG}-pregame-leakfree"
      ARRAYS="data/eval2/calib_${SIM_TAG}_arrays.npz"
      MODEL_PREGAME_CHUNK="${PREGAME_CHUNK:-250}"
      [[ "$model" != pa_transformer || -n "$PREGAME_CHUNK" ]] || MODEL_PREGAME_CHUNK=75
      PLAYER_RECAL_ARGS=(--recal "$RECAL_FILE")
      ((NO_RECAL)) && PLAYER_RECAL_ARGS=(--no-recal)
      run_logged playercorr "$PYTHON" -m diamondworldjax.scripts.prod_playercorr --ckpt "$CKPT" "${PLAYER_RECAL_ARGS[@]}" --skill-mode mean --train-end 2023 --test-seasons 2024 --max-pa-per-game "$V16_MAX_PA" --tag "$TAG" "${COMPARISON_FLAGS[@]}"
      run_logged bootstrap "$PYTHON" -m diamondworldjax.scripts.bootstrap_playercorr --rates "${BASELINE_TAG}=${BASELINE_RATES}" --rates "${TAG}=${RATES}" --baseline "$BASELINE_TAG" --reps "$BOOTSTRAP_REPS" --out "data/eval2/bootstrap_${TAG}.txt" --json-out "data/eval2/bootstrap_${TAG}.json"
      PREGAME_RECAL_ARGS=(--recal-file "$RECAL_FILE" --recal-key "$RECAL_KEY" --recal-scale "$RECAL_SCALE")
      ((NO_RECAL)) && PREGAME_RECAL_ARGS+=(--no-recal)
      run_logged pregame_sim "$PYTHON" -m diamondworldjax.scripts.run_pregame_sim --pregame-staff --r "$PREGAME_REPS" --chunk "$MODEL_PREGAME_CHUNK" --ckpt "$CKPT" --train-end 2023 --tag "$SIM_TAG" "${COMPARISON_FLAGS[@]}" "${PREGAME_RECAL_ARGS[@]}"
      run_logged benchmarks "$PYTHON" -m diamondworldjax.scripts.simulator_benchmarks --arrays "$ARRAYS" --tag "$SIM_TAG" --season-2024-only
    else
      run_logged players "$PYTHON" -m diamondworldjax.scripts.eval_players --ckpt "$CKPT" --min-pa "$MIN_PA" --samples "$PA_SIM_REPS" --skill-mode "$SKILL_MODE" --seed "$SEED"
      for ((rep=0; rep<PA_SIM_REPS; rep++)); do
        rep_seed=$((SEED + rep * 1000003))
        run_logged "games_$rep" "$PYTHON" -m diamondworldjax.scripts.simulate_games --ckpt "$CKPT" --player-stats --min-pa "$MIN_PA" --limit-games "$LIMIT_GAMES" --skill-mode "$SKILL_MODE" --seed "$rep_seed" --dump-scores "$RUN_DIR/scores_$rep.npz"
      done
    fi
  else
    PARAMS_DIR="checkpoints/pitchformer"
    ABCD_COMMON=(--history-reset "$HISTORY_RESET" --dropout "$ABCD_DROPOUT" --d-hr-weight "$ABCD_D_HR_WEIGHT"
      --position-encoding sinusoidal --window-size "$ABCD_WINDOW" --context-len "$ABCD_WINDOW"
      --max-len "$ABCD_MAX_LEN" --pitch-history --missing-samples "$MISSING_SAMPLES" --c-event-mode bundles)
    MODEL_FLAGS=(--shared-emb --player-skills none)
    TRAIN_ENV=()
    TRAIN_BATCH="$ABCD_BATCH"
    TRAIN_FLAGS=()
    EVAL_SKILLS=mean
    case "$model" in
      abcd_id) MODEL_FLAGS=(--shared-emb --player-skills id) ;;
      abcd_bayesian)
        MODEL_FLAGS=(--player-skills bayesian --skill-prior walk --skill-feature-mode "$SKILL_FEATURE_MODE")
        [[ "$SKILL_FEATURE_MODE" != pa ]] || MODEL_FLAGS+=("${PA_FEATURE_ARGS[@]}")
        TRAIN_BATCH="$BAYESIAN_BATCH"
        TRAIN_FLAGS=(--update-chunk-size "$BAYESIAN_UPDATE_CHUNK_SIZE" --prefetch-depth 1)
        TRAIN_ENV=(env "XLA_PYTHON_CLIENT_MEM_FRACTION=$BAYESIAN_MEM_FRACTION")
        EVAL_SKILLS="$SKILL_MODE"
        [[ -z "$SKILL_FEATURES" ]] || MODEL_FLAGS+=(--skill-features "$SKILL_FEATURES") ;;
    esac
    if ((SKIP_TRAIN)); then
      if ((!DRY_RUN)); then
        for letter in A B C D; do
          [[ -f "$PARAMS_DIR/${letter}_${TAG}_params.pkl" ]] || { echo "Missing $letter checkpoint for $TAG" >&2; exit 1; }
        done
        [[ -f "$PARAMS_DIR/${TAG}_metadata.pkl" ]] || { echo "Missing metadata for $TAG" >&2; exit 1; }
        if [[ "$model" == abcd_bayesian ]]; then
          [[ -f "$PARAMS_DIR/bayesian_${TAG}.pkl" ]] || { echo "Missing Bayesian posterior for $TAG" >&2; exit 1; }
        fi
      fi
    else
      if ((!DRY_RUN)) && compgen -G "$PARAMS_DIR/*_${TAG}*.pkl" > /dev/null; then echo "Checkpoint artifacts exist; use --skip-train or a new tag: $TAG" >&2; exit 1; fi
      run_logged train "${TRAIN_ENV[@]}" "$PYTHON" -m diamondworldjax.scripts.train_pitchformer --stack abcd --tag "$TAG" --out "$PARAMS_DIR" --train-seasons 2015,2016,2017,2018,2019,2020,2021,2022,2023 --test-season 2024 --seed "$SEED" --steps "$ABCD_STEPS" --bs "$TRAIN_BATCH" "${TRAIN_FLAGS[@]}" "${ABCD_COMMON[@]}" "${MODEL_FLAGS[@]}"
    fi
    if ((COMPARE_V16)); then
      RATES="data/eval2/prod_rates_${TAG}.npz"
      SIM_TAG="${TAG}-observed-schedule"
      ARRAYS="data/eval2/calib_${SIM_TAG}_arrays.npz"
      run_logged playercorr "$PYTHON" -m diamondworldjax.scripts.eval_pitchformer_games --params-dir "$PARAMS_DIR" --tag "$TAG" --season 2024 --c-events --player-stats --min-pa "$MIN_PA" --reps "$ABCD_SIM_REPS" --limit-games 0 --batch-games "$GAME_BATCH" --max-half-continuations "$ABCD_HALF_CONTINUATIONS" --skill-mode "$EVAL_SKILLS" --seed "$SEED" --rates-max-pa-per-game "$V16_MAX_PA" --rates-out "$RATES" --arrays-out "$ARRAYS" --out "$RUN_DIR/games.json"
      run_logged bootstrap "$PYTHON" -m diamondworldjax.scripts.bootstrap_playercorr --rates "${BASELINE_TAG}=${BASELINE_RATES}" --rates "${TAG}=${RATES}" --baseline "$BASELINE_TAG" --reps "$BOOTSTRAP_REPS" --out "data/eval2/bootstrap_${TAG}.txt" --json-out "data/eval2/bootstrap_${TAG}.json"
      run_logged benchmarks "$PYTHON" -m diamondworldjax.scripts.simulator_benchmarks --arrays "$ARRAYS" --tag "$SIM_TAG" --season-2024-only
    else
      run_logged games "$PYTHON" -m diamondworldjax.scripts.eval_pitchformer_games --params-dir "$PARAMS_DIR" --tag "$TAG" --season 2024 --c-events --player-stats --min-pa "$MIN_PA" --reps "$ABCD_SIM_REPS" --limit-games "$LIMIT_GAMES" --batch-games "$GAME_BATCH" --max-half-continuations "$ABCD_HALF_CONTINUATIONS" --skill-mode "$EVAL_SKILLS" --seed "$SEED" --out "$RUN_DIR/games.json"
    fi
  fi
done

if ((!DRY_RUN)); then
  echo "Finished selected models. Review ABCD games.json completion_diagnostics and training-report c_support_coverage before comparing scores."
fi
