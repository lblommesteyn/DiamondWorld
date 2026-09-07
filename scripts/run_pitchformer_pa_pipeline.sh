#!/usr/bin/env bash
# Train, test/evaluate, and simulate a PA-level Pitchformer model.
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAG="pitchformer_pa"
STEPS=50000
BATCH=64
SAMPLES=8
SIM_GAMES=512
SEED=0
TRAIN_END=2022
PITCHFORMER_DIM=128
PITCHFORMER_LAYERS=2
PITCHFORMER_HEADS=4
PITCHFORMER_DROPOUT=0.0
SKIP_TRAIN=0
SKIP_TEST=0
SKIP_EVAL=0
SKIP_SIMS=0
TRAIN_EXTRA=()
EVAL_EXTRA=()
SIM_EXTRA=()

usage() {
  cat <<'EOF'
Usage: scripts/run_pitchformer_pa_pipeline.sh [options]

Runs: train_pa -> eval_pa (held-out test) -> eval_players -> simulate_games.

Core options:
  --tag NAME                 Output/checkpoint tag (default: pitchformer_pa)
  --steps N                  Training steps (default: 50000)
  --batch N                  Games per train/eval batch (default: 64)
  --samples N                Samples per held-out game (default: 8)
  --sim-games N              Games to simulate (default: 512)
  --seed N                   Random seed (default: 0)
  --train-end YEAR           Last training season (default: 2022)
  --dim N                    Pitchformer hidden width (default: 128)
  --layers N                 Pitchformer layers (default: 2)
  --heads N                  Pitchformer heads (default: 4)
  --dropout X                Training dropout (default: 0.0)
  --skip-train|--skip-test|--skip-eval|--skip-sims

Extra phase arguments (repeat once per argument/token):
  --train-arg ARG            Append ARG to train_pa
  --eval-arg ARG             Append ARG to eval_pa and eval_players
  --sim-arg ARG              Append ARG to simulate_games

Example:
  scripts/run_pitchformer_pa_pipeline.sh --tag pf_try --steps 10000 \
    --train-arg=--fatigue --eval-arg=--fatigue --eval-arg=--use-park \
    --sim-arg=--fatigue --sim-arg=--use-park
EOF
}

while (($#)); do
  case "$1" in
    --tag) TAG="$2"; shift 2;;
    --steps) STEPS="$2"; shift 2;;
    --batch) BATCH="$2"; shift 2;;
    --samples) SAMPLES="$2"; shift 2;;
    --sim-games) SIM_GAMES="$2"; shift 2;;
    --seed) SEED="$2"; shift 2;;
    --train-end) TRAIN_END="$2"; shift 2;;
    --dim) PITCHFORMER_DIM="$2"; shift 2;;
    --layers) PITCHFORMER_LAYERS="$2"; shift 2;;
    --heads) PITCHFORMER_HEADS="$2"; shift 2;;
    --dropout) PITCHFORMER_DROPOUT="$2"; shift 2;;
    --skip-train) SKIP_TRAIN=1; shift;;
    --skip-test) SKIP_TEST=1; shift;;
    --skip-eval) SKIP_EVAL=1; shift;;
    --skip-sims) SKIP_SIMS=1; shift;;
    --train-arg) TRAIN_EXTRA+=("$2"); shift 2;;
    --train-arg=*) TRAIN_EXTRA+=("${1#*=}"); shift;;
    --eval-arg) EVAL_EXTRA+=("$2"); shift 2;;
    --eval-arg=*) EVAL_EXTRA+=("${1#*=}"); shift;;
    --sim-arg) SIM_EXTRA+=("$2"); shift 2;;
    --sim-arg=*) SIM_EXTRA+=("${1#*=}"); shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2;;
  esac
done

cd "$ROOT_DIR"
RUN_DIR="data/runs/${TAG}"
CKPT="checkpoints/dwjax_pa_${TAG}/$(printf 'dwjax_step_%07d.pkl' "$STEPS")"
mkdir -p "$RUN_DIR"

MODEL_ARGS=(--outcome-only --pitchformer
  --pitchformer-dim "$PITCHFORMER_DIM"
  --pitchformer-layers "$PITCHFORMER_LAYERS"
  --pitchformer-heads "$PITCHFORMER_HEADS"
  --pitchformer-dropout "$PITCHFORMER_DROPOUT")

run_logged() {
  local name="$1"; shift
  echo "[$(date -u +%FT%TZ)] $name"
  "$@" 2>&1 | tee "$RUN_DIR/${name}.log"
}

if (( ! SKIP_TRAIN )); then
  run_logged train python -m diamondworldjax.scripts.train_pa \
    --tag "$TAG" --steps "$STEPS" --batch "$BATCH" --seed "$SEED" \
    --train-end "$TRAIN_END" "${MODEL_ARGS[@]}" "${TRAIN_EXTRA[@]}"
fi

[[ -f "$CKPT" ]] || { echo "Checkpoint not found: $CKPT" >&2; exit 1; }

if (( ! SKIP_TEST )); then
  run_logged test python -m diamondworldjax.scripts.eval_pa \
    --ckpt "$CKPT" --batch "$BATCH" --samples "$SAMPLES" --seed "$SEED" \
    --out "$RUN_DIR/test.json" "${MODEL_ARGS[@]}" "${EVAL_EXTRA[@]}"
fi

if (( ! SKIP_EVAL )); then
  run_logged evaluate_players python -m diamondworldjax.scripts.eval_players \
    --ckpt "$CKPT" --batch "$BATCH" --samples "$SAMPLES" --seed "$SEED" \
    "${MODEL_ARGS[@]}" "${EVAL_EXTRA[@]}"
fi

if (( ! SKIP_SIMS )); then
  run_logged simulate python -m diamondworldjax.scripts.simulate_games \
    --ckpt "$CKPT" --limit-games "$SIM_GAMES" --seed "$SEED" \
    --dump-runs "$RUN_DIR/sim_runs.npy" "${MODEL_ARGS[@]}" "${SIM_EXTRA[@]}"
fi

echo "Pipeline complete. Outputs: $RUN_DIR"
