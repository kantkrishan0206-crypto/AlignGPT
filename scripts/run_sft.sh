#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# run_sft.sh
#
# Robust launcher script to run Supervised Fine-Tuning (SFT) training for
# the rlhf-lab project. Designed to be safe, reproducible, and work well for
# local development (VS Code) as well as small-scale single-node GPU runs.
#
# Features:
# - argument parsing and config overrides
# - conda / venv auto-activation support
# - optional deepspeed / torchrun support for multi-GPU
# - automatic environment validation (python, CUDA, GPUs)
# - logging to rolling log files and stdout
# - resume/load-from-checkpoint support
# - dry-run mode to check everything before launching
#
# Usage examples:
#   ./scripts/run_sft.sh --config configs/sft_gpt2.yaml
#   ./scripts/run_sft.sh --config configs/sft_gpt2.yaml --epochs 3 --batch_size 8
#   ./scripts/run_sft.sh --config configs/sft_gpt2.yaml --resume checkpoints/sft/step_1000
#   ./scripts/run_sft.sh --dry-run
#
# NOTE: This script is intentionally conservative. It will not overwrite
# checkpoints unless you explicitly pass --force.
# ----------------------------------------------------------------------------

set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR%/scripts}"
LOG_DIR="$ROOT_DIR/logs"
ENV_FILE="$ROOT_DIR/environment.yml"
PYTHON_CMD="python"
CONDA_ENV_NAME="rlhf-lab"

# Default runtime values (can be overridden by args)
CONFIG_PATH="configs/sft_gpt2.yaml"
TRAIN_FILE="data/sft.jsonl"
OUTPUT_DIR="checkpoints/sft"
NUM_EPOCHS="1"
BATCH_SIZE="4"
GRAD_ACCUM="1"
LEARNING_RATE="5e-5"
FP16="true"
RESUME_PATH=""
FORCE_OVERWRITE="false"
DRY_RUN="false"
USE_DEEPSPEED="false"
DEEPSPEED_CONFIG=""
NUM_NODES=1
NUM_GPUS=0
TORCHRUN=""
EXTRA_PY_ARGS=""

# Helper: print usage
print_usage() {
  cat <<'USAGE'
Usage: run_sft.sh [options]

Options:
  --config PATH          Path to YAML config file (default: configs/sft_gpt2.yaml)
  --train_file PATH      Path to training JSONL file (default: data/sft.jsonl)
  --output_dir PATH      Output/checkpoint directory (default: checkpoints/sft)
  --epochs N             Number of training epochs
  --batch_size N         Per-device train batch size
  --grad_accum N         Gradient accumulation steps
  --lr FLOAT             Learning rate
  --fp16 true|false      Use mixed precision (default: true)
  --resume PATH          Resume from checkpoint dir
  --force                Overwrite output_dir if exists (dangerous)
  --dry-run              Validate environment and print the command but don't run
  --deepspeed PATH       Use deepspeed and point to deepspeed config JSON
  --num_gpus N           Force number of GPUs for torchrun (overrides auto detection)
  --extra_args '...'     Extra args passed to python trainer
  -h, --help             Show this help message

Example:
  ./scripts/run_sft.sh --config configs/sft_gpt2.yaml --epochs 3 --batch_size 8 --deepspeed ds_config.json
USAGE
}

# Parse args
while [[ ${#} -gt 0 ]]; do
  case "$1" in
    --config) CONFIG_PATH="$2"; shift 2;;
    --train_file) TRAIN_FILE="$2"; shift 2;;
    --output_dir) OUTPUT_DIR="$2"; shift 2;;
    --epochs) NUM_EPOCHS="$2"; shift 2;;
    --batch_size) BATCH_SIZE="$2"; shift 2;;
    --grad_accum) GRAD_ACCUM="$2"; shift 2;;
    --lr) LEARNING_RATE="$2"; shift 2;;
    --fp16) FP16="$2"; shift 2;;
    --resume) RESUME_PATH="$2"; shift 2;;
    --force) FORCE_OVERWRITE="true"; shift;;
    --dry-run) DRY_RUN="true"; shift;;
    --deepspeed) USE_DEEPSPEED="true"; DEEPSPEED_CONFIG="$2"; shift 2;;
    --num_gpus) NUM_GPUS="$2"; shift 2;;
    --extra_args) EXTRA_PY_ARGS="$2"; shift 2;;
    -h|--help) print_usage; exit 0;;
    *) echo "Unknown arg: $1"; print_usage; exit 1;;
  esac
done

mkdir -p "$LOG_DIR"

now_ts() { date +"%Y%m%d_%H%M%S"; }

# Determine python / conda environment
find_python() {
  # prefer conda env activation if environment.yml is present
  if command -v conda >/dev/null 2>&1 && [[ -f "$ENV_FILE" ]]; then
    # try to activate the named env if exists, else create it (only on user's explicit request)
    if conda env list | grep -q "^$CONDA_ENV_NAME[[:space:]]"; then
      echo "CONDA"
    else
      # do not auto-create env; just fallback to system python
      echo "SYSTEM"
    fi
  else
    echo "SYSTEM"
  fi
}

PY_ENV_TYPE=$(find_python)
if [[ "$PY_ENV_TYPE" == "CONDA" ]]; then
  # try to activate
  if [[ -n "${CONDA_EXE:-}" ]]; then
    # shell may not have conda initialized; try to source conda.sh
    CONDA_BASE=$(conda info --base 2>/dev/null || echo "")
    if [[ -n "$CONDA_BASE" && -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
      # shellcheck disable=SC1090
      source "$CONDA_BASE/etc/profile.d/conda.sh"
      conda activate "$CONDA_ENV_NAME" || echo "Warning: failed to activate conda env $CONDA_ENV_NAME"
      PYTHON_CMD="python"
    fi
  fi
fi

# Detect GPUs
detect_gpus() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    NV_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l || true)
    echo "$NV_COUNT"
  else
    echo "0"
  fi
}

AUTO_GPU_COUNT=$(detect_gpus)
if [[ -n "$NUM_GPUS" && "$NUM_GPUS" -gt 0 ]]; then
  GPU_COUNT="$NUM_GPUS"
else
  GPU_COUNT="$AUTO_GPU_COUNT"
fi

logger() { echo "[$(date +"%Y-%m-%d %H:%M:%S")] $*"; }

# Validate important files
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "ERROR: Config file not found: $CONFIG_PATH" >&2
  exit 2
fi
if [[ ! -f "$TRAIN_FILE" ]]; then
  echo "WARNING: Training file not found: $TRAIN_FILE" >&2
  # not fatal; user might use other data sources
fi

# Check output dir
if [[ -d "$OUTPUT_DIR" && "$FORCE_OVERWRITE" != "true" && -z "$RESUME_PATH" ]]; then
  echo "Output dir $OUTPUT_DIR already exists. Use --resume to continue or --force to overwrite." >&2
  exit 3
fi

# Build python command
PY_ARGS=("$ROOT_DIR/src/training/sft_trainer.py" --config "$CONFIG_PATH")
# Add overrides
PY_ARGS+=(--train_file "$TRAIN_FILE")
PY_ARGS+=(--output_dir "$OUTPUT_DIR")
PY_ARGS+=(--local_files_only)
# pass some hyperparams via env vars or extra args
PY_ARGS+=(--extra_args "--num_train_epochs ${NUM_EPOCHS} --per_device_train_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${GRAD_ACCUM} --learning_rate ${LEARNING_RATE} --fp16 ${FP16}")

# Compose the final command
if [[ "$USE_DEEPSPEED" == "true" ]]; then
  if ! command -v deepspeed >/dev/null 2>&1; then
    echo "ERROR: deepspeed requested but not installed in PATH." >&2
    exit 4
  fi
  if [[ -z "$DEEPSPEED_CONFIG" ]]; then
    echo "ERROR: --deepspeed requires a deepspeed config JSON path." >&2
    exit 5
  fi
  CMD=(deepspeed --num_gpus ${GPU_COUNT} "${PY_ARGS[@]}" --deepspeed_config "$DEEPSPEED_CONFIG")
elif [[ "$GPU_COUNT" -gt 1 ]]; then
  # use torchrun for multi-gpu single-node
  if command -v torchrun >/dev/null 2>&1; then
    CMD=(torchrun --nproc_per_node=${GPU_COUNT} "${PY_ARGS[@]}")
  elif command -v python -m torch.distributed.run >/dev/null 2>&1; then
    CMD=(python -m torch.distributed.run --nproc_per_node=${GPU_COUNT} "${PY_ARGS[@]}")
  else
    echo "Multi-GPU requested (${GPU_COUNT}) but torchrun not available. Falling back to single-GPU." >&2
    CMD=($PYTHON_CMD "${PY_ARGS[@]}")
  fi
else
  CMD=($PYTHON_CMD "${PY_ARGS[@]}")
fi

# Flatten CMD into a string for logging
CMD_STR="${CMD[*]} $EXTRA_PY_ARGS"

# Dry run
if [[ "$DRY_RUN" == "true" ]]; then
  logger "DRY RUN: would execute -> $CMD_STR"
  exit 0
fi

# Ensure logs directory exists
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/sft_$(now_ts).log"

logger "Starting SFT run"
logger "Config: $CONFIG_PATH"
logger "Train file: $TRAIN_FILE"
logger "Output dir: $OUTPUT_DIR"
logger "GPUs detected: $AUTO_GPU_COUNT | GPUs used: $GPU_COUNT"
logger "Command: $CMD_STR"
logger "Logs: $LOG_FILE"

# Run the command and tee logs
# Use a subshell so traps and exits don't break the caller shell
(
  set -o pipefail
  "${CMD[@]}" ${EXTRA_PY_ARGS} 2>&1 | tee "$LOG_FILE"
)

EXIT_CODE=${PIPESTATUS[0]:-0}
if [[ $EXIT_CODE -ne 0 ]]; then
  logger "Training finished with EXIT CODE $EXIT_CODE. Check log: $LOG_FILE"
  exit $EXIT_CODE
fi

logger "Training finished successfully. Logs at: $LOG_FILE"

# Post-run: save a copy of config used and a short run-summary
SUMMARY_FILE="$OUTPUT_DIR/run_summary_$(now_ts).json"
cat > "$SUMMARY_FILE" <<EOF
{
  "cmd": "$CMD_STR",
  "config": "${CONFIG_PATH}",
  "train_file": "${TRAIN_FILE}",
  "output_dir": "${OUTPUT_DIR}",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "log_file": "${LOG_FILE}"
}
EOF
logger "Wrote run summary to $SUMMARY_FILE"

exit 0

