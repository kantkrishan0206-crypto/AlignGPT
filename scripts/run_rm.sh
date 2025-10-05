
set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR%/scripts}"
LOG_DIR="$ROOT_DIR/logs"
ENV_FILE="$ROOT_DIR/environment.yml"
PYTHON_CMD="python"
CONDA_ENV_NAME="rlhf-lab"

# Defaults
CONFIG_PATH="configs/rm_gpt2.yaml"
TRAIN_FILE="data/pref_pairs.jsonl"
VAL_FILE=""
OUTPUT_DIR="checkpoints/rm"
NUM_EPOCHS="3"
BATCH_SIZE="16"
LR="2e-5"
FP16="true"
RESUME_PATH=""
FORCE_OVERWRITE="false"
DRY_RUN="false"
NUM_GPUS=0
LOCAL_ONLY="false"
EXTRA_PY_ARGS=""
DOWNLOAD_SAMPLE="false"
SAMPLE_URL="https://raw.githubusercontent.com/your-repo/rlhf-lab/main/data/sample_pref_pairs.jsonl"

print_usage() {
  cat <<'USAGE'
Usage: run_rm.sh [options]

Options:
  --config PATH          Path to YAML config file (default: configs/rm_gpt2.yaml)
  --train_file PATH      Path to training JSONL file (default: data/pref_pairs.jsonl)
  --val_file PATH        Path to validation JSONL file (optional)
  --output_dir PATH      Output/checkpoint directory (default: checkpoints/rm)
  --epochs N             Hint for overriding epochs (passed to trainer via extra args)
  --batch_size N         Train batch size override
  --lr FLOAT             Learning rate override
  --fp16 true|false      Use mixed precision
  --resume PATH          Resume from checkpoint dir
  --force                Overwrite output_dir if exists (dangerous)
  --dry-run              Validate environment and print the command but don't run
  --num_gpus N           Force number of GPUs for torchrun (overrides auto detection)
  --local-only           Load HF files from local files only (no hub downloads)
  --download-sample      Download a tiny sample pref_pairs.jsonl if train file missing
  --extra_args '...'     Extra args passed to python trainer
  -h, --help             Show this help message

Example:
  ./scripts/run_rm.sh --config configs/rm_gpt2.yaml --train_file data/pref_pairs.jsonl --output_dir checkpoints/rm_test
USAGE
}

# Parse args
while [[ ${#} -gt 0 ]]; do
  case "$1" in
    --config) CONFIG_PATH="$2"; shift 2;;
    --train_file) TRAIN_FILE="$2"; shift 2;;
    --val_file) VAL_FILE="$2"; shift 2;;
    --output_dir) OUTPUT_DIR="$2"; shift 2;;
    --epochs) NUM_EPOCHS="$2"; shift 2;;
    --batch_size) BATCH_SIZE="$2"; shift 2;;
    --lr) LR="$2"; shift 2;;
    --fp16) FP16="$2"; shift 2;;
    --resume) RESUME_PATH="$2"; shift 2;;
    --force) FORCE_OVERWRITE="true"; shift;;
    --dry-run) DRY_RUN="true"; shift;;
    --num_gpus) NUM_GPUS="$2"; shift 2;;
    --local-only) LOCAL_ONLY="true"; shift;;
    --download-sample) DOWNLOAD_SAMPLE="true"; shift;;
    --extra_args) EXTRA_PY_ARGS="$2"; shift 2;;
    -h|--help) print_usage; exit 0;;
    *) echo "Unknown arg: $1"; print_usage; exit 1;;
  esac
done

mkdir -p "$LOG_DIR"

now_ts() { date +"%Y%m%d_%H%M%S"; }

# Determine python / conda environment
find_python() {
  if command -v conda >/dev/null 2>&1 && [[ -f "$ENV_FILE" ]]; then
    if conda env list | grep -q "^$CONDA_ENV_NAME[[:space:]]"; then
      echo "CONDA"
    else
      echo "SYSTEM"
    fi
  else
    echo "SYSTEM"
  fi
}

PY_ENV_TYPE=$(find_python)
if [[ "$PY_ENV_TYPE" == "CONDA" ]]; then
  if [[ -n "${CONDA_EXE:-}" ]]; then
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

# Validate config
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "ERROR: Config file not found: $CONFIG_PATH" >&2
  exit 2
fi

# If train file absent and user asked to download sample, do so
if [[ ! -f "$TRAIN_FILE" ]]; then
  if [[ "$DOWNLOAD_SAMPLE" == "true" ]]; then
    mkdir -p "$(dirname "$TRAIN_FILE")"
    logger "Downloading sample preference pairs to $TRAIN_FILE"
    if command -v curl >/dev/null 2>&1; then
      curl -fsSL "$SAMPLE_URL" -o "$TRAIN_FILE" || true
    elif command -v wget >/dev/null 2>&1; then
      wget -qO "$TRAIN_FILE" "$SAMPLE_URL" || true
    else
      logger "No curl/wget available to download sample file; continuing (trainer has fallback)"
    fi
  else
    logger "WARNING: Training file not found: $TRAIN_FILE -- trainer may use synthetic fallback"
  fi
fi

# Check output dir
if [[ -d "$OUTPUT_DIR" && "$FORCE_OVERWRITE" != "true" && -z "$RESUME_PATH" ]]; then
  echo "Output dir $OUTPUT_DIR already exists. Use --resume to continue or --force to overwrite." >&2
  exit 3
fi

# Build python invocation
PY_ARGS=("$ROOT_DIR/src/training/rm_trainer.py" --config "$CONFIG_PATH" --train_file "$TRAIN_FILE")
if [[ -n "$VAL_FILE" ]]; then
  PY_ARGS+=(--val_file "$VAL_FILE")
fi
if [[ -n "$RESUME_PATH" ]]; then
  PY_ARGS+=(--resume "$RESUME_PATH")
fi
if [[ "$LOCAL_ONLY" == "true" ]]; then
  PY_ARGS+=(--local_files_only)
fi
# pass overrides via environment variable or extra_args
if [[ -n "$NUM_EPOCHS" ]]; then
  EXTRA_PY_ARGS="--extra_args \"--max_epochs ${NUM_EPOCHS} --train_batch_size ${BATCH_SIZE} --lr ${LR}\" $EXTRA_PY_ARGS"
fi

# Compose launcher for multi-GPU or single GPU
if [[ "$GPU_COUNT" -gt 1 ]]; then
  if command -v torchrun >/dev/null 2>&1; then
    CMD=(torchrun --nproc_per_node=${GPU_COUNT} "${PY_ARGS[@]}")
  else
    logger "torchrun not available; falling back to single-process run"
    CMD=($PYTHON_CMD "${PY_ARGS[@]}")
  fi
else
  CMD=($PYTHON_CMD "${PY_ARGS[@]}")
fi

CMD_STR="${CMD[*]} $EXTRA_PY_ARGS"

if [[ "$DRY_RUN" == "true" ]]; then
  logger "DRY RUN: would execute -> $CMD_STR"
  exit 0
fi

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/rm_$(now_ts).log"

logger "Starting RM training run"
logger "Config: $CONFIG_PATH"
logger "Train file: $TRAIN_FILE"
logger "Output dir: $OUTPUT_DIR"
logger "GPUs detected: $AUTO_GPU_COUNT | GPUs used: $GPU_COUNT"
logger "Command: $CMD_STR"
logger "Logs: $LOG_FILE"

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

