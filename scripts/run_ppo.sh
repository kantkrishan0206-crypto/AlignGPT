

set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR%/scripts}"
LOG_DIR="$ROOT_DIR/logs"
ENV_FILE="$ROOT_DIR/environment.yml"
PYTHON_CMD="python"
CONDA_ENV_NAME="rlhf-lab"

# Defaults
CONFIG_PATH="configs/ppo_gpt2.yaml"
PROMPTS_FILE="data/prompts.jsonl"
REWARD_MODEL_PATH="checkpoints/rm/final"
POLICY_MODEL_PATH="gpt2"
OUTPUT_DIR="checkpoints/ppo"
TOTAL_STEPS=1000
ROLLOUT_SIZE=64
BATCH_SIZE=8
SAVE_EVERY=500
FP16="true"
RESUME_PATH=""
FORCE_OVERWRITE="false"
DRY_RUN="false"
LOCAL_ONLY="false"
USE_DEEPSPEED="false"
DEEPSPEED_CONFIG=""
NUM_GPUS=0
EXTRA_PY_ARGS=""
DOWNLOAD_SAMPLE="false"
SAMPLE_PROMPTS_URL="https://raw.githubusercontent.com/your-repo/rlhf-lab/main/data/sample_prompts.jsonl"

print_usage() {
  cat <<'USAGE'
Usage: run_ppo.sh [options]

Options:
  --config PATH            Path to YAML config file (default: configs/ppo_gpt2.yaml)
  --prompts PATH           Prompts JSONL (default: data/prompts.jsonl)
  --reward PATH            Path to trained reward model (required unless in smoke-mode)
  --policy MODEL_OR_PATH   Policy model name or path (default: gpt2)
  --output_dir PATH        Output dir for PPO checkpoints (default: checkpoints/ppo)
  --total_steps N          Total training steps override
  --rollout_size N         Number of rollouts collected per update
  --batch_size N           Generation batch size (prompts per generation)
  --save_every N           Checkpoint save frequency (steps)
  --fp16 true|false        Use mixed precision (default: true)
  --resume PATH            Resume from checkpoint
  --force                  Overwrite output dir if exists
  --dry-run                Validate and print commands but don't run
  --local-only             Load models/tokenizers from local files only
  --deepspeed PATH         Use deepspeed with config JSON
  --num_gpus N             Force number of GPUs (overrides detection)
  --download-sample        Download a tiny prompts file if prompts missing
  --extra_args '...'       Extra args forwarded to python trainer
  -h, --help               Show this help message

Example:
  ./scripts/run_ppo.sh --config configs/ppo_gpt2.yaml --prompts data/prompts.jsonl \
      --reward checkpoints/rm/final --output_dir checkpoints/ppo_expt1
USAGE
}

# Parse arguments
while [[ ${#} -gt 0 ]]; do
  case "$1" in
    --config) CONFIG_PATH="$2"; shift 2;;
    --prompts) PROMPTS_FILE="$2"; shift 2;;
    --reward) REWARD_MODEL_PATH="$2"; shift 2;;
    --policy) POLICY_MODEL_PATH="$2"; shift 2;;
    --output_dir) OUTPUT_DIR="$2"; shift 2;;
    --total_steps) TOTAL_STEPS="$2"; shift 2;;
    --rollout_size) ROLLOUT_SIZE="$2"; shift 2;;
    --batch_size) BATCH_SIZE="$2"; shift 2;;
    --save_every) SAVE_EVERY="$2"; shift 2;;
    --fp16) FP16="$2"; shift 2;;
    --resume) RESUME_PATH="$2"; shift 2;;
    --force) FORCE_OVERWRITE="true"; shift;;
    --dry-run) DRY_RUN="true"; shift;;
    --local-only) LOCAL_ONLY="true"; shift;;
    --deepspeed) USE_DEEPSPEED="true"; DEEPSPEED_CONFIG="$2"; shift 2;;
    --num_gpus) NUM_GPUS="$2"; shift 2;;
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

# GPU detection
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

# Validate files and optional sample download
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "ERROR: Config file not found: $CONFIG_PATH" >&2
  exit 2
fi

if [[ ! -f "$PROMPTS_FILE" ]]; then
  if [[ "$DOWNLOAD_SAMPLE" == "true" ]]; then
    mkdir -p "$(dirname "$PROMPTS_FILE")"
    logger "Downloading sample prompts to $PROMPTS_FILE"
    if command -v curl >/dev/null 2>&1; then
      curl -fsSL "$SAMPLE_PROMPTS_URL" -o "$PROMPTS_FILE" || true
    elif command -v wget >/dev/null 2>&1; then
      wget -qO "$PROMPTS_FILE" "$SAMPLE_PROMPTS_URL" || true
    else
      logger "No curl/wget available to download sample prompts; trainer may still run with builtin defaults"
    fi
  else
    logger "WARNING: Prompts file not found: $PROMPTS_FILE. Trainer can still run with default prompts if implemented."
  fi
fi

if [[ -z "$REWARD_MODEL_PATH" ]]; then
  logger "ERROR: reward model path must be provided via --reward or in config" >&2
  exit 3
fi

# Check output dir
if [[ -d "$OUTPUT_DIR" && "$FORCE_OVERWRITE" != "true" && -z "$RESUME_PATH" ]]; then
  echo "Output dir $OUTPUT_DIR already exists. Use --resume to continue or --force to overwrite." >&2
  exit 4
fi

# Build python invocation
PY_ARGS=("$ROOT_DIR/src/training/ppo_trainer.py" --config "$CONFIG_PATH")
PY_ARGS+=(--prompts "$PROMPTS_FILE")
PY_ARGS+=(--out_dir "$OUTPUT_DIR")
PY_ARGS+=(--local_files_only)
PY_ARGS+=(--extra_args "--total_steps ${TOTAL_STEPS} --rollout_size ${ROLLOUT_SIZE} --batch_size ${BATCH_SIZE} --save_every ${SAVE_EVERY} --fp16 ${FP16} --reward_model_path ${REWARD_MODEL_PATH} --model_name_or_path ${POLICY_MODEL_PATH}")
if [[ -n "$RESUME_PATH" ]]; then
  PY_ARGS+=(--resume "$RESUME_PATH")
fi

# Compose launcher command
if [[ "$USE_DEEPSPEED" == "true" ]]; then
  if ! command -v deepspeed >/dev/null 2>&1; then
    echo "ERROR: deepspeed requested but not installed in PATH." >&2
    exit 5
  fi
  if [[ -z "$DEEPSPEED_CONFIG" ]]; then
    echo "ERROR: --deepspeed requires a deepspeed config JSON path." >&2
    exit 6
  fi
  CMD=(deepspeed --num_gpus ${GPU_COUNT} "${PY_ARGS[@]}" --deepspeed_config "$DEEPSPEED_CONFIG")
elif [[ "$GPU_COUNT" -gt 1 ]]; then
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
LOG_FILE="$LOG_DIR/ppo_$(now_ts).log"

logger "Starting PPO run"
logger "Config: $CONFIG_PATH"
logger "Prompts: $PROMPTS_FILE"
logger "Reward model: $REWARD_MODEL_PATH"
logger "Policy model: $POLICY_MODEL_PATH"
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
  logger "PPO finished with EXIT CODE $EXIT_CODE. Check log: $LOG_FILE"
  exit $EXIT_CODE
fi

logger "PPO finished successfully. Logs at: $LOG_FILE"

SUMMARY_FILE="$OUTPUT_DIR/run_summary_$(now_ts).json"
cat > "$SUMMARY_FILE" <<EOF
{
  "cmd": "$CMD_STR",
  "config": "${CONFIG_PATH}",
  "prompts": "${PROMPTS_FILE}",
  "reward_model": "${REWARD_MODEL_PATH}",
  "policy_model": "${POLICY_MODEL_PATH}",
  "output_dir": "${OUTPUT_DIR}",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "log_file": "${LOG_FILE}"
}
EOF
logger "Wrote run summary to $SUMMARY_FILE"

exit 0

