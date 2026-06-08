#!/bin/bash -l
#SBATCH --job-name=sgvd-final
#SBATCH --partition=gpu
#SBATCH --array=0-29%6
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/lustre/scratch/%u/schwartz-geometry-value-detection/logs/%x-%A_%a.out
#SBATCH --error=/lustre/scratch/%u/schwartz-geometry-value-detection/logs/%x-%A_%a.err
#SBATCH --hint=nomultithread
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=vicyesmo@upv.es

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/schwartz-geometry-value-detection}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
PYTHON_BIN="$VENV_DIR/bin/python"
TRAIN_SCRIPT="$PROJECT_DIR/scripts/train_deberta.py"
EVAL_SCRIPT="$PROJECT_DIR/scripts/eval_deberta.py"

FINAL_CONFIGS="${FINAL_CONFIGS:-configs/deberta_bce.yaml configs/deberta_asl.yaml configs/deberta_empirical_structure.yaml configs/deberta_random_geoloss.yaml configs/deberta_schwartz_geoloss.yaml configs/deberta_schwartz_geosmooth.yaml}"
FINAL_SEEDS="${FINAL_SEEDS:-42 7 1701 11 1984}"
FINAL_RESULTS_DIR="${FINAL_RESULTS_DIR:-$PROJECT_DIR/results}"
FINAL_CONFIG_DIR="${FINAL_CONFIG_DIR:-$FINAL_RESULTS_DIR/final_configs}"
THRESHOLD_MODE="${THRESHOLD_MODE:-per_label}"
THRESHOLD_STEP="${THRESHOLD_STEP:-0.01}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
RETRY_COLLAPSED="${RETRY_COLLAPSED:-0}"
SAVE_CHECKPOINTS="${SAVE_CHECKPOINTS:-1}"
SAVE_HF_MODEL="${SAVE_HF_MODEL:-0}"
CHECKPOINT_EVERY_EPOCHS="${CHECKPOINT_EVERY_EPOCHS:-0}"
FORCE_FINAL="${FORCE_FINAL:-0}"
DRY_RUN="${DRY_RUN:-0}"
DEBUG="${DEBUG:-0}"

if [ "$DRY_RUN" != "1" ] || [ -n "${SLURM_JOB_ID:-}" ]; then
  mkdir -p "/lustre/scratch/$USER/schwartz-geometry-value-detection/logs"
fi
cd "$PROJECT_DIR"

FINAL_CONFIGS="${FINAL_CONFIGS//,/ }"
FINAL_SEEDS="${FINAL_SEEDS//,/ }"
read -r -a CONFIG_ARGS <<< "$FINAL_CONFIGS"
read -r -a SEED_ARGS <<< "$FINAL_SEEDS"

N_CONFIGS="${#CONFIG_ARGS[@]}"
N_SEEDS="${#SEED_ARGS[@]}"
TOTAL_TASKS=$((N_CONFIGS * N_SEEDS))

print_plan() {
  echo "======================================================================"
  echo "Sirius final supervised plan"
  echo "PROJECT_DIR=$PROJECT_DIR"
  echo "FINAL_RESULTS_DIR=$FINAL_RESULTS_DIR"
  echo "FINAL_CONFIGS=$FINAL_CONFIGS"
  echo "FINAL_SEEDS=$FINAL_SEEDS"
  echo "TOTAL_TASKS=$TOTAL_TASKS"
  echo "Array recommendation: --array=0-$((TOTAL_TASKS - 1))%6"
  echo "======================================================================"
  local task_id config_index seed_index
  for task_id in $(seq 0 $((TOTAL_TASKS - 1))); do
    config_index=$((task_id / N_SEEDS))
    seed_index=$((task_id % N_SEEDS))
    printf "%02d  config=%s  seed=%s\n" \
      "$task_id" "${CONFIG_ARGS[$config_index]}" "${SEED_ARGS[$seed_index]}"
  done
}

if [ "$DRY_RUN" = "1" ] && [ -z "${SLURM_ARRAY_TASK_ID:-}" ] && [ -z "${TASK_ID:-}" ]; then
  print_plan
  exit 0
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:-${TASK_ID:-}}"
if [ -z "$TASK_ID" ]; then
  echo "No Slurm array task id found. Submit with sbatch, or set TASK_ID for a single local task." >&2
  echo "For a local plan only: DRY_RUN=1 bash scripts/run_sirius_final_supervised.sh" >&2
  exit 1
fi

if [ "$TASK_ID" -lt 0 ] || [ "$TASK_ID" -ge "$TOTAL_TASKS" ]; then
  echo "Task id $TASK_ID is outside final-run range 0-$((TOTAL_TASKS - 1)); nothing to do."
  exit 0
fi

CONFIG_INDEX=$((TASK_ID / N_SEEDS))
SEED_INDEX=$((TASK_ID % N_SEEDS))
BASE_CONFIG="${CONFIG_ARGS[$CONFIG_INDEX]}"
SEED="${SEED_ARGS[$SEED_INDEX]}"
BASE_STEM="$(basename "$BASE_CONFIG" .yaml)"
TASK_CONFIG="$FINAL_CONFIG_DIR/${BASE_STEM}_seed${SEED}_final.yaml"

echo "======================================================================"
echo "Sirius final supervised task"
echo "TASK_ID=$TASK_ID/$((TOTAL_TASKS - 1))"
echo "BASE_CONFIG=$BASE_CONFIG"
echo "SEED=$SEED"
echo "FINAL_RESULTS_DIR=$FINAL_RESULTS_DIR"
echo "TASK_CONFIG=$TASK_CONFIG"
echo "THRESHOLD_MODE=$THRESHOLD_MODE"
echo "THRESHOLD_STEP=$THRESHOLD_STEP"
echo "EVAL_SPLIT=$EVAL_SPLIT"
echo "MAX_SAMPLES=${MAX_SAMPLES:-<none>}"
echo "RETRY_COLLAPSED=$RETRY_COLLAPSED"
echo "SAVE_CHECKPOINTS=$SAVE_CHECKPOINTS"
echo "SAVE_HF_MODEL=$SAVE_HF_MODEL"
echo "CHECKPOINT_EVERY_EPOCHS=$CHECKPOINT_EVERY_EPOCHS"
echo "FORCE_FINAL=$FORCE_FINAL"
echo "DRY_RUN=$DRY_RUN"
echo "======================================================================"

if [ "$DRY_RUN" = "1" ]; then
  exit 0
fi

HF_TOKEN_FILE="${HF_TOKEN_FILE:-$HOME/.config/huggingface/token}"
if [ -z "${HF_TOKEN:-}" ] && [ -r "$HF_TOKEN_FILE" ]; then
  HF_TOKEN="$(cat "$HF_TOKEN_FILE")"
fi
export HF_TOKEN

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Missing venv Python: $PYTHON_BIN" >&2
  echo "Either create/activate .venv from a login session, or run:" >&2
  echo "  sbatch scripts/bootstrap_slurm_venv.sh" >&2
  echo "If auto-detection fails, pass a real Python path with:" >&2
  echo "  sbatch --export=ALL,BASE_PYTHON=/real/path/to/python3.11 scripts/bootstrap_slurm_venv.sh" >&2
  exit 1
fi

source "$VENV_DIR/bin/activate"

PY_VER="$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
NVIDIA_LIB_BASE="$VENV_DIR/lib/python${PY_VER}/site-packages/nvidia"
if [ -d "$NVIDIA_LIB_BASE" ]; then
  for d in "$NVIDIA_LIB_BASE"/*/lib; do
    [ -d "$d" ] || continue
    export LD_LIBRARY_PATH="$d${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  done
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

"$PYTHON_BIN" -V
"$PYTHON_BIN" - <<'PY'
import torch
import transformers

try:
    import safetensors
except Exception:
    safetensors = None

print("torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__)
if safetensors is not None:
    print("safetensors", safetensors.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
else:
    raise SystemExit("CUDA is not available in this Slurm job.")
PY

mkdir -p "$FINAL_RESULTS_DIR" "$FINAL_CONFIG_DIR"

export BASE_CONFIG TASK_CONFIG FINAL_RESULTS_DIR SAVE_CHECKPOINTS SAVE_HF_MODEL
export CHECKPOINT_EVERY_EPOCHS
"$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path

import yaml

base = Path(os.environ["BASE_CONFIG"])
out = Path(os.environ["TASK_CONFIG"])
raw = yaml.safe_load(base.read_text(encoding="utf-8")) or {}
if not isinstance(raw, dict):
    raise SystemExit(f"Config root must be a mapping: {base}")

raw["results_dir"] = os.environ["FINAL_RESULTS_DIR"]
raw["save_checkpoints"] = os.environ["SAVE_CHECKPOINTS"] == "1"
training = dict(raw.get("training", {}))
training["save_hf_model"] = os.environ["SAVE_HF_MODEL"] == "1"
training["checkpoint_every_epochs"] = int(os.environ["CHECKPOINT_EVERY_EPOCHS"])
raw["training"] = training

out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
print(out)
PY

artifact_prefix() {
  local cfg="$1"
  local seed="$2"
  "$PYTHON_BIN" - "$cfg" "$seed" <<'PY'
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "src"))
logging.disable(logging.CRITICAL)

from schwartz_value_geometry.utils.config import load_config
from schwartz_value_geometry.utils.naming import artifact_prefix

cfg = load_config(sys.argv[1])
print(artifact_prefix(cfg, seed=int(sys.argv[2])))
PY
}

PREFIX="$(artifact_prefix "$TASK_CONFIG" "$SEED")"
METRICS_PATH="$FINAL_RESULTS_DIR/logs/${PREFIX}_${EVAL_SPLIT}_metrics.json"

if [ -f "$METRICS_PATH" ] && [ "$FORCE_FINAL" != "1" ]; then
  echo "Skipping completed final run: $METRICS_PATH"
  exit 0
fi

train_cmd=(
  "$PYTHON_BIN" "$TRAIN_SCRIPT"
  --config "$TASK_CONFIG"
  --seed "$SEED"
  --retry_collapsed "$RETRY_COLLAPSED"
)

eval_cmd=(
  "$PYTHON_BIN" "$EVAL_SCRIPT"
  --config "$TASK_CONFIG"
  --seed "$SEED"
  --split "$EVAL_SPLIT"
  --use_validation_thresholds
  --threshold_mode "$THRESHOLD_MODE"
  --threshold_step "$THRESHOLD_STEP"
)

if [ -n "$MAX_SAMPLES" ]; then
  train_cmd+=(--max_samples "$MAX_SAMPLES")
  eval_cmd+=(--max_samples "$MAX_SAMPLES")
fi

if [ "$DEBUG" = "1" ]; then
  train_cmd+=(--debug)
  eval_cmd+=(--debug)
fi

echo "Training command: ${train_cmd[*]}"
"${train_cmd[@]}"

echo "Evaluation command: ${eval_cmd[*]}"
"${eval_cmd[@]}"

echo "Final supervised task complete: $METRICS_PATH"
