#!/bin/bash -l
#SBATCH --job-name=sgvd-roberta-lr
#SBATCH --partition=gpu
#SBATCH --array=0-2
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/lustre/scratch/%u/schwartz-geometry-value-detection/logs/%x-%A_%a.out
#SBATCH --error=/lustre/scratch/%u/schwartz-geometry-value-detection/logs/%x-%A_%a.err
#SBATCH --hint=nomultithread
#SBATCH --mail-type=END,FAIL

# Short learning-rate check for the second backbone (revision).
# One array task per learning rate: BCE loss, seed 42, validation Macro-F1 only (no test).
# Each task writes its own CSV; pick the best rate and set it in
# configs/roberta_large_{bce,schwartz_geoloss}.yaml before the final runs.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/schwartz-geometry-value-detection}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
PYTHON_BIN="$VENV_DIR/bin/python"
LR_CONFIG="${LR_CONFIG:-configs/roberta_large_bce.yaml}"
LR_VALUES="${LR_VALUES:-5e-6 1e-5 2e-5}"
LR_WEIGHT_DECAY="${LR_WEIGHT_DECAY:-0.15}"
LR_BATCH_SIZE="${LR_BATCH_SIZE:-8}"
LR_MAX_LENGTH="${LR_MAX_LENGTH:-512}"
LR_OUTPUT_DIR="${LR_OUTPUT_DIR:-$PROJECT_DIR/results/tuning/roberta_large_lr_check}"
DRY_RUN="${DRY_RUN:-0}"

read -r -a LR_ARGS <<< "${LR_VALUES//,/ }"
TASK_ID="${SLURM_ARRAY_TASK_ID:-${TASK_ID:-0}}"
if [ "$TASK_ID" -ge "${#LR_ARGS[@]}" ]; then
  echo "Task $TASK_ID outside 0-$((${#LR_ARGS[@]} - 1)); nothing to do."
  exit 0
fi
LR="${LR_ARGS[$TASK_ID]}"
OUTPUT="$LR_OUTPUT_DIR/lr_check_lr${LR}.csv"

cd "$PROJECT_DIR"
cmd=(
  "$PYTHON_BIN" scripts/grid_deberta_hparams.py
  --config "$LR_CONFIG"
  --output "$OUTPUT"
  --learning_rates "$LR"
  --weight_decays "$LR_WEIGHT_DECAY"
  --batch_sizes "$LR_BATCH_SIZE"
  --max_lengths "$LR_MAX_LENGTH"
  --retry_collapsed 1
)
echo "Task $TASK_ID: ${cmd[*]}"
if [ "$DRY_RUN" = "1" ]; then
  exit 0
fi
mkdir -p "/lustre/scratch/$USER/schwartz-geometry-value-detection/logs" "$LR_OUTPUT_DIR"

HF_TOKEN_FILE="${HF_TOKEN_FILE:-$HOME/.config/huggingface/token}"
if [ -z "${HF_TOKEN:-}" ] && [ -r "$HF_TOKEN_FILE" ]; then
  HF_TOKEN="$(cat "$HF_TOKEN_FILE")"
fi
export HF_TOKEN

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

"$PYTHON_BIN" -c 'import torch; print("torch", torch.__version__, "gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)'
"${cmd[@]}"
echo "LR check task complete: $OUTPUT"
