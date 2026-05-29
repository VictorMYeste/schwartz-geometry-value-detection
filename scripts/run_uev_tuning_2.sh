#!/bin/bash -l
#SBATCH --job-name=sgvd-tune-dry
#SBATCH --partition=grupo_pro
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/home/hpc/34045staff/output/schwartz-geometry-value-detection/%x-%j.out
#SBATCH --error=/home/hpc/34045staff/output/schwartz-geometry-value-detection/%x-%j.err
#SBATCH --hint=nomultithread
#SBATCH --mail-type=ALL
#SBATCH --mail-user=victor.yeste@universidadeuropea.es

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/hpc/34045staff/schwartz-geometry-value-detection}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
PYTHON_BIN="$VENV_DIR/bin/python"
GRID_SCRIPT="$PROJECT_DIR/scripts/grid_loss_hparams.py"
UEV_OUTPUT_DIR="${UEV_OUTPUT_DIR:-/home/hpc/34045staff/output/schwartz-geometry-value-detection}"

# Low-resource dry run for the same ASL tuning command as run_uev_tuning.sh.
METHODS="${METHODS:-asl}"
TUNING_SEEDS="${TUNING_SEEDS:-42 7 1701}"
TUNING_RESULTS_DIR="${TUNING_RESULTS_DIR:-$PROJECT_DIR/results/tuning}"
TUNING_OUTPUT="${TUNING_OUTPUT:-$TUNING_RESULTS_DIR/grid_loss_hparams.csv}"
RETRY_COLLAPSED="${RETRY_COLLAPSED:-1}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
LIMIT="${LIMIT:-}"
DRY_RUN="${DRY_RUN:-1}"
SAVE_CHECKPOINTS="${SAVE_CHECKPOINTS:-0}"
SAVE_HF_MODEL="${SAVE_HF_MODEL:-0}"
DEBUG="${DEBUG:-0}"

mkdir -p "$UEV_OUTPUT_DIR"
cd "$PROJECT_DIR"

HF_TOKEN_FILE="${HF_TOKEN_FILE:-$HOME/.config/huggingface/token}"
if [ -z "${HF_TOKEN:-}" ] && [ -r "$HF_TOKEN_FILE" ]; then
  HF_TOKEN="$(cat "$HF_TOKEN_FILE")"
fi
export HF_TOKEN

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Missing Python in venv: $PYTHON_BIN" >&2
  echo "Create the UEV venv from a login session, then retry:" >&2
  echo "  cd $PROJECT_DIR" >&2
  echo "  python3.11 -m venv .venv" >&2
  echo "  source .venv/bin/activate" >&2
  echo "  python -m pip install -r requirements.txt" >&2
  echo "  python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124" >&2
  echo "  python -m pip install -e ." >&2
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
import re

import torch
import transformers
import safetensors

print("torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("safetensors", safetensors.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
else:
    raise SystemExit("CUDA is not available in this Slurm job.")

match = re.match(r"^(\d+)\.(\d+)", torch.__version__)
torch_major_minor = tuple(int(part) for part in match.groups()) if match else (0, 0)
if torch_major_minor < (2, 6):
    print(
        "WARNING: torch<2.6. This is acceptable only if Transformers loads "
        "model.safetensors. If loading falls back to pytorch_model.bin, upgrade "
        "with: python -m pip install --upgrade --force-reinstall torch==2.6.0 "
        "--index-url https://download.pytorch.org/whl/cu124"
    )
PY

mkdir -p "$TUNING_RESULTS_DIR"

METHODS="${METHODS//,/ }"
TUNING_SEEDS="${TUNING_SEEDS//,/ }"
read -r -a METHOD_ARGS <<< "$METHODS"
read -r -a SEED_ARGS <<< "$TUNING_SEEDS"

cmd=(
  "$PYTHON_BIN" "$GRID_SCRIPT"
  --methods "${METHOD_ARGS[@]}"
  --seeds "${SEED_ARGS[@]}"
  --output "$TUNING_OUTPUT"
  --results_dir "$TUNING_RESULTS_DIR"
  --retry_collapsed "$RETRY_COLLAPSED"
)

if [ -n "$MAX_SAMPLES" ]; then
  cmd+=(--max_samples "$MAX_SAMPLES")
fi

if [ -n "$LIMIT" ]; then
  cmd+=(--limit "$LIMIT")
fi

if [ "$DRY_RUN" = "1" ]; then
  cmd+=(--dry_run)
fi

if [ "$SAVE_CHECKPOINTS" = "1" ]; then
  cmd+=(--save_checkpoints)
fi

if [ "$SAVE_HF_MODEL" = "1" ]; then
  cmd+=(--save_hf_model)
fi

if [ "$DEBUG" = "1" ]; then
  cmd+=(--debug)
fi

echo "======================================================================"
echo "UEV tuning dry run"
echo "PROJECT_DIR=$PROJECT_DIR"
echo "METHODS=$METHODS"
echo "TUNING_SEEDS=$TUNING_SEEDS"
echo "TUNING_RESULTS_DIR=$TUNING_RESULTS_DIR"
echo "TUNING_OUTPUT=$TUNING_OUTPUT"
echo "MAX_SAMPLES=${MAX_SAMPLES:-<none>}"
echo "LIMIT=${LIMIT:-<none>}"
echo "DRY_RUN=$DRY_RUN"
echo "SAVE_CHECKPOINTS=$SAVE_CHECKPOINTS"
echo "SAVE_HF_MODEL=$SAVE_HF_MODEL"
echo "Command: ${cmd[*]}"
echo "======================================================================"

"${cmd[@]}"

echo "Dry-run tuning job complete. CSV: $TUNING_OUTPUT"
