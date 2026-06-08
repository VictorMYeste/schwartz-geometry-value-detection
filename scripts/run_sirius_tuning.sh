#!/bin/bash -l
#SBATCH --job-name=sgvd-tune
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=96:00:00
#SBATCH --output=/lustre/scratch/%u/schwartz-geometry-value-detection/logs/%x-%j.out
#SBATCH --error=/lustre/scratch/%u/schwartz-geometry-value-detection/logs/%x-%j.err
#SBATCH --hint=nomultithread
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=vicyesmo@upv.es

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/schwartz-geometry-value-detection}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
PYTHON_BIN="$VENV_DIR/bin/python"
GRID_SCRIPT="$PROJECT_DIR/scripts/grid_loss_hparams.py"

METHODS="${METHODS:-asl}"
TUNING_SEEDS="${TUNING_SEEDS:-42 7 1701}"
TUNING_RESULTS_DIR="${TUNING_RESULTS_DIR:-$PROJECT_DIR/results/tuning}"
TUNING_OUTPUT="${TUNING_OUTPUT:-$TUNING_RESULTS_DIR/grid_loss_hparams.csv}"
RETRY_COLLAPSED="${RETRY_COLLAPSED:-1}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
LIMIT="${LIMIT:-}"
DRY_RUN="${DRY_RUN:-0}"
SAVE_CHECKPOINTS="${SAVE_CHECKPOINTS:-0}"
SAVE_HF_MODEL="${SAVE_HF_MODEL:-0}"
DEBUG="${DEBUG:-0}"

mkdir -p "/lustre/scratch/$USER/schwartz-geometry-value-detection/logs"
cd "$PROJECT_DIR"

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

PY_VER="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
NVIDIA_LIB_BASE="$VENV_DIR/lib/python${PY_VER}/site-packages/nvidia"
if [ -d "$NVIDIA_LIB_BASE" ]; then
  for d in "$NVIDIA_LIB_BASE"/*/lib; do
    [ -d "$d" ] || continue
    export LD_LIBRARY_PATH="$d${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  done
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

python -V
python - <<'PY'
import torch
import transformers

print("torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
else:
    raise SystemExit("CUDA is not available in this Slurm job.")
PY

mkdir -p "$TUNING_RESULTS_DIR"

METHODS="${METHODS//,/ }"
TUNING_SEEDS="${TUNING_SEEDS//,/ }"
read -r -a METHOD_ARGS <<< "$METHODS"
read -r -a SEED_ARGS <<< "$TUNING_SEEDS"

cmd=(
  python "$GRID_SCRIPT"
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
echo "Sirius tuning job"
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

echo "Tuning job complete. CSV: $TUNING_OUTPUT"
