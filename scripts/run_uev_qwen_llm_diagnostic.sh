#!/bin/bash -l
#SBATCH --job-name=sgvd-qwenllm
#SBATCH --partition=grupo_pro
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=124G
#SBATCH --time=168:00:00
#SBATCH --output=/home/hpc/34045staff/output/schwartz-geometry-value-detection/%x-%j.out
#SBATCH --error=/home/hpc/34045staff/output/schwartz-geometry-value-detection/%x-%j.err
#SBATCH --hint=nomultithread
#SBATCH --mail-type=ALL
#SBATCH --mail-user=victor.yeste@universidadeuropea.es

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/hpc/34045staff/schwartz-geometry-value-detection}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
PYTHON_BIN="$VENV_DIR/bin/python"
RUN_SCRIPT="$PROJECT_DIR/scripts/run_qwen_llm_diagnostic.py"
ASSETS_SCRIPT="$PROJECT_DIR/scripts/make_llm_diagnostic_assets.py"
UEV_OUTPUT_DIR="${UEV_OUTPUT_DIR:-/home/hpc/34045staff/output/schwartz-geometry-value-detection}"

LLM_CONFIGS="${LLM_CONFIGS:-configs/llm_qwen_definitions_only.yaml configs/llm_qwen_schwartz_continuum.yaml}"
LLM_SPLIT="${LLM_SPLIT:-test}"
LLM_RESULTS_DIR="${LLM_RESULTS_DIR:-$PROJECT_DIR/results/llm_diagnostic}"
LLM_MAX_SAMPLES="${LLM_MAX_SAMPLES:-}"
DRY_RUN="${DRY_RUN:-0}"

if [ "$DRY_RUN" != "1" ] || [ -n "${SLURM_JOB_ID:-}" ]; then
  mkdir -p "$UEV_OUTPUT_DIR"
fi
cd "$PROJECT_DIR"

LLM_CONFIGS="${LLM_CONFIGS//,/ }"
read -r -a CONFIG_ARGS <<< "$LLM_CONFIGS"

cmd_base=("$PYTHON_BIN" "$RUN_SCRIPT" --split "$LLM_SPLIT" --output_dir "$LLM_RESULTS_DIR" --eval)
if [ -n "$LLM_MAX_SAMPLES" ]; then
  cmd_base+=(--max_samples "$LLM_MAX_SAMPLES")
fi

echo "======================================================================"
echo "UEV Qwen2.5-72B LLM diagnostic"
echo "PROJECT_DIR=$PROJECT_DIR"
echo "LLM_RESULTS_DIR=$LLM_RESULTS_DIR"
echo "LLM_CONFIGS=$LLM_CONFIGS"
echo "LLM_SPLIT=$LLM_SPLIT"
echo "LLM_MAX_SAMPLES=${LLM_MAX_SAMPLES:-<full split>}"
echo "DRY_RUN=$DRY_RUN"
for cfg in "${CONFIG_ARGS[@]}"; do
  echo "Command: ${cmd_base[*]} --config $PROJECT_DIR/$cfg"
done
echo "Assets: $PYTHON_BIN $ASSETS_SCRIPT --logs_dir $LLM_RESULTS_DIR/logs"
echo "======================================================================"

if [ "$DRY_RUN" = "1" ]; then
  exit 0
fi

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Missing Python in venv: $PYTHON_BIN" >&2
  echo "Create/update the UEV venv from a login session, then retry:" >&2
  echo "  cd $PROJECT_DIR" >&2
  echo "  python3.11 -m venv .venv" >&2
  echo "  source .venv/bin/activate" >&2
  echo "  python -m pip install -r requirements.txt" >&2
  echo "  python -m pip install -U torch --index-url https://download.pytorch.org/whl/cu128" >&2
  echo "  python -m pip install -U transformers accelerate bitsandbytes safetensors sentencepiece protobuf" >&2
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

HF_TOKEN_FILE="${HF_TOKEN_FILE:-$HOME/.config/huggingface/token}"
if [ -z "${HF_TOKEN:-}" ] && [ -r "$HF_TOKEN_FILE" ]; then
  HF_TOKEN="$(cat "$HF_TOKEN_FILE")"
fi
export HF_TOKEN

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"

"$PYTHON_BIN" -V
"$PYTHON_BIN" - <<'PY'
import importlib.metadata as meta
import torch

print("torch", torch.__version__, "cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0), "cc", torch.cuda.get_device_capability(0))
for pkg in ("transformers", "accelerate", "bitsandbytes", "safetensors"):
    try:
        print(pkg, meta.version(pkg))
    except Exception:
        print(pkg, "<missing>")
PY

mkdir -p "$LLM_RESULTS_DIR/predictions" "$LLM_RESULTS_DIR/logs"

infer_paths() {
  local cfg_path="$1"
  "$PYTHON_BIN" - "$cfg_path" "$LLM_SPLIT" "$LLM_RESULTS_DIR" <<'PY'
import sys
from pathlib import Path
from schwartz_value_geometry.utils.config import load_config
from scripts.run_qwen_llm_diagnostic import output_path_for

cfg = load_config(sys.argv[1])
pred = output_path_for(cfg, split=sys.argv[2], output_dir=Path(sys.argv[3]))
metrics = pred.parents[1] / "logs" / f"{pred.stem}_metrics.json"
print(pred)
print(metrics)
PY
}

for cfg in "${CONFIG_ARGS[@]}"; do
  mapfile -t paths < <(infer_paths "$cfg")
  pred_path="${paths[0]}"
  metrics_path="${paths[1]}"
  if [ -f "$metrics_path" ]; then
    echo "Skipping completed LLM diagnostic: cfg=$cfg metrics=$metrics_path"
    continue
  fi
  if [ -f "$pred_path" ]; then
    echo "Resuming interrupted LLM diagnostic: cfg=$cfg predictions=$pred_path"
  else
    echo "Starting LLM diagnostic: cfg=$cfg"
  fi
  "${cmd_base[@]}" --config "$cfg"
done

"$PYTHON_BIN" "$ASSETS_SCRIPT" --logs_dir "$LLM_RESULTS_DIR/logs"

echo "Qwen LLM diagnostic complete. Results: $LLM_RESULTS_DIR"
