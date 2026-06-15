#!/bin/bash -l
#SBATCH --job-name=sgvd-qwenllm
#SBATCH --partition=gpu
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=96:00:00
#SBATCH --output=/lustre/scratch/%u/schwartz-geometry-value-detection/logs/%x-%j.out
#SBATCH --error=/lustre/scratch/%u/schwartz-geometry-value-detection/logs/%x-%j.err
#SBATCH --hint=nomultithread
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vicyesmo@upv.es

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/schwartz-geometry-value-detection}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
PYTHON_BIN="$VENV_DIR/bin/python"
RUN_SCRIPT="$PROJECT_DIR/scripts/run_qwen_llm_diagnostic.py"
ASSETS_SCRIPT="$PROJECT_DIR/scripts/make_llm_diagnostic_assets.py"

LLM_CONFIGS="${LLM_CONFIGS:-configs/llm_qwen_definitions_only.yaml configs/llm_qwen_schwartz_continuum.yaml}"
LLM_SPLIT="${LLM_SPLIT:-test}"
LLM_RESULTS_DIR="${LLM_RESULTS_DIR:-$PROJECT_DIR/results/llm_diagnostic}"
LLM_MAX_SAMPLES="${LLM_MAX_SAMPLES:-}"
DRY_RUN="${DRY_RUN:-0}"

if [ "$DRY_RUN" != "1" ] || [ -n "${SLURM_JOB_ID:-}" ]; then
  mkdir -p "/lustre/scratch/$USER/schwartz-geometry-value-detection/logs"
fi
cd "$PROJECT_DIR"

LLM_CONFIGS="${LLM_CONFIGS//,/ }"
read -r -a CONFIG_ARGS <<< "$LLM_CONFIGS"

cmd_base=("$PYTHON_BIN" "$RUN_SCRIPT" --split "$LLM_SPLIT" --output_dir "$LLM_RESULTS_DIR" --eval)
if [ -n "$LLM_MAX_SAMPLES" ]; then
  cmd_base+=(--max_samples "$LLM_MAX_SAMPLES")
fi

echo "======================================================================"
echo "Sirius Qwen2.5-72B LLM diagnostic"
echo "PROJECT_DIR=$PROJECT_DIR"
echo "VENV_DIR=$VENV_DIR"
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

HF_TOKEN_FILE="${HF_TOKEN_FILE:-$HOME/.config/huggingface/token}"
if [ -z "${HF_TOKEN:-}" ] && [ -r "$HF_TOKEN_FILE" ]; then
  HF_TOKEN="$(cat "$HF_TOKEN_FILE")"
fi
export HF_TOKEN

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Missing Python in venv: $PYTHON_BIN" >&2
  echo "Run once: sbatch scripts/bootstrap_slurm_venv.sh" >&2
  echo "If auto-detection fails, pass a real Python path with:" >&2
  echo "  sbatch --export=ALL,BASE_PYTHON=/real/path/to/python3.11 scripts/bootstrap_slurm_venv.sh" >&2
  exit 1
fi

if [ ! -f "$VENV_DIR/.bootstrap_complete" ]; then
  echo "Venv not marked as bootstrapped: $VENV_DIR/.bootstrap_complete missing" >&2
  echo "Run: sbatch --export=ALL,FORCE_REBUILD=1,BASE_PYTHON=/real/path/to/python3.11 scripts/bootstrap_slurm_venv.sh" >&2
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
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

"$PYTHON_BIN" -V
"$PYTHON_BIN" - <<'PY'
import importlib.metadata as meta

import torch


def parse_version(raw):
    parts = []
    for token in raw.replace("-", ".").split("."):
        if token.isdigit():
            parts.append(int(token))
        else:
            digits = "".join(ch for ch in token if ch.isdigit())
            if digits:
                parts.append(int(digits))
        if len(parts) == 3:
            break
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


print("torch", torch.__version__, "cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu_count", torch.cuda.device_count())
    for idx in range(torch.cuda.device_count()):
        print("gpu", idx, torch.cuda.get_device_name(idx), "cc", torch.cuda.get_device_capability(idx))
else:
    raise SystemExit("CUDA is not available in this Slurm job.")
for pkg in ("transformers", "accelerate", "bitsandbytes", "safetensors"):
    try:
        version = meta.version(pkg)
        print(pkg, version)
        if pkg == "bitsandbytes" and parse_version(version) < (0, 46, 1):
            raise SystemExit("bitsandbytes>=0.46.1 is required for Qwen 4-bit inference")
    except Exception:
        if pkg == "bitsandbytes":
            raise SystemExit("bitsandbytes>=0.46.1 is required for Qwen 4-bit inference")
        print(pkg, "<missing>")
PY

mkdir -p "$LLM_RESULTS_DIR/predictions" "$LLM_RESULTS_DIR/logs"

infer_paths() {
  local cfg_path="$1"
  "$PYTHON_BIN" - "$cfg_path" "$LLM_SPLIT" "$LLM_RESULTS_DIR" <<'PY'
import importlib.util
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "src"))
logging.disable(logging.CRITICAL)

from schwartz_value_geometry.utils.config import load_config

script_path = Path.cwd() / "scripts" / "run_qwen_llm_diagnostic.py"
spec = importlib.util.spec_from_file_location("run_qwen_llm_diagnostic", script_path)
if spec is None or spec.loader is None:
    raise SystemExit(f"Cannot import {script_path}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

cfg = load_config(sys.argv[1])
pred = module.output_path_for(cfg, split=sys.argv[2], output_dir=Path(sys.argv[3]))
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
