#!/bin/bash -l
#SBATCH --job-name=sgvd-smoke
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/scratch/%u/schwartz-geometry-value-detection/logs/%x-%j.out
#SBATCH --error=/lustre/scratch/%u/schwartz-geometry-value-detection/logs/%x-%j.err
#SBATCH --hint=nomultithread
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=vicyesmo@upv.es

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/schwartz-geometry-value-detection}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
PYTHON_BIN="$VENV_DIR/bin/python"
TRAIN_SCRIPT="$PROJECT_DIR/scripts/train_deberta.py"
EVAL_SCRIPT="$PROJECT_DIR/scripts/eval_deberta.py"
AGG_SCRIPT="$PROJECT_DIR/scripts/aggregate_results.py"

SMOKE_SEED="${SMOKE_SEED:-42}"
MAX_SAMPLES="${MAX_SAMPLES:-32}"
SMOKE_EPOCHS="${SMOKE_EPOCHS:-1}"
SMOKE_BATCH_SIZE="${SMOKE_BATCH_SIZE:-2}"
SMOKE_MAX_LENGTH="${SMOKE_MAX_LENGTH:-256}"
SMOKE_GRAD_ACCUM_STEPS="${SMOKE_GRAD_ACCUM_STEPS:-1}"
THRESHOLD_STEP="${THRESHOLD_STEP:-0.10}"
SMOKE_RESULTS_DIR="${SMOKE_RESULTS_DIR:-$PROJECT_DIR/results/smoke}"
SMOKE_CONFIG_DIR="${SMOKE_CONFIG_DIR:-$SMOKE_RESULTS_DIR/configs}"
FORCE_SMOKE="${FORCE_SMOKE:-0}"

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

export PROJECT_DIR SMOKE_CONFIG_DIR SMOKE_RESULTS_DIR SMOKE_EPOCHS
export SMOKE_BATCH_SIZE SMOKE_MAX_LENGTH SMOKE_GRAD_ACCUM_STEPS
python - <<'PY'
import os
from pathlib import Path

import yaml

project_dir = Path(os.environ["PROJECT_DIR"])
config_dir = Path(os.environ["SMOKE_CONFIG_DIR"])
results_dir = Path(os.environ["SMOKE_RESULTS_DIR"])
epochs = int(os.environ["SMOKE_EPOCHS"])
batch_size = int(os.environ["SMOKE_BATCH_SIZE"])
max_length = int(os.environ["SMOKE_MAX_LENGTH"])
grad_accum_steps = int(os.environ["SMOKE_GRAD_ACCUM_STEPS"])
config_dir.mkdir(parents=True, exist_ok=True)
results_dir.mkdir(parents=True, exist_ok=True)

configs = [
    "configs/deberta_bce.yaml",
    "configs/deberta_asl.yaml",
    "configs/deberta_random_geoloss.yaml",
    "configs/deberta_empirical_structure.yaml",
    "configs/deberta_schwartz_geoloss.yaml",
    "configs/deberta_schwartz_geosmooth.yaml",
]

for rel_path in configs:
    src = project_dir / rel_path
    raw = yaml.safe_load(src.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise SystemExit(f"Config root must be a mapping: {src}")
    raw["results_dir"] = str(results_dir)
    raw["save_checkpoints"] = True
    training = dict(raw.get("training", {}))
    training["num_epochs"] = epochs
    training["batch_size"] = batch_size
    training["max_length"] = max_length
    training["grad_accum_steps"] = grad_accum_steps
    training["early_stopping_patience"] = 0
    training["save_hf_model"] = False
    raw["training"] = training
    out = config_dir / f"{src.stem}_smoke.yaml"
    out.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    print(out)
PY

mapfile -t SMOKE_CONFIGS < <(find "$SMOKE_CONFIG_DIR" -maxdepth 1 -name '*_smoke.yaml' | sort)

if [ "${#SMOKE_CONFIGS[@]}" -eq 0 ]; then
  echo "No smoke configs found in $SMOKE_CONFIG_DIR" >&2
  exit 1
fi

artifact_prefix() {
  local cfg="$1"
  python - "$cfg" "$SMOKE_SEED" <<'PY'
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "src"))
logging.disable(logging.CRITICAL)
from schwartz_value_geometry.utils.config import load_config
from schwartz_value_geometry.utils.naming import artifact_prefix

cfg = load_config(sys.argv[1])
print(artifact_prefix(cfg, seed=int(sys.argv[2])))
PY
}

for cfg in "${SMOKE_CONFIGS[@]}"; do
  prefix="$(artifact_prefix "$cfg")"
  metrics_path="$SMOKE_RESULTS_DIR/logs/${prefix}_test_metrics.json"

  if [ -f "$metrics_path" ] && [ "$FORCE_SMOKE" != "1" ]; then
    echo "Skipping completed smoke run: cfg=$cfg seed=$SMOKE_SEED"
    continue
  fi

  echo "======================================================================"
  echo "Smoke training: cfg=$cfg seed=$SMOKE_SEED max_samples=$MAX_SAMPLES epochs=$SMOKE_EPOCHS"
  python "$TRAIN_SCRIPT" \
    --config "$cfg" \
    --seed "$SMOKE_SEED" \
    --max_samples "$MAX_SAMPLES" \
    --retry_collapsed 0

  echo "Smoke evaluation with validation-tuned thresholds: cfg=$cfg"
  python "$EVAL_SCRIPT" \
    --config "$cfg" \
    --seed "$SMOKE_SEED" \
    --split test \
    --max_samples "$MAX_SAMPLES" \
    --use_validation_thresholds \
    --threshold_mode per_label \
    --threshold_step "$THRESHOLD_STEP"
done

echo "Aggregating smoke outputs"
python "$AGG_SCRIPT" \
  --results_dir "$SMOKE_RESULTS_DIR/logs" \
  --output_dir "$SMOKE_RESULTS_DIR/analysis" \
  --split test

echo "Smoke job complete. Outputs: $SMOKE_RESULTS_DIR"
