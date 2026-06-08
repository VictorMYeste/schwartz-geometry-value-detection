#!/bin/bash -l
#SBATCH --job-name=sgvd-decctrl
#SBATCH --partition=grupo_pro
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/home/hpc/34045staff/output/schwartz-geometry-value-detection/%x-%j.out
#SBATCH --error=/home/hpc/34045staff/output/schwartz-geometry-value-detection/%x-%j.err
#SBATCH --hint=nomultithread
#SBATCH --mail-type=ALL
#SBATCH --mail-user=victor.yeste@universidadeuropea.es

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/hpc/34045staff/schwartz-geometry-value-detection}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
PYTHON_BIN="$VENV_DIR/bin/python"
CONTROL_SCRIPT="$PROJECT_DIR/scripts/bootstrap_energy_decoder_controls.py"
UEV_OUTPUT_DIR="${UEV_OUTPUT_DIR:-/home/hpc/34045staff/output/schwartz-geometry-value-detection}"

DECODER_OUTPUT_DIR="${DECODER_OUTPUT_DIR:-$PROJECT_DIR/results/analysis/schwartz_energy_decoder_bootstrap_examples_full}"
CONTROL_OUTPUT_DIR="${CONTROL_OUTPUT_DIR:-$PROJECT_DIR/results/analysis/schwartz_energy_decoder_control_bootstrap}"
CONTROL_METHODS="${CONTROL_METHODS:-bce}"
CONTROL_SEEDS="${CONTROL_SEEDS:-}"
CONTROL_MODEL_SLUG="${CONTROL_MODEL_SLUG:-deberta-v3-base}"
CONTROL_FAMILY="${CONTROL_FAMILY:-full}"
CONTROL_OBJECTIVE="${CONTROL_OBJECTIVE:-pareto_99}"
CONTROL_GEOMETRIES="${CONTROL_GEOMETRIES:-random empirical}"
CONTROL_METRICS="${CONTROL_METRICS:-macro_f1 micro_f1 opposite_error_rate decoder_geometry_cost}"
CONTROL_BOOTSTRAP_ITERATIONS="${CONTROL_BOOTSTRAP_ITERATIONS:-2000}"
CONTROL_BOOTSTRAP_SEED="${CONTROL_BOOTSTRAP_SEED:-42}"
DRY_RUN="${DRY_RUN:-0}"

if [ "$DRY_RUN" != "1" ] || [ -n "${SLURM_JOB_ID:-}" ]; then
  mkdir -p "$UEV_OUTPUT_DIR"
fi
cd "$PROJECT_DIR"

CONTROL_METHODS="${CONTROL_METHODS//,/ }"
CONTROL_SEEDS="${CONTROL_SEEDS//,/ }"
CONTROL_GEOMETRIES="${CONTROL_GEOMETRIES//,/ }"
CONTROL_METRICS="${CONTROL_METRICS//,/ }"

read -r -a METHOD_ARGS <<< "$CONTROL_METHODS"
read -r -a GEOMETRY_ARGS <<< "$CONTROL_GEOMETRIES"
read -r -a METRIC_ARGS <<< "$CONTROL_METRICS"

cmd=(
  "$PYTHON_BIN" "$CONTROL_SCRIPT"
  --decoder_output_dir "$DECODER_OUTPUT_DIR"
  --output_dir "$CONTROL_OUTPUT_DIR"
  --methods "${METHOD_ARGS[@]}"
  --model_slug "$CONTROL_MODEL_SLUG"
  --family "$CONTROL_FAMILY"
  --objective "$CONTROL_OBJECTIVE"
  --controls "${GEOMETRY_ARGS[@]}"
  --metrics "${METRIC_ARGS[@]}"
  --bootstrap_iterations "$CONTROL_BOOTSTRAP_ITERATIONS"
  --bootstrap_seed "$CONTROL_BOOTSTRAP_SEED"
)

if [ -n "$CONTROL_SEEDS" ]; then
  read -r -a SEED_ARGS <<< "$CONTROL_SEEDS"
  cmd+=(--seeds "${SEED_ARGS[@]}")
fi

echo "======================================================================"
echo "UEV Schwartz-vs-control energy decoder bootstrap"
echo "PROJECT_DIR=$PROJECT_DIR"
echo "DECODER_OUTPUT_DIR=$DECODER_OUTPUT_DIR"
echo "CONTROL_OUTPUT_DIR=$CONTROL_OUTPUT_DIR"
echo "CONTROL_METHODS=$CONTROL_METHODS"
echo "CONTROL_SEEDS=${CONTROL_SEEDS:-<all discovered>}"
echo "CONTROL_MODEL_SLUG=$CONTROL_MODEL_SLUG"
echo "CONTROL_FAMILY=$CONTROL_FAMILY"
echo "CONTROL_OBJECTIVE=$CONTROL_OBJECTIVE"
echo "CONTROL_GEOMETRIES=$CONTROL_GEOMETRIES"
echo "CONTROL_METRICS=$CONTROL_METRICS"
echo "CONTROL_BOOTSTRAP_ITERATIONS=$CONTROL_BOOTSTRAP_ITERATIONS"
echo "DRY_RUN=$DRY_RUN"
echo "Command: ${cmd[*]}"
echo "======================================================================"

if [ "$DRY_RUN" = "1" ]; then
  exit 0
fi

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Missing Python in venv: $PYTHON_BIN" >&2
  echo "Create the UEV venv from a login session, then retry:" >&2
  echo "  cd $PROJECT_DIR" >&2
  echo "  python3.11 -m venv .venv" >&2
  echo "  source .venv/bin/activate" >&2
  echo "  python -m pip install -r requirements.txt" >&2
  echo "  python -m pip install -e ." >&2
  exit 1
fi

source "$VENV_DIR/bin/activate"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

"$PYTHON_BIN" -V
"$PYTHON_BIN" - <<'PY'
import numpy
import pandas

print("numpy", numpy.__version__)
print("pandas", pandas.__version__)
PY

mkdir -p "$CONTROL_OUTPUT_DIR"

"${cmd[@]}"

echo "Decoder control bootstrap complete. Output: $CONTROL_OUTPUT_DIR"
