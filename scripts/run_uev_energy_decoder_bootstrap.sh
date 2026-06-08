#!/bin/bash -l
#SBATCH --job-name=sgvd-decboot
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
DECODER_SCRIPT="$PROJECT_DIR/scripts/schwartz_energy_decoder.py"
UEV_OUTPUT_DIR="${UEV_OUTPUT_DIR:-/home/hpc/34045staff/output/schwartz-geometry-value-detection}"

# Targeted paper diagnostic: full decoder, standard vs validation-selected
# pareto_99, with random and empirical controls.
DECODER_GEOMETRIES="${DECODER_GEOMETRIES:-schwartz random empirical}"
DECODER_FAMILIES="${DECODER_FAMILIES:-full}"
DECODER_OBJECTIVES="${DECODER_OBJECTIVES:-standard pareto_99}"
DECODER_BOOTSTRAP_METRICS="${DECODER_BOOTSTRAP_METRICS:-macro_f1 micro_f1 decoder_geometry_cost}"
DECODER_BOOTSTRAP_ITERATIONS="${DECODER_BOOTSTRAP_ITERATIONS:-2000}"
DECODER_BOOTSTRAP_SEED="${DECODER_BOOTSTRAP_SEED:-42}"
DECODER_OUTPUT_DIR="${DECODER_OUTPUT_DIR:-$PROJECT_DIR/results/analysis/schwartz_energy_decoder_bootstrap_examples_full}"
DECODER_MAX_ERROR_EXAMPLES="${DECODER_MAX_ERROR_EXAMPLES:-25}"
DECODER_METHODS="${DECODER_METHODS:-bce}"
DECODER_SEEDS="${DECODER_SEEDS:-}"
DECODER_MODEL_SLUG="${DECODER_MODEL_SLUG:-deberta-v3-base}"
DECODER_RANDOM_SEED="${DECODER_RANDOM_SEED:-42}"
DECODER_EMPIRICAL_METRIC="${DECODER_EMPIRICAL_METRIC:-jaccard}"
DECODER_ALPHA_VALUES="${DECODER_ALPHA_VALUES:-0.10}"
DECODER_BETA_VALUES="${DECODER_BETA_VALUES:-0.20}"
DECODER_GAMMA_VALUES="${DECODER_GAMMA_VALUES:-0.02}"
DECODER_TOP_K="${DECODER_TOP_K:-8}"
DECODER_MAX_CANDIDATES="${DECODER_MAX_CANDIDATES:-8}"
DECODER_MAX_LABELS="${DECODER_MAX_LABELS:-5}"
DECODER_THRESHOLD_FACTOR="${DECODER_THRESHOLD_FACTOR:-0.5}"
DECODER_MIN_PROB="${DECODER_MIN_PROB:-0.01}"
DECODER_MARGINAL_TEMPERATURE="${DECODER_MARGINAL_TEMPERATURE:-1.0}"
DECODER_NEIGHBOR_STEPS="${DECODER_NEIGHBOR_STEPS:-2}"
DRY_RUN="${DRY_RUN:-0}"

if [ "$DRY_RUN" != "1" ] || [ -n "${SLURM_JOB_ID:-}" ]; then
  mkdir -p "$UEV_OUTPUT_DIR"
fi
cd "$PROJECT_DIR"

DECODER_GEOMETRIES="${DECODER_GEOMETRIES//,/ }"
DECODER_FAMILIES="${DECODER_FAMILIES//,/ }"
DECODER_OBJECTIVES="${DECODER_OBJECTIVES//,/ }"
DECODER_BOOTSTRAP_METRICS="${DECODER_BOOTSTRAP_METRICS//,/ }"
DECODER_METHODS="${DECODER_METHODS//,/ }"
DECODER_SEEDS="${DECODER_SEEDS//,/ }"

read -r -a GEOMETRY_ARGS <<< "$DECODER_GEOMETRIES"
read -r -a FAMILY_ARGS <<< "$DECODER_FAMILIES"
read -r -a OBJECTIVE_ARGS <<< "$DECODER_OBJECTIVES"
read -r -a BOOTSTRAP_METRIC_ARGS <<< "$DECODER_BOOTSTRAP_METRICS"
read -r -a METHOD_ARGS <<< "$DECODER_METHODS"

cmd=(
  "$PYTHON_BIN" "$DECODER_SCRIPT"
  --geometries "${GEOMETRY_ARGS[@]}"
  --families "${FAMILY_ARGS[@]}"
  --objectives "${OBJECTIVE_ARGS[@]}"
  --bootstrap_iterations "$DECODER_BOOTSTRAP_ITERATIONS"
  --bootstrap_seed "$DECODER_BOOTSTRAP_SEED"
  --bootstrap_metrics "${BOOTSTRAP_METRIC_ARGS[@]}"
  --max_error_examples "$DECODER_MAX_ERROR_EXAMPLES"
  --methods "${METHOD_ARGS[@]}"
  --model_slug "$DECODER_MODEL_SLUG"
  --random_seed "$DECODER_RANDOM_SEED"
  --empirical_metric "$DECODER_EMPIRICAL_METRIC"
  --alpha_values "$DECODER_ALPHA_VALUES"
  --beta_values "$DECODER_BETA_VALUES"
  --gamma_values "$DECODER_GAMMA_VALUES"
  --top_k "$DECODER_TOP_K"
  --max_candidates "$DECODER_MAX_CANDIDATES"
  --max_labels "$DECODER_MAX_LABELS"
  --threshold_factor "$DECODER_THRESHOLD_FACTOR"
  --min_prob "$DECODER_MIN_PROB"
  --marginal_temperature "$DECODER_MARGINAL_TEMPERATURE"
  --neighbor_steps "$DECODER_NEIGHBOR_STEPS"
  --output_dir "$DECODER_OUTPUT_DIR"
)

if [ -n "$DECODER_SEEDS" ]; then
  read -r -a SEED_ARGS <<< "$DECODER_SEEDS"
  cmd+=(--seeds "${SEED_ARGS[@]}")
fi

echo "======================================================================"
echo "UEV Schwartz energy decoder bootstrap"
echo "PROJECT_DIR=$PROJECT_DIR"
echo "DECODER_OUTPUT_DIR=$DECODER_OUTPUT_DIR"
echo "DECODER_GEOMETRIES=$DECODER_GEOMETRIES"
echo "DECODER_FAMILIES=$DECODER_FAMILIES"
echo "DECODER_OBJECTIVES=$DECODER_OBJECTIVES"
echo "DECODER_BOOTSTRAP_METRICS=$DECODER_BOOTSTRAP_METRICS"
echo "DECODER_BOOTSTRAP_ITERATIONS=$DECODER_BOOTSTRAP_ITERATIONS"
echo "DECODER_MAX_ERROR_EXAMPLES=$DECODER_MAX_ERROR_EXAMPLES"
echo "DECODER_METHODS=$DECODER_METHODS"
echo "DECODER_SEEDS=${DECODER_SEEDS:-<all discovered>}"
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

mkdir -p "$DECODER_OUTPUT_DIR"

"${cmd[@]}"

echo "Decoder bootstrap complete. Output: $DECODER_OUTPUT_DIR"
