# Experiment Protocol

This project uses a two-stage protocol so exploratory tuning remains affordable
while final comparisons are robust enough for a TACL-style submission.

## Random Seeds

Preliminary tuning and debugging use three seeds, matching the previous
Schwartz value-detection paper:

```text
42, 7, 1701
```

Final supervised experiments use five fixed seeds:

```text
42, 7, 1701, 11, 1984
```

Rationale:

- `42`: conventional reference seed.
- `7`: previous-project seed and Harry Potter reference.
- `1701`: previous-project seed and Star Trek reference.
- `11`: compact additional fixed seed and the author's favorite number.
- `1984`: literary reference to Orwell's novel and a clearly memorable fixed seed.

## Reproducible Run Sequence

Run the experiment in this order.

### 1. Prepare The Sirius Environment

Clone or update the repository on Sirius:

```bash
cd "$HOME/schwartz-geometry-value-detection"
git pull
```

Use a project-local virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e .
```

If Slurm cannot see this `.venv` or CUDA libraries from inside `sbatch`, use the
bootstrap fallback:

```bash
sbatch --export=ALL,BASE_PYTHON=/path/to/python3.11 scripts/bootstrap_slurm_venv.sh
```

The smoke script does not require a bootstrap marker. It only requires that
`.venv/bin/python` exists and can import PyTorch, Transformers, and CUDA inside
the Slurm job.

### 2. Run Development Smoke Tests

Before any tuning or final runs, submit:

```bash
sbatch scripts/run_sirius_smoke.sh
```

This checks the end-to-end path for:

- BCE;
- ASL;
- random GeoLoss;
- empirical structure;
- Schwartz GeoLoss;
- Schwartz GeoSmooth;
- validation-tuned per-label thresholds;
- metric JSON writing;
- paper-table aggregation.

The script defaults to:

```text
SMOKE_SEED=42
MAX_SAMPLES=32
SMOKE_EPOCHS=1
SMOKE_BATCH_SIZE=2
SMOKE_MAX_LENGTH=256
SMOKE_GRAD_ACCUM_STEPS=1
THRESHOLD_STEP=0.10
SMOKE_RESULTS_DIR=results/smoke
```

These settings intentionally test code paths, not model quality. They keep GPU
memory low and make the job suitable for a generic single-GPU Slurm allocation.

If needed:

```bash
sbatch --export=ALL,MAX_SAMPLES=64,FORCE_SMOKE=1 scripts/run_sirius_smoke.sh
```

After completion, inspect:

```bash
ls results/smoke/logs
ls results/smoke/analysis
```

Do not start grid search until the smoke job completes without errors.

### 3. Tune Method Hyperparameters

After the smoke tests pass, tune ASL:

```bash
python3 scripts/grid_loss_hparams.py --methods asl
```

Then tune the structured methods:

```bash
python3 scripts/grid_loss_hparams.py --methods geoloss schwartz_geosmooth
```

The tuning grid uses the tuning seeds `42, 7, 1701` by default and writes:

```text
results/analysis/grid_loss_hparams.csv
```

Use `--dry_run` before launching a grid and `--max_samples` only for debugging:

```bash
python3 scripts/grid_loss_hparams.py --methods asl --dry_run
python3 scripts/grid_loss_hparams.py --methods asl --max_samples 128 --limit 2
```

Select hyperparameters using validation metrics only. Prioritize Macro-F1, then
Macro-AUPRC, then circular-error metrics as tie-breakers.

### 4. Freeze Final Configs

After tuning, update the corresponding YAML files with the selected validation
hyperparameters:

```text
configs/deberta_asl.yaml
configs/deberta_random_geoloss.yaml
configs/deberta_empirical_structure.yaml
configs/deberta_schwartz_geoloss.yaml
configs/deberta_schwartz_geosmooth.yaml
```

Do not change these configs again after final seed runs begin, except to fix a
documented implementation bug.

### 5. Run Final Supervised Seeds

Run every final supervised condition with the five final seeds:

```text
42, 7, 1701, 11, 1984
```

The supervised configs are:

```text
configs/deberta_bce.yaml
configs/deberta_asl.yaml
configs/deberta_empirical_structure.yaml
configs/deberta_random_geoloss.yaml
configs/deberta_schwartz_geoloss.yaml
configs/deberta_schwartz_geosmooth.yaml
```

For each config and seed, train once:

```bash
python3 scripts/train_deberta.py --config CONFIG_PATH --seed SEED
```

Then evaluate the saved best checkpoint with validation-frozen per-label
thresholds:

```bash
python3 scripts/eval_deberta.py \
  --config CONFIG_PATH \
  --seed SEED \
  --split test \
  --use_validation_thresholds \
  --threshold_mode per_label
```

Do not tune thresholds on the test split.

### 6. Aggregate Paper Tables

After seed-level evaluations are complete, aggregate paper-ready CSVs:

```bash
python3 scripts/aggregate_results.py
```

This writes:

- `seed_level_results.csv`: one row per metrics JSON file;
- `main_supervised_results.csv`: compact paper-facing mean ± std table;
- `mean_std_summary.csv`: full mean/std/count summary for all scalar metrics;
- `per_label_results.csv`: one row per seed and value label;
- `per_label_summary.csv`: mean/std/count per value label;
- `geometry_metrics.csv`: seed-level circular/error-geometry metrics;
- `threshold_table.csv`: global or per-label thresholds used by each run.

Only after these supervised tables are stable should the LLM diagnostic be run.

## Tuning Stage

Use the three preliminary seeds for:

- checking training stability;
- choosing method-specific hyperparameters;
- selecting GeoLoss `lambda`;
- selecting GeoSmooth `tau`;
- selecting ASL parameters if exposed;
- validating any empirical-structure or random-geometry settings.

Model selection should use validation performance, prioritizing:

1. Macro-F1;
2. Macro-AUPRC, once implemented;
3. Schwartz-aware metrics such as circular error, once implemented.

Do not select models using test-set metrics.

### ASL Hyperparameter Selection

The ASL config starts from the standard literature-inspired setting:

```yaml
loss:
  name: asl
  gamma_pos: 0.0
  gamma_neg: 4.0
  clip: 0.05
  eps: 1.0e-8
```

Interpretation:

- `gamma_pos`: downweights easy positive labels. Keep fixed at `0.0` unless
  validation results clearly suggest otherwise, because positive labels are
  sparse and should retain strong gradients.
- `gamma_neg`: downweights easy negative labels. Tune this because most labels
  are negative for most sentences.
- `clip`: negative probability margin. Tune this because it controls how
  aggressively very easy negatives are ignored.
- `eps`: numerical stability for logarithms. Do not tune this.

Use the tuning seeds `42, 7, 1701` and select ASL parameters on the validation
split only. A compact grid is:

```text
gamma_pos: {0.0}
gamma_neg: {2.0, 3.0, 4.0, 5.0}
clip: {0.0, 0.03, 0.05, 0.10}
eps: {1.0e-8}
```

After selecting the best ASL setting, freeze it before the final five-seed
comparison. Do not retune ASL parameters on the test split.

The tuning script for this stage is:

```bash
python3 scripts/grid_loss_hparams.py --methods asl
```

### Geometry-Aware Method Selection

The structured method configs are:

```text
configs/deberta_random_geoloss.yaml
configs/deberta_empirical_structure.yaml
configs/deberta_schwartz_geoloss.yaml
configs/deberta_schwartz_geosmooth.yaml
```

By default, these structured methods use ASL as their base loss. This keeps the
main comparison focused on whether adding label structure improves over the
strong imbalance-aware baseline. If needed, BCE-based structured variants can be
added later as ablations.

Method roles:

- `random_geoloss`: negative control using the GeoLoss formula with a fixed
  seeded random circular order.
- `empirical_structure`: generic label-structure baseline using training-set
  label co-occurrence.
- `schwartz_geoloss`: main proposed theory-aware loss using the true Schwartz
  circular distance matrix.
- `schwartz_geosmooth`: secondary theory-aware variant using Schwartz-smoothed
  soft targets.

Tune method-specific parameters on validation only:

```text
lambda_geo: {0.01, 0.05, 0.1, 0.2}
tau: {0.1, 0.2, 0.5, 1.0}
```

For the random-geometry negative control, keep the random circular order fixed
across training seeds unless explicitly running an additional random-order
sensitivity analysis.

Use:

```bash
python3 scripts/grid_loss_hparams.py --methods geoloss schwartz_geosmooth
```

Use `--dry_run` to inspect the planned grid and `--max_samples` only for
debugging.

## Final Stage

After hyperparameters are frozen, run all main supervised conditions with the
five final seeds:

- DeBERTa-v3-base + BCE;
- DeBERTa-v3-base + ASL;
- DeBERTa-v3-base + empirical label structure;
- DeBERTa-v3-base + random circular geometry;
- DeBERTa-v3-base + Schwartz GeoLoss;
- DeBERTa-v3-base + Schwartz GeoSmooth.

For each seed and condition:

- tune thresholds on the validation split;
- evaluate once on the test split;
- save probabilities, binary predictions, labels, config, and seed metadata.

Final tables should report mean ± standard deviation over the five seeds.

Run artifacts should use explicit loss names, for example `deberta_bce`,
`deberta_asl`, `deberta_geoloss`, and `deberta_geosmooth`.

## LLM Diagnostic

The LLM comparison is diagnostic, not the main contribution. If decoding is
deterministic (`temperature = 0`), one run per prompt/model condition is
acceptable, with invalid-output rate reported. Additional LLM repetitions are
optional only if nondeterminism is observed or decoding uses sampling.
