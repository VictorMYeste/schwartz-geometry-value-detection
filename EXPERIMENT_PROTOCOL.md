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
sbatch scripts/bootstrap_slurm_venv.sh
```

The bootstrap script auto-detects Python 3.11+. If auto-detection fails, pass a
real executable path, not the placeholder:

```bash
sbatch --export=ALL,BASE_PYTHON=/real/path/to/python3.11 scripts/bootstrap_slurm_venv.sh
```

The Slurm scripts do not require a bootstrap marker. They only require that
`.venv/bin/python` exists and can import PyTorch, Transformers, and CUDA inside
the Slurm job.

UEV does not use the Sirius bootstrap script. Create or update the `.venv` from
the login session. Use `safetensors` and PyTorch `>=2.6` when possible because
recent Transformers versions block `torch.load` on older PyTorch versions when a
model falls back to `pytorch_model.bin`:

```bash
cd /home/hpc/34045staff/schwartz-geometry-value-detection
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --upgrade --force-reinstall torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip uninstall -y torchvision torchaudio
python -m pip install -e .
```

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

After the smoke tests pass, run a dry-run plan on Sirius:

```bash
sbatch --export=ALL,DRY_RUN=1 scripts/run_sirius_tuning.sh
```

Then launch ASL tuning:

```bash
sbatch scripts/run_sirius_tuning.sh
```

This is equivalent to:

```bash
python3 scripts/grid_loss_hparams.py --methods asl
```

but inside the Sirius Slurm environment, with CUDA checks, `.venv` activation,
and tuning artifacts isolated under:

```text
results/tuning/
results/tuning/grid_loss_hparams.csv
```

After selecting the best plain loss on validation/test-stable final baseline
results, use the stronger plain loss as the base for the main structured
comparison. In the current protocol, BCE is stronger than ASL, so the top-level
structured configs omit `base` and therefore use the default BCE base loss.
ASL-based structured configs are kept separately under `configs/asl_based/` as
an appendix/ablation condition.

Then tune the geometry-aware methods:

```bash
METHODS="geoloss schwartz_geosmooth" \
TUNING_OUTPUT="$PWD/results/tuning/grid_loss_hparams_geometry.csv" \
sbatch --export=ALL scripts/run_sirius_tuning.sh
```

This order matters because the geometry-aware configs should be tuned with the
same base loss that will be used in the final structured comparison. If the base
loss changes, final geometry tuning should be repeated from a clean geometry
tuning CSV.
The grid script appends to the selected CSV rather than overwriting it, but use
a separate geometry CSV so ASL selection and geometry selection remain distinct:

```text
results/tuning/grid_loss_hparams.csv
results/tuning/grid_loss_hparams_geometry.csv
```

The full exploratory `METHODS=all` grid contains 96 training runs:

- ASL: 16 parameter combinations x 3 tuning seeds = 48 runs;
- random GeoLoss: 4 lambda values x 3 tuning seeds = 12 runs;
- empirical structure: 4 lambda values x 3 tuning seeds = 12 runs;
- Schwartz GeoLoss: 4 lambda values x 3 tuning seeds = 12 runs;
- Schwartz GeoSmooth: 4 tau values x 3 tuning seeds = 12 runs.

Use it only if you intentionally want a single exploratory pass with the current
ASL defaults:

```bash
METHODS=all sbatch --export=ALL scripts/run_sirius_tuning.sh
```

The Slurm script is resumable through the CSV file: completed rows are skipped
when the job is relaunched. Do not run two jobs writing to the same tuning CSV at
the same time.

Useful Sirius variants:

```bash
METHODS=asl sbatch --export=ALL scripts/run_sirius_tuning.sh
METHODS="geoloss schwartz_geosmooth" TUNING_OUTPUT="$PWD/results/tuning/grid_loss_hparams_geometry.csv" sbatch --export=ALL scripts/run_sirius_tuning.sh
sbatch --export=ALL,LIMIT=10 scripts/run_sirius_tuning.sh
sbatch --export=ALL,MAX_SAMPLES=128,LIMIT=2 scripts/run_sirius_tuning.sh
```

By default, tuning does not save checkpoints or Hugging Face model bundles,
because the selection signal is the validation CSV. If checkpoints are needed
for debugging, use `SAVE_CHECKPOINTS=1`; only use `SAVE_HF_MODEL=1` if model
bundles are explicitly needed.

Select hyperparameters using validation metrics only. Prioritize Macro-F1, then
Macro-AUPRC, then circular-error metrics as tie-breakers.

The default maximum is `30` epochs with early stopping. If a tuning run was
started with an older lower epoch cap, restart from a clean tuning CSV; otherwise
the resume logic will skip completed rows from the old protocol.

Current ASL selection from validation tuning:

```yaml
gamma_pos: 0.0
gamma_neg: 2.0
clip: 0.0
eps: 1.0e-8
```

This setting had the highest mean validation Macro-F1 across the ASL tuning
seeds `42, 7, 1701` in `results/tuning/grid_loss_hparams.csv`. However, final
baseline results showed BCE outperforming ASL, so the main structured configs
use BCE as their base. The ASL-based structured runs are retained as an
appendix/ablation.

Current geometry-aware selections from validation tuning:

```yaml
random_geoloss:
  lambda_geo: 0.01
empirical_structure:
  lambda_geo: 0.01
schwartz_geoloss:
  lambda_geo: 0.01
schwartz_geosmooth:
  tau: 0.1
```

These settings had the highest mean validation Macro-F1 within each method over
the tuning seeds `42, 7, 1701` in
`results/tuning/grid_loss_hparams_geometry.csv`, using the BCE-based top-level
structured configs.

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

On Sirius, submit the full final supervised array:

```bash
mkdir -p "/lustre/scratch/$USER/schwartz-geometry-value-detection/logs"
sbatch scripts/run_sirius_final_supervised.sh
```

On UEV, submit:

```bash
mkdir -p /home/hpc/34045staff/output/schwartz-geometry-value-detection
sbatch scripts/run_uev_final_supervised.sh
```

Both launchers run the 30 final jobs as a Slurm array:

```text
6 configs x 5 seeds = 30 tasks
array: 0-29%6
```

Each task trains one config/seed pair, saves the best PyTorch checkpoint, then
evaluates once on the test split using validation-frozen per-label thresholds.
The launchers set `SAVE_HF_MODEL=0` by default to avoid writing 30 full
Hugging Face model bundles; set `SAVE_HF_MODEL=1` only if those bundles are
needed. They also set `CHECKPOINT_EVERY_EPOCHS=0` by default, so only the best
checkpoint needed for evaluation is kept. Set `CHECKPOINT_EVERY_EPOCHS=1` if
partial-run resume checkpoints are needed and storage is available.

Before submitting the real array, inspect the local plan without allocating
GPUs:

```bash
DRY_RUN=1 bash scripts/run_sirius_final_supervised.sh
DRY_RUN=1 bash scripts/run_uev_final_supervised.sh
```

If cluster load is high, reduce concurrency with a command-line array override:

```bash
sbatch --array=0-29%2 scripts/run_sirius_final_supervised.sh
sbatch --array=0-29%2 scripts/run_uev_final_supervised.sh
```

The final launcher is resumable at the task level: if the expected test metrics
JSON already exists, that task is skipped unless `FORCE_FINAL=1` is set.

### 6. Aggregate Paper Tables

After seed-level evaluations are complete, aggregate paper-ready CSVs:

```bash
python3 scripts/aggregate_results.py \
  --results_dir results/logs \
  --output_dir results/analysis/paper_tables \
  --split test
```

This writes:

- `seed_level_results.csv`: one row per metrics JSON file;
- `main_supervised_results.csv`: compact paper-facing mean ± std table;
- `mean_std_summary.csv`: full mean/std/count summary for all scalar metrics;
- `per_label_results.csv`: one row per seed and value label;
- `per_label_summary.csv`: mean/std/count per value label;
- `geometry_metrics.csv`: seed-level circular/error-geometry metrics;
- `threshold_table.csv`: global or per-label thresholds used by each run.

### 7. Run Seed-Level Bootstrap Significance

After the paper tables are stable, compare each method against the selected
baseline with paired seed-level bootstrap. This tests whether the reported
five-seed mean difference is stable over the matched final seeds:

```bash
python3 scripts/bootstrap_seed_significance.py \
  --seed_level_csv results/analysis/paper_tables/seed_level_results.csv \
  --output results/analysis/paper_tables/bootstrap_seed_significance.csv \
  --baseline bce \
  --split test \
  --threshold_source provided \
  --n_iterations 2000 \
  --seed 42
```

The output reports method-baseline deltas, confidence intervals, two-sided
bootstrap p-values, and seed win counts. Selection decisions must not be made
from these test-set significance results; use them only after final runs are
frozen.

### 8. Run Geometry-Aware Calibration Diagnostic

If training-time structure-aware losses do not clearly improve the BCE
baseline, run the validation-only calibration diagnostic before starting a new
training campaign:

```bash
python3 scripts/geometry_aware_calibration.py \
  --predictions_dir results/predictions \
  --logs_dir results/logs \
  --output_dir results/analysis/geometry_calibration
```

This step does not update model weights. For each run, it:

- reads saved validation and test probabilities;
- starts from the validation-frozen per-label thresholds already used in final
  evaluation;
- searches a small validation grid over geometry-aware probability suppression,
  threshold offset, and opposite-value conflict filtering;
- selects `standard`, `f1`, `pareto_99`, `pareto_98`, and `pareto_95`
  settings on validation only;
- applies each selected setting once to the test split.

The calibration outputs are:

- `geometry_calibration_validation_grid.csv`: validation metrics for every
  calibration setting;
- `geometry_calibration_selected_seed_results.csv`: selected validation setting
  and test metrics for each seed;
- `geometry_calibration_selected_mean_std.csv`: mean/std/count summaries;
- `geometry_calibration_delta_vs_standard.csv`: test deltas against the
  original validation-frozen thresholds;
- `geometry_calibration_thresholds.csv`: thresholds selected per seed and
  label.

Treat this as a diagnostic or possible pivot method only if it gives a
validation-selected Pareto improvement on test, meaning geometry metrics improve
without a meaningful Macro-F1 loss. Do not choose calibration settings directly
from test metrics.

### 9. Run Schwartz Structured Energy Decoder

If lightweight calibration is not strong enough, evaluate a structured
label-set decoder over the already trained BCE model:

```bash
python3 scripts/schwartz_energy_decoder.py \
  --predictions_dir results/predictions \
  --logs_dir results/logs \
  --output_dir results/analysis/schwartz_energy_decoder
```

This is also a post-training method: it does not update DeBERTa weights. It
starts from the validation-frozen per-label thresholds and decodes the final
predicted label set jointly with this energy:

```text
score(y | x) =
  independent threshold-adjusted label utility
  + alpha_neighbor * nearby-value compatibility
  - beta_opposite * opposite-value conflict
  - gamma_cardinality * excess-label penalty
```

The default run evaluates BCE only, because BCE is the current strongest base
model. By default it evaluates three geometry controls:

- `schwartz`: true Schwartz circular structure;
- `random`: fixed random circular order negative control;
- `empirical`: training-set label co-occurrence structure.

It also evaluates decoder component ablations:

- `cardinality_only`;
- `neighbor_only`;
- `opposite_only`;
- `neighbor_opposite`;
- `full`.

To run it over other trained methods, pass `--methods`, for example:

```bash
python3 scripts/schwartz_energy_decoder.py \
  --methods bce schwartz_geoloss empirical_structure \
  --output_dir results/analysis/schwartz_energy_decoder_all
```

For the main controls and ablations without expensive bootstrap/example
generation, run:

```bash
python3 scripts/schwartz_energy_decoder.py \
  --bootstrap_iterations 0 \
  --max_error_examples 0 \
  --output_dir results/analysis/schwartz_energy_decoder_controls
```

For targeted sample-level bootstrap and qualitative error analysis of the main
full decoder family, run:

```bash
python3 scripts/schwartz_energy_decoder.py \
  --geometries schwartz random empirical \
  --families full \
  --objectives standard pareto_99 \
  --bootstrap_iterations 2000 \
  --bootstrap_metrics macro_f1 micro_f1 \
  --max_error_examples 25 \
  --output_dir results/analysis/schwartz_energy_decoder_bootstrap_examples
```

The decoder-geometry-cost sample bootstrap is supported but slower; run it on
the cluster by adding `decoder_geometry_cost` to `--bootstrap_metrics`.

On UEV, use the dedicated Slurm launcher for the 2000-iteration bootstrap with
geometry cost included:

```bash
DRY_RUN=1 bash scripts/run_uev_energy_decoder_bootstrap.sh
sbatch scripts/run_uev_energy_decoder_bootstrap.sh
```

The launcher is CPU-focused and defaults to:

```text
geometries: schwartz random empirical
families: full
objectives: standard pareto_99
bootstrap_metrics: macro_f1 micro_f1 decoder_geometry_cost
bootstrap_iterations: 2000
output_dir: results/analysis/schwartz_energy_decoder_bootstrap_examples_full
```

The decoder outputs are:

- `schwartz_energy_decoder_validation_grid.csv`: validation metrics for every
  decoder setting;
- `schwartz_energy_decoder_selected_seed_results.csv`: selected validation
  setting and test metrics for each seed;
- `schwartz_energy_decoder_selected_mean_std.csv`: mean/std/count summaries;
- `schwartz_energy_decoder_delta_vs_standard.csv`: test deltas against the
  original independent validation-frozen thresholds;
- `schwartz_energy_decoder_sample_bootstrap.csv`: sample-level paired bootstrap
  rows for requested metrics;
- `schwartz_energy_decoder_error_examples.csv`: readable examples where the
  decoder changed the predicted label set.

After this, run the direct Schwartz-vs-control bootstrap:

```bash
python3 scripts/bootstrap_energy_decoder_controls.py \
  --decoder_output_dir results/analysis/schwartz_energy_decoder_bootstrap_examples_full \
  --output_dir results/analysis/schwartz_energy_decoder_control_bootstrap
```

This tests the contrast:

```text
(Schwartz decoder - standard thresholding)
-
(control decoder - standard thresholding)
```

For a fixed bootstrap resample the standard-thresholding term cancels, so the
implemented paired bootstrap compares the Schwartz decoder directly with the
random and empirical control decoders. The main metric is
`decoder_geometry_cost`; secondary metrics are `macro_f1`, `micro_f1`, and
`opposite_error_rate`.

If the full 2000-iteration control bootstrap is too slow locally, use the UEV
launcher:

```bash
DRY_RUN=1 bash scripts/run_uev_energy_decoder_control_bootstrap.sh
sbatch scripts/run_uev_energy_decoder_control_bootstrap.sh
```

The control-bootstrap outputs are:

- `schwartz_energy_decoder_control_delta_seed_results.csv`: seed-level direct
  Schwartz-minus-control deltas;
- `schwartz_energy_decoder_control_delta_mean_std.csv`: mean/std/count summary
  of the direct control deltas;
- `schwartz_energy_decoder_control_sample_bootstrap.csv`: per-seed sample-level
  paired bootstrap confidence intervals and p-values.

The frozen final decoder setup is documented in
`configs/schwartz_energy_decoder.yaml`. Treat this config as fixed once the
decoder control bootstrap is reported.

Finally, generate the paper-facing decoder tables, SVG figures, and curated
qualitative examples:

```bash
python3 scripts/make_decoder_paper_assets.py
```

This writes compact outputs under:

```text
results/analysis/paper_tables/
results/analysis/paper_figures/
```

Use validation Macro-F1 as the primary selection signal. For Pareto objectives,
the script keeps settings within a validation Macro-F1 tolerance and then
minimizes a binary decoder-geometry cost based on opposite errors,
neighbor-error rate, and confusion-distance correlation.

Important interpretation: this decoder changes the final predicted label set.
It can improve binary set-level structure even if probability-mass metrics such
as circular error or opposite-value activation worsen when computed from the
decoder's structured marginals. Report this distinction explicitly.

Only after supervised tables, bootstrap analysis, calibration diagnostics, and
any structured-decoder diagnostic are stable should the LLM diagnostic be run.

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
comparison. Also copy the selected ASL parameters into the structured loss
configs before tuning `lambda_geo` or `tau`. Do not retune ASL parameters on the
test split.

The direct local command for this stage is:

```bash
python3 scripts/grid_loss_hparams.py --methods asl
```

On Sirius, prefer:

```bash
METHODS=asl sbatch --export=ALL scripts/run_sirius_tuning.sh
```

### Geometry-Aware Method Selection

The structured method configs are:

```text
configs/deberta_random_geoloss.yaml
configs/deberta_empirical_structure.yaml
configs/deberta_schwartz_geoloss.yaml
configs/deberta_schwartz_geosmooth.yaml
```

By default, these structured methods use BCE as their base loss because the
top-level configs omit `loss.base` and BCE was the stronger plain baseline in
the completed final runs. ASL-based structured variants are retained under
`configs/asl_based/` for appendix/ablation reporting.

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

The direct local command is:

```bash
python3 scripts/grid_loss_hparams.py --methods geoloss schwartz_geosmooth
```

On Sirius, prefer:

```bash
METHODS="geoloss schwartz_geosmooth" \
TUNING_OUTPUT="$PWD/results/tuning/grid_loss_hparams_geometry.csv" \
sbatch --export=ALL scripts/run_sirius_tuning.sh
```

Use `DRY_RUN=1` to inspect the planned grid and `MAX_SAMPLES` only for
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

Use one main model in the paper:

```text
Qwen/Qwen2.5-72B-Instruct
```

The 72B diagnostic uses 4-bit loading and requires at least two visible 24 GB
GPUs, or an equivalent total visible GPU memory budget. With the current
Transformers/bitsandbytes implementation, a single RTX 4090 is expected to fail
because `device_map="auto"` would need to dispatch part of the quantized model to
CPU or disk.

Run two fixed prompt conditions over the full test split:

- `definitions_only`: the model receives the 19 refined Schwartz labels and
  one-line definitions;
- `schwartz_continuum`: the model receives the same definitions plus the
  Schwartz circular order and instructions about nearby compatibility and
  opposite-value conflict.

Both prompts require strict JSON output:

```json
{"labels": ["Achievement", "Power: resources"]}
```

If no value is expressed, the required output is:

```json
{"labels": []}
```

Local/debug commands:

```bash
python3 scripts/run_qwen_llm_diagnostic.py \
  --config configs/llm_qwen_definitions_only.yaml \
  --split test \
  --max_samples 16 \
  --eval

python3 scripts/run_qwen_llm_diagnostic.py \
  --config configs/llm_qwen_schwartz_continuum.yaml \
  --split test \
  --max_samples 16 \
  --eval
```

Full UEV run:

```bash
DRY_RUN=1 bash scripts/run_uev_qwen_llm_diagnostic.sh
sbatch scripts/run_uev_qwen_llm_diagnostic.sh
```

Full Sirius run:

```bash
DRY_RUN=1 bash scripts/run_sirius_qwen_llm_diagnostic.sh
sbatch scripts/run_sirius_qwen_llm_diagnostic.sh
```

The Sirius launcher follows the Qwen2.5-72B setup used in the previous paper:
`gpu` partition, 2 GPUs, 8 CPU threads, 128 GB RAM, and 96 hours. It expects the
Sirius `.venv` to have been created with `scripts/bootstrap_slurm_venv.sh`, which
also writes the `.venv/.bootstrap_complete` marker checked by the launcher.
For 4-bit Qwen inference, the venv must include `bitsandbytes>=0.46.1`. If an
older already-bootstrapped venv fails at model loading, update it from the Sirius
login node with:

```bash
source .venv/bin/activate
python -m pip install -U "bitsandbytes>=0.46.1"
```

The UEV and Sirius launchers run both prompt conditions, resume partial JSONL
prediction files, skip completed metrics files, and then run:

```bash
python3 scripts/make_llm_diagnostic_assets.py
```

LLM predictions are written to:

```text
results/llm_diagnostic/predictions/
```

LLM metric JSONs are written to:

```text
results/llm_diagnostic/logs/
```

Paper-facing outputs are:

```text
results/analysis/paper_tables/llm_diagnostic_results.csv
results/analysis/paper_figures/figure_llm_diagnostic.svg
```

Run the sample-level paired bootstrap for the LLM diagnostic with:

```bash
python3 scripts/bootstrap_llm_diagnostic.py \
  --bootstrap_iterations 2000 \
  --output_dir results/analysis/llm_diagnostic_bootstrap

cp results/analysis/llm_diagnostic_bootstrap/llm_diagnostic_bootstrap_summary.csv \
  results/analysis/paper_tables/llm_diagnostic_bootstrap_summary.csv
cp results/analysis/llm_diagnostic_bootstrap/llm_diagnostic_sample_bootstrap.csv \
  results/analysis/paper_tables/llm_diagnostic_sample_bootstrap.csv
```

The main-paper bootstrap comparisons are:

- Qwen continuum vs Qwen definitions;
- Qwen continuum vs BCE thresholding, tested against each BCE seed;
- Qwen continuum vs BCE + Schwartz decoder, tested against each BCE seed.

The script also writes definitions-vs-supervised contrasts for appendix or
sanity checking. Metrics are `macro_f1`, `micro_f1`, `opposite_error_rate`, and
`decoder_geometry_cost`.

Report:

- Macro-F1;
- Micro-F1;
- opposite-error rate;
- decoder geometry cost;
- invalid-output rate;
- repaired-output rate;
- empty-prediction rate;
- average number of predicted labels.

Do not report Macro-AUPRC for the LLM rows unless calibrated label scores are
added later. The current diagnostic uses label-only JSON outputs, so AUPRC is
not a valid comparison.

QLoRA is intentionally out of scope for the main diagnostic. It would turn the
LLM section into another supervised training method, making the paper less
clean. If added later, treat it as appendix-only future work or a separate
experiment, not as part of the main prompt-vs-decoder diagnostic.
