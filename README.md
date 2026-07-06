# Beyond Independent Labels: Geometry-Aware Human Value Detection Along the Schwartz Continuum

This repository contains code and experiment configurations for a project on
**theory-aware multi-label human value detection**.

The project studies whether neural value classifiers can be made more coherent
with **Schwartz's circular motivational continuum**. Instead of treating the 19
refined Schwartz values as independent labels, the proposed models encode their
theoretical proximity and opposition as an output-space geometry.

## Overview

The task is sentence-level multi-label classification over the 19 refined
Schwartz values. The original attained/constrained annotations are collapsed
into binary value presence labels:

```text
value is active = attained OR constrained
```

The main hypothesis is that incorporating Schwartz-continuum geometry into
training can preserve or improve standard predictive performance while making
model errors more theoretically plausible.

## Research Questions

1. Does Schwartz-continuum geometry improve or preserve multi-label value
   detection performance compared with strong DeBERTa-v3-base baselines?
2. When the model is wrong, are its errors closer to the gold values in the
   Schwartz continuum?
3. Does the model reduce probability mass assigned to theoretically distant or
   conflicting values?
4. Does the true Schwartz geometry outperform empirical label structure and
   random circular geometry?
5. Can prompted LLMs with Schwartz definitions or continuum information match
   supervised training-time geometry?

## Planned Methods

The controlled backbone is `microsoft/deberta-v3-base`.

Planned supervised conditions:

- DeBERTa-v3-base + BCE
- DeBERTa-v3-base + Asymmetric Loss
- DeBERTa-v3-base + empirical label-structure regularization
- DeBERTa-v3-base + random circular geometry
- DeBERTa-v3-base + Schwartz GeoLoss
- DeBERTa-v3-base + Schwartz GeoSmooth

A small LLM diagnostic comparison will test:

- zero-shot value-definition prompting
- Schwartz-continuum prompting
- supervised DeBERTa-v3-base baselines
- supervised Schwartz-aware models

## Core Idea

Each value is assigned a position on the Schwartz circle. The resulting circular
distance matrix is used to penalize probability mass assigned to values far from
the gold labels.

The main proposed loss is:

```text
L = L_base + lambda * L_geo
```

where `L_base` is BCE or Asymmetric Loss, and `L_geo` measures the expected
Schwartz circular distance between gold values and predicted probability mass.

## Evaluation

Standard metrics:

- Macro-F1
- Micro-F1
- Macro-AUPRC
- Per-label F1

Schwartz-aware metrics:

- Expected circular error
- Opposite-value activation
- Neighbor-error rate
- Confusion-distance correlation

LLM-specific metric:

- Invalid-output rate

## Data

The project uses the same human value detection benchmark format as prior work
on Schwartz value detection. Raw benchmark texts are not redistributed in this
repository.

Expected local layout:

```text
data/raw/
  training/
    sentences.tsv
    labels.tsv
  validation/
    sentences.tsv
    labels.tsv
  test/
    sentences.tsv
    labels.tsv
```

Users must obtain the dataset separately and comply with the original data usage
agreement.

## Repository Structure

```text
schwartz-geometry-value-detection/
  configs/              # YAML experiment configurations
  data/                 # Local data location, not redistributed
  scripts/              # Training, evaluation, analysis, and launch scripts
  src/schwartz_value_geometry/
    data/               # Dataset loading and label collapsing
    eval/               # Multi-label metrics and statistical tests
    models/             # DeBERTa model and training utilities
    utils/              # Config, logging, and seed helpers
  tests/                # Unit tests
  results/              # Generated metrics and analysis outputs
  paper/                # Paper assets and derived tables/figures
```

Model predictions are released under `results/predictions/` as gzipped
`.jsonl.gz` files; run `gunzip -k results/predictions/*.jsonl.gz` before the
evaluation and decoding scripts, which read the raw `.jsonl`.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e ".[dev]"
```

For GPU training, install the PyTorch build appropriate for the target CUDA
environment before running the training scripts.

## Quickstart

Run the DeBERTa-v3-base BCE baseline:

```bash
python3 scripts/train_deberta.py \
  --config configs/deberta_bce.yaml \
  --seed 42
```

Run the same model with asymmetric loss:

```bash
python3 scripts/train_deberta.py \
  --config configs/deberta_asl.yaml \
  --seed 42
```

Structured-loss configs are also available:

```text
configs/deberta_random_geoloss.yaml
configs/deberta_empirical_structure.yaml
configs/deberta_schwartz_geoloss.yaml
configs/deberta_schwartz_geosmooth.yaml
```

Inspect the planned ASL/GeoLoss/GeoSmooth tuning grid:

```bash
python3 scripts/grid_loss_hparams.py --dry_run
```

Aggregate trained seed results into paper-ready CSVs:

```bash
python3 scripts/aggregate_results.py
```

Run paired seed-level bootstrap tests against BCE:

```bash
python3 scripts/bootstrap_seed_significance.py \
  --seed_level_csv results/analysis/paper_tables/seed_level_results.csv \
  --output results/analysis/paper_tables/bootstrap_seed_significance.csv \
  --baseline bce \
  --n_iterations 2000
```

Run the validation-only geometry-aware calibration diagnostic:

```bash
python3 scripts/geometry_aware_calibration.py \
  --predictions_dir results/predictions \
  --logs_dir results/logs \
  --output_dir results/analysis/geometry_calibration
```

This does not retrain models. It searches a small calibration grid on
validation predictions, applies the selected settings to test predictions, and
writes seed-level and mean/std CSVs for Pareto analysis.

Run the Schwartz structured energy decoder over the final BCE probabilities:

```bash
python3 scripts/schwartz_energy_decoder.py \
  --predictions_dir results/predictions \
  --logs_dir results/logs \
  --output_dir results/analysis/schwartz_energy_decoder
```

The decoder keeps the trained classifier fixed, but predicts the final label
set jointly. It rewards nearby Schwartz values, penalizes opposite-value
co-activation, applies a small cardinality penalty, tunes those weights on
validation, and applies the selected decoder once to test.

Run the decoder controls and ablations:

```bash
python3 scripts/schwartz_energy_decoder.py \
  --bootstrap_iterations 0 \
  --max_error_examples 0 \
  --output_dir results/analysis/schwartz_energy_decoder_controls
```

This evaluates Schwartz, random circular, and empirical co-occurrence
geometries with cardinality-only, neighbor-only, opposite-only,
neighbor+opposite, and full decoder families.

Run targeted sample-level bootstrap and error analysis:

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

The full decoder-geometry-cost bootstrap is also supported, but it is slower;
run it on the cluster by adding `decoder_geometry_cost` to
`--bootstrap_metrics`.

On UEV, submit the 2000-iteration bootstrap including geometry cost with:

```bash
sbatch scripts/run_uev_energy_decoder_bootstrap.sh
```

Inspect the command without submitting work from a login session with:

```bash
DRY_RUN=1 bash scripts/run_uev_energy_decoder_bootstrap.sh
```

After the full decoder bootstrap has been produced, run the direct
Schwartz-vs-control bootstrap:

```bash
python3 scripts/bootstrap_energy_decoder_controls.py \
  --decoder_output_dir results/analysis/schwartz_energy_decoder_bootstrap_examples_full \
  --output_dir results/analysis/schwartz_energy_decoder_control_bootstrap
```

This tests the paper-critical contrast between the Schwartz decoder and the
random/empirical controls directly. On UEV, use:

```bash
DRY_RUN=1 bash scripts/run_uev_energy_decoder_control_bootstrap.sh
sbatch scripts/run_uev_energy_decoder_control_bootstrap.sh
```

The frozen decoder configuration is documented in
`configs/schwartz_energy_decoder.yaml`.

After the decoder and control-bootstrap outputs are complete, build the
paper-ready decoder tables, SVG figures, and curated qualitative examples with:

```bash
python3 scripts/make_decoder_paper_assets.py
```

Run the Qwen2.5-72B-Instruct LLM diagnostic with two prompt conditions:

The 72B diagnostic uses 4-bit loading and requires at least two visible 24 GB
GPUs, or an equivalent total visible GPU memory budget. A single RTX 4090
cannot fit this model with the current Transformers/bitsandbytes path.

```bash
python3 scripts/run_qwen_llm_diagnostic.py \
  --config configs/llm_qwen_definitions_only.yaml \
  --split test \
  --eval

python3 scripts/run_qwen_llm_diagnostic.py \
  --config configs/llm_qwen_schwartz_continuum.yaml \
  --split test \
  --eval
```

On UEV, run both prompt conditions over the full test split with:

```bash
DRY_RUN=1 bash scripts/run_uev_qwen_llm_diagnostic.sh
sbatch scripts/run_uev_qwen_llm_diagnostic.sh
```

On Sirius, use the launcher adapted from the previous Qwen2.5-72B run:

```bash
DRY_RUN=1 bash scripts/run_sirius_qwen_llm_diagnostic.sh
sbatch scripts/run_sirius_qwen_llm_diagnostic.sh
```

The Sirius launcher expects the bootstrapped project venv at `.venv` and the
`.venv/.bootstrap_complete` marker created by `scripts/bootstrap_slurm_venv.sh`.
If an older Sirius venv fails with a missing 4-bit quantization dependency,
update it once with:

```bash
source .venv/bin/activate
python -m pip install -U "bitsandbytes>=0.46.1"
```

After both LLM runs finish, generate the paper table and SVG figure:

```bash
python3 scripts/make_llm_diagnostic_assets.py
```

Run the LLM diagnostic paired bootstrap:

```bash
python3 scripts/bootstrap_llm_diagnostic.py \
  --bootstrap_iterations 2000 \
  --output_dir results/analysis/llm_diagnostic_bootstrap

cp results/analysis/llm_diagnostic_bootstrap/llm_diagnostic_bootstrap_summary.csv \
  results/analysis/paper_tables/llm_diagnostic_bootstrap_summary.csv
cp results/analysis/llm_diagnostic_bootstrap/llm_diagnostic_sample_bootstrap.csv \
  results/analysis/paper_tables/llm_diagnostic_sample_bootstrap.csv
```

This tests Qwen continuum vs Qwen definitions, Qwen continuum vs BCE
thresholding, Qwen continuum vs the Schwartz decoder, and appendix/sanity-check
definitions-vs-supervised contrasts.

Run the Sirius development smoke job:

```bash
sbatch scripts/run_sirius_smoke.sh
```

Use `--max_samples` for a smoke run and `--eval` to evaluate the saved best
checkpoint on the test split.

## Status

Research code under active development.

## License

Apache License 2.0.
