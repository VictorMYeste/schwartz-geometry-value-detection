# Beyond Independent Labels: Schwartz-Geometry Decoding for Human Value Detection

Code and experiment configurations for:

> **Beyond Independent Labels: Schwartz-Geometry Decoding for Human Value Detection**
> Víctor Yeste and Paolo Rosso, 2026
> arXiv: _preprint to be released; the link will be added here upon publication_

This repository supports the paper's experiments on **theory-aware, sentence-level
Schwartz value detection**: encoding the 19 refined Schwartz values as a circular
output-space geometry and using it as a soft inductive bias through training-time
objectives, a post-hoc structured energy decoder, and a bounded zero-shot LLM
diagnostic.

If you use this code, configurations, released predictions, or derived results,
please cite the paper. See [Citation](#citation).

---

## Contents

- [Overview](#overview)
- [What Is Included](#what-is-included)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Data](#data)
- [Schwartz Value Geometry](#schwartz-value-geometry)
- [Quickstart](#quickstart)
- [Reproducing the Paper](#reproducing-the-paper)
- [Configuration Inventory](#configuration-inventory)
- [Cluster Launchers](#cluster-launchers)
- [Final Result Extraction](#final-result-extraction)
- [Released Predictions](#released-predictions)
- [Artifact and Data Release Policy](#artifact-and-data-release-policy)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## Overview

Human value detection is commonly posed as sentence-level **multi-label
classification** over the 19 refined Schwartz values, typically predicted as
independent labels. Schwartz theory, however, describes the values as a **circular
motivational continuum** in which adjacent values are compatible and opposing
values are in tension. The paper asks whether this structure can be operationalized
as an explicit output-space geometry and used as a soft bias rather than a hard
constraint.

The original attained/constrained annotations are collapsed into one binary
presence label per value:

```text
value is active = attained OR constrained
```

On a fixed `microsoft/deberta-v3-base` classifier, the paper compares two ways of
injecting the geometry:

- **Training-time geometry-aware objectives** (`GeoLoss`, `GeoSmooth`), which add a
  circular-distance penalty or geometry-smoothed soft targets to a BCE or
  Asymmetric-Loss base.
- **A post-hoc Schwartz-aware energy decoder**, which keeps the trained classifier
  fixed and re-decodes whole label sets to reward compatible neighbors and penalize
  opposing values.

A bounded **Qwen2.5-72B-Instruct** diagnostic tests whether supplying the continuum
at inference time can match supervised structured prediction. The central finding is
that the decoder makes label sets measurably more coherent with the continuum at no
cost to Macro-F1/Micro-F1, and only for the **true** Schwartz ordering — not for a
random circular permutation or an empirical co-occurrence graph run through the
identical decoder.

---

## What Is Included

This repository contains:

- Data loading and attained/constrained label-collapsing utilities.
- The Schwartz circular geometry: angular positions, the circular-distance matrix,
  and its neighbor and opposite relations, plus random and empirical control
  geometries.
- Supervised DeBERTa-v3-base training/evaluation with BCE and Asymmetric Loss.
- Geometry-aware training objectives: `GeoLoss` and `GeoSmooth`.
- The post-hoc Schwartz-aware structured **energy decoder** and its controls.
- Theory-aware evaluation metrics and paired bootstrap significance tests.
- A zero-shot Qwen2.5-72B-Instruct diagnostic with definitions-only and continuum
  prompts.
- Slurm launchers for the Sirius and UEV cluster environments.
- Analysis scripts that generate the paper tables, figures, and qualitative
  examples.
- Released model predictions (test and validation) as gzipped artifacts.
- Artifact documentation for responsible release and reproducibility.

This repository does **not** redistribute the benchmark texts or fine-tuned model
weights. See [Data](#data) and
[Artifact and Data Release Policy](#artifact-and-data-release-policy).

---

## Repository Structure

```text
schwartz-geometry-value-detection/
  configs/                    # YAML configs for all supervised, decoder, and LLM conditions
    asl_based/                # Geometry-aware objectives on the Asymmetric-Loss base
  data/
    raw/                      # Expected location for the restricted dataset files
  scripts/
    train_deberta.py          # Supervised DeBERTa training/evaluation
    eval_deberta.py           # Evaluation of saved predictions
    schwartz_energy_decoder.py # Structured energy decoder, controls, and ablations
    geometry_aware_calibration.py
    grid_deberta_hparams.py   # Backbone hyperparameter grid
    grid_loss_hparams.py      # ASL/GeoLoss/GeoSmooth tuning grid
    aggregate_results.py      # Seed-level -> paper-ready CSVs
    bootstrap_seed_significance.py
    bootstrap_energy_decoder_controls.py
    bootstrap_llm_diagnostic.py
    compute_decoder_edit_rate.py
    run_qwen_llm_diagnostic.py # Qwen2.5-72B zero-shot diagnostic
    eval_llm_diagnostic.py
    make_decoder_paper_assets.py
    make_llm_diagnostic_assets.py
    run_sirius_*.sh           # Sirius Slurm launchers
    run_uev_*.sh              # UEV Slurm launchers
    bootstrap_slurm_venv.sh   # One-time cluster venv bootstrap
  src/schwartz_value_geometry/
    geometry.py               # Circular geometry, controls, distance/neighbor/opposite
    data/                     # Dataset loading and label collapsing
    models/                   # DeBERTa model, losses (BCE/ASL/GeoLoss/GeoSmooth), training
    eval/                     # Multi-label + theory-aware metrics, paired bootstrap tests
    llm/                      # Prompt templates and LLM client/parsing
    utils/                    # Config, logging, and seed helpers
  tests/                      # Unit tests
  results/
    predictions/              # Released model predictions (gzipped .jsonl.gz)
    analysis/paper_tables/    # Derived tables backing the paper
  paper/                      # Paper source and derived tables/figures
  INSTALL.md                  # Environment setup
  EXPERIMENT_PROTOCOL.md      # End-to-end experiment protocol
  artifact_documentation.md   # Dataset/model/artifact documentation
```

Model predictions are released under `results/predictions/` as gzipped
`.jsonl.gz` files; run `gunzip -k results/predictions/*.jsonl.gz` before the
evaluation and decoding scripts, which read the raw `.jsonl`.

---

## Installation

For full setup details, see [`INSTALL.md`](INSTALL.md).

Minimal local setup:

```bash
git clone https://github.com/VictorMYeste/schwartz-geometry-value-detection.git
cd schwartz-geometry-value-detection

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e ".[dev]"
```

Recommended Python version: `>=3.11`.

For GPU training, install the PyTorch build appropriate for the target CUDA
environment before running the training scripts. The Qwen2.5-72B diagnostic uses
4-bit loading and requires at least two visible 24 GB GPUs (or an equivalent total
visible GPU-memory budget).

Optional Hugging Face authentication is recommended for faster model downloads:

```bash
export HF_TOKEN=your_token_here
```

---

## Data

The experiments use the **Touché24-ValueEval** benchmark for human value detection,
distributed by The ValuesML Team under a restricted **Data Usage Agreement**:

- Zenodo: <https://doi.org/10.5281/zenodo.13283288>

The agreement permits scientific research use for human value detection but
prohibits redistribution of the dataset in part or in full, so the raw texts are
**not** included in this repository. Users must obtain the dataset separately from
the official organizers and follow the original access conditions.

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

Each split provides sentence identifiers (`text_id`, `sent_id`), sentence order,
sentence text, and the value-label files. Sentences are grouped into documents by
`text_id`; splits are partitioned by document so that no document is shared across
splits.

---

## Schwartz Value Geometry

The 19 refined values are placed in canonical order on the unit circle, value
`v_k` at angle `θ_k = 2π·k/19`. The normalized circular distance between two
values is the shorter arc between their positions:

```text
d(v_i, v_j) = (2/19) * min(|i - j|, 19 - |i - j|)
```

so `d = 0` for identical values and `d → 1` for near-antipodal values (the maximum
attainable distance is `18/19 ≈ 0.95`). From this single distance matrix the code
derives two relations reused by the objectives, the decoder, and the metrics:

- **Neighbor compatibility**: value pairs within two steps on the circle.
- **Opposite tension**: value pairs with `d > 0.75` (at least eight steps apart).

To test whether any benefit is specific to the *true* structure, the same machinery
builds two controls: a **random circular geometry** (a seeded permutation of the 19
values) and an **empirical co-occurrence geometry** (`1 - Jaccard` from training-split
label co-occurrence). All geometries are implemented in
`src/schwartz_value_geometry/geometry.py`.

---

## Quickstart

### Smoke Test

Run a small DeBERTa training pass without persisting full outputs:

```bash
python3 scripts/train_deberta.py \
  --config configs/deberta_bce.yaml \
  --seed 42 \
  --max_samples 16 \
  --dry_run
```

On Sirius, the development smoke job is:

```bash
sbatch scripts/run_sirius_smoke.sh
```

### Train One Model

```bash
python3 scripts/train_deberta.py \
  --config configs/deberta_bce.yaml \
  --seed 42 \
  --eval
```

Swap the config for any supervised condition (`deberta_asl.yaml`,
`deberta_schwartz_geoloss.yaml`, `deberta_schwartz_geosmooth.yaml`,
`deberta_random_geoloss.yaml`, `deberta_empirical_structure.yaml`).

### Run the Structured Energy Decoder

```bash
python3 scripts/schwartz_energy_decoder.py \
  --predictions_dir results/predictions \
  --logs_dir results/logs \
  --output_dir results/analysis/schwartz_energy_decoder
```

The decoder keeps the trained classifier fixed and predicts the final label set
jointly: it rewards nearby Schwartz values, penalizes opposite-value co-activation,
applies a small cardinality penalty, tunes those weights on validation under a
Pareto rule (retain validation Macro-F1, then minimize the geometry cost), and
applies the selected decoder once to test.

### Run One LLM Diagnostic Condition

```bash
python3 scripts/run_qwen_llm_diagnostic.py \
  --config configs/llm_qwen_definitions_only.yaml \
  --split test \
  --eval
```

---

## Reproducing the Paper

The controlled backbone is `microsoft/deberta-v3-base`. Each supervised
configuration is run with five seeds (`42, 7, 1701, 11, 1984`); mean and standard
deviation are reported over seeds. The full protocol is documented in
[`EXPERIMENT_PROTOCOL.md`](EXPERIMENT_PROTOCOL.md).

### 1. Prepare Environment and Data

```bash
python -m pip install -r requirements.txt
python -m pip install -e ".[dev]"
# place the Touché24-ValueEval files under data/raw/ (see Data)
```

If you use the released predictions instead of retraining, decompress them first:

```bash
gunzip -k results/predictions/*.jsonl.gz
```

### 2. Train Supervised Conditions

Run each config over the five reported seeds:

```bash
for seed in 42 7 1701 11 1984; do
  python3 scripts/train_deberta.py --config configs/deberta_bce.yaml --seed $seed --eval
done
```

Repeat for `deberta_asl.yaml`, `deberta_schwartz_geoloss.yaml`,
`deberta_schwartz_geosmooth.yaml`, `deberta_random_geoloss.yaml`, and
`deberta_empirical_structure.yaml`. The `configs/asl_based/` variants repeat the
geometry-aware objectives on the Asymmetric-Loss base (Appendix).

### 3. Aggregate Supervised Results and Significance

```bash
python3 scripts/aggregate_results.py
python3 scripts/bootstrap_seed_significance.py \
  --seed_level_csv results/analysis/paper_tables/seed_level_results.csv \
  --output results/analysis/paper_tables/bootstrap_seed_significance.csv \
  --baseline bce \
  --n_iterations 2000
```

### 4. Run the Decoder, Controls, and Direct Contrast

```bash
# Schwartz, random, and empirical geometries across decoder families
python3 scripts/schwartz_energy_decoder.py \
  --bootstrap_iterations 0 --max_error_examples 0 \
  --output_dir results/analysis/schwartz_energy_decoder_controls

# Sample-level bootstrap incl. decoder geometry cost (heavy; run on the cluster)
sbatch scripts/run_uev_energy_decoder_bootstrap.sh

# Direct Schwartz-vs-control contrast (the paper-critical test)
python3 scripts/bootstrap_energy_decoder_controls.py \
  --decoder_output_dir results/analysis/schwartz_energy_decoder_bootstrap_examples_full \
  --output_dir results/analysis/schwartz_energy_decoder_control_bootstrap
```

### 5. Run the LLM Diagnostic

```bash
python3 scripts/run_qwen_llm_diagnostic.py --config configs/llm_qwen_definitions_only.yaml --split test --eval
python3 scripts/run_qwen_llm_diagnostic.py --config configs/llm_qwen_schwartz_continuum.yaml --split test --eval
python3 scripts/bootstrap_llm_diagnostic.py --bootstrap_iterations 2000 \
  --output_dir results/analysis/llm_diagnostic_bootstrap
```

### 6. Build Paper Tables and Figures

```bash
python3 scripts/make_decoder_paper_assets.py
python3 scripts/make_llm_diagnostic_assets.py
```

---

## Configuration Inventory

### Supervised (BCE base)

```text
configs/deberta_bce.yaml
configs/deberta_asl.yaml
configs/deberta_empirical_structure.yaml
configs/deberta_random_geoloss.yaml
configs/deberta_schwartz_geoloss.yaml
configs/deberta_schwartz_geosmooth.yaml
```

### Geometry-Aware Objectives (Asymmetric-Loss base, Appendix)

```text
configs/asl_based/deberta_empirical_structure.yaml
configs/asl_based/deberta_random_geoloss.yaml
configs/asl_based/deberta_schwartz_geoloss.yaml
configs/asl_based/deberta_schwartz_geosmooth.yaml
```

### Structured Energy Decoder

```text
configs/schwartz_energy_decoder.yaml
```

### Zero-Shot LLM Diagnostic

```text
configs/llm_qwen_definitions_only.yaml
configs/llm_qwen_schwartz_continuum.yaml
```

---

## Cluster Launchers

The repository includes Slurm launchers used during the project. They are
cluster-specific templates and should be inspected before submission, because the
config loops may be edited for resumed or partial runs. Every launcher accepts a
dry run, e.g. `DRY_RUN=1 bash scripts/run_uev_final_supervised.sh`.

Bootstrap the cluster environment once:

```bash
sbatch scripts/bootstrap_slurm_venv.sh
```

### Sirius Cluster

```bash
sbatch scripts/run_sirius_tuning.sh
sbatch scripts/run_sirius_final_supervised.sh
sbatch scripts/run_sirius_qwen_llm_diagnostic.sh
```

### UEV Cluster

```bash
sbatch scripts/run_uev_tuning.sh
sbatch scripts/run_uev_final_supervised.sh
sbatch scripts/run_uev_energy_decoder_bootstrap.sh
sbatch scripts/run_uev_energy_decoder_control_bootstrap.sh
sbatch scripts/run_uev_qwen_llm_diagnostic.sh
```

Notes:

- The launchers expect the bootstrapped project venv at `.venv` and the
  `.venv/.bootstrap_complete` marker created by `scripts/bootstrap_slurm_venv.sh`.
- Large LLM launchers require Hugging Face access to Qwen2.5-72B-Instruct.
- Resumability is implemented through prediction/metrics file existence checks.

---

## Final Result Extraction

The canonical final-analysis commands are:

```bash
python3 scripts/aggregate_results.py
python3 scripts/make_decoder_paper_assets.py
python3 scripts/make_llm_diagnostic_assets.py
```

Derived tables are written under `results/analysis/paper_tables/`, including:

```text
results/analysis/paper_tables/main_supervised_results.csv
results/analysis/paper_tables/seed_level_results.csv
results/analysis/paper_tables/threshold_table.csv
results/analysis/paper_tables/decoder_main_results.csv
results/analysis/paper_tables/decoder_control_bootstrap_summary.csv
results/analysis/paper_tables/bootstrap_seed_significance.csv
results/analysis/paper_tables/llm_diagnostic_results.csv
results/analysis/paper_tables/qualitative_decoder_examples_selected.csv
```

Qualitative example files can contain restricted benchmark text. Only paraphrased
or identifier-based examples are released publicly; do not redistribute files with
verbatim target sentences unless the dataset usage agreement permits it.

---

## Released Predictions

This repository does not redistribute fine-tuned model weights. Instead, it releases
the **model predictions** needed to reproduce every table without retraining, under
`results/predictions/`:

- `*_test.jsonl.gz` — test-split predictions (`text_id`, `sent_id`, `gold_labels`,
  `pred_labels`, `probabilities`) for each condition and seed.
- `*_validation_thresholds.jsonl.gz` — validation-split predictions used to tune the
  per-label thresholds; the resulting threshold values are also in
  `results/analysis/paper_tables/threshold_table.csv`.

Decompress before running the analysis scripts:

```bash
gunzip -k results/predictions/*.jsonl.gz
```

---

## Artifact and Data Release Policy

For detailed artifact documentation, see
[`artifact_documentation.md`](artifact_documentation.md).

Publicly releasable artifacts:

- Code.
- Configurations.
- Prompt templates.
- Aggregate metrics and derived analysis tables.
- Model predictions (test and validation), released as gzipped files.

Not publicly redistributed:

- Raw benchmark texts and any artifact reproducing verbatim restricted text.
- Fine-tuned model weights and third-party LLM weights.

Users must obtain the dataset and gated models separately and comply with their
licenses and access terms.

---

## Citation

If you use this repository, please cite the paper (preprint to be released):

```bibtex
@misc{yeste2026schwartzgeometry,
  title  = {Beyond Independent Labels: Schwartz-Geometry Decoding for Human Value Detection},
  author = {Víctor Yeste and Paolo Rosso},
  year   = {2026},
  note   = {Preprint; arXiv identifier to be added upon publication}
}
```

---

## License

The code in this repository is released under the **Apache License 2.0**.
See [`LICENSE`](LICENSE) for details.

This license does not grant any rights over the underlying benchmark data or
third-party model weights. Please respect the corresponding dataset, model, and
software licenses.

---

## Contact

For questions, open a GitHub issue or contact:

Víctor Yeste — vicyesmo [at] upv [dot] es
