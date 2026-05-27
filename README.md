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

Run the current DeBERTa-v3-base BCE baseline:

```bash
python3 scripts/train_deberta.py \
  --config configs/deberta_bce.yaml \
  --seed 42
```

Use `--max_samples` for a smoke run and `--eval` to evaluate the saved best
checkpoint on the test split.

## Status

Research code under active development.

## License

Apache License 2.0.
