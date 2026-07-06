# Artifact Documentation

This document summarizes the artifacts associated with the paper experiments on
**theory-aware, sentence-level Schwartz value detection** with a circular
output-space geometry, training-time geometry-aware objectives, a post-hoc
structured energy decoder, and a bounded zero-shot LLM diagnostic. It is intended
to support reproducibility, responsible release, and the ACL/EMNLP author-checklist
recommendation to document data and model artifacts.

For anonymous review, public URLs may be omitted or replaced by anonymized
supplemental material. For the camera-ready version, this document should be
included in the public GitHub repository linked from the paper.

## 1. Released Artifacts

The repository is intended to release:

- Source code for data loading, the Schwartz geometry, training objectives,
  the structured decoder, LLM inference, evaluation, and result aggregation.
- Configuration files for all reported supervised, decoder, and LLM conditions.
- Prompt templates for the zero-shot LLM diagnostic.
- Analysis scripts that generate the paper tables, figures, and qualitative
  bundles.
- Aggregate metrics and derived result tables.
- Model predictions (test and validation) as gzipped files, where permitted by the
  dataset usage agreement.

The repository is not intended to release:

- Raw benchmark texts.
- Any derived artifact that would reproduce verbatim restricted dataset text.
- Fine-tuned DeBERTa checkpoints or third-party instruction-tuned LLM weights
  (e.g., Qwen).

## 2. Dataset Documentation

### Dataset Source

The experiments use the **Touché24-ValueEval** benchmark for human value detection,
distributed by The ValuesML Team on Zenodo:

- <https://doi.org/10.5281/zenodo.13283288>

Users must obtain the dataset from the official source under the original access
conditions.

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

### Access and Redistribution

The benchmark data is governed by a restricted Data Usage Agreement. The agreement
permits scientific research use for human value detection but prohibits
redistribution or sharing of the dataset in part or in full.

Because of this restriction:

- Raw text files are not redistributed in this repository.
- Public qualitative examples use sentence identifiers and paraphrased descriptions
  instead of verbatim target sentences.
- Any derived artifact that would reproduce restricted text is not publicly
  released.
- Users with authorized dataset access can regenerate all text-dependent artifacts
  by running the released scripts locally.

### Task Definition

The prediction unit is a sentence identified by `text_id` and `sent_id`. Sentences
are grouped into documents by `text_id`, and splits are partitioned by document so
that no document is shared across splits.

The task is multi-label classification over the 19 refined Schwartz values, in
canonical continuum order:

- Self-direction: thought
- Self-direction: action
- Stimulation
- Hedonism
- Achievement
- Power: dominance
- Power: resources
- Face
- Security: personal
- Security: societal
- Tradition
- Conformity: rules
- Conformity: interpersonal
- Humility
- Benevolence: dependability
- Benevolence: caring
- Universalism: concern
- Universalism: nature
- Universalism: tolerance

The official labels distinguish attained and constrained values. The experiments
collapse both variants into value-presence labels, producing one binary target per
value:

```text
value is active = attained OR constrained
```

### Dataset Statistics Used in the Paper

Splits are partitioned by document (`text_id`); the label space is the 19 refined
values in all splits.

| Split | Documents | Sentences | ≥1 value | (% of split) | >1 value | Values / positive sentence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Training | 1,603 | 44,758 | 23,062 | 51.5% | 2,640 | 1.13 |
| Validation | 523 | 14,904 | 7,600 | 51.0% | 876 | 1.13 |
| Test | 522 | 14,569 | 7,402 | 50.8% | 901 | 1.14 |

Per-label support is strongly imbalanced, from *Humility* (0.24%) to
*Security: societal* (8.6%) over the full corpus.

### Intended Use

The dataset and derived artifacts are intended for research on value detection,
theory-aware structured prediction, output-space geometry, and model comparison
under controlled experimental conditions.

### Out-of-Scope Use

The dataset and models should not be used for individual-level profiling, inferring
a person's values or beliefs, automated moderation, surveillance, or high-stakes
decisions about people, political speakers, or social groups.

## 3. Schwartz Value Geometry Documentation

The label geometry is defined in `src/schwartz_value_geometry/geometry.py`.

- Each value `v_k` is placed on the unit circle at angle `θ_k = 2π·k/19`, in
  canonical order.
- The normalized circular distance is
  `d(v_i, v_j) = (2/19) · min(|i - j|, 19 - |i - j|)`, so `d = 0` for identical
  values and the maximum attainable distance is `18/19 ≈ 0.95`.
- **Neighbor compatibility**: pairs within two steps on the circle.
- **Opposite tension**: pairs with `d > 0.75` (at least eight steps apart).

Two control geometries are built with the same machinery to test whether any benefit
is specific to the true ordering:

- **Random circular geometry**: a seeded random permutation of the 19 values,
  preserving the circular form but destroying the theory-derived ordering.
- **Empirical co-occurrence geometry**: distances `1 - Jaccard(i, j)` from
  training-split label co-occurrence.

The distance matrix and its neighbor/opposite relations are the single source from
which every geometry-aware objective, decoder term, and theory-aware metric is
derived.

## 4. Model Artifact Documentation

### Fine-Tuned DeBERTa Models

All supervised systems share one architecture: a `microsoft/deberta-v3-base` encoder
with a linear head mapping the pooled sentence representation to 19 logits, with
sigmoid outputs. They differ only in the training objective and an optional post-hoc
decoding step.

Main training settings:

- Multi-label binary classification with sigmoid outputs.
- Optimizer: AdamW. Base hyperparameters selected once on validation by grid search
  over learning rate (`{6,7,8,9,10}×10⁻⁶`) and weight decay (`{0.10,…,0.20}`),
  yielding `1e-5` and `0.15`.
- Effective batch size 16, sequence length 1024 tokens, gradient clipping 1.0.
- Up to 30 epochs with early stopping (patience 3).
- Per-label thresholds tuned on validation by sweeping `[0, 1]` in steps of `0.01`
  to maximize each label's F1.
- Seeds: `42, 7, 1701, 11, 1984`; mean and standard deviation reported over seeds.

Training objectives:

- **BCE**: binary cross-entropy over the 19 labels (primary baseline).
- **ASL**: Asymmetric Loss (imbalance-aware). Search `γ_neg ∈ {2,3,4,5}`,
  `clip ∈ {0,0.03,0.05,0.1}`, `γ_pos = 0`.
- **GeoLoss**: base loss plus a distance-weighted penalty on probability mass placed
  far from the gold values on the circle. Search `λ ∈ {0.01,0.05,0.1,0.2}`.
- **GeoSmooth**: geometry-smoothed soft targets (a distance-aware label smoothing).
  Search `τ ∈ {0.1,0.2,0.5,1.0}`.
- Structure controls swap the distance matrix `D` for the random or empirical
  geometry (`random GeoLoss`, `empirical structure`).

Structured energy decoder:

- Keeps the trained classifier fixed and re-decodes whole label sets, combining
  classifier log-odds margins with pairwise neighbor rewards and opposite penalties
  and a cardinality penalty.
- Fixed component magnitudes: neighbor `α = 0.1`, opposite `β = 0.2`, cardinality
  `γ = 0.02`, with a two-step neighbor window; candidate pool capped at 8 and the
  decoded set at 5 labels per sentence.
- Weights selected on validation under a Pareto rule (retain validation Macro-F1
  within a small tolerance of the best, then minimize the validation geometry cost);
  the final runs keep the neighbor and opposite terms and set the cardinality term
  to `γ = 0`.

Checkpoint release policy:

- Fine-tuned weights are not redistributed in this repository.
- The released **predictions** (test and validation) allow every table to be
  reproduced without retraining. Users must obtain the official dataset separately to
  reproduce training itself.

### Zero-Shot LLM Diagnostic Configuration

The diagnostic evaluates a zero-shot instruction-tuned LLM using prompts and
configuration files, not fine-tuning:

- `Qwen/Qwen2.5-72B-Instruct`

LLM configuration notes:

- 4-bit quantization; requires at least two visible 24 GB GPUs (or an equivalent
  total visible GPU-memory budget).
- Deterministic decoding: temperature `0.0`, at most 128 new tokens.
- Two prompt conditions: **definitions-only** and **continuum** (which additionally
  describes the circular ordering and neighbor/opposite expectations).
- Both prompts require a strict JSON label list drawn only from the allowed values;
  responses that cannot be parsed without repair are counted in an
  **invalid-output rate** and contribute no labels.

LLM artifact release policy:

- Third-party model weights are not redistributed.
- The repository releases only prompts, configs, scripts, and derived outputs that
  are permitted under the dataset and model-provider terms.
- Users must follow the model provider's license and access requirements.

## 5. Evaluation and Result Artifacts

Standard metrics: Macro-F1, Micro-F1, and threshold-free Macro-AUPRC.

Theory-aware metrics introduced for this label space:

- **Expected circular error** (probability-mass, for supervised models).
- **Opposite-value activation**.
- **Opposite-error rate** and **neighbor-error rate** (label-set, for the decoder).
- **Confusion–distance correlation**.
- **Decoder geometry cost**: the composite sum of opposite-error rate, neighbor-error
  rate, and confusion–distance correlation.

Significance testing uses paired bootstrap tests (2,000 resamples): a seed-level
bootstrap against the BCE baseline for the supervised systems, and sample-level
bootstrap tests on the shared test set for the decoder-control and LLM comparisons.

The canonical final-result workflow is:

```bash
python3 scripts/aggregate_results.py
python3 scripts/make_decoder_paper_assets.py
python3 scripts/make_llm_diagnostic_assets.py
```

Main derived outputs are written to `results/analysis/paper_tables/`:

- `main_supervised_results.csv`
- `seed_level_results.csv`
- `threshold_table.csv`
- `decoder_main_results.csv`
- `decoder_control_bootstrap_summary.csv`
- `bootstrap_seed_significance.csv`
- `llm_diagnostic_results.csv`
- `qualitative_decoder_examples_selected.csv`

Model predictions are released under `results/predictions/` as gzipped `.jsonl.gz`
files (`*_test.jsonl.gz` and `*_validation_thresholds.jsonl.gz`). When sharing
result files publicly, inspect them for restricted text; if an artifact includes raw
target sentences, release the script needed to regenerate it instead of the artifact
itself.

## 6. Reproducibility Notes

To reproduce the paper results, users need:

- Authorized access to the official Touché24-ValueEval dataset.
- This repository's source code and configuration files.
- The Python environment specified by `requirements.txt` and `pyproject.toml`.
- Access to `microsoft/deberta-v3-base` and `Qwen/Qwen2.5-72B-Instruct` from Hugging
  Face.
- Sufficient GPU resources (at least two visible 24 GB GPUs for the 72B diagnostic).

Recommended setup and launch commands are documented in `README.md`,
[`INSTALL.md`](INSTALL.md), and [`EXPERIMENT_PROTOCOL.md`](EXPERIMENT_PROTOCOL.md).

Cluster launchers are provided for the Sirius and UEV environments (tuning, final
supervised runs, the energy-decoder bootstraps, and the Qwen diagnostic). The
released predictions allow the full analysis pipeline to be reproduced without
retraining, after decompressing them with
`gunzip -k results/predictions/*.jsonl.gz`.

## 7. Limitations of the Artifacts

- Scope is deliberately narrow: one dataset family (Touché24-ValueEval), English,
  sentence-level inputs, and one backbone (DeBERTa-v3-base).
- The improvement measured is label-set **coherence**, not accuracy: the decoder
  lowers theory-aware costs while leaving Macro-F1 and Micro-F1 within seed noise.
- The **decoder geometry cost** is a composite defined here rather than an
  established benchmark, and the circular operationalization (equal angular spacing,
  canonical order, opposite threshold `D > 0.75`, two-step neighbor window) is one
  reasonable choice among several.
- The decoder reranks candidate sets from the classifier's own probabilities, so it
  cannot recover values the base model never surfaces and inherits any miscalibration
  of those probabilities and thresholds.
- The LLM diagnostic uses a single model and two prompts under deterministic
  decoding; its numbers may shift with the model, prompt, or decoding settings.
- The benchmark is sparse and imbalanced; macro-averaged metrics should be
  interpreted alongside per-label results.

## 8. Ethical and Responsible Use

These artifacts are intended for transparent research on aggregate patterns in value
expression. Human value detection can support research on social, political, and
moral language, but it can also be misused to profile individuals or infer sensitive
beliefs.

Sentence-level value attributions are uncertain and culturally variable. The systems
should be read as tools for aggregate analysis and annotation support, not as
verdicts about individual speakers. Theory-aware decoding improves structural
consistency with the Schwartz taxonomy; it does not remove annotation noise,
ambiguity, or cross-cultural differences, and it should not be treated as evidence
that a person holds a value.

Researchers reusing the artifacts should:

- Respect the dataset usage agreement and base-model licenses.
- Report whether they use released predictions, regenerated predictions, or only
  configuration files.
- Document any changes to preprocessing, objectives, decoder settings, prompts,
  thresholds, or hardware.
- Inspect qualitative examples and error cases before drawing substantive
  conclusions about value expression.
