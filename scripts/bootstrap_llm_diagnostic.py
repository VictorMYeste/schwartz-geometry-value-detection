"""Sample-level paired bootstrap tests for the Qwen LLM diagnostic."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from geometry_aware_calibration import (  # noqa: E402
    PredictionSet,
    discover_runs,
    load_prediction_jsonl,
    load_validation_thresholds,
)
from schwartz_energy_decoder import (  # noqa: E402
    EnergySetting,
    build_geometry_matrices,
    decode_and_evaluate,
)
from schwartz_value_geometry.geometry import (  # noqa: E402
    OPPOSITE_DISTANCE_THRESHOLD,
    circular_distance_matrix,
    circular_step_distance_matrix,
    validate_label_names,
)
from schwartz_value_geometry.utils.logging import get_logger  # noqa: E402

LOGGER = get_logger(__name__)

DEFAULT_METRICS = [
    "macro_f1",
    "micro_f1",
    "opposite_error_rate",
    "decoder_geometry_cost",
]
LOWER_IS_BETTER = {"opposite_error_rate", "decoder_geometry_cost"}
PAPER_COMPARISONS = {
    "qwen_continuum_vs_qwen_definitions",
    "qwen_continuum_vs_bce_thresholding",
    "qwen_continuum_vs_schwartz_decoder",
}


@dataclass(frozen=True)
class Condition:
    name: str
    family: str
    y_pred: np.ndarray
    seed: int | None = None


def parse_int_set(raw: str) -> set[int] | None:
    if not raw.strip():
        return None
    return {int(value.strip()) for value in raw.replace(",", " ").split() if value.strip()}


def parse_metric_list(raw: str) -> list[str]:
    values = [value.strip() for value in raw.replace(",", " ").split() if value.strip()]
    if not values:
        raise ValueError("At least one metric is required")
    unknown = sorted(set(values) - set(DEFAULT_METRICS))
    if unknown:
        raise ValueError(f"Unsupported LLM bootstrap metrics: {unknown}")
    return values


def _key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["text_id"]), str(row["sent_id"])


def load_llm_prediction_jsonl(
    path: Path,
    *,
    label_names: list[str] | None = None,
) -> tuple[list[tuple[str, str]], np.ndarray, np.ndarray, list[str]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"LLM prediction file is empty: {path}")

    if label_names is None:
        ordered: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for label in list(row.get("gold_labels", []) or []) + list(
                row.get("pred_labels", []) or []
            ):
                label = str(label)
                if label not in seen:
                    seen.add(label)
                    ordered.append(label)
        label_names = ordered
    validate_label_names(label_names)
    label_index = {label: idx for idx, label in enumerate(label_names)}

    keys: list[tuple[str, str]] = []
    y_true = np.zeros((len(rows), len(label_names)), dtype=np.int8)
    y_pred = np.zeros_like(y_true)
    for row_idx, row in enumerate(rows):
        keys.append(_key(row))
        for label in row.get("gold_labels", []) or []:
            y_true[row_idx, label_index[str(label)]] = 1
        for label in row.get("pred_labels", []) or []:
            label = str(label)
            if label in label_index:
                y_pred[row_idx, label_index[label]] = 1
    return keys, y_true, y_pred, label_names


def load_prediction_keys(path: Path) -> list[tuple[str, str]]:
    keys: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                keys.append(_key(json.loads(line)))
    if not keys:
        raise ValueError(f"Prediction file is empty: {path}")
    return keys


def reorder_by_keys(
    *,
    source_keys: list[tuple[str, str]],
    source_array: np.ndarray,
    target_keys: list[tuple[str, str]],
    name: str,
) -> np.ndarray:
    lookup = {key: idx for idx, key in enumerate(source_keys)}
    missing = [key for key in target_keys if key not in lookup]
    if missing:
        raise ValueError(f"{name} is missing {len(missing)} target keys; first={missing[0]}")
    return source_array[[lookup[key] for key in target_keys]]


def load_selected_decoder_rows(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing selected decoder table: {path}")
    selected = pd.read_csv(path)
    required = {
        "method",
        "seed",
        "model_slug",
        "geometry",
        "family",
        "objective",
        "top_k",
        "max_candidates",
        "max_labels",
        "threshold_factor",
        "min_prob",
        "marginal_temperature",
        "neighbor_steps",
        "random_seed",
        "alpha_neighbor",
        "beta_opposite",
        "gamma_cardinality",
    }
    missing = sorted(required - set(selected.columns))
    if missing:
        raise ValueError(f"Selected decoder table is missing columns: {missing}")
    return selected


def setting_from_row(row: pd.Series) -> EnergySetting:
    return EnergySetting(
        alpha_neighbor=float(row["alpha_neighbor"]),
        beta_opposite=float(row["beta_opposite"]),
        gamma_cardinality=float(row["gamma_cardinality"]),
    )


def bootstrap_weights(n_samples: int, n_iterations: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    weights = np.zeros((n_iterations, n_samples), dtype=np.uint16)
    for idx in range(n_iterations):
        sampled = rng.integers(0, n_samples, size=n_samples)
        weights[idx] = np.bincount(sampled, minlength=n_samples).astype(np.uint16)
    return weights.astype(np.float32, copy=False)


def _f1_from_counts(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> np.ndarray:
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) > 0)
    return np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision, dtype=float),
        where=(precision + recall) > 0,
    )


def macro_f1_scores(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> tuple[float, np.ndarray]:
    tp = ((y_true == 1) & (y_pred == 1)).astype(np.float32)
    fp = ((y_true == 0) & (y_pred == 1)).astype(np.float32)
    fn = ((y_true == 1) & (y_pred == 0)).astype(np.float32)
    obs = float(_f1_from_counts(tp.sum(axis=0), fp.sum(axis=0), fn.sum(axis=0)).mean())
    boot = _f1_from_counts(weights @ tp, weights @ fp, weights @ fn).mean(axis=1)
    return obs, boot.astype(float)


def micro_f1_scores(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> tuple[float, np.ndarray]:
    tp = ((y_true == 1) & (y_pred == 1)).sum(axis=1).astype(np.float32)
    fp = ((y_true == 0) & (y_pred == 1)).sum(axis=1).astype(np.float32)
    fn = ((y_true == 1) & (y_pred == 0)).sum(axis=1).astype(np.float32)
    obs = float(_f1_from_counts(np.asarray(tp.sum()), np.asarray(fp.sum()), np.asarray(fn.sum())))
    boot = _f1_from_counts(weights @ tp, weights @ fp, weights @ fn)
    return obs, np.asarray(boot, dtype=float)


def _fp_geometry_components(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    label_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    distances = circular_distance_matrix(label_names)
    steps = circular_step_distance_matrix(label_names)
    off_diagonal = ~np.eye(len(label_names), dtype=bool)
    n_pairs = int(off_diagonal.sum())

    fp_total = np.zeros(y_true.shape[0], dtype=np.float32)
    neighbor_fp = np.zeros_like(fp_total)
    opposite_fp = np.zeros_like(fp_total)
    confusion = np.zeros((y_true.shape[0], n_pairs), dtype=np.float32)

    for row_idx, (gold_row, pred_row) in enumerate(zip(y_true, y_pred, strict=False)):
        gold_idx = np.flatnonzero(gold_row)
        fp_idx = np.flatnonzero((pred_row == 1) & (gold_row == 0))
        fp_total[row_idx] = float(fp_idx.size)
        if gold_idx.size == 0 or fp_idx.size == 0:
            opposite_fp[row_idx] = float(fp_idx.size)
            continue
        min_steps = steps[np.ix_(gold_idx, fp_idx)].min(axis=0)
        min_distances = distances[np.ix_(gold_idx, fp_idx)].min(axis=0)
        neighbor_fp[row_idx] = float((min_steps <= 2).sum())
        opposite_fp[row_idx] = float((min_distances > OPPOSITE_DISTANCE_THRESHOLD).sum())
        local = np.zeros((len(label_names), len(label_names)), dtype=np.float32)
        for gold_label in gold_idx:
            local[gold_label, fp_idx] += 1.0
        confusion[row_idx] = local[off_diagonal]
    return fp_total, neighbor_fp, opposite_fp, confusion


def _rate(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    return np.divide(num, den, out=np.zeros_like(num, dtype=float), where=den > 0)


def _correlation_from_counts(counts: np.ndarray, label_names: list[str]) -> np.ndarray:
    distances = circular_distance_matrix(label_names)
    off_diagonal = ~np.eye(len(label_names), dtype=bool)
    x = distances[off_diagonal].astype(float)
    x_centered = x - x.mean()
    x_ss = float(np.square(x_centered).sum())

    if counts.ndim == 1:
        y = counts.astype(float)
        y_centered = y - y.mean()
        denom = np.sqrt(x_ss * np.square(y_centered).sum())
        return np.asarray(0.0 if denom == 0.0 else float((x_centered * y_centered).sum() / denom))

    y = counts.astype(float)
    y_centered = y - y.mean(axis=1, keepdims=True)
    denom = np.sqrt(x_ss * np.square(y_centered).sum(axis=1))
    num = y_centered @ x_centered
    return np.divide(num, denom, out=np.zeros_like(num, dtype=float), where=denom > 0)


def geometry_scores(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    label_names: list[str],
    weights: np.ndarray,
) -> dict[str, tuple[float, np.ndarray]]:
    fp_total, neighbor_fp, opposite_fp, confusion = _fp_geometry_components(
        y_true,
        y_pred,
        label_names=label_names,
    )
    obs_neighbor = float(_rate(np.asarray(neighbor_fp.sum()), np.asarray(fp_total.sum())))
    obs_opposite = float(_rate(np.asarray(opposite_fp.sum()), np.asarray(fp_total.sum())))
    obs_corr = float(_correlation_from_counts(confusion.sum(axis=0), label_names))
    boot_fp = weights @ fp_total
    boot_neighbor = _rate(weights @ neighbor_fp, boot_fp)
    boot_opposite = _rate(weights @ opposite_fp, boot_fp)
    boot_corr = _correlation_from_counts(weights @ confusion, label_names)
    return {
        "opposite_error_rate": (obs_opposite, boot_opposite),
        "decoder_geometry_cost": (
            obs_neighbor + obs_opposite + obs_corr,
            boot_neighbor + boot_opposite + boot_corr,
        ),
    }


def condition_scores(
    y_true: np.ndarray,
    condition: Condition,
    *,
    label_names: list[str],
    metrics: list[str],
    weights: np.ndarray,
) -> dict[str, tuple[float, np.ndarray]]:
    scores: dict[str, tuple[float, np.ndarray]] = {}
    if "macro_f1" in metrics:
        scores["macro_f1"] = macro_f1_scores(y_true, condition.y_pred, weights)
    if "micro_f1" in metrics:
        scores["micro_f1"] = micro_f1_scores(y_true, condition.y_pred, weights)
    if "opposite_error_rate" in metrics or "decoder_geometry_cost" in metrics:
        scores.update(
            geometry_scores(
                y_true,
                condition.y_pred,
                label_names=label_names,
                weights=weights,
            )
        )
    return {metric: scores[metric] for metric in metrics}


def comparison_rows(
    *,
    comparison: str,
    condition_a: Condition,
    condition_b: Condition,
    scores_a: dict[str, tuple[float, np.ndarray]],
    scores_b: dict[str, tuple[float, np.ndarray]],
    metrics: list[str],
    n_samples: int,
    n_iterations: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        obs_a, boot_a = scores_a[metric]
        obs_b, boot_b = scores_b[metric]
        deltas = boot_a - boot_b
        delta = float(obs_a - obs_b)
        ci_low = float(np.quantile(deltas, 0.025))
        ci_high = float(np.quantile(deltas, 0.975))
        p_value = min(1.0, float(2 * min((deltas <= 0).mean(), (deltas >= 0).mean())))
        lower_is_better = metric in LOWER_IS_BETTER
        improvement = -delta if lower_is_better else delta
        improvement_ci_low = -ci_high if lower_is_better else ci_low
        improvement_ci_high = -ci_low if lower_is_better else ci_high
        rows.append(
            {
                "comparison": comparison,
                "main_paper_comparison": comparison in PAPER_COMPARISONS,
                "condition_a": condition_a.name,
                "condition_b": condition_b.name,
                "condition_a_family": condition_a.family,
                "condition_b_family": condition_b.family,
                "reference_seed": condition_b.seed,
                "metric": metric,
                "higher_is_better": not lower_is_better,
                "score_a": obs_a,
                "score_b": obs_b,
                "delta_a_minus_b": delta,
                "delta_ci_low": ci_low,
                "delta_ci_high": ci_high,
                "improvement_a_over_b": improvement,
                "improvement_ci_low": improvement_ci_low,
                "improvement_ci_high": improvement_ci_high,
                "p_value_two_sided": p_value,
                "n_samples": n_samples,
                "n_iterations": n_iterations,
                "bootstrap_seed": bootstrap_seed,
                "significant_0.05": p_value < 0.05,
            }
        )
    return rows


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    out_rows: list[dict[str, Any]] = []
    group_cols = [
        "comparison",
        "main_paper_comparison",
        "condition_a",
        "condition_b_family",
        "metric",
        "higher_is_better",
    ]
    for key, group in rows.groupby(group_cols, dropna=False):
        out: dict[str, Any] = dict(zip(group_cols, key, strict=False))
        out["n_comparisons"] = int(len(group))
        for column in [
            "score_a",
            "score_b",
            "delta_a_minus_b",
            "improvement_a_over_b",
            "p_value_two_sided",
        ]:
            out[f"{column}_mean"] = float(group[column].mean())
            out[f"{column}_std"] = float(group[column].std(ddof=1)) if len(group) > 1 else 0.0
        out["significant_0.05_count"] = int(group["significant_0.05"].sum())
        out["significant_0.05_total"] = int(len(group))
        out["min_improvement_ci_low"] = float(group["improvement_ci_low"].min())
        out["max_improvement_ci_high"] = float(group["improvement_ci_high"].max())
        out["max_p_value"] = float(group["p_value_two_sided"].max())
        out_rows.append(out)
    return pd.DataFrame(out_rows)


def run(
    *,
    llm_definitions_path: Path,
    llm_continuum_path: Path,
    predictions_dir: Path,
    logs_dir: Path,
    decoder_output_dir: Path,
    output_dir: Path,
    methods: set[str],
    seeds: set[int] | None,
    model_slug: str,
    family: str,
    objective: str,
    metrics: list[str],
    n_iterations: int,
    bootstrap_seed: int,
    include_definitions_supervised: bool,
) -> dict[str, Path]:
    runs = discover_runs(
        predictions_dir=predictions_dir,
        logs_dir=logs_dir,
        methods=methods,
        seeds=seeds,
        model_slug=model_slug,
    )
    runs = [run for run in runs if run.method == "bce"]
    if not runs:
        raise ValueError("No BCE supervised runs found for LLM bootstrap comparisons")

    first_test = load_prediction_jsonl(runs[0].test_predictions)
    label_names = first_test.label_names
    llm_keys, y_true, qwen_def, _ = load_llm_prediction_jsonl(
        llm_definitions_path,
        label_names=label_names,
    )
    continuum_keys, continuum_true, qwen_cont, _ = load_llm_prediction_jsonl(
        llm_continuum_path,
        label_names=label_names,
    )
    qwen_cont = reorder_by_keys(
        source_keys=continuum_keys,
        source_array=qwen_cont,
        target_keys=llm_keys,
        name="Qwen continuum predictions",
    )
    continuum_true = reorder_by_keys(
        source_keys=continuum_keys,
        source_array=continuum_true,
        target_keys=llm_keys,
        name="Qwen continuum gold labels",
    )
    if not np.array_equal(y_true, continuum_true):
        raise ValueError("Qwen definitions and continuum files disagree on gold labels")

    weights = bootstrap_weights(y_true.shape[0], n_iterations, bootstrap_seed)
    conditions: dict[str, Condition] = {
        "qwen_definitions": Condition("qwen_definitions", "llm", qwen_def),
        "qwen_continuum": Condition("qwen_continuum", "llm", qwen_cont),
    }

    selected = load_selected_decoder_rows(
        decoder_output_dir / "schwartz_energy_decoder_selected_seed_results.csv"
    )
    selected = selected[
        (selected["method"] == "bce")
        & (selected["model_slug"] == model_slug)
        & (selected["geometry"] == "schwartz")
        & (selected["family"] == family)
        & (selected["objective"] == objective)
    ].copy()

    for supervised_run in runs:
        test = load_prediction_jsonl(supervised_run.test_predictions)
        test_keys = load_prediction_keys(supervised_run.test_predictions)
        if test.label_names != label_names:
            raise ValueError(f"Label names differ for {supervised_run.test_predictions}")
        thresholds = load_validation_thresholds(supervised_run.validation_metrics, label_names)
        standard = decode_and_evaluate(
            test,
            thresholds=thresholds,
            setting=EnergySetting(0.0, 0.0, 0.0),
            top_k=8,
            max_candidates=8,
            max_labels=5,
            threshold_factor=0.5,
            min_prob=0.01,
            marginal_temperature=1.0,
            neighbor_steps=2,
        )
        standard_pred = reorder_by_keys(
            source_keys=test_keys,
            source_array=standard.y_pred,
            target_keys=llm_keys,
            name=f"BCE thresholding seed {supervised_run.seed}",
        )
        conditions[f"bce_thresholding_seed{supervised_run.seed}"] = Condition(
            f"bce_thresholding_seed{supervised_run.seed}",
            "bce_thresholding",
            standard_pred,
            seed=supervised_run.seed,
        )

        row_match = selected[selected["seed"] == supervised_run.seed]
        if row_match.empty:
            raise ValueError(f"Missing selected Schwartz decoder row for seed {supervised_run.seed}")
        row = row_match.iloc[0]
        matrices = build_geometry_matrices(
            label_names,
            geometry="schwartz",
            neighbor_steps=int(row["neighbor_steps"]),
            random_seed=int(row["random_seed"]),
        )
        decoded = decode_and_evaluate(
            test,
            thresholds=thresholds,
            setting=setting_from_row(row),
            geometry_matrices=matrices,
            top_k=int(row["top_k"]),
            max_candidates=int(row["max_candidates"]),
            max_labels=int(row["max_labels"]),
            threshold_factor=float(row["threshold_factor"]),
            min_prob=float(row["min_prob"]),
            marginal_temperature=float(row["marginal_temperature"]),
            neighbor_steps=int(row["neighbor_steps"]),
        )
        decoded_pred = reorder_by_keys(
            source_keys=test_keys,
            source_array=decoded.y_pred,
            target_keys=llm_keys,
            name=f"Schwartz decoder seed {supervised_run.seed}",
        )
        conditions[f"schwartz_decoder_seed{supervised_run.seed}"] = Condition(
            f"schwartz_decoder_seed{supervised_run.seed}",
            "schwartz_decoder",
            decoded_pred,
            seed=supervised_run.seed,
        )

    score_cache = {
        name: condition_scores(
            y_true,
            condition,
            label_names=label_names,
            metrics=metrics,
            weights=weights,
        )
        for name, condition in conditions.items()
    }

    rows: list[dict[str, Any]] = []
    rows.extend(
        comparison_rows(
            comparison="qwen_continuum_vs_qwen_definitions",
            condition_a=conditions["qwen_continuum"],
            condition_b=conditions["qwen_definitions"],
            scores_a=score_cache["qwen_continuum"],
            scores_b=score_cache["qwen_definitions"],
            metrics=metrics,
            n_samples=y_true.shape[0],
            n_iterations=n_iterations,
            bootstrap_seed=bootstrap_seed,
        )
    )

    for supervised_run in runs:
        for reference in ("bce_thresholding", "schwartz_decoder"):
            b_name = f"{reference}_seed{supervised_run.seed}"
            rows.extend(
                comparison_rows(
                    comparison=f"qwen_continuum_vs_{reference}",
                    condition_a=conditions["qwen_continuum"],
                    condition_b=conditions[b_name],
                    scores_a=score_cache["qwen_continuum"],
                    scores_b=score_cache[b_name],
                    metrics=metrics,
                    n_samples=y_true.shape[0],
                    n_iterations=n_iterations,
                    bootstrap_seed=bootstrap_seed,
                )
            )
            if include_definitions_supervised:
                rows.extend(
                    comparison_rows(
                        comparison=f"qwen_definitions_vs_{reference}",
                        condition_a=conditions["qwen_definitions"],
                        condition_b=conditions[b_name],
                        scores_a=score_cache["qwen_definitions"],
                        scores_b=score_cache[b_name],
                        metrics=metrics,
                        n_samples=y_true.shape[0],
                        n_iterations=n_iterations,
                        bootstrap_seed=bootstrap_seed,
                    )
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    details = pd.DataFrame(rows)
    summary = summarize(details)
    paths = {
        "details": output_dir / "llm_diagnostic_sample_bootstrap.csv",
        "summary": output_dir / "llm_diagnostic_bootstrap_summary.csv",
    }
    details.to_csv(paths["details"], index=False)
    summary.to_csv(paths["summary"], index=False)
    return paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run sample-level paired bootstrap tests for LLM diagnostic outputs."
    )
    parser.add_argument(
        "--llm_definitions_path",
        default="results/llm_diagnostic/predictions/qwen_Qwen2.5-72B-Instruct_definitions_only_test.jsonl",
    )
    parser.add_argument(
        "--llm_continuum_path",
        default="results/llm_diagnostic/predictions/qwen_Qwen2.5-72B-Instruct_schwartz_continuum_test.jsonl",
    )
    parser.add_argument("--predictions_dir", default="results/predictions")
    parser.add_argument("--logs_dir", default="results/logs")
    parser.add_argument(
        "--decoder_output_dir",
        default="results/analysis/schwartz_energy_decoder_bootstrap_examples_full",
    )
    parser.add_argument(
        "--output_dir",
        default="results/analysis/llm_diagnostic_bootstrap",
    )
    parser.add_argument("--seeds", default="", help="Optional seed filter, e.g. '42 7 1701'.")
    parser.add_argument("--model_slug", default="deberta-v3-base")
    parser.add_argument("--family", default="full")
    parser.add_argument("--objective", default="pareto_99")
    parser.add_argument("--metrics", default=" ".join(DEFAULT_METRICS))
    parser.add_argument("--bootstrap_iterations", type=int, default=2000)
    parser.add_argument("--bootstrap_seed", type=int, default=42)
    parser.add_argument(
        "--skip_definitions_supervised",
        action="store_true",
        help="Only run definitions-vs-continuum plus continuum-vs-supervised comparisons.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    paths = run(
        llm_definitions_path=Path(args.llm_definitions_path),
        llm_continuum_path=Path(args.llm_continuum_path),
        predictions_dir=Path(args.predictions_dir),
        logs_dir=Path(args.logs_dir),
        decoder_output_dir=Path(args.decoder_output_dir),
        output_dir=Path(args.output_dir),
        methods={"bce"},
        seeds=parse_int_set(args.seeds),
        model_slug=str(args.model_slug),
        family=str(args.family),
        objective=str(args.objective),
        metrics=parse_metric_list(args.metrics),
        n_iterations=int(args.bootstrap_iterations),
        bootstrap_seed=int(args.bootstrap_seed),
        include_definitions_supervised=not bool(args.skip_definitions_supervised),
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
