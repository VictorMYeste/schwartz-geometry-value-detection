"""Validation-tuned Schwartz structured energy decoder.

This script uses saved validation/test probabilities from a trained classifier
and replaces independent thresholding with a structured label-set decoder.
Model weights are not updated.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import dataclass
from functools import lru_cache
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
    METRIC_COLUMNS,
    OBJECTIVE_TOLERANCES,
    PredictionSet,
    discover_runs,
    load_prediction_jsonl,
    load_validation_thresholds,
    parse_float_list,
)
from schwartz_value_geometry.data.dataset import load_split  # noqa: E402
from schwartz_value_geometry.eval.metrics import (  # noqa: E402
    binarize_probs,
    compute_all_metrics,
    compute_confusion_distance_correlation,
    compute_global_metrics,
    compute_neighbor_error_rate,
    compute_opposite_error_rate,
    macro_f1_from_arrays,
)
from schwartz_value_geometry.eval.stats import paired_bootstrap_delta  # noqa: E402
from schwartz_value_geometry.geometry import (  # noqa: E402
    OPPOSITE_DISTANCE_THRESHOLD,
    circular_distance_matrix,
    circular_step_distance_matrix,
    distance_matrix_for_order,
    empirical_cooccurrence_distance_matrix,
    random_circular_order,
)
from schwartz_value_geometry.utils.logging import get_logger  # noqa: E402

LOGGER = get_logger(__name__)

DECODER_GEOMETRIES = {"schwartz", "random", "empirical"}
DECODER_FAMILIES = {
    "cardinality_only",
    "neighbor_only",
    "opposite_only",
    "neighbor_opposite",
    "full",
}

BINARY_BOOTSTRAP_METRICS = [
    "macro_f1",
    "micro_f1",
    "neighbor_error_rate",
    "opposite_error_rate",
    "confusion_distance_correlation",
    "decoder_geometry_cost",
]

LOWER_IS_BETTER = {
    "neighbor_error_rate",
    "opposite_error_rate",
    "confusion_distance_correlation",
    "decoder_geometry_cost",
}


@dataclass(frozen=True)
class EnergySetting:
    alpha_neighbor: float
    beta_opposite: float
    gamma_cardinality: float

    @property
    def is_standard(self) -> bool:
        return (
            self.alpha_neighbor == 0.0
            and self.beta_opposite == 0.0
            and self.gamma_cardinality == 0.0
        )


@dataclass(frozen=True)
class GeometryMatrices:
    name: str
    neighbor_similarity: np.ndarray
    opposite_mask: np.ndarray
    distance_matrix: np.ndarray


@dataclass(frozen=True)
class DecodedOutputs:
    metrics: dict[str, Any]
    y_pred: np.ndarray
    y_scores: np.ndarray


def _logit(values: np.ndarray, *, eps: float = 1e-7) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=float), eps, 1.0 - eps)
    return np.log(values / (1.0 - values))


@lru_cache(maxsize=None)
def subset_masks(n_candidates: int, max_labels: int) -> tuple[np.ndarray, np.ndarray]:
    """Return binary subset masks and cardinalities for a candidate size."""
    if n_candidates < 0:
        raise ValueError("n_candidates must be non-negative")
    effective_max = n_candidates if max_labels <= 0 else min(max_labels, n_candidates)
    masks: list[np.ndarray] = []
    for size in range(effective_max + 1):
        for combo in itertools.combinations(range(n_candidates), size):
            mask = np.zeros(n_candidates, dtype=float)
            if combo:
                mask[list(combo)] = 1.0
            masks.append(mask)
    if not masks:
        arr = np.zeros((1, n_candidates), dtype=float)
    else:
        arr = np.vstack(masks).astype(float)
    return arr, arr.sum(axis=1)


def _neighbor_similarity_from_distances(
    distances: np.ndarray,
    *,
    neighbor_steps: int,
) -> np.ndarray:
    """Convert a distance matrix into a soft near-neighbor compatibility matrix."""
    distances = np.asarray(distances, dtype=float)
    if neighbor_steps <= 0:
        return np.zeros_like(distances, dtype=float)
    n_labels = distances.shape[0]
    max_neighbor_distance = 2.0 * float(neighbor_steps) / float(n_labels)
    neighbor = np.zeros_like(distances, dtype=float)
    mask = (distances > 0.0) & (distances <= max_neighbor_distance + 1e-12)
    neighbor[mask] = 1.0 - (distances[mask] / (max_neighbor_distance + 1e-12))
    np.fill_diagonal(neighbor, 0.0)
    return neighbor


def schwartz_geometry_matrices(
    label_names: list[str],
    *,
    neighbor_steps: int,
) -> GeometryMatrices:
    """Build geometry matrices from the canonical Schwartz circle."""
    step_distances = circular_step_distance_matrix(label_names)
    distances = circular_distance_matrix(label_names)
    neighbor = np.zeros_like(distances, dtype=float)
    if neighbor_steps > 0:
        neighbor_mask = (step_distances > 0) & (step_distances <= neighbor_steps)
        neighbor[neighbor_mask] = (
            (neighbor_steps + 1 - step_distances[neighbor_mask]) / neighbor_steps
        )
    opposite = (distances > OPPOSITE_DISTANCE_THRESHOLD).astype(float)
    np.fill_diagonal(neighbor, 0.0)
    np.fill_diagonal(opposite, 0.0)
    return GeometryMatrices(
        name="schwartz",
        neighbor_similarity=neighbor,
        opposite_mask=opposite,
        distance_matrix=distances,
    )


def random_geometry_matrices(
    label_names: list[str],
    *,
    neighbor_steps: int,
    random_seed: int,
) -> GeometryMatrices:
    """Build a fixed random circular geometry negative-control matrix."""
    order = random_circular_order(label_names, seed=random_seed)
    distances = distance_matrix_for_order(label_names, order)
    neighbor = _neighbor_similarity_from_distances(
        distances,
        neighbor_steps=neighbor_steps,
    )
    opposite = (distances > OPPOSITE_DISTANCE_THRESHOLD).astype(float)
    np.fill_diagonal(opposite, 0.0)
    return GeometryMatrices(
        name="random",
        neighbor_similarity=neighbor,
        opposite_mask=opposite,
        distance_matrix=distances,
    )


def empirical_geometry_matrices(
    label_names: list[str],
    *,
    neighbor_steps: int,
    empirical_labels: np.ndarray,
    empirical_metric: str,
) -> GeometryMatrices:
    """Build a data-driven label-structure control from training co-occurrence."""
    del neighbor_steps
    distances = empirical_cooccurrence_distance_matrix(
        empirical_labels,
        metric=empirical_metric,
    )
    neighbor = 1.0 - distances
    np.fill_diagonal(neighbor, 0.0)
    opposite = (distances > OPPOSITE_DISTANCE_THRESHOLD).astype(float)
    np.fill_diagonal(opposite, 0.0)
    return GeometryMatrices(
        name="empirical",
        neighbor_similarity=neighbor,
        opposite_mask=opposite,
        distance_matrix=distances,
    )


def build_geometry_matrices(
    label_names: list[str],
    *,
    geometry: str,
    neighbor_steps: int,
    random_seed: int,
    empirical_labels: np.ndarray | None = None,
    empirical_metric: str = "jaccard",
) -> GeometryMatrices:
    """Build geometry-control matrices by name."""
    geometry = geometry.strip().lower()
    if geometry == "schwartz":
        return schwartz_geometry_matrices(label_names, neighbor_steps=neighbor_steps)
    if geometry == "random":
        return random_geometry_matrices(
            label_names,
            neighbor_steps=neighbor_steps,
            random_seed=random_seed,
        )
    if geometry == "empirical":
        if empirical_labels is None:
            empirical_labels = load_empirical_training_labels(label_names)
        return empirical_geometry_matrices(
            label_names,
            neighbor_steps=neighbor_steps,
            empirical_labels=empirical_labels,
            empirical_metric=empirical_metric,
        )
    raise ValueError(f"Unknown decoder geometry: {geometry}")


def load_empirical_training_labels(label_names: list[str]) -> np.ndarray:
    """Load training labels in the prediction label order for empirical controls."""
    training_df = load_split("training")
    missing = [label for label in label_names if label not in training_df.columns]
    if missing:
        raise ValueError(f"Training split is missing labels for empirical decoder: {missing}")
    return training_df[label_names].to_numpy(dtype=float)


def build_pairwise_matrix(
    label_names: list[str],
    *,
    alpha_neighbor: float,
    beta_opposite: float,
    neighbor_steps: int,
    geometry: str = "schwartz",
    random_seed: int = 42,
    empirical_labels: np.ndarray | None = None,
    empirical_metric: str = "jaccard",
) -> np.ndarray:
    """Build pairwise label compatibility matrix for the energy decoder."""
    matrices = build_geometry_matrices(
        label_names,
        geometry=geometry,
        neighbor_steps=neighbor_steps,
        random_seed=random_seed,
        empirical_labels=empirical_labels,
        empirical_metric=empirical_metric,
    )
    return pairwise_from_matrices(
        matrices,
        alpha_neighbor=alpha_neighbor,
        beta_opposite=beta_opposite,
    )


def pairwise_from_matrices(
    matrices: GeometryMatrices,
    *,
    alpha_neighbor: float,
    beta_opposite: float,
) -> np.ndarray:
    """Apply decoder weights to precomputed geometry matrices."""
    pairwise = (
        float(alpha_neighbor) * matrices.neighbor_similarity
        - float(beta_opposite) * matrices.opposite_mask
    )
    np.fill_diagonal(pairwise, 0.0)
    return pairwise


def select_candidate_indices(
    probs: np.ndarray,
    thresholds: np.ndarray,
    *,
    top_k: int,
    max_candidates: int,
    threshold_factor: float,
    min_prob: float,
) -> np.ndarray:
    """Choose a small candidate set while preserving threshold-positive labels."""
    probs = np.asarray(probs, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    utility = _logit(probs) - _logit(thresholds)

    base_positive = set(np.flatnonzero(probs >= thresholds).tolist())
    relaxed = set(np.flatnonzero(probs >= thresholds * threshold_factor).tolist())
    probable = set(np.flatnonzero(probs >= min_prob).tolist())
    top_count = min(max(top_k, 0), probs.size)
    top = set(np.argsort(-probs, kind="mergesort")[:top_count].tolist())
    candidates = base_positive | relaxed | probable | top

    if not candidates:
        return np.asarray([], dtype=int)
    if max_candidates <= 0 or len(candidates) <= max_candidates:
        return np.asarray(
            sorted(candidates, key=lambda idx: (-utility[idx], -probs[idx], idx)),
            dtype=int,
        )

    ordered_base = sorted(
        base_positive,
        key=lambda idx: (-utility[idx], -probs[idx], idx),
    )
    keep: list[int] = ordered_base[:max_candidates]
    kept = set(keep)
    remaining = sorted(
        candidates - kept,
        key=lambda idx: (-utility[idx], -probs[idx], idx),
    )
    for idx in remaining:
        if len(keep) >= max_candidates:
            break
        keep.append(idx)
        kept.add(idx)
    return np.asarray(keep, dtype=int)


def decode_one(
    probs: np.ndarray,
    thresholds: np.ndarray,
    *,
    setting: EnergySetting,
    pairwise: np.ndarray,
    top_k: int,
    max_candidates: int,
    max_labels: int,
    threshold_factor: float,
    min_prob: float,
    marginal_temperature: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode one sample and return binary predictions plus structured marginals."""
    n_labels = probs.size
    if setting.is_standard:
        pred = (probs >= thresholds).astype(int)
        return pred, probs.copy()

    candidates = select_candidate_indices(
        probs,
        thresholds,
        top_k=top_k,
        max_candidates=max_candidates,
        threshold_factor=threshold_factor,
        min_prob=min_prob,
    )
    if candidates.size == 0:
        return np.zeros(n_labels, dtype=int), np.zeros(n_labels, dtype=float)

    utility = _logit(probs[candidates]) - _logit(thresholds[candidates])
    local_pairwise = pairwise[np.ix_(candidates, candidates)]
    masks, cardinalities = subset_masks(int(candidates.size), int(max_labels))
    pair_scores = 0.5 * np.sum((masks @ local_pairwise) * masks, axis=1)
    cardinality_penalty = setting.gamma_cardinality * np.maximum(cardinalities - 1.0, 0.0)
    scores = masks @ utility + pair_scores - cardinality_penalty

    best_idx = int(np.argmax(scores))
    local_pred = masks[best_idx].astype(int)
    pred = np.zeros(n_labels, dtype=int)
    pred[candidates] = local_pred

    marginals = np.zeros(n_labels, dtype=float)
    if marginal_temperature <= 0.0:
        marginals[candidates] = local_pred.astype(float)
    else:
        scaled = (scores - float(scores.max())) / float(marginal_temperature)
        weights = np.exp(scaled)
        weights /= weights.sum()
        marginals[candidates] = weights @ masks
    return pred, marginals


def decode_predictions(
    predictions: PredictionSet,
    *,
    thresholds: np.ndarray,
    setting: EnergySetting,
    geometry_matrices: GeometryMatrices | None = None,
    top_k: int,
    max_candidates: int,
    max_labels: int,
    threshold_factor: float,
    min_prob: float,
    marginal_temperature: float,
    neighbor_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode all samples for one setting."""
    if setting.is_standard:
        return (
            binarize_probs(predictions.y_probs, threshold=thresholds),
            predictions.y_probs.copy(),
        )

    if geometry_matrices is None:
        geometry_matrices = build_geometry_matrices(
            predictions.label_names,
            geometry="schwartz",
            neighbor_steps=neighbor_steps,
            random_seed=42,
        )
    pairwise = pairwise_from_matrices(
        geometry_matrices,
        alpha_neighbor=setting.alpha_neighbor,
        beta_opposite=setting.beta_opposite,
    )
    y_pred = np.zeros_like(predictions.y_true, dtype=int)
    y_scores = np.zeros_like(predictions.y_probs, dtype=float)
    for row_idx, probs in enumerate(predictions.y_probs):
        pred_row, score_row = decode_one(
            probs,
            thresholds,
            setting=setting,
            pairwise=pairwise,
            top_k=top_k,
            max_candidates=max_candidates,
            max_labels=max_labels,
            threshold_factor=threshold_factor,
            min_prob=min_prob,
            marginal_temperature=marginal_temperature,
        )
        y_pred[row_idx] = pred_row
        y_scores[row_idx] = score_row
    return y_pred, y_scores


def decoder_geometry_cost(metrics: dict[str, Any]) -> float:
    """Binary-output geometry cost used only for validation-side selection."""
    return float(
        metrics["opposite_error_rate"]
        + metrics["neighbor_error_rate"]
        + metrics["confusion_distance_correlation"]
    )


def evaluate_setting(
    predictions: PredictionSet,
    *,
    thresholds: np.ndarray,
    setting: EnergySetting,
    geometry_matrices: GeometryMatrices | None = None,
    top_k: int,
    max_candidates: int,
    max_labels: int,
    threshold_factor: float,
    min_prob: float,
    marginal_temperature: float,
    neighbor_steps: int,
) -> dict[str, Any]:
    """Decode and compute all metrics for one setting."""
    return decode_and_evaluate(
        predictions,
        thresholds=thresholds,
        setting=setting,
        geometry_matrices=geometry_matrices,
        top_k=top_k,
        max_candidates=max_candidates,
        max_labels=max_labels,
        threshold_factor=threshold_factor,
        min_prob=min_prob,
        marginal_temperature=marginal_temperature,
        neighbor_steps=neighbor_steps,
    ).metrics


def decode_and_evaluate(
    predictions: PredictionSet,
    *,
    thresholds: np.ndarray,
    setting: EnergySetting,
    geometry_matrices: GeometryMatrices | None = None,
    top_k: int,
    max_candidates: int,
    max_labels: int,
    threshold_factor: float,
    min_prob: float,
    marginal_temperature: float,
    neighbor_steps: int,
) -> DecodedOutputs:
    """Decode predictions and retain metrics, binary labels, and scores."""
    y_pred, y_scores = decode_predictions(
        predictions,
        thresholds=thresholds,
        setting=setting,
        geometry_matrices=geometry_matrices,
        top_k=top_k,
        max_candidates=max_candidates,
        max_labels=max_labels,
        threshold_factor=threshold_factor,
        min_prob=min_prob,
        marginal_temperature=marginal_temperature,
        neighbor_steps=neighbor_steps,
    )
    metrics = compute_all_metrics(
        predictions.y_true,
        y_pred,
        y_scores,
        label_names=predictions.label_names,
    )
    metrics["decoder_geometry_cost"] = decoder_geometry_cost(metrics)
    metrics["n_predictions"] = float(y_pred.sum())
    metrics["prediction_rate"] = float(y_pred.mean()) if y_pred.size else 0.0
    return DecodedOutputs(metrics=metrics, y_pred=y_pred, y_scores=y_scores)


def scalar_metric_row(metrics: dict[str, Any], *, prefix: str = "") -> dict[str, float]:
    row = {f"{prefix}{metric}": float(metrics[metric]) for metric in METRIC_COLUMNS}
    row[f"{prefix}decoder_geometry_cost"] = float(metrics["decoder_geometry_cost"])
    row[f"{prefix}n_predictions"] = float(metrics["n_predictions"])
    row[f"{prefix}prediction_rate"] = float(metrics["prediction_rate"])
    return row


def setting_row(setting: EnergySetting) -> dict[str, float]:
    return {
        "alpha_neighbor": float(setting.alpha_neighbor),
        "beta_opposite": float(setting.beta_opposite),
        "gamma_cardinality": float(setting.gamma_cardinality),
    }


def build_grid(
    *,
    alpha_values: list[float],
    beta_values: list[float],
    gamma_values: list[float],
) -> list[EnergySetting]:
    settings: list[EnergySetting] = []
    seen: set[tuple[float, float, float]] = set()
    for alpha in [0.0, *alpha_values]:
        for beta in [0.0, *beta_values]:
            for gamma in [0.0, *gamma_values]:
                key = (float(alpha), float(beta), float(gamma))
                if key in seen:
                    continue
                seen.add(key)
                settings.append(EnergySetting(*key))
    settings.sort(key=lambda item: (item.alpha_neighbor, item.beta_opposite, item.gamma_cardinality))
    return settings


def build_family_grid(
    *,
    families: list[str],
    alpha_values: list[float],
    beta_values: list[float],
    gamma_values: list[float],
) -> list[tuple[str, EnergySetting]]:
    """Build decoder ablation grids grouped by component family."""
    grid: list[tuple[str, EnergySetting]] = []
    for family in families:
        family = family.strip().lower()
        if family not in DECODER_FAMILIES:
            raise ValueError(f"Unknown decoder family: {family}")
        if family == "cardinality_only":
            settings = [
                EnergySetting(0.0, 0.0, gamma)
                for gamma in sorted({0.0, *gamma_values})
            ]
        elif family == "neighbor_only":
            settings = [
                EnergySetting(alpha, 0.0, 0.0)
                for alpha in sorted({0.0, *alpha_values})
            ]
        elif family == "opposite_only":
            settings = [
                EnergySetting(0.0, beta, 0.0)
                for beta in sorted({0.0, *beta_values})
            ]
        elif family == "neighbor_opposite":
            settings = [
                EnergySetting(alpha, beta, 0.0)
                for alpha in sorted({0.0, *alpha_values})
                for beta in sorted({0.0, *beta_values})
            ]
        elif family == "full":
            settings = build_grid(
                alpha_values=alpha_values,
                beta_values=beta_values,
                gamma_values=gamma_values,
            )
        grid.extend((family, setting) for setting in settings)
    return grid


def select_setting(
    validation_rows: list[dict[str, Any]],
    *,
    objective: str,
) -> dict[str, Any]:
    """Select a setting using validation metrics only."""
    if objective == "standard":
        for row in validation_rows:
            if (
                float(row["alpha_neighbor"]) == 0.0
                and float(row["beta_opposite"]) == 0.0
                and float(row["gamma_cardinality"]) == 0.0
            ):
                return row
        raise ValueError("Standard decoder setting not found")

    if objective == "f1":
        return max(
            validation_rows,
            key=lambda row: (
                float(row["validation_macro_f1"]),
                -float(row["validation_decoder_geometry_cost"]),
            ),
        )

    tolerance = OBJECTIVE_TOLERANCES.get(objective)
    if tolerance is None:
        raise ValueError(f"Unknown decoder objective: {objective}")
    best_macro = max(float(row["validation_macro_f1"]) for row in validation_rows)
    min_macro = best_macro * tolerance
    candidates = [
        row for row in validation_rows if float(row["validation_macro_f1"]) >= min_macro
    ]
    return min(
        candidates,
        key=lambda row: (
            float(row["validation_decoder_geometry_cost"]),
            -float(row["validation_macro_f1"]),
        ),
    )


def summarize_selected(seed_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_cols = [
        column
        for column in seed_df.columns
        if column.startswith("test_") and pd.api.types.is_numeric_dtype(seed_df[column])
    ]
    group_cols = ["method", "model_slug", "geometry", "family", "objective"]
    summary = seed_df.groupby(group_cols, dropna=False)[metric_cols].agg(
        ["mean", "std", "count"]
    )
    summary.columns = [
        "_".join(str(part) for part in column if part)
        for column in summary.columns
    ]
    summary = summary.reset_index()

    standard = seed_df[seed_df["objective"] == "standard"].copy()
    deltas: list[dict[str, Any]] = []
    for _, row in seed_df[seed_df["objective"] != "standard"].iterrows():
        mask = (
            (standard["method"] == row["method"])
            & (standard["model_slug"] == row["model_slug"])
            & (standard["geometry"] == row["geometry"])
            & (standard["family"] == row["family"])
            & (standard["seed"] == row["seed"])
        )
        baseline = standard[mask]
        if baseline.empty:
            continue
        baseline_row = baseline.iloc[0]
        delta_row: dict[str, Any] = {
            "method": row["method"],
            "model_slug": row["model_slug"],
            "geometry": row["geometry"],
            "family": row["family"],
            "seed": row["seed"],
            "objective": row["objective"],
        }
        for metric in METRIC_COLUMNS + [
            "decoder_geometry_cost",
            "n_predictions",
            "prediction_rate",
        ]:
            column = f"test_{metric}"
            delta_row[f"delta_{metric}"] = float(row[column]) - float(
                baseline_row[column]
            )
        deltas.append(delta_row)

    delta_df = pd.DataFrame(deltas)
    if delta_df.empty:
        return summary, pd.DataFrame()

    delta_metric_cols = [
        column
        for column in delta_df.columns
        if column.startswith("delta_") and pd.api.types.is_numeric_dtype(delta_df[column])
    ]
    delta_summary = delta_df.groupby(
        ["method", "model_slug", "geometry", "family", "objective"], dropna=False
    )[delta_metric_cols].agg(["mean", "std", "count"])
    delta_summary.columns = [
        "_".join(str(part) for part in column if part)
        for column in delta_summary.columns
    ]
    return summary, delta_summary.reset_index()


def metric_function(metric: str, *, label_names: list[str]):
    """Return a binary-prediction metric function for sample-level bootstrap."""
    if metric == "macro_f1":
        return macro_f1_from_arrays
    if metric == "micro_f1":
        return lambda y_true, y_pred: float(
            compute_global_metrics(y_true, y_pred)["micro_f1"]
        )
    if metric == "neighbor_error_rate":
        return lambda y_true, y_pred: compute_neighbor_error_rate(
            y_true,
            y_pred,
            label_names=label_names,
        )
    if metric == "opposite_error_rate":
        return lambda y_true, y_pred: compute_opposite_error_rate(
            y_true,
            y_pred,
            label_names=label_names,
        )
    if metric == "confusion_distance_correlation":
        return lambda y_true, y_pred: compute_confusion_distance_correlation(
            y_true,
            y_pred,
            label_names=label_names,
        )
    if metric == "decoder_geometry_cost":
        return lambda y_true, y_pred: float(
            compute_neighbor_error_rate(y_true, y_pred, label_names=label_names)
            + compute_opposite_error_rate(y_true, y_pred, label_names=label_names)
            + compute_confusion_distance_correlation(
                y_true,
                y_pred,
                label_names=label_names,
            )
        )
    raise ValueError(f"Unsupported sample bootstrap metric: {metric}")


def sample_level_bootstrap_rows(
    *,
    y_true: np.ndarray,
    decoded_pred: np.ndarray,
    standard_pred: np.ndarray,
    label_names: list[str],
    base_row: dict[str, Any],
    metrics: list[str],
    n_iterations: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Bootstrap paired sample-level deltas decoded - standard."""
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        result = paired_bootstrap_delta(
            y_true,
            decoded_pred,
            standard_pred,
            metric_fn=metric_function(metric, label_names=label_names),
            n_iterations=n_iterations,
            seed=seed,
        )
        lower_is_better = metric in LOWER_IS_BETTER
        improvement = -result.delta if lower_is_better else result.delta
        improvement_ci_low = -result.ci_high if lower_is_better else result.ci_low
        improvement_ci_high = -result.ci_low if lower_is_better else result.ci_high
        rows.append(
            {
                **base_row,
                "metric": metric,
                "higher_is_better": not lower_is_better,
                "delta_decoded_minus_standard": result.delta,
                "delta_ci_low": result.ci_low,
                "delta_ci_high": result.ci_high,
                "improvement_over_standard": improvement,
                "improvement_ci_low": improvement_ci_low,
                "improvement_ci_high": improvement_ci_high,
                "p_value_two_sided": result.p_value,
                "n_samples": result.n_samples,
                "n_iterations": result.n_iterations,
                "bootstrap_seed": seed,
                "significant_0.05": result.p_value < 0.05,
            }
        )
    return rows


def _sample_f1(gold: np.ndarray, pred: np.ndarray) -> float:
    tp = int(((gold == 1) & (pred == 1)).sum())
    fp = int(((gold == 0) & (pred == 1)).sum())
    fn = int(((gold == 1) & (pred == 0)).sum())
    if tp + fp + fn == 0:
        return 1.0
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    if precision + recall == 0.0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def _labels_from_row(row: np.ndarray, label_names: list[str]) -> list[str]:
    return [label_names[idx] for idx in np.flatnonzero(row)]


def _load_prediction_metadata(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_text_lookup() -> dict[tuple[str, str], str]:
    try:
        test_df = load_split("test")
    except Exception as exc:  # pragma: no cover - optional diagnostic context
        LOGGER.warning("Could not load test text for error analysis: %s", exc)
        return {}
    return {
        (str(row.text_id), str(row.sent_id)): str(row.text)
        for row in test_df.itertuples(index=False)
    }


def error_example_rows(
    *,
    metadata_rows: list[dict[str, Any]],
    text_lookup: dict[tuple[str, str], str],
    y_true: np.ndarray,
    decoded_pred: np.ndarray,
    standard_pred: np.ndarray,
    y_probs: np.ndarray,
    label_names: list[str],
    base_row: dict[str, Any],
    max_examples: int,
) -> list[dict[str, Any]]:
    """Build readable examples where structured decoding changes predictions."""
    distances = circular_distance_matrix(label_names)
    candidates: list[dict[str, Any]] = []
    for idx, (gold, decoded, standard) in enumerate(
        zip(y_true, decoded_pred, standard_pred, strict=False)
    ):
        if np.array_equal(decoded, standard):
            continue
        standard_f1 = _sample_f1(gold, standard)
        decoded_f1 = _sample_f1(gold, decoded)
        removed = np.flatnonzero((standard == 1) & (decoded == 0))
        added = np.flatnonzero((standard == 0) & (decoded == 1))
        lost_true = [int(label) for label in removed if gold[label] == 1]
        gained_true = [int(label) for label in added if gold[label] == 1]

        gold_idx = np.flatnonzero(gold)
        removed_opposite_fp: list[int] = []
        if gold_idx.size:
            for label in removed:
                if (
                    gold[label] == 0
                    and np.min(distances[np.ix_(gold_idx, [label])])
                    > OPPOSITE_DISTANCE_THRESHOLD
                ):
                    removed_opposite_fp.append(int(label))

        if decoded_f1 <= standard_f1 and not removed_opposite_fp and not gained_true:
            continue

        metadata = metadata_rows[idx] if idx < len(metadata_rows) else {}
        text_id = str(metadata.get("text_id", ""))
        sent_id = str(metadata.get("sent_id", ""))
        top_idx = np.argsort(-y_probs[idx], kind="mergesort")[:5]
        candidates.append(
            {
                **base_row,
                "text_id": text_id,
                "sent_id": sent_id,
                "text": text_lookup.get((text_id, sent_id), ""),
                "gold_labels": " | ".join(_labels_from_row(gold, label_names)),
                "standard_labels": " | ".join(_labels_from_row(standard, label_names)),
                "decoded_labels": " | ".join(_labels_from_row(decoded, label_names)),
                "removed_labels": " | ".join(label_names[label] for label in removed),
                "added_labels": " | ".join(label_names[label] for label in added),
                "removed_opposite_false_positives": " | ".join(
                    label_names[label] for label in removed_opposite_fp
                ),
                "lost_true_positives": " | ".join(
                    label_names[label] for label in lost_true
                ),
                "gained_true_positives": " | ".join(
                    label_names[label] for label in gained_true
                ),
                "sample_f1_standard": standard_f1,
                "sample_f1_decoded": decoded_f1,
                "sample_f1_delta": decoded_f1 - standard_f1,
                "top_probabilities": " | ".join(
                    f"{label_names[label]}={float(y_probs[idx, label]):.4f}"
                    for label in top_idx
                ),
                "_rank_removed_opposite": len(removed_opposite_fp),
            }
        )

    candidates.sort(
        key=lambda row: (
            -float(row["sample_f1_delta"]),
            -int(row["_rank_removed_opposite"]),
            row["text_id"],
            row["sent_id"],
        )
    )
    output = candidates[:max_examples]
    for row in output:
        row.pop("_rank_removed_opposite", None)
    return output


def _matches_filter(value: str, allowed: set[str] | None) -> bool:
    return allowed is None or value in allowed


def run_decoder(
    *,
    predictions_dir: Path,
    logs_dir: Path,
    output_dir: Path,
    methods: set[str] | None,
    seeds: set[int] | None,
    model_slug: str | None,
    geometries: list[str],
    families: list[str],
    alpha_values: list[float],
    beta_values: list[float],
    gamma_values: list[float],
    objectives: list[str],
    top_k: int,
    max_candidates: int,
    max_labels: int,
    threshold_factor: float,
    min_prob: float,
    marginal_temperature: float,
    neighbor_steps: int,
    random_seed: int,
    empirical_metric: str,
    bootstrap_iterations: int,
    bootstrap_seed: int,
    bootstrap_metrics: list[str],
    bootstrap_geometries: set[str] | None,
    bootstrap_families: set[str] | None,
    bootstrap_objectives: set[str] | None,
    max_error_examples: int,
    example_geometries: set[str] | None,
    example_families: set[str] | None,
    example_objectives: set[str] | None,
) -> dict[str, Path]:
    runs = discover_runs(
        predictions_dir=predictions_dir,
        logs_dir=logs_dir,
        methods=methods,
        seeds=seeds,
        model_slug=model_slug,
    )
    if not runs:
        raise ValueError("No prediction/metric run pairs found")

    output_dir.mkdir(parents=True, exist_ok=True)
    grid = build_family_grid(
        families=families,
        alpha_values=alpha_values,
        beta_values=beta_values,
        gamma_values=gamma_values,
    )
    geometries = [geometry.strip().lower() for geometry in geometries]
    unknown_geometries = sorted(set(geometries) - DECODER_GEOMETRIES)
    if unknown_geometries:
        raise ValueError(f"Unknown decoder geometries: {unknown_geometries}")
    LOGGER.info(
        "Discovered %d runs; evaluating %d settings x %d geometries",
        len(runs),
        len(grid),
        len(geometries),
    )

    validation_grid_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    example_rows: list[dict[str, Any]] = []
    text_lookup = _load_text_lookup() if max_error_examples > 0 else {}

    for run in runs:
        LOGGER.info("Decoding method=%s seed=%s", run.method, run.seed)
        validation = load_prediction_jsonl(run.validation_predictions)
        test = load_prediction_jsonl(run.test_predictions)
        if validation.label_names != test.label_names:
            raise ValueError(f"Validation/test labels differ for {run}")
        thresholds = load_validation_thresholds(run.validation_metrics, validation.label_names)
        metadata_rows = (
            _load_prediction_metadata(run.test_predictions)
            if max_error_examples > 0
            else []
        )
        empirical_labels = None
        if "empirical" in geometries:
            empirical_labels = load_empirical_training_labels(validation.label_names)
        standard_outputs = decode_and_evaluate(
            test,
            thresholds=thresholds,
            setting=EnergySetting(0.0, 0.0, 0.0),
            top_k=top_k,
            max_candidates=max_candidates,
            max_labels=max_labels,
            threshold_factor=threshold_factor,
            min_prob=min_prob,
            marginal_temperature=marginal_temperature,
            neighbor_steps=neighbor_steps,
        )

        for geometry in geometries:
            matrices = build_geometry_matrices(
                validation.label_names,
                geometry=geometry,
                neighbor_steps=neighbor_steps,
                random_seed=random_seed,
                empirical_labels=empirical_labels,
                empirical_metric=empirical_metric,
            )
            for family in families:
                family_rows: list[dict[str, Any]] = []
                validation_metrics_by_setting: dict[
                    tuple[float, float, float],
                    dict[str, Any],
                ] = {}
                for grid_family, setting in grid:
                    if grid_family != family:
                        continue
                    metrics = evaluate_setting(
                        validation,
                        thresholds=thresholds,
                        setting=setting,
                        geometry_matrices=matrices,
                        top_k=top_k,
                        max_candidates=max_candidates,
                        max_labels=max_labels,
                        threshold_factor=threshold_factor,
                        min_prob=min_prob,
                        marginal_temperature=marginal_temperature,
                        neighbor_steps=neighbor_steps,
                    )
                    key = (
                        setting.alpha_neighbor,
                        setting.beta_opposite,
                        setting.gamma_cardinality,
                    )
                    validation_metrics_by_setting[key] = metrics
                    row = {
                        "method": run.method,
                        "seed": run.seed,
                        "model_slug": run.model_slug,
                        "geometry": geometry,
                        "family": family,
                        **setting_row(setting),
                        **scalar_metric_row(metrics, prefix="validation_"),
                    }
                    validation_grid_rows.append(row)
                    family_rows.append(row)

                for objective in objectives:
                    selected = select_setting(family_rows, objective=objective)
                    selected_setting = EnergySetting(
                        alpha_neighbor=float(selected["alpha_neighbor"]),
                        beta_opposite=float(selected["beta_opposite"]),
                        gamma_cardinality=float(selected["gamma_cardinality"]),
                    )
                    key = (
                        selected_setting.alpha_neighbor,
                        selected_setting.beta_opposite,
                        selected_setting.gamma_cardinality,
                    )
                    validation_metrics = validation_metrics_by_setting[key]
                    test_outputs = decode_and_evaluate(
                        test,
                        thresholds=thresholds,
                        setting=selected_setting,
                        geometry_matrices=matrices,
                        top_k=top_k,
                        max_candidates=max_candidates,
                        max_labels=max_labels,
                        threshold_factor=threshold_factor,
                        min_prob=min_prob,
                        marginal_temperature=marginal_temperature,
                        neighbor_steps=neighbor_steps,
                    )
                    selected_base = {
                        "method": run.method,
                        "seed": run.seed,
                        "model_slug": run.model_slug,
                        "geometry": geometry,
                        "family": family,
                        "objective": objective,
                        "top_k": top_k,
                        "max_candidates": max_candidates,
                        "max_labels": max_labels,
                        "threshold_factor": threshold_factor,
                        "min_prob": min_prob,
                        "marginal_temperature": marginal_temperature,
                        "neighbor_steps": neighbor_steps,
                        "random_seed": random_seed,
                        "empirical_metric": empirical_metric,
                        **setting_row(selected_setting),
                    }
                    selected_rows.append(
                        {
                            **selected_base,
                            **scalar_metric_row(validation_metrics, prefix="validation_"),
                            **scalar_metric_row(test_outputs.metrics, prefix="test_"),
                        }
                    )

                    do_bootstrap = (
                        bootstrap_iterations > 0
                        and objective != "standard"
                        and _matches_filter(geometry, bootstrap_geometries)
                        and _matches_filter(family, bootstrap_families)
                        and _matches_filter(objective, bootstrap_objectives)
                    )
                    if do_bootstrap:
                        bootstrap_rows.extend(
                            sample_level_bootstrap_rows(
                                y_true=test.y_true,
                                decoded_pred=test_outputs.y_pred,
                                standard_pred=standard_outputs.y_pred,
                                label_names=test.label_names,
                                base_row=selected_base,
                                metrics=bootstrap_metrics,
                                n_iterations=bootstrap_iterations,
                                seed=bootstrap_seed,
                            )
                        )
                    do_examples = (
                        max_error_examples > 0
                        and objective != "standard"
                        and _matches_filter(geometry, example_geometries)
                        and _matches_filter(family, example_families)
                        and _matches_filter(objective, example_objectives)
                    )
                    if do_examples:
                        example_rows.extend(
                            error_example_rows(
                                metadata_rows=metadata_rows,
                                text_lookup=text_lookup,
                                y_true=test.y_true,
                                decoded_pred=test_outputs.y_pred,
                                standard_pred=standard_outputs.y_pred,
                                y_probs=test.y_probs,
                                label_names=test.label_names,
                                base_row=selected_base,
                                max_examples=max_error_examples,
                            )
                        )

    validation_grid_df = pd.DataFrame(validation_grid_rows)
    selected_df = pd.DataFrame(selected_rows)
    bootstrap_df = pd.DataFrame(bootstrap_rows)
    examples_df = pd.DataFrame(example_rows)
    summary_df, delta_summary_df = summarize_selected(selected_df)

    paths = {
        "validation_grid": output_dir / "schwartz_energy_decoder_validation_grid.csv",
        "selected_seed_results": output_dir / "schwartz_energy_decoder_selected_seed_results.csv",
        "selected_mean_std": output_dir / "schwartz_energy_decoder_selected_mean_std.csv",
        "delta_vs_standard": output_dir / "schwartz_energy_decoder_delta_vs_standard.csv",
        "sample_bootstrap": output_dir / "schwartz_energy_decoder_sample_bootstrap.csv",
        "error_examples": output_dir / "schwartz_energy_decoder_error_examples.csv",
    }
    validation_grid_df.to_csv(paths["validation_grid"], index=False)
    selected_df.to_csv(paths["selected_seed_results"], index=False)
    summary_df.to_csv(paths["selected_mean_std"], index=False)
    delta_summary_df.to_csv(paths["delta_vs_standard"], index=False)
    bootstrap_df.to_csv(paths["sample_bootstrap"], index=False)
    examples_df.to_csv(paths["error_examples"], index=False)
    return paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tune a Schwartz structured label-set energy decoder on validation "
            "probabilities and apply it once to test probabilities."
        )
    )
    parser.add_argument("--predictions_dir", default="results/predictions")
    parser.add_argument("--logs_dir", default="results/logs")
    parser.add_argument("--output_dir", default="results/analysis/schwartz_energy_decoder")
    parser.add_argument(
        "--methods",
        nargs="*",
        default=["bce"],
        help="Methods to decode. Defaults to BCE, the current strongest base model.",
    )
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--model_slug", default="deberta-v3-base")
    parser.add_argument(
        "--geometries",
        nargs="+",
        default=["schwartz", "random", "empirical"],
        choices=sorted(DECODER_GEOMETRIES),
        help="Geometry controls to evaluate.",
    )
    parser.add_argument(
        "--families",
        nargs="+",
        default=[
            "cardinality_only",
            "neighbor_only",
            "opposite_only",
            "neighbor_opposite",
            "full",
        ],
        choices=sorted(DECODER_FAMILIES),
        help="Decoder component ablation families to evaluate.",
    )
    parser.add_argument("--alpha_values", default="0.10")
    parser.add_argument("--beta_values", default="0.20")
    parser.add_argument("--gamma_values", default="0.02")
    parser.add_argument(
        "--objectives",
        nargs="+",
        default=["standard", "f1", "pareto_99", "pareto_98", "pareto_95"],
        choices=sorted(OBJECTIVE_TOLERANCES),
    )
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--max_candidates", type=int, default=8)
    parser.add_argument("--max_labels", type=int, default=5)
    parser.add_argument("--threshold_factor", type=float, default=0.5)
    parser.add_argument("--min_prob", type=float, default=0.01)
    parser.add_argument("--marginal_temperature", type=float, default=1.0)
    parser.add_argument("--neighbor_steps", type=int, default=2)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--empirical_metric", default="jaccard", choices=["jaccard", "cosine"])
    parser.add_argument("--bootstrap_iterations", type=int, default=2000)
    parser.add_argument("--bootstrap_seed", type=int, default=42)
    parser.add_argument(
        "--bootstrap_metrics",
        nargs="+",
        default=["macro_f1", "micro_f1", "decoder_geometry_cost"],
        choices=BINARY_BOOTSTRAP_METRICS,
    )
    parser.add_argument(
        "--bootstrap_geometries",
        nargs="*",
        default=["schwartz", "random", "empirical"],
        choices=sorted(DECODER_GEOMETRIES),
    )
    parser.add_argument(
        "--bootstrap_families",
        nargs="*",
        default=["full"],
        choices=sorted(DECODER_FAMILIES),
    )
    parser.add_argument(
        "--bootstrap_objectives",
        nargs="*",
        default=["pareto_99"],
        choices=sorted(OBJECTIVE_TOLERANCES),
    )
    parser.add_argument("--max_error_examples", type=int, default=25)
    parser.add_argument(
        "--example_geometries",
        nargs="*",
        default=["schwartz"],
        choices=sorted(DECODER_GEOMETRIES),
    )
    parser.add_argument(
        "--example_families",
        nargs="*",
        default=["full"],
        choices=sorted(DECODER_FAMILIES),
    )
    parser.add_argument(
        "--example_objectives",
        nargs="*",
        default=["pareto_99"],
        choices=sorted(OBJECTIVE_TOLERANCES),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    paths = run_decoder(
        predictions_dir=Path(args.predictions_dir),
        logs_dir=Path(args.logs_dir),
        output_dir=Path(args.output_dir),
        methods=set(args.methods) if args.methods else None,
        seeds=set(args.seeds) if args.seeds else None,
        model_slug=str(args.model_slug) if args.model_slug else None,
        geometries=list(args.geometries),
        families=list(args.families),
        alpha_values=parse_float_list(str(args.alpha_values)),
        beta_values=parse_float_list(str(args.beta_values)),
        gamma_values=parse_float_list(str(args.gamma_values)),
        objectives=list(args.objectives),
        top_k=int(args.top_k),
        max_candidates=int(args.max_candidates),
        max_labels=int(args.max_labels),
        threshold_factor=float(args.threshold_factor),
        min_prob=float(args.min_prob),
        marginal_temperature=float(args.marginal_temperature),
        neighbor_steps=int(args.neighbor_steps),
        random_seed=int(args.random_seed),
        empirical_metric=str(args.empirical_metric),
        bootstrap_iterations=int(args.bootstrap_iterations),
        bootstrap_seed=int(args.bootstrap_seed),
        bootstrap_metrics=list(args.bootstrap_metrics),
        bootstrap_geometries=set(args.bootstrap_geometries) if args.bootstrap_geometries else None,
        bootstrap_families=set(args.bootstrap_families) if args.bootstrap_families else None,
        bootstrap_objectives=set(args.bootstrap_objectives) if args.bootstrap_objectives else None,
        max_error_examples=int(args.max_error_examples),
        example_geometries=set(args.example_geometries) if args.example_geometries else None,
        example_families=set(args.example_families) if args.example_families else None,
        example_objectives=set(args.example_objectives) if args.example_objectives else None,
    )
    for name, path in paths.items():
        LOGGER.info("Wrote %s to %s", name, path)


if __name__ == "__main__":
    main()
