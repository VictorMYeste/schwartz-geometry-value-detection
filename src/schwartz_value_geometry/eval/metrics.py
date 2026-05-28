"""Evaluation metrics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from schwartz_value_geometry.geometry import (
    NEIGHBOR_STEP_DISTANCE,
    OPPOSITE_DISTANCE_THRESHOLD,
    circular_distance_matrix,
    circular_step_distance_matrix,
)
from schwartz_value_geometry.utils.logging import get_logger

LOGGER = get_logger(__name__)


def _as_2d(array: np.ndarray) -> np.ndarray:
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {array.shape}")
    return array


def _threshold_array(
    threshold: float | Sequence[float] | np.ndarray,
    n_labels: int,
) -> float | np.ndarray:
    if np.isscalar(threshold):
        return float(threshold)
    arr = np.asarray(threshold, dtype=float)
    if arr.shape != (n_labels,):
        raise ValueError(f"Threshold array must have shape ({n_labels},), got {arr.shape}")
    return arr


def binarize_probs(
    probs: np.ndarray,
    *,
    threshold: float | Sequence[float] | np.ndarray = 0.5,
) -> np.ndarray:
    """Binarize probability outputs using a scalar or per-label threshold."""
    probs = _as_2d(np.asarray(probs, dtype=float))
    thresholds = _threshold_array(threshold, probs.shape[1])
    return (probs >= thresholds).astype(int)


def compute_global_metrics(gold: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """Compute global micro/macro precision, recall, and F1."""
    gold = _as_2d(np.asarray(gold)).astype(int)
    pred = _as_2d(np.asarray(pred)).astype(int)
    LOGGER.debug("Computing global metrics (samples=%d, labels=%d)", *gold.shape)

    if gold.size == 0:
        return {
            "micro_precision": 0.0,
            "micro_recall": 0.0,
            "micro_f1": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
        }

    tp = (gold & pred).sum()
    fp = ((1 - gold) & pred).sum()
    fn = (gold & (1 - pred)).sum()

    micro_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    micro_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0
        else 0.0
    )

    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    for col in range(gold.shape[1]):
        tp_c = (gold[:, col] & pred[:, col]).sum()
        fp_c = ((1 - gold[:, col]) & pred[:, col]).sum()
        fn_c = (gold[:, col] & (1 - pred[:, col])).sum()
        precision_c = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
        recall_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
        f1_c = (
            2 * precision_c * recall_c / (precision_c + recall_c)
            if (precision_c + recall_c) > 0
            else 0.0
        )
        precisions.append(float(precision_c))
        recalls.append(float(recall_c))
        f1s.append(float(f1_c))

    macro_precision = float(np.mean(precisions)) if precisions else 0.0
    macro_recall = float(np.mean(recalls)) if recalls else 0.0
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0

    return {
        "micro_precision": float(micro_precision),
        "micro_recall": float(micro_recall),
        "micro_f1": float(micro_f1),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
    }


def compute_per_label_f1(
    gold: np.ndarray,
    pred: np.ndarray,
    *,
    label_names: list[str],
) -> dict[str, float]:
    """Compute per-label F1 scores."""
    gold = _as_2d(np.asarray(gold)).astype(int)
    pred = _as_2d(np.asarray(pred)).astype(int)
    LOGGER.debug("Computing per-label F1 for %d labels", len(label_names))
    if gold.size == 0:
        return {name: 0.0 for name in label_names}

    per_label_f1: dict[str, float] = {}
    for col, name in enumerate(label_names):
        tp_c = (gold[:, col] & pred[:, col]).sum()
        fp_c = ((1 - gold[:, col]) & pred[:, col]).sum()
        fn_c = (gold[:, col] & (1 - pred[:, col])).sum()
        precision_c = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
        recall_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
        f1_c = (
            2 * precision_c * recall_c / (precision_c + recall_c)
            if (precision_c + recall_c) > 0
            else 0.0
        )
        per_label_f1[name] = float(f1_c)
    return per_label_f1


def compute_per_label_support(
    gold: np.ndarray,
    pred: np.ndarray,
    *,
    label_names: list[str],
) -> dict[str, dict[str, float]]:
    """Compute per-label support and prediction rates."""
    gold = _as_2d(np.asarray(gold)).astype(int)
    pred = _as_2d(np.asarray(pred)).astype(int)
    total = int(gold.shape[0]) if gold.ndim == 2 else 0
    LOGGER.debug("Computing per-label support for %d labels", len(label_names))
    if gold.size == 0:
        return {
            name: {"gold": 0.0, "pred": 0.0, "gold_rate": 0.0, "pred_rate": 0.0}
            for name in label_names
        }

    stats: dict[str, dict[str, float]] = {}
    for col, name in enumerate(label_names):
        gold_count = int(gold[:, col].sum())
        pred_count = int(pred[:, col].sum())
        gold_rate = gold_count / total if total > 0 else 0.0
        pred_rate = pred_count / total if total > 0 else 0.0
        stats[name] = {
            "gold": float(gold_count),
            "pred": float(pred_count),
            "gold_rate": float(gold_rate),
            "pred_rate": float(pred_rate),
        }
    return stats


def macro_f1_from_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute macro-F1 only, for efficient paired tests."""
    metrics = compute_global_metrics(y_true, y_pred)
    return float(metrics["macro_f1"])


def average_precision_score_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute average precision for one binary label without sklearn."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    positives = int(y_true.sum())
    if positives == 0:
        return 0.0
    order = np.argsort(-y_score, kind="mergesort")
    sorted_true = y_true[order]
    true_positive_cumsum = np.cumsum(sorted_true)
    ranks = np.arange(1, len(sorted_true) + 1)
    precision_at_k = true_positive_cumsum / ranks
    return float((precision_at_k * sorted_true).sum() / positives)


def compute_auprc_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    *,
    label_names: list[str],
) -> dict[str, object]:
    """Compute macro/micro AUPRC and per-label average precision."""
    y_true = _as_2d(np.asarray(y_true)).astype(int)
    y_probs = _as_2d(np.asarray(y_probs)).astype(float)
    if y_true.shape != y_probs.shape:
        raise ValueError("y_true and y_probs must have identical shapes")
    if y_true.size == 0:
        return {
            "macro_auprc": 0.0,
            "micro_auprc": 0.0,
            "per_label_auprc": {name: 0.0 for name in label_names},
        }

    per_label = {
        name: average_precision_score_binary(y_true[:, idx], y_probs[:, idx])
        for idx, name in enumerate(label_names)
    }
    macro = float(np.mean(list(per_label.values()))) if per_label else 0.0
    micro = average_precision_score_binary(y_true.ravel(), y_probs.ravel())
    return {
        "macro_auprc": macro,
        "micro_auprc": float(micro),
        "per_label_auprc": per_label,
    }


def compute_f1_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    label_names: list[str],
) -> dict[str, object]:
    """Compute micro/macro F1 and per-label F1."""
    global_metrics = compute_global_metrics(y_true, y_pred)
    per_label_f1 = compute_per_label_f1(y_true, y_pred, label_names=label_names)
    per_label_support = compute_per_label_support(
        y_true, y_pred, label_names=label_names
    )
    macro_f1 = global_metrics["macro_f1"]
    micro_f1 = global_metrics["micro_f1"]
    return {
        "micro_f1": float(micro_f1),
        "macro_f1": float(macro_f1),
        "per_label_f1": per_label_f1,
        "per_label_support": per_label_support,
    }


def compute_circular_error(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    *,
    label_names: list[str],
    distance_matrix: np.ndarray | None = None,
    eps: float = 1e-12,
) -> float:
    """Expected circular distance from gold labels to predicted probability mass."""
    y_true = _as_2d(np.asarray(y_true)).astype(float)
    y_probs = _as_2d(np.asarray(y_probs)).astype(float)
    if y_true.shape != y_probs.shape:
        raise ValueError("y_true and y_probs must have identical shapes")
    distances = (
        circular_distance_matrix(label_names)
        if distance_matrix is None
        else np.asarray(distance_matrix, dtype=float)
    )
    weighted_distances = np.einsum("bi,ij,bj->b", y_true, distances, y_probs)
    gold_counts = y_true.sum(axis=1)
    per_sample = weighted_distances / (gold_counts + eps)
    return float(per_sample.mean()) if per_sample.size else 0.0


def compute_opposite_value_activation(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    *,
    label_names: list[str],
    distance_matrix: np.ndarray | None = None,
    threshold: float = OPPOSITE_DISTANCE_THRESHOLD,
    eps: float = 1e-12,
) -> float:
    """Probability mass assigned to values distant from the gold values."""
    y_true = _as_2d(np.asarray(y_true)).astype(float)
    y_probs = _as_2d(np.asarray(y_probs)).astype(float)
    if y_true.shape != y_probs.shape:
        raise ValueError("y_true and y_probs must have identical shapes")
    distances = (
        circular_distance_matrix(label_names)
        if distance_matrix is None
        else np.asarray(distance_matrix, dtype=float)
    )
    opposite = (distances > threshold).astype(float)
    weighted_mass = np.einsum("bi,ij,bj->b", y_true, opposite, y_probs)
    gold_counts = y_true.sum(axis=1)
    per_sample = weighted_mass / (gold_counts + eps)
    return float(per_sample.mean()) if per_sample.size else 0.0


def compute_neighbor_error_rate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    label_names: list[str],
    max_steps: int = NEIGHBOR_STEP_DISTANCE,
) -> float:
    """Proportion of false positives that are near any gold value."""
    y_true = _as_2d(np.asarray(y_true)).astype(int)
    y_pred = _as_2d(np.asarray(y_pred)).astype(int)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have identical shapes")
    step_distances = circular_step_distance_matrix(label_names)
    false_positive_total = 0
    neighbor_false_positive_total = 0

    for gold_row, pred_row in zip(y_true, y_pred, strict=False):
        gold_idx = np.flatnonzero(gold_row)
        false_positive_idx = np.flatnonzero((pred_row == 1) & (gold_row == 0))
        false_positive_total += int(false_positive_idx.size)
        if gold_idx.size == 0 or false_positive_idx.size == 0:
            continue
        min_steps = step_distances[np.ix_(gold_idx, false_positive_idx)].min(axis=0)
        neighbor_false_positive_total += int((min_steps <= max_steps).sum())

    if false_positive_total == 0:
        return 0.0
    return float(neighbor_false_positive_total / false_positive_total)


def compute_opposite_error_rate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    label_names: list[str],
    distance_matrix: np.ndarray | None = None,
    threshold: float = OPPOSITE_DISTANCE_THRESHOLD,
) -> float:
    """Proportion of false positives that are distant from all gold values."""
    y_true = _as_2d(np.asarray(y_true)).astype(int)
    y_pred = _as_2d(np.asarray(y_pred)).astype(int)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have identical shapes")
    distances = (
        circular_distance_matrix(label_names)
        if distance_matrix is None
        else np.asarray(distance_matrix, dtype=float)
    )
    false_positive_total = 0
    opposite_false_positive_total = 0

    for gold_row, pred_row in zip(y_true, y_pred, strict=False):
        gold_idx = np.flatnonzero(gold_row)
        false_positive_idx = np.flatnonzero((pred_row == 1) & (gold_row == 0))
        false_positive_total += int(false_positive_idx.size)
        if gold_idx.size == 0 or false_positive_idx.size == 0:
            opposite_false_positive_total += int(false_positive_idx.size)
            continue
        min_distances = distances[np.ix_(gold_idx, false_positive_idx)].min(axis=0)
        opposite_false_positive_total += int((min_distances > threshold).sum())

    if false_positive_total == 0:
        return 0.0
    return float(opposite_false_positive_total / false_positive_total)


def compute_confusion_distance_correlation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    label_names: list[str],
    distance_matrix: np.ndarray | None = None,
) -> float:
    """Correlate label-pair circular distance with confusion frequency."""
    y_true = _as_2d(np.asarray(y_true)).astype(int)
    y_pred = _as_2d(np.asarray(y_pred)).astype(int)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have identical shapes")
    distances = (
        circular_distance_matrix(label_names)
        if distance_matrix is None
        else np.asarray(distance_matrix, dtype=float)
    )
    n_labels = y_true.shape[1]
    confusion_counts = np.zeros((n_labels, n_labels), dtype=float)

    for gold_row, pred_row in zip(y_true, y_pred, strict=False):
        gold_idx = np.flatnonzero(gold_row)
        false_positive_idx = np.flatnonzero((pred_row == 1) & (gold_row == 0))
        for gold_label in gold_idx:
            confusion_counts[gold_label, false_positive_idx] += 1.0

    off_diagonal = ~np.eye(n_labels, dtype=bool)
    distance_values = distances[off_diagonal]
    confusion_values = confusion_counts[off_diagonal]
    if np.all(confusion_values == confusion_values[0]):
        return 0.0
    correlation = np.corrcoef(distance_values, confusion_values)[0, 1]
    if np.isnan(correlation):
        return 0.0
    return float(correlation)


def compute_geometry_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    *,
    label_names: list[str],
    distance_matrix: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute Schwartz-continuum-aware metrics."""
    distances = (
        circular_distance_matrix(label_names)
        if distance_matrix is None
        else np.asarray(distance_matrix, dtype=float)
    )
    return {
        "circular_error": compute_circular_error(
            y_true,
            y_probs,
            label_names=label_names,
            distance_matrix=distances,
        ),
        "opposite_value_activation": compute_opposite_value_activation(
            y_true,
            y_probs,
            label_names=label_names,
            distance_matrix=distances,
        ),
        "neighbor_error_rate": compute_neighbor_error_rate(
            y_true,
            y_pred,
            label_names=label_names,
        ),
        "opposite_error_rate": compute_opposite_error_rate(
            y_true,
            y_pred,
            label_names=label_names,
            distance_matrix=distances,
        ),
        "confusion_distance_correlation": compute_confusion_distance_correlation(
            y_true,
            y_pred,
            label_names=label_names,
            distance_matrix=distances,
        ),
    }


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    *,
    label_names: list[str],
) -> dict[str, object]:
    """Compute standard, probability-based, and Schwartz-aware metrics."""
    metrics = compute_f1_metrics(y_true, y_pred, label_names=label_names)
    metrics.update(compute_auprc_metrics(y_true, y_probs, label_names=label_names))
    metrics.update(
        compute_geometry_metrics(
            y_true,
            y_pred,
            y_probs,
            label_names=label_names,
        )
    )
    return metrics


def sweep_thresholds(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    *,
    label_names: list[str],
    start: float = 0.0,
    stop: float = 1.0,
    step: float = 0.01,
) -> dict[str, object]:
    """Sweep thresholds and return the best macro-F1."""
    if y_true.size == 0:
        return {
            "best_threshold": 0.5,
            "best_metrics": compute_f1_metrics(y_true, y_true, label_names=label_names),
            "sweep": [],
        }

    thresholds = np.arange(start, stop + 1e-9, step)
    best_threshold = 0.5
    best_macro = -1.0
    sweep_rows: list[dict[str, float]] = []
    for thr in thresholds:
        y_pred = (y_probs >= thr).astype(int)
        metrics = compute_f1_metrics(y_true, y_pred, label_names=label_names)
        macro = float(metrics["macro_f1"])
        micro = float(metrics["micro_f1"])
        sweep_rows.append(
            {"threshold": float(thr), "macro_f1": macro, "micro_f1": micro}
        )
        if macro > best_macro:
            best_macro = macro
            best_threshold = float(thr)

    best_pred = (y_probs >= best_threshold).astype(int)
    best_metrics = compute_f1_metrics(y_true, best_pred, label_names=label_names)
    return {
        "best_threshold": best_threshold,
        "best_metrics": best_metrics,
        "sweep": sweep_rows,
    }


def sweep_per_label_thresholds(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    *,
    label_names: list[str],
    start: float = 0.0,
    stop: float = 1.0,
    step: float = 0.01,
) -> dict[str, object]:
    """Tune one threshold per label on validation data."""
    y_true = _as_2d(np.asarray(y_true)).astype(int)
    y_probs = _as_2d(np.asarray(y_probs)).astype(float)
    if y_true.shape != y_probs.shape:
        raise ValueError("y_true and y_probs must have identical shapes")
    if y_true.shape[1] != len(label_names):
        raise ValueError("label_names length does not match prediction width")
    if y_true.size == 0:
        thresholds = np.full(len(label_names), 0.5, dtype=float)
        y_pred = binarize_probs(y_probs, threshold=thresholds)
        return {
            "best_thresholds": thresholds.tolist(),
            "best_thresholds_by_label": dict(zip(label_names, thresholds.tolist(), strict=False)),
            "best_metrics": compute_all_metrics(y_true, y_pred, y_probs, label_names=label_names),
            "sweep": [],
        }

    threshold_values = np.arange(start, stop + 1e-9, step)
    best_thresholds = np.zeros(len(label_names), dtype=float)
    sweep_rows: list[dict[str, float | str]] = []

    for idx, label_name in enumerate(label_names):
        best_f1 = -1.0
        best_threshold = 0.5
        gold_col = y_true[:, idx]
        prob_col = y_probs[:, idx]
        for thr in threshold_values:
            pred_col = (prob_col >= thr).astype(int)
            label_f1 = compute_per_label_f1(
                gold_col.reshape(-1, 1),
                pred_col.reshape(-1, 1),
                label_names=[label_name],
            )[label_name]
            sweep_rows.append(
                {
                    "label": label_name,
                    "threshold": float(thr),
                    "f1": float(label_f1),
                }
            )
            if label_f1 > best_f1 or (
                np.isclose(label_f1, best_f1) and thr > best_threshold
            ):
                best_f1 = float(label_f1)
                best_threshold = float(thr)
        best_thresholds[idx] = best_threshold

    y_pred = binarize_probs(y_probs, threshold=best_thresholds)
    best_metrics = compute_all_metrics(y_true, y_pred, y_probs, label_names=label_names)
    return {
        "best_thresholds": best_thresholds.tolist(),
        "best_thresholds_by_label": dict(
            zip(label_names, best_thresholds.tolist(), strict=False)
        ),
        "best_metrics": best_metrics,
        "sweep": sweep_rows,
    }
