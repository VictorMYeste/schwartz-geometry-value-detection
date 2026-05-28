"""Schwartz-continuum label geometry.

The canonical order below follows the project's theory-facing 19-value
Schwartz continuum. Model outputs may use a different column order; helper
functions in this module reorder the geometry to match any provided label list.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

SCHWARTZ_VALUE_ORDER: tuple[str, ...] = (
    "Self-direction: thought",
    "Self-direction: action",
    "Stimulation",
    "Hedonism",
    "Achievement",
    "Power: dominance",
    "Power: resources",
    "Face",
    "Security: personal",
    "Security: societal",
    "Tradition",
    "Conformity: rules",
    "Conformity: interpersonal",
    "Humility",
    "Benevolence: dependability",
    "Benevolence: caring",
    "Universalism: concern",
    "Universalism: nature",
    "Universalism: tolerance",
)

SCHWARTZ_INDEX: dict[str, int] = {
    value: idx for idx, value in enumerate(SCHWARTZ_VALUE_ORDER)
}

OPPOSITE_DISTANCE_THRESHOLD = 0.75
NEIGHBOR_STEP_DISTANCE = 2


def validate_label_names(label_names: Sequence[str]) -> None:
    """Validate that labels match the 19 refined Schwartz values."""
    actual = set(label_names)
    expected = set(SCHWARTZ_VALUE_ORDER)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"extra={extra}")
        raise ValueError("Label names do not match Schwartz values: " + "; ".join(details))
    if len(label_names) != len(SCHWARTZ_VALUE_ORDER):
        raise ValueError(
            f"Expected {len(SCHWARTZ_VALUE_ORDER)} labels, got {len(label_names)}"
        )


def theory_positions(label_names: Sequence[str] | None = None) -> np.ndarray:
    """Return each label's integer position in the Schwartz continuum."""
    labels = SCHWARTZ_VALUE_ORDER if label_names is None else tuple(label_names)
    validate_label_names(labels)
    return np.asarray([SCHWARTZ_INDEX[label] for label in labels], dtype=int)


def circular_angles(label_names: Sequence[str] | None = None) -> np.ndarray:
    """Return each label's angle in radians on the Schwartz circle."""
    positions = theory_positions(label_names)
    return 2.0 * math.pi * positions / len(SCHWARTZ_VALUE_ORDER)


def circular_step_distance_matrix(
    label_names: Sequence[str] | None = None,
) -> np.ndarray:
    """Return pairwise circular step distances in the supplied label order."""
    positions = theory_positions(label_names)
    n_labels = len(SCHWARTZ_VALUE_ORDER)
    raw = np.abs(positions[:, None] - positions[None, :])
    return np.minimum(raw, n_labels - raw).astype(int)


def circular_distance_matrix(label_names: Sequence[str] | None = None) -> np.ndarray:
    """Return normalized pairwise circular distances in [0, 1).

    This implements min angular distance divided by pi. With 19 labels, there
    is no exact antipodal value, so the largest observed distance is 18/19.
    """
    step_distances = circular_step_distance_matrix(label_names).astype(float)
    return 2.0 * step_distances / len(SCHWARTZ_VALUE_ORDER)


def circular_similarity_matrix(label_names: Sequence[str] | None = None) -> np.ndarray:
    """Return a simple theory-derived similarity matrix."""
    return 1.0 - circular_distance_matrix(label_names)


def opposite_value_mask(
    label_names: Sequence[str] | None = None,
    *,
    threshold: float = OPPOSITE_DISTANCE_THRESHOLD,
) -> np.ndarray:
    """Return mask for theoretically distant/opposing value pairs."""
    distances = circular_distance_matrix(label_names)
    return (distances > threshold).astype(int)


def random_circular_order(
    label_names: Sequence[str],
    *,
    seed: int,
) -> tuple[str, ...]:
    """Return a seeded random circular order over the same label set."""
    validate_label_names(label_names)
    rng = np.random.default_rng(seed)
    ordered = list(label_names)
    rng.shuffle(ordered)
    return tuple(ordered)


def distance_matrix_for_order(
    label_names: Sequence[str],
    circular_order: Sequence[str],
) -> np.ndarray:
    """Build a normalized circular distance matrix for a custom order."""
    validate_label_names(label_names)
    validate_label_names(circular_order)
    if len(set(circular_order)) != len(circular_order):
        raise ValueError("circular_order contains duplicate labels")

    order_index = {label: idx for idx, label in enumerate(circular_order)}
    n_labels = len(circular_order)
    positions = np.asarray([order_index[label] for label in label_names], dtype=int)
    raw = np.abs(positions[:, None] - positions[None, :])
    steps = np.minimum(raw, n_labels - raw).astype(float)
    return 2.0 * steps / n_labels


def empirical_cooccurrence_similarity_matrix(
    labels: np.ndarray,
    *,
    metric: str = "jaccard",
) -> np.ndarray:
    """Return a symmetric label-similarity matrix from training co-occurrence."""
    y = np.asarray(labels, dtype=float)
    if y.ndim != 2:
        raise ValueError(f"Expected a 2D label matrix, got shape {y.shape}")
    y = (y > 0).astype(float)
    n_labels = y.shape[1]
    cooccurrence = y.T @ y
    support = np.diag(cooccurrence)

    metric = metric.strip().lower()
    if metric == "jaccard":
        union = support[:, None] + support[None, :] - cooccurrence
        similarity = np.divide(
            cooccurrence,
            union,
            out=np.zeros_like(cooccurrence, dtype=float),
            where=union > 0,
        )
    elif metric == "cosine":
        denom = np.sqrt(support[:, None] * support[None, :])
        similarity = np.divide(
            cooccurrence,
            denom,
            out=np.zeros_like(cooccurrence, dtype=float),
            where=denom > 0,
        )
    else:
        raise ValueError("metric must be one of {'jaccard', 'cosine'}")

    similarity = np.clip(similarity, 0.0, 1.0)
    if n_labels:
        np.fill_diagonal(similarity, 1.0)
    return similarity


def empirical_cooccurrence_distance_matrix(
    labels: np.ndarray,
    *,
    metric: str = "jaccard",
) -> np.ndarray:
    """Return an empirical distance matrix derived from label co-occurrence."""
    similarity = empirical_cooccurrence_similarity_matrix(labels, metric=metric)
    distances = 1.0 - similarity
    if distances.size:
        np.fill_diagonal(distances, 0.0)
    return distances
