import numpy as np
import pytest
from schwartz_value_geometry.eval.metrics import (
    binarize_probs,
    compute_all_metrics,
    compute_global_metrics,
    compute_opposite_error_rate,
    compute_opposite_error_rate_value_positive,
    macro_f1_from_arrays,
    sweep_per_label_thresholds,
)
from schwartz_value_geometry.geometry import SCHWARTZ_VALUE_ORDER


def test_macro_f1_from_arrays():
    y_true = np.array([[1, 0], [0, 1]])
    y_pred = np.array([[1, 0], [1, 0]])
    macro_f1 = macro_f1_from_arrays(y_true, y_pred)
    assert 0.0 <= macro_f1 <= 1.0


def test_compute_global_metrics_keys():
    y_true = np.array([[1, 0], [0, 1]])
    y_pred = np.array([[1, 0], [1, 0]])
    metrics = compute_global_metrics(y_true, y_pred)
    assert "micro_f1" in metrics
    assert "macro_f1" in metrics


def test_binarize_probs_accepts_per_label_thresholds():
    probs = np.array([[0.4, 0.6], [0.7, 0.2]])
    thresholds = np.array([0.5, 0.5])
    pred = binarize_probs(probs, threshold=thresholds)
    assert pred.tolist() == [[0, 1], [1, 0]]


def test_compute_all_metrics_includes_geometry_metrics():
    label_names = list(SCHWARTZ_VALUE_ORDER)
    y_true = np.zeros((2, 19), dtype=int)
    y_true[0, 0] = 1
    y_true[1, 9] = 1
    y_probs = np.zeros((2, 19), dtype=float)
    y_probs[0, 1] = 0.9
    y_probs[1, 18] = 0.8
    y_pred = (y_probs >= 0.5).astype(int)
    metrics = compute_all_metrics(y_true, y_pred, y_probs, label_names=label_names)
    assert "macro_auprc" in metrics
    assert "circular_error" in metrics
    assert "opposite_value_activation" in metrics
    assert "neighbor_error_rate" in metrics
    assert "confusion_distance_correlation" in metrics


def test_sweep_per_label_thresholds_returns_one_threshold_per_label():
    label_names = list(SCHWARTZ_VALUE_ORDER)
    y_true = np.zeros((3, 19), dtype=int)
    y_probs = np.zeros((3, 19), dtype=float)
    y_true[:, 0] = [1, 0, 1]
    y_true[:, 1] = [0, 1, 0]
    y_probs[:, 0] = [0.9, 0.2, 0.7]
    y_probs[:, 1] = [0.1, 0.8, 0.4]
    sweep = sweep_per_label_thresholds(
        y_true,
        y_probs,
        label_names=label_names,
        start=0.0,
        stop=1.0,
        step=0.1,
    )
    assert len(sweep["best_thresholds"]) == 19
    assert set(sweep["best_thresholds_by_label"]) == set(label_names)


def _three_sample_case():
    """Gold Self-direction: thought (index 0) in samples 1-2, no gold value in sample 3.

    False positives: Security: societal (index 9, nine steps from index 0, so opposite),
    Self-direction: action (index 1, a neighbor), and Achievement (index 4) on the value-free
    sample, which the plain rate counts as opposite.
    """
    y_true = np.zeros((3, 19), dtype=int)
    y_pred = np.zeros((3, 19), dtype=int)
    y_true[0, 0] = y_pred[0, 0] = 1
    y_pred[0, 9] = 1
    y_true[1, 0] = 1
    y_pred[1, 1] = 1
    y_pred[2, 4] = 1
    return y_true, y_pred


def test_opposite_error_rate_value_positive_drops_value_free_samples():
    label_names = list(SCHWARTZ_VALUE_ORDER)
    y_true, y_pred = _three_sample_case()
    assert compute_opposite_error_rate(y_true, y_pred, label_names=label_names) == pytest.approx(2 / 3)
    assert compute_opposite_error_rate_value_positive(
        y_true, y_pred, label_names=label_names) == pytest.approx(1 / 2)


def test_opposite_error_rate_value_positive_matches_plain_rate_on_positive_rows():
    label_names = list(SCHWARTZ_VALUE_ORDER)
    rng = np.random.default_rng(0)
    y_true = (rng.random((200, 19)) < 0.06).astype(int)
    y_pred = (rng.random((200, 19)) < 0.08).astype(int)
    keep = y_true.sum(axis=1) > 0
    assert 0 < keep.sum() < len(keep)
    expected = compute_opposite_error_rate(y_true[keep], y_pred[keep], label_names=label_names)
    assert compute_opposite_error_rate_value_positive(
        y_true, y_pred, label_names=label_names) == pytest.approx(expected)
    # With no value-free samples, the two rates coincide.
    assert compute_opposite_error_rate_value_positive(
        y_true[keep], y_pred[keep], label_names=label_names
    ) == pytest.approx(compute_opposite_error_rate(y_true[keep], y_pred[keep], label_names=label_names))


def test_opposite_error_rate_value_positive_edge_cases():
    label_names = list(SCHWARTZ_VALUE_ORDER)
    no_gold = np.zeros((2, 19), dtype=int)
    pred = np.ones((2, 19), dtype=int)
    assert compute_opposite_error_rate_value_positive(no_gold, pred, label_names=label_names) == 0.0
    with pytest.raises(ValueError):
        compute_opposite_error_rate_value_positive(no_gold, pred[:1], label_names=label_names)
