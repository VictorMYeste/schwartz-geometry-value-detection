import numpy as np
from schwartz_value_geometry.geometry import (
    SCHWARTZ_VALUE_ORDER,
    circular_distance_matrix,
    circular_step_distance_matrix,
    distance_matrix_for_order,
    empirical_cooccurrence_distance_matrix,
    theory_positions,
)


def test_theory_positions_follow_canonical_order():
    positions = theory_positions(SCHWARTZ_VALUE_ORDER)
    assert positions.tolist() == list(range(19))


def test_distance_matrix_wraps_around_circle():
    distances = circular_distance_matrix(SCHWARTZ_VALUE_ORDER)
    assert distances.shape == (19, 19)
    assert np.allclose(np.diag(distances), 0.0)
    assert distances[0, 1] == distances[0, 18]
    assert distances[0, 9] > distances[0, 2]


def test_geometry_reorders_to_model_label_order():
    model_order = list(SCHWARTZ_VALUE_ORDER)
    model_order[14], model_order[15] = model_order[15], model_order[14]
    step_distances = circular_step_distance_matrix(model_order)
    idx_caring = model_order.index("Benevolence: caring")
    idx_dependability = model_order.index("Benevolence: dependability")
    assert step_distances[idx_caring, idx_dependability] == 1


def test_custom_order_distance_matrix():
    custom_order = tuple(reversed(SCHWARTZ_VALUE_ORDER))
    distances = distance_matrix_for_order(SCHWARTZ_VALUE_ORDER, custom_order)
    assert distances.shape == (19, 19)
    assert np.allclose(distances, distances.T)


def test_empirical_cooccurrence_distance_matrix():
    labels = np.zeros((4, 3), dtype=float)
    labels[0, [0, 1]] = 1
    labels[1, [0, 1]] = 1
    labels[2, [0, 2]] = 1
    labels[3, 2] = 1

    distances = empirical_cooccurrence_distance_matrix(labels)

    assert distances.shape == (3, 3)
    assert np.allclose(np.diag(distances), 0.0)
    assert distances[0, 1] < distances[1, 2]
