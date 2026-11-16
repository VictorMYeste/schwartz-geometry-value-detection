import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_module():
    root = Path(__file__).resolve().parents[1]
    for path in (root / "src", root / "scripts"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    path = root / "scripts" / "eval_decoder_edits_external.py"
    spec = importlib.util.spec_from_file_location("eval_decoder_edits_external", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_edit_outcomes_toy_counts():
    module = _load_module()
    gold = np.array([
        [1, 0, 0, 0],  # removal of a false positive -> improved
        [0, 1, 0, 0],  # addition of a gold label -> improved
        [1, 0, 0, 0],  # addition of a non-gold label -> worsened
        [0, 0, 0, 0],  # removal on a value-free sentence -> improved (empty == empty)
        [1, 1, 0, 0],  # swap: one non-gold out, one non-gold in -> unchanged F1, mixed
        [0, 0, 1, 0],  # not edited
    ])
    y_thr = np.array([
        [1, 1, 0, 0],
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [1, 0, 1, 0],
        [0, 0, 1, 0],
    ])
    y_dec = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 1],
        [0, 0, 0, 0],
        [1, 0, 0, 1],
        [0, 0, 1, 0],
    ])
    out = module.edit_outcomes(gold, y_thr, y_dec)
    assert out["n_sentences"] == 6
    assert out["n_edited"] == 5
    assert (out["n_edited_addition_only"], out["n_edited_removal_only"], out["n_edited_mixed"]) == (2, 2, 1)
    assert out["n_edited_gold_empty"] == 1
    assert (out["n_improved"], out["n_worsened"], out["n_unchanged"]) == (3, 1, 1)
    assert (out["n_added"], out["n_added_in_gold"]) == (3, 1)
    assert (out["n_removed"], out["n_removed_not_in_gold"]) == (3, 3)
    assert np.isclose(out["share_added_in_gold"], 1 / 3)
    assert np.isclose(out["share_removed_not_in_gold"], 1.0)
    assert np.isclose(out["win_rate"], 3 / 4)


def test_matched_construction_and_vectorized_metrics_match_repository():
    module = _load_module()
    from schwartz_energy_decoder import schwartz_geometry_matrices
    from schwartz_value_geometry.geometry import SCHWARTZ_VALUE_ORDER

    # Label columns in a non-theory order, as in the prediction files.
    names = list(SCHWARTZ_VALUE_ORDER)
    names[14], names[15] = names[15], names[14]
    matched = module.matched_geometry_for_order(names, SCHWARTZ_VALUE_ORDER, neighbor_steps=2)
    reference = schwartz_geometry_matrices(names, neighbor_steps=2)
    assert np.array_equal(matched.neighbor_similarity, reference.neighbor_similarity)
    assert np.array_equal(matched.opposite_mask, reference.opposite_mask)

    rng = np.random.default_rng(0)
    gold = (rng.random((300, 19)) < 0.08).astype(int)
    pred = (rng.random((300, 19)) < 0.10).astype(int)
    module.SchwartzMetrics(names).verify(gold, pred)  # raises if any term differs
