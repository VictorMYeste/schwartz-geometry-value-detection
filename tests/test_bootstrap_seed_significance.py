import importlib.util
import pytest
import sys
from pathlib import Path


def _load_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "bootstrap_seed_significance.py"
    )
    spec = importlib.util.spec_from_file_location("bootstrap_seed_significance", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_significance_rows_uses_paired_seed_deltas():
    module = _load_module()
    rows = [
        {"method": "bce", "seed": "1", "split": "test", "threshold_source": "provided", "macro_f1": "0.4"},
        {"method": "bce", "seed": "2", "split": "test", "threshold_source": "provided", "macro_f1": "0.5"},
        {"method": "schwartz_geoloss", "seed": "1", "split": "test", "threshold_source": "provided", "macro_f1": "0.5"},
        {"method": "schwartz_geoloss", "seed": "2", "split": "test", "threshold_source": "provided", "macro_f1": "0.7"},
    ]

    output = module.build_significance_rows(
        rows,
        baseline="bce",
        methods=["schwartz_geoloss"],
        metrics=["macro_f1"],
        n_iterations=100,
        seed=42,
        alpha=0.05,
    )

    assert len(output) == 1
    assert output[0]["n_seed_pairs"] == 2
    assert output[0]["delta_method_minus_baseline"] == pytest.approx(0.15)
    assert output[0]["wins_over_baseline"] == 2
