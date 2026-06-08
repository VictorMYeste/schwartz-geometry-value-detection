import importlib.util
import sys
from pathlib import Path

import pandas as pd


def _load_control_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "bootstrap_energy_decoder_controls.py"
    )
    spec = importlib.util.spec_from_file_location("bootstrap_energy_decoder_controls", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_seed_level_control_comparison_without_sample_bootstrap(tmp_path):
    module = _load_control_module()
    decoder_output_dir = tmp_path / "decoder"
    output_dir = tmp_path / "control"
    decoder_output_dir.mkdir()

    rows = []
    for geometry, macro_f1, cost in [
        ("schwartz", 0.31, 0.40),
        ("random", 0.30, 0.50),
        ("empirical", 0.29, 0.45),
    ]:
        rows.append(
            {
                "method": "bce",
                "seed": 42,
                "model_slug": "deberta-v3-base",
                "geometry": geometry,
                "family": "full",
                "objective": "pareto_99",
                "top_k": 8,
                "max_candidates": 8,
                "max_labels": 5,
                "threshold_factor": 0.5,
                "min_prob": 0.01,
                "marginal_temperature": 1.0,
                "neighbor_steps": 2,
                "random_seed": 42,
                "empirical_metric": "jaccard",
                "alpha_neighbor": 0.1,
                "beta_opposite": 0.2,
                "gamma_cardinality": 0.0,
                "test_macro_f1": macro_f1,
                "test_decoder_geometry_cost": cost,
            }
        )
    pd.DataFrame(rows).to_csv(
        decoder_output_dir / "schwartz_energy_decoder_selected_seed_results.csv",
        index=False,
    )

    paths = module.run_control_bootstrap(
        decoder_output_dir=decoder_output_dir,
        predictions_dir=tmp_path / "predictions",
        logs_dir=tmp_path / "logs",
        output_dir=output_dir,
        methods={"bce"},
        seeds={42},
        model_slug="deberta-v3-base",
        family="full",
        objective="pareto_99",
        controls=["random", "empirical"],
        metrics=["macro_f1", "decoder_geometry_cost"],
        bootstrap_iterations=0,
        bootstrap_seed=42,
    )

    seed_deltas = pd.read_csv(paths["seed_deltas"])
    assert set(seed_deltas["comparison"]) == {
        "schwartz_vs_random",
        "schwartz_vs_empirical",
    }
    random_row = seed_deltas[seed_deltas["control_geometry"] == "random"].iloc[0]
    assert round(float(random_row["delta_macro_f1"]), 6) == 0.01
    assert round(float(random_row["schwartz_improvement_decoder_geometry_cost"]), 6) == 0.10
