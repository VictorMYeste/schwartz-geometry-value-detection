import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from schwartz_value_geometry.geometry import SCHWARTZ_VALUE_ORDER


def _load_calibration_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "geometry_aware_calibration.py"
    )
    spec = importlib.util.spec_from_file_location("geometry_aware_calibration", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _probabilities(active: dict[str, float]) -> dict[str, float]:
    return {label: float(active.get(label, 0.01)) for label in SCHWARTZ_VALUE_ORDER}


def _write_predictions(path: Path) -> None:
    rows = [
        {
            "text_id": "a",
            "sent_id": "1",
            "gold_labels": ["Self-direction: thought"],
            "pred_labels": [],
            "probabilities": _probabilities({"Self-direction: thought": 0.9}),
        },
        {
            "text_id": "a",
            "sent_id": "2",
            "gold_labels": ["Security: societal"],
            "pred_labels": [],
            "probabilities": _probabilities({"Security: societal": 0.8}),
        },
        {
            "text_id": "a",
            "sent_id": "3",
            "gold_labels": [],
            "pred_labels": [],
            "probabilities": _probabilities({"Power: dominance": 0.7}),
        },
    ]
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def _write_threshold_metrics(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "meta": {
                    "threshold": {label: 0.5 for label in SCHWARTZ_VALUE_ORDER},
                    "seed": 42,
                    "loss": {"name": "bce"},
                    "split": "validation",
                }
            }
        ),
        encoding="utf-8",
    )


def test_opposite_suppression_reduces_far_probability_more_than_near():
    module = _load_calibration_module()
    label_names = list(SCHWARTZ_VALUE_ORDER)
    probs = np.zeros((1, len(label_names)), dtype=float)
    probs[0, 0] = 0.9
    probs[0, 1] = 0.4
    probs[0, 9] = 0.4

    calibrated = module.apply_opposite_suppression(
        probs,
        label_names=label_names,
        rho=2.0,
    )

    near_ratio = calibrated[0, 1] / probs[0, 1]
    far_ratio = calibrated[0, 9] / probs[0, 9]
    assert far_ratio < near_ratio


def test_run_calibration_writes_selected_results(tmp_path):
    module = _load_calibration_module()
    predictions_dir = tmp_path / "predictions"
    logs_dir = tmp_path / "logs"
    output_dir = tmp_path / "analysis"
    predictions_dir.mkdir()
    logs_dir.mkdir()

    validation_path = (
        predictions_dir
        / "deberta_bce_seed42_deberta-v3-base_validation_thresholds.jsonl"
    )
    test_path = predictions_dir / "deberta_bce_seed42_deberta-v3-base_test.jsonl"
    metrics_path = logs_dir / "deberta_bce_seed42_deberta-v3-base_validation_thresholds.json"
    _write_predictions(validation_path)
    _write_predictions(test_path)
    _write_threshold_metrics(metrics_path)

    paths = module.run_calibration(
        predictions_dir=predictions_dir,
        logs_dir=logs_dir,
        output_dir=output_dir,
        methods={"bce"},
        seeds={42},
        model_slug="deberta-v3-base",
        rho_values=[0.0, 0.5],
        threshold_deltas=[0.0],
        opposite_margins=[None, 0.0],
        objectives=["standard", "f1", "pareto_99"],
    )

    assert set(paths) == {
        "validation_grid",
        "selected_seed_results",
        "selected_mean_std",
        "delta_vs_standard",
        "thresholds",
    }
    selected = pd.read_csv(paths["selected_seed_results"])
    assert set(selected["objective"]) == {"standard", "f1", "pareto_99"}
    assert selected["test_macro_f1"].notna().all()
