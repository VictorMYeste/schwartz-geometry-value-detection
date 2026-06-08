import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from schwartz_value_geometry.geometry import SCHWARTZ_VALUE_ORDER


def _load_decoder_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "schwartz_energy_decoder.py"
    spec = importlib.util.spec_from_file_location("schwartz_energy_decoder", path)
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
            "probabilities": _probabilities(
                {
                    "Self-direction: thought": 0.85,
                    "Security: societal": 0.70,
                }
            ),
        },
        {
            "text_id": "a",
            "sent_id": "2",
            "gold_labels": ["Security: societal"],
            "pred_labels": [],
            "probabilities": _probabilities({"Security: societal": 0.80}),
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


def test_zero_energy_setting_reproduces_threshold_predictions():
    module = _load_decoder_module()
    label_names = list(SCHWARTZ_VALUE_ORDER)
    probs = np.zeros((1, len(label_names)), dtype=float)
    probs[0, 0] = 0.8
    probs[0, 9] = 0.7
    thresholds = np.full(len(label_names), 0.5, dtype=float)
    predictions = module.PredictionSet(
        label_names=label_names,
        y_true=np.zeros_like(probs, dtype=int),
        y_probs=probs,
    )

    y_pred, y_scores = module.decode_predictions(
        predictions,
        thresholds=thresholds,
        setting=module.EnergySetting(0.0, 0.0, 0.0),
        top_k=8,
        max_candidates=8,
        max_labels=5,
        threshold_factor=0.5,
        min_prob=0.01,
        marginal_temperature=1.0,
        neighbor_steps=2,
    )

    assert y_pred.tolist() == (probs >= thresholds).astype(int).tolist()
    assert np.allclose(y_scores, probs)


def test_opposite_penalty_removes_lower_confidence_conflict():
    module = _load_decoder_module()
    label_names = list(SCHWARTZ_VALUE_ORDER)
    probs = np.full(len(label_names), 0.01, dtype=float)
    probs[0] = 0.85
    probs[9] = 0.70
    thresholds = np.full(len(label_names), 0.5, dtype=float)
    pairwise = module.build_pairwise_matrix(
        label_names,
        alpha_neighbor=0.0,
        beta_opposite=2.0,
        neighbor_steps=2,
    )

    y_pred, _ = module.decode_one(
        probs,
        thresholds,
        setting=module.EnergySetting(0.0, 2.0, 0.0),
        pairwise=pairwise,
        top_k=8,
        max_candidates=8,
        max_labels=5,
        threshold_factor=0.5,
        min_prob=0.01,
        marginal_temperature=1.0,
    )

    assert y_pred[0] == 1
    assert y_pred[9] == 0


def test_run_decoder_writes_selected_results(tmp_path):
    module = _load_decoder_module()
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

    paths = module.run_decoder(
        predictions_dir=predictions_dir,
        logs_dir=logs_dir,
        output_dir=output_dir,
        methods={"bce"},
        seeds={42},
        model_slug="deberta-v3-base",
        geometries=["schwartz"],
        families=["full"],
        alpha_values=[0.0],
        beta_values=[0.0, 2.0],
        gamma_values=[0.0],
        objectives=["standard", "f1", "pareto_99"],
        top_k=8,
        max_candidates=8,
        max_labels=5,
        threshold_factor=0.5,
        min_prob=0.01,
        marginal_temperature=1.0,
        neighbor_steps=2,
        random_seed=42,
        empirical_metric="jaccard",
        bootstrap_iterations=10,
        bootstrap_seed=42,
        bootstrap_metrics=["macro_f1", "decoder_geometry_cost"],
        bootstrap_geometries={"schwartz"},
        bootstrap_families={"full"},
        bootstrap_objectives={"pareto_99"},
        max_error_examples=2,
        example_geometries={"schwartz"},
        example_families={"full"},
        example_objectives={"pareto_99"},
    )

    assert set(paths) == {
        "validation_grid",
        "selected_seed_results",
        "selected_mean_std",
        "delta_vs_standard",
        "sample_bootstrap",
        "error_examples",
    }
    selected = pd.read_csv(paths["selected_seed_results"])
    assert set(selected["objective"]) == {"standard", "f1", "pareto_99"}
    assert selected["test_macro_f1"].notna().all()
    bootstrap = pd.read_csv(paths["sample_bootstrap"])
    assert set(bootstrap["metric"]) == {"macro_f1", "decoder_geometry_cost"}
