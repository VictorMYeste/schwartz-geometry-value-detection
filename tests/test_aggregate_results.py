import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd


def _load_aggregate_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "aggregate_results.py"
    spec = importlib.util.spec_from_file_location("aggregate_results", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_metric(path: Path, *, seed: int, macro_f1: float) -> None:
    payload = {
        "macro_f1": macro_f1,
        "micro_f1": macro_f1 + 0.1,
        "macro_auprc": macro_f1 + 0.2,
        "micro_auprc": macro_f1 + 0.3,
        "circular_error": 1.0 - macro_f1,
        "opposite_value_activation": 0.2,
        "neighbor_error_rate": 0.5,
        "confusion_distance_correlation": -0.1,
        "per_label_f1": {"A": macro_f1, "B": macro_f1 / 2},
        "per_label_auprc": {"A": macro_f1 + 0.1, "B": macro_f1 / 2 + 0.1},
        "per_label_support": {
            "A": {"gold": 10, "pred": 9, "gold_rate": 0.1, "pred_rate": 0.09},
            "B": {"gold": 5, "pred": 6, "gold_rate": 0.05, "pred_rate": 0.06},
        },
        "threshold_tuning": {
            "mode": "per_label",
            "thresholds": [0.2, 0.3],
            "thresholds_by_label": {"A": 0.2, "B": 0.3},
        },
        "meta": {
            "model_name": "microsoft/deberta-v3-base",
            "loss": {"name": "schwartz_geoloss"},
            "seed": seed,
            "split": "test",
            "threshold": {"A": 0.2, "B": 0.3},
            "threshold_source": "provided",
        },
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_write_tables_creates_expected_csvs(tmp_path):
    module = _load_aggregate_module()
    metrics_dir = tmp_path / "logs"
    output_dir = tmp_path / "analysis"
    metrics_dir.mkdir()
    metric_1 = metrics_dir / "deberta_schwartz_geoloss_seed42_deberta-v3-base_test_metrics.json"
    metric_2 = metrics_dir / "deberta_schwartz_geoloss_seed7_deberta-v3-base_test_metrics.json"
    _write_metric(metric_1, seed=42, macro_f1=0.4)
    _write_metric(metric_2, seed=7, macro_f1=0.6)

    paths = module.write_tables(
        metrics_paths=[metric_1, metric_2],
        output_dir=output_dir,
        split="test",
    )

    assert set(paths) == {
        "seed_level_results",
        "main_supervised_results",
        "mean_std_summary",
        "per_label_results",
        "per_label_summary",
        "geometry_metrics",
        "threshold_table",
    }
    seed_df = pd.read_csv(paths["seed_level_results"])
    main_df = pd.read_csv(paths["main_supervised_results"])
    per_label_df = pd.read_csv(paths["per_label_results"])
    threshold_df = pd.read_csv(paths["threshold_table"])

    assert len(seed_df) == 2
    assert len(per_label_df) == 4
    assert len(threshold_df) == 4
    assert main_df.loc[0, "macro_f1_mean_std"].startswith("0.5000")


def test_find_metrics_files_is_recursive(tmp_path):
    module = _load_aggregate_module()
    nested = tmp_path / "results" / "logs" / "nested"
    nested.mkdir(parents=True)
    metric_path = nested / "example_metrics.json"
    metric_path.write_text('{"macro_f1": 0.5, "meta": {"loss": {"name": "bce"}}}\n')

    paths = module.find_metrics_files(tmp_path / "results", "*.json")

    assert paths == [metric_path]
