"""Evaluate LLM diagnostic predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from schwartz_value_geometry.data.dataset import get_label_names  # noqa: E402
from schwartz_value_geometry.eval.metrics import (  # noqa: E402
    compute_f1_metrics,
    compute_geometry_metrics,
)
from schwartz_value_geometry.utils.logging import get_logger  # noqa: E402

LOGGER = get_logger(__name__)


def load_prediction_arrays(path: Path, label_names: list[str]) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    y_true: list[list[int]] = []
    y_pred: list[list[int]] = []
    label_set = set(label_names)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(row)
            gold = set(row.get("gold_labels", []) or [])
            pred = set(row.get("pred_labels", []) or [])
            unknown_gold = sorted(gold - label_set)
            unknown_pred = sorted(pred - label_set)
            if unknown_gold or unknown_pred:
                raise ValueError(
                    f"Unknown labels in {path}: gold={unknown_gold} pred={unknown_pred}"
                )
            y_true.append([1 if label in gold else 0 for label in label_names])
            y_pred.append([1 if label in pred else 0 for label in label_names])
    return np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int), rows


def infer_metrics_path(pred_path: Path) -> Path:
    root = pred_path.parents[1]
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir / f"{pred_path.stem}_metrics.json"


def evaluate_predictions(pred_path: Path) -> dict[str, Any]:
    label_names = get_label_names()
    y_true, y_pred, rows = load_prediction_arrays(pred_path, label_names)
    y_scores = y_pred.astype(float)
    metrics: dict[str, Any] = {}
    metrics.update(compute_f1_metrics(y_true, y_pred, label_names=label_names))
    metrics.update(compute_geometry_metrics(y_true, y_pred, y_scores, label_names=label_names))
    metrics["decoder_geometry_cost"] = float(
        metrics["neighbor_error_rate"]
        + metrics["opposite_error_rate"]
        + metrics["confusion_distance_correlation"]
    )

    total = len(rows)
    invalid = sum(bool(row.get("invalid_output", False)) for row in rows)
    repaired = sum(bool(row.get("repaired_output", False)) for row in rows)
    empty_pred = int((y_pred.sum(axis=1) == 0).sum()) if y_pred.size else 0
    metrics.update(
        {
            "n_samples": total,
            "invalid_output_count": invalid,
            "invalid_output_rate": invalid / total if total else 0.0,
            "repaired_output_count": repaired,
            "repaired_output_rate": repaired / total if total else 0.0,
            "empty_prediction_count": empty_pred,
            "empty_prediction_rate": empty_pred / total if total else 0.0,
            "avg_predicted_labels": float(y_pred.sum(axis=1).mean()) if y_pred.size else 0.0,
            "macro_auprc": None,
            "micro_auprc": None,
            "auprc_note": "not_applicable_for_label_only_llm_outputs",
            "meta": {
                "prediction_file": str(pred_path),
                "model_name": rows[0].get("model_name") if rows else None,
                "model_slug": rows[0].get("model_slug") if rows else None,
                "prompt_type": rows[0].get("prompt_type") if rows else None,
                "split": rows[0].get("split") if rows else None,
            },
        }
    )

    metrics_path = infer_metrics_path(pred_path)
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    per_label_rows = []
    per_label_f1 = metrics.get("per_label_f1", {})
    support = metrics.get("per_label_support", {})
    for label in label_names:
        row = {"label": label, "f1": per_label_f1.get(label, 0.0)}
        row.update(support.get(label, {}))
        per_label_rows.append(row)
    pd.DataFrame(per_label_rows).to_csv(
        metrics_path.with_name(metrics_path.stem + "_per_label.csv"),
        index=False,
    )
    LOGGER.info(
        "LLM metrics macro_f1=%.4f micro_f1=%.4f invalid=%.4f geometry_cost=%.4f",
        metrics["macro_f1"],
        metrics["micro_f1"],
        metrics["invalid_output_rate"],
        metrics["decoder_geometry_cost"],
    )
    LOGGER.info("Saved LLM metrics to %s", metrics_path)
    return metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LLM diagnostic predictions.")
    parser.add_argument("--predictions", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    pred_path = Path(args.predictions)
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)
    evaluate_predictions(pred_path)


if __name__ == "__main__":
    main()
