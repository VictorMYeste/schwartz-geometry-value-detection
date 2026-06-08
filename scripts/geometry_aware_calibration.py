"""Validation-only geometry-aware output calibration.

This script reuses saved validation/test probability JSONL files. It tunes a
small post-processing grid on validation, then applies the selected calibration
once to test. No model weights are updated.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from schwartz_value_geometry.eval.metrics import (  # noqa: E402
    binarize_probs,
    compute_all_metrics,
)
from schwartz_value_geometry.geometry import (  # noqa: E402
    OPPOSITE_DISTANCE_THRESHOLD,
    circular_distance_matrix,
    validate_label_names,
)
from schwartz_value_geometry.utils.logging import get_logger  # noqa: E402

LOGGER = get_logger(__name__)

METRIC_COLUMNS = [
    "macro_f1",
    "micro_f1",
    "macro_auprc",
    "micro_auprc",
    "circular_error",
    "opposite_value_activation",
    "neighbor_error_rate",
    "opposite_error_rate",
    "confusion_distance_correlation",
]

OBJECTIVE_TOLERANCES = {
    "standard": None,
    "f1": None,
    "pareto_99": 0.99,
    "pareto_98": 0.98,
    "pareto_95": 0.95,
}

PREDICTION_RE = re.compile(
    r"^deberta_(?P<method>.+)_seed(?P<seed>\d+)_(?P<model_slug>.+)_"
    r"(?P<split>validation_thresholds|test)\.jsonl$"
)


@dataclass(frozen=True)
class PredictionSet:
    label_names: list[str]
    y_true: np.ndarray
    y_probs: np.ndarray


@dataclass(frozen=True)
class RunFiles:
    method: str
    seed: int
    model_slug: str
    validation_predictions: Path
    test_predictions: Path
    validation_metrics: Path


@dataclass(frozen=True)
class CalibrationSetting:
    rho: float
    threshold_delta: float
    opposite_margin: float | None

    @property
    def transform(self) -> str:
        return "none" if self.rho == 0.0 else "opposite_suppression"

    @property
    def conflict_filter(self) -> str:
        return "none" if self.opposite_margin is None else "opposite_margin"


def parse_float_list(raw: str) -> list[float]:
    """Parse comma-separated floats from the CLI."""
    if not raw.strip():
        return []
    return [float(value.strip()) for value in raw.split(",") if value.strip()]


def parse_margin_list(raw: str) -> list[float | None]:
    """Parse comma-separated margins. Use 'none' for no conflict filtering."""
    margins: list[float | None] = []
    for value in raw.split(","):
        token = value.strip().lower()
        if not token:
            continue
        if token in {"none", "na", "null"}:
            margins.append(None)
        else:
            margins.append(float(token))
    return margins


def load_prediction_jsonl(path: Path) -> PredictionSet:
    """Load labels and probabilities from saved prediction JSONL."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"Prediction file is empty: {path}")

    first_probs = rows[0].get("probabilities", {})
    if not isinstance(first_probs, dict) or not first_probs:
        raise ValueError(f"Prediction rows must contain probabilities: {path}")
    label_names = list(first_probs)
    validate_label_names(label_names)
    label_index = {label: idx for idx, label in enumerate(label_names)}

    y_true = np.zeros((len(rows), len(label_names)), dtype=int)
    y_probs = np.zeros((len(rows), len(label_names)), dtype=float)
    for row_idx, row in enumerate(rows):
        probs = row.get("probabilities", {})
        if list(probs) != label_names:
            raise ValueError(f"Inconsistent probability label order in {path}")
        for label, value in probs.items():
            y_probs[row_idx, label_index[label]] = float(value)
        for label in row.get("gold_labels", []) or []:
            y_true[row_idx, label_index[str(label)]] = 1
    return PredictionSet(label_names=label_names, y_true=y_true, y_probs=y_probs)


def load_validation_thresholds(path: Path, label_names: list[str]) -> np.ndarray:
    """Load validation-frozen thresholds from a metrics JSON file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    threshold = meta.get("threshold") if isinstance(meta, dict) else None
    if threshold is None:
        tuning = payload.get("threshold_tuning", {})
        threshold = tuning.get("thresholds_by_label") or tuning.get("thresholds")

    if isinstance(threshold, dict):
        return np.asarray([float(threshold[label]) for label in label_names], dtype=float)
    if isinstance(threshold, list):
        arr = np.asarray(threshold, dtype=float)
        if arr.shape != (len(label_names),):
            raise ValueError(
                f"Threshold list in {path} has shape {arr.shape}, "
                f"expected ({len(label_names)},)"
            )
        return arr
    if isinstance(threshold, int | float):
        return np.full(len(label_names), float(threshold), dtype=float)
    raise ValueError(f"Could not load validation thresholds from {path}")


def discover_runs(
    *,
    predictions_dir: Path,
    logs_dir: Path,
    methods: set[str] | None,
    seeds: set[int] | None,
    model_slug: str | None,
) -> list[RunFiles]:
    """Find validation/test prediction pairs and matching validation metrics."""
    validation_files: dict[tuple[str, int, str], Path] = {}
    test_files: dict[tuple[str, int, str], Path] = {}

    for path in sorted(predictions_dir.glob("*.jsonl")):
        match = PREDICTION_RE.match(path.name)
        if not match:
            continue
        method = match.group("method")
        seed = int(match.group("seed"))
        slug = match.group("model_slug")
        split = match.group("split")
        if methods is not None and method not in methods:
            continue
        if seeds is not None and seed not in seeds:
            continue
        if model_slug is not None and slug != model_slug:
            continue
        key = (method, seed, slug)
        if split == "validation_thresholds":
            validation_files[key] = path
        elif split == "test":
            test_files[key] = path

    runs: list[RunFiles] = []
    for key, validation_path in sorted(validation_files.items()):
        test_path = test_files.get(key)
        if test_path is None:
            LOGGER.warning("Skipping %s: missing test predictions", validation_path.name)
            continue
        method, seed, slug = key
        metrics_path = (
            logs_dir / f"deberta_{method}_seed{seed}_{slug}_validation_thresholds.json"
        )
        if not metrics_path.exists():
            LOGGER.warning("Skipping %s: missing %s", validation_path.name, metrics_path)
            continue
        runs.append(
            RunFiles(
                method=method,
                seed=seed,
                model_slug=slug,
                validation_predictions=validation_path,
                test_predictions=test_path,
                validation_metrics=metrics_path,
            )
        )
    return runs


def apply_opposite_suppression(
    probs: np.ndarray,
    *,
    label_names: list[str],
    rho: float,
) -> np.ndarray:
    """Suppress labels far from the sample's own predicted probability mass."""
    probs = np.asarray(probs, dtype=float)
    if rho <= 0.0:
        return probs.copy()

    distances = circular_distance_matrix(label_names)
    mass = probs.sum(axis=1, keepdims=True)
    weights = np.divide(
        probs,
        mass,
        out=np.zeros_like(probs, dtype=float),
        where=mass > 0.0,
    )
    expected_distance = weights @ distances
    calibrated = probs * np.exp(-float(rho) * expected_distance)
    return np.clip(calibrated, 0.0, 1.0)


def apply_opposite_conflict_filter(
    y_pred: np.ndarray,
    probs: np.ndarray,
    *,
    label_names: list[str],
    margin: float | None,
) -> np.ndarray:
    """Remove lower-confidence predictions that conflict with stronger ones."""
    if margin is None:
        return y_pred.copy()

    filtered = np.asarray(y_pred, dtype=int).copy()
    probs = np.asarray(probs, dtype=float)
    opposite_mask = circular_distance_matrix(label_names) > OPPOSITE_DISTANCE_THRESHOLD
    margin_value = float(margin)

    for row_idx in range(filtered.shape[0]):
        selected = np.flatnonzero(filtered[row_idx])
        if selected.size < 2:
            continue
        selected_probs = probs[row_idx, selected]
        for selected_pos, label_idx in enumerate(selected):
            opponents = selected[opposite_mask[label_idx, selected]]
            if opponents.size == 0:
                continue
            stronger = probs[row_idx, opponents] > selected_probs[selected_pos] + margin_value
            if np.any(stronger):
                filtered[row_idx, label_idx] = 0
    return filtered


def evaluate_setting(
    predictions: PredictionSet,
    *,
    base_thresholds: np.ndarray,
    setting: CalibrationSetting,
) -> dict[str, Any]:
    """Apply one calibration setting and compute metrics."""
    probs = apply_opposite_suppression(
        predictions.y_probs,
        label_names=predictions.label_names,
        rho=setting.rho,
    )
    thresholds = np.clip(base_thresholds + setting.threshold_delta, 0.0, 1.0)
    y_pred = binarize_probs(probs, threshold=thresholds)
    y_pred = apply_opposite_conflict_filter(
        y_pred,
        probs,
        label_names=predictions.label_names,
        margin=setting.opposite_margin,
    )
    metrics = compute_all_metrics(
        predictions.y_true,
        y_pred,
        probs,
        label_names=predictions.label_names,
    )
    metrics["geometry_cost"] = geometry_cost(metrics)
    metrics["n_predictions"] = float(y_pred.sum())
    metrics["prediction_rate"] = float(y_pred.mean()) if y_pred.size else 0.0
    metrics["thresholds"] = thresholds.tolist()
    return metrics


def geometry_cost(metrics: dict[str, Any]) -> float:
    """Single scalar used only for validation-side Pareto selection."""
    return float(
        metrics["circular_error"]
        + metrics["opposite_value_activation"]
        + metrics["opposite_error_rate"]
    )


def scalar_metric_row(metrics: dict[str, Any], *, prefix: str = "") -> dict[str, float]:
    """Keep scalar metrics needed by output CSVs."""
    row = {f"{prefix}{metric}": float(metrics[metric]) for metric in METRIC_COLUMNS}
    row[f"{prefix}geometry_cost"] = float(metrics["geometry_cost"])
    row[f"{prefix}n_predictions"] = float(metrics["n_predictions"])
    row[f"{prefix}prediction_rate"] = float(metrics["prediction_rate"])
    return row


def setting_row(setting: CalibrationSetting) -> dict[str, Any]:
    """Serialize calibration setting for CSV rows."""
    return {
        "transform": setting.transform,
        "rho": setting.rho,
        "threshold_delta": setting.threshold_delta,
        "conflict_filter": setting.conflict_filter,
        "opposite_margin": "" if setting.opposite_margin is None else setting.opposite_margin,
    }


def build_grid(
    *,
    rho_values: list[float],
    threshold_deltas: list[float],
    opposite_margins: list[float | None],
) -> list[CalibrationSetting]:
    """Build deterministic calibration settings, with standard first."""
    settings: list[CalibrationSetting] = []
    seen: set[tuple[float, float, float | None]] = set()
    for rho in [0.0, *rho_values]:
        for threshold_delta in [0.0, *threshold_deltas]:
            for margin in [None, *opposite_margins]:
                key = (float(rho), float(threshold_delta), margin)
                if key in seen:
                    continue
                seen.add(key)
                settings.append(
                    CalibrationSetting(
                        rho=float(rho),
                        threshold_delta=float(threshold_delta),
                        opposite_margin=margin,
                    )
                )
    settings.sort(
        key=lambda item: (
            item.rho != 0.0,
            item.rho,
            item.threshold_delta != 0.0,
            item.threshold_delta,
            item.opposite_margin is not None,
            -1.0 if item.opposite_margin is None else item.opposite_margin,
        )
    )
    return settings


def select_setting(
    validation_rows: list[dict[str, Any]],
    *,
    objective: str,
) -> dict[str, Any]:
    """Select a validation row for the requested objective."""
    if objective == "standard":
        for row in validation_rows:
            if (
                float(row["rho"]) == 0.0
                and float(row["threshold_delta"]) == 0.0
                and str(row["conflict_filter"]) == "none"
            ):
                return row
        raise ValueError("Standard calibration setting not found")

    if objective == "f1":
        return max(
            validation_rows,
            key=lambda row: (
                float(row["validation_macro_f1"]),
                -float(row["validation_geometry_cost"]),
            ),
        )

    tolerance = OBJECTIVE_TOLERANCES.get(objective)
    if tolerance is None:
        raise ValueError(f"Unknown calibration objective: {objective}")
    best_macro = max(float(row["validation_macro_f1"]) for row in validation_rows)
    min_macro = best_macro * tolerance
    candidates = [
        row for row in validation_rows if float(row["validation_macro_f1"]) >= min_macro
    ]
    return min(
        candidates,
        key=lambda row: (
            float(row["validation_geometry_cost"]),
            -float(row["validation_macro_f1"]),
        ),
    )


def summarize_selected(seed_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build mean/std and delta-vs-standard tables."""
    metric_cols = [
        column
        for column in seed_df.columns
        if column.startswith("test_") and pd.api.types.is_numeric_dtype(seed_df[column])
    ]
    group_cols = ["method", "model_slug", "objective"]
    summary = seed_df.groupby(group_cols, dropna=False)[metric_cols].agg(
        ["mean", "std", "count"]
    )
    summary.columns = [
        "_".join(str(part) for part in column if part)
        for column in summary.columns
    ]
    summary = summary.reset_index()

    standard = seed_df[seed_df["objective"] == "standard"].copy()
    deltas: list[dict[str, Any]] = []
    key_cols = ["method", "model_slug", "seed"]
    for _, row in seed_df[seed_df["objective"] != "standard"].iterrows():
        mask = np.ones(len(standard), dtype=bool)
        for column in key_cols:
            mask &= standard[column].to_numpy() == row[column]
        baseline = standard[mask]
        if baseline.empty:
            continue
        baseline_row = baseline.iloc[0]
        delta_row: dict[str, Any] = {
            "method": row["method"],
            "model_slug": row["model_slug"],
            "seed": row["seed"],
            "objective": row["objective"],
        }
        for metric in METRIC_COLUMNS + [
            "geometry_cost",
            "n_predictions",
            "prediction_rate",
        ]:
            column = f"test_{metric}"
            delta_row[f"delta_{metric}"] = float(row[column]) - float(
                baseline_row[column]
            )
        deltas.append(delta_row)

    delta_df = pd.DataFrame(deltas)
    if not delta_df.empty:
        delta_metric_cols = [
            column
            for column in delta_df.columns
            if column.startswith("delta_")
            and pd.api.types.is_numeric_dtype(delta_df[column])
        ]
        delta_summary = delta_df.groupby(
            ["method", "model_slug", "objective"], dropna=False
        )[delta_metric_cols].agg(["mean", "std", "count"])
        delta_summary.columns = [
            "_".join(str(part) for part in column if part)
            for column in delta_summary.columns
        ]
        delta_summary = delta_summary.reset_index()
    else:
        delta_summary = pd.DataFrame()
    return summary, delta_summary


def run_calibration(
    *,
    predictions_dir: Path,
    logs_dir: Path,
    output_dir: Path,
    methods: set[str] | None,
    seeds: set[int] | None,
    model_slug: str | None,
    rho_values: list[float],
    threshold_deltas: list[float],
    opposite_margins: list[float | None],
    objectives: list[str],
) -> dict[str, Path]:
    """Run calibration for discovered final runs and write CSV outputs."""
    runs = discover_runs(
        predictions_dir=predictions_dir,
        logs_dir=logs_dir,
        methods=methods,
        seeds=seeds,
        model_slug=model_slug,
    )
    if not runs:
        raise ValueError("No prediction/metric run pairs found")

    output_dir.mkdir(parents=True, exist_ok=True)
    grid = build_grid(
        rho_values=rho_values,
        threshold_deltas=threshold_deltas,
        opposite_margins=opposite_margins,
    )
    LOGGER.info("Discovered %d runs; evaluating %d calibration settings", len(runs), len(grid))

    validation_grid_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []

    for run in runs:
        LOGGER.info("Calibrating method=%s seed=%s", run.method, run.seed)
        validation = load_prediction_jsonl(run.validation_predictions)
        test = load_prediction_jsonl(run.test_predictions)
        if validation.label_names != test.label_names:
            raise ValueError(f"Validation/test labels differ for {run}")
        base_thresholds = load_validation_thresholds(
            run.validation_metrics,
            validation.label_names,
        )

        run_validation_rows: list[dict[str, Any]] = []
        metrics_by_setting: dict[tuple[float, float, float | None], dict[str, Any]] = {}
        for setting in grid:
            metrics = evaluate_setting(
                validation,
                base_thresholds=base_thresholds,
                setting=setting,
            )
            key = (setting.rho, setting.threshold_delta, setting.opposite_margin)
            metrics_by_setting[key] = metrics
            row = {
                "method": run.method,
                "seed": run.seed,
                "model_slug": run.model_slug,
                **setting_row(setting),
                **scalar_metric_row(metrics, prefix="validation_"),
            }
            validation_grid_rows.append(row)
            run_validation_rows.append(row)

        for objective in objectives:
            selected = select_setting(run_validation_rows, objective=objective)
            selected_setting = CalibrationSetting(
                rho=float(selected["rho"]),
                threshold_delta=float(selected["threshold_delta"]),
                opposite_margin=(
                    None
                    if str(selected["opposite_margin"]) == ""
                    else float(selected["opposite_margin"])
                ),
            )
            validation_metrics = metrics_by_setting[
                (
                    selected_setting.rho,
                    selected_setting.threshold_delta,
                    selected_setting.opposite_margin,
                )
            ]
            test_metrics = evaluate_setting(
                test,
                base_thresholds=base_thresholds,
                setting=selected_setting,
            )
            row = {
                "method": run.method,
                "seed": run.seed,
                "model_slug": run.model_slug,
                "objective": objective,
                **setting_row(selected_setting),
                **scalar_metric_row(validation_metrics, prefix="validation_"),
                **scalar_metric_row(test_metrics, prefix="test_"),
            }
            selected_rows.append(row)

            thresholds = np.asarray(test_metrics["thresholds"], dtype=float)
            for label, threshold in zip(validation.label_names, thresholds, strict=False):
                threshold_rows.append(
                    {
                        "method": run.method,
                        "seed": run.seed,
                        "model_slug": run.model_slug,
                        "objective": objective,
                        "label": label,
                        "threshold": float(threshold),
                        **setting_row(selected_setting),
                    }
                )

    validation_grid_df = pd.DataFrame(validation_grid_rows)
    selected_df = pd.DataFrame(selected_rows)
    thresholds_df = pd.DataFrame(threshold_rows)
    summary_df, delta_summary_df = summarize_selected(selected_df)

    paths = {
        "validation_grid": output_dir / "geometry_calibration_validation_grid.csv",
        "selected_seed_results": output_dir / "geometry_calibration_selected_seed_results.csv",
        "selected_mean_std": output_dir / "geometry_calibration_selected_mean_std.csv",
        "delta_vs_standard": output_dir / "geometry_calibration_delta_vs_standard.csv",
        "thresholds": output_dir / "geometry_calibration_thresholds.csv",
    }
    validation_grid_df.to_csv(paths["validation_grid"], index=False)
    selected_df.to_csv(paths["selected_seed_results"], index=False)
    summary_df.to_csv(paths["selected_mean_std"], index=False)
    delta_summary_df.to_csv(paths["delta_vs_standard"], index=False)
    thresholds_df.to_csv(paths["thresholds"], index=False)
    return paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tune a lightweight Schwartz-aware calibration layer on validation "
            "predictions and apply it to test predictions."
        )
    )
    parser.add_argument(
        "--predictions_dir",
        default="results/predictions",
        help="Directory with saved validation/test prediction JSONL files.",
    )
    parser.add_argument(
        "--logs_dir",
        default="results/logs",
        help="Directory with validation threshold metrics JSON files.",
    )
    parser.add_argument(
        "--output_dir",
        default="results/analysis/geometry_calibration",
        help="Directory where calibration CSVs are written.",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="Optional method filter, e.g. bce schwartz_geoloss.",
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=None,
        help="Optional seed filter.",
    )
    parser.add_argument(
        "--model_slug",
        default="deberta-v3-base",
        help="Filename model slug to include.",
    )
    parser.add_argument(
        "--rho_values",
        default="0.0,0.5,1.0",
        help="Comma-separated opposite-suppression strengths.",
    )
    parser.add_argument(
        "--threshold_deltas",
        default="-0.05,0.0,0.05",
        help="Comma-separated offsets added to validation-frozen thresholds.",
    )
    parser.add_argument(
        "--opposite_margins",
        default="none,0.0,0.10",
        help="Comma-separated conflict-filter margins; include 'none' for no filter.",
    )
    parser.add_argument(
        "--objectives",
        nargs="+",
        default=["standard", "f1", "pareto_99", "pareto_98", "pareto_95"],
        choices=sorted(OBJECTIVE_TOLERANCES),
        help="Validation selection objectives to report.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    paths = run_calibration(
        predictions_dir=Path(args.predictions_dir),
        logs_dir=Path(args.logs_dir),
        output_dir=Path(args.output_dir),
        methods=set(args.methods) if args.methods else None,
        seeds=set(args.seeds) if args.seeds else None,
        model_slug=str(args.model_slug) if args.model_slug else None,
        rho_values=parse_float_list(str(args.rho_values)),
        threshold_deltas=parse_float_list(str(args.threshold_deltas)),
        opposite_margins=parse_margin_list(str(args.opposite_margins)),
        objectives=list(args.objectives),
    )
    for name, path in paths.items():
        LOGGER.info("Wrote %s to %s", name, path)


if __name__ == "__main__":
    main()
