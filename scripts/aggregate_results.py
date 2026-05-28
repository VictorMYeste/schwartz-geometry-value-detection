"""Aggregate seed-level metrics into paper-ready CSV tables."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from schwartz_value_geometry.utils.logging import get_logger  # noqa: E402

LOGGER = get_logger(__name__)

STANDARD_METRICS = [
    "macro_f1",
    "micro_f1",
    "macro_auprc",
    "micro_auprc",
]

GEOMETRY_METRICS = [
    "circular_error",
    "opposite_value_activation",
    "neighbor_error_rate",
    "opposite_error_rate",
    "confusion_distance_correlation",
]

PAPER_METRICS = [
    "macro_f1",
    "micro_f1",
    "macro_auprc",
    "circular_error",
    "opposite_value_activation",
    "neighbor_error_rate",
    "confusion_distance_correlation",
]

GROUP_COLUMNS = ["method", "model_name", "split", "threshold_source"]


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _mean_std_text(mean_value: float, std_value: float | None) -> str:
    if pd.isna(std_value):
        std_value = 0.0
    return f"{mean_value:.4f} ± {float(std_value):.4f}"


def _read_metrics(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        LOGGER.warning("Skipping invalid JSON file: %s", path)
        return None
    if not isinstance(payload, dict):
        return None
    if "meta" not in payload and not any(metric in payload for metric in PAPER_METRICS):
        return None
    return payload


def find_metrics_files(results_dir: Path, pattern: str) -> list[Path]:
    """Find JSON metrics files recursively."""
    return sorted(path for path in results_dir.rglob(pattern) if path.is_file())


def _method_from_payload(payload: dict[str, Any], path: Path) -> str:
    meta = payload.get("meta", {})
    loss = meta.get("loss", {}) if isinstance(meta, dict) else {}
    if isinstance(loss, dict) and loss.get("name"):
        return str(loss["name"])
    stem = path.stem
    if stem.startswith("deberta_"):
        after_prefix = stem[len("deberta_") :]
        if "_seed" in after_prefix:
            return after_prefix.split("_seed", maxsplit=1)[0]
    return "unknown"


def _seed_from_payload(payload: dict[str, Any], path: Path) -> int | None:
    meta = payload.get("meta", {})
    if isinstance(meta, dict) and meta.get("seed") is not None:
        try:
            return int(meta["seed"])
        except (TypeError, ValueError):
            pass
    for token in path.stem.split("_"):
        if token.startswith("seed"):
            try:
                return int(token.replace("seed", ""))
            except ValueError:
                return None
    return None


def _split_from_payload(payload: dict[str, Any], path: Path) -> str:
    meta = payload.get("meta", {})
    if isinstance(meta, dict) and meta.get("split"):
        return str(meta["split"])
    stem = path.stem
    if "_validation" in stem:
        return "validation"
    if "_test" in stem:
        return "test"
    if "_training" in stem:
        return "training"
    return "unknown"


def _model_name_from_payload(payload: dict[str, Any]) -> str:
    meta = payload.get("meta", {})
    if isinstance(meta, dict):
        return str(meta.get("model_name", "unknown"))
    return "unknown"


def _threshold_source_from_payload(payload: dict[str, Any]) -> str:
    meta = payload.get("meta", {})
    if isinstance(meta, dict):
        return str(meta.get("threshold_source", "unknown"))
    return "unknown"


def _scalar_metric_rows(payload: dict[str, Any]) -> dict[str, float]:
    return {key: float(value) for key, value in payload.items() if _is_number(value)}


def _ordered_union_keys(*mappings: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    for mapping in mappings:
        for key in mapping:
            if key not in seen:
                labels.append(key)
                seen.add(key)
    return labels


def load_seed_level_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    """Load one seed-level row per metrics JSON file."""
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = _read_metrics(path)
        if payload is None:
            continue
        row = {
            "method": _method_from_payload(payload, path),
            "model_name": _model_name_from_payload(payload),
            "split": _split_from_payload(payload, path),
            "seed": _seed_from_payload(payload, path),
            "threshold_source": _threshold_source_from_payload(payload),
            "source_file": str(path),
        }
        row.update(_scalar_metric_rows(payload))
        rows.append(row)
    return rows


def load_per_label_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    """Load per-label F1/AUPRC/support rows."""
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = _read_metrics(path)
        if payload is None:
            continue
        per_label_f1 = payload.get("per_label_f1", {}) or {}
        per_label_auprc = payload.get("per_label_auprc", {}) or {}
        per_label_support = payload.get("per_label_support", {}) or {}
        labels = _ordered_union_keys(per_label_f1, per_label_auprc, per_label_support)
        for label in labels:
            support = per_label_support.get(label, {}) or {}
            rows.append(
                {
                    "method": _method_from_payload(payload, path),
                    "model_name": _model_name_from_payload(payload),
                    "split": _split_from_payload(payload, path),
                    "seed": _seed_from_payload(payload, path),
                    "threshold_source": _threshold_source_from_payload(payload),
                    "label": label,
                    "f1": per_label_f1.get(label),
                    "auprc": per_label_auprc.get(label),
                    "gold": support.get("gold"),
                    "pred": support.get("pred"),
                    "gold_rate": support.get("gold_rate"),
                    "pred_rate": support.get("pred_rate"),
                    "source_file": str(path),
                }
            )
    return rows


def load_threshold_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    """Load global or per-label threshold tuning rows."""
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = _read_metrics(path)
        if payload is None:
            continue
        meta = payload.get("meta", {}) if isinstance(payload.get("meta"), dict) else {}
        tuning = payload.get("threshold_tuning", {})
        threshold = meta.get("threshold")
        mode = tuning.get("mode") if isinstance(tuning, dict) else None

        base = {
            "method": _method_from_payload(payload, path),
            "model_name": _model_name_from_payload(payload),
            "split": _split_from_payload(payload, path),
            "seed": _seed_from_payload(payload, path),
            "threshold_source": _threshold_source_from_payload(payload),
            "threshold_mode": mode or "provided",
            "source_file": str(path),
        }

        if isinstance(threshold, dict):
            for label, value in threshold.items():
                rows.append({**base, "label": label, "threshold": float(value)})
        elif threshold is not None:
            rows.append({**base, "label": "", "threshold": float(threshold)})
    return rows


def _filter_split(df: pd.DataFrame, split: str) -> pd.DataFrame:
    if df.empty or split == "all":
        return df
    return df[df["split"] == split].copy()


def build_mean_std_summary(
    seed_df: pd.DataFrame,
    *,
    metrics: list[str],
    group_columns: list[str] = GROUP_COLUMNS,
) -> pd.DataFrame:
    """Build mean/std/n summaries for scalar metrics."""
    if seed_df.empty:
        return pd.DataFrame()
    available_metrics = [metric for metric in metrics if metric in seed_df.columns]
    if not available_metrics:
        return pd.DataFrame()

    grouped = seed_df.groupby(group_columns, dropna=False)
    summary = grouped[available_metrics].agg(["mean", "std", "count"]).reset_index()
    summary.columns = [
        "_".join(str(part) for part in column if part)
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]

    for metric in available_metrics:
        summary[f"{metric}_mean_std"] = [
            _mean_std_text(mean_value, std_value)
            for mean_value, std_value in zip(
                summary[f"{metric}_mean"],
                summary[f"{metric}_std"],
                strict=False,
            )
        ]
    return summary


def build_paper_main_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Return compact paper-facing main supervised table."""
    if summary_df.empty:
        return pd.DataFrame()
    columns = GROUP_COLUMNS.copy()
    columns.extend(
        f"{metric}_mean_std"
        for metric in PAPER_METRICS
        if f"{metric}_mean_std" in summary_df.columns
    )
    columns.extend(
        f"{metric}_count"
        for metric in PAPER_METRICS
        if f"{metric}_count" in summary_df.columns
    )
    return summary_df[columns].copy()


def write_tables(
    *,
    metrics_paths: list[Path],
    output_dir: Path,
    split: str,
) -> dict[str, Path]:
    """Write all aggregation CSV tables and return their paths."""
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_df = pd.DataFrame(load_seed_level_rows(metrics_paths))
    per_label_df = pd.DataFrame(load_per_label_rows(metrics_paths))
    threshold_df = pd.DataFrame(load_threshold_rows(metrics_paths))

    seed_df = _filter_split(seed_df, split)
    per_label_df = _filter_split(per_label_df, split)

    geometry_columns = [
        column for column in GROUP_COLUMNS + ["seed", *GEOMETRY_METRICS, "source_file"]
        if column in seed_df.columns
    ]
    geometry_df = seed_df[geometry_columns].copy() if geometry_columns else pd.DataFrame()

    scalar_metrics = [
        column
        for column in seed_df.columns
        if column not in {*GROUP_COLUMNS, "seed", "source_file"}
        and pd.api.types.is_numeric_dtype(seed_df[column])
    ]
    summary_df = build_mean_std_summary(seed_df, metrics=scalar_metrics)
    paper_main_df = build_paper_main_table(summary_df)

    per_label_summary = pd.DataFrame()
    if not per_label_df.empty:
        metrics = [
            metric
            for metric in ["f1", "auprc", "gold", "pred", "gold_rate", "pred_rate"]
            if metric in per_label_df.columns
        ]
        per_label_summary = build_mean_std_summary(
            per_label_df,
            metrics=metrics,
            group_columns=[*GROUP_COLUMNS, "label"],
        )

    paths = {
        "seed_level_results": output_dir / "seed_level_results.csv",
        "main_supervised_results": output_dir / "main_supervised_results.csv",
        "mean_std_summary": output_dir / "mean_std_summary.csv",
        "per_label_results": output_dir / "per_label_results.csv",
        "per_label_summary": output_dir / "per_label_summary.csv",
        "geometry_metrics": output_dir / "geometry_metrics.csv",
        "threshold_table": output_dir / "threshold_table.csv",
    }

    seed_df.to_csv(paths["seed_level_results"], index=False)
    paper_main_df.to_csv(paths["main_supervised_results"], index=False)
    summary_df.to_csv(paths["mean_std_summary"], index=False)
    per_label_df.to_csv(paths["per_label_results"], index=False)
    per_label_summary.to_csv(paths["per_label_summary"], index=False)
    geometry_df.to_csv(paths["geometry_metrics"], index=False)
    threshold_df.to_csv(paths["threshold_table"], index=False)
    return paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate paper result CSV tables.")
    parser.add_argument(
        "--results_dir",
        default="results/logs",
        help="Directory containing metrics JSON files.",
    )
    parser.add_argument(
        "--pattern",
        default="*.json",
        help="Recursive glob pattern for metrics JSON files.",
    )
    parser.add_argument(
        "--output_dir",
        default="results/analysis/paper_tables",
        help="Directory where CSV tables will be written.",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Split to keep for result tables, or 'all'. Threshold table keeps all splits.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.debug:
        LOGGER.setLevel("DEBUG")

    metrics_paths = find_metrics_files(Path(args.results_dir), args.pattern)
    LOGGER.info("Found %d candidate JSON files", len(metrics_paths))
    paths = write_tables(
        metrics_paths=metrics_paths,
        output_dir=Path(args.output_dir),
        split=str(args.split),
    )
    for name, path in paths.items():
        LOGGER.info("Wrote %s to %s", name, path)


if __name__ == "__main__":
    main()
