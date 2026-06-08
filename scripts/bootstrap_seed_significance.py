"""Bootstrap paired seed-level deltas against a baseline method."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from schwartz_value_geometry.utils.logging import get_logger  # noqa: E402

LOGGER = get_logger(__name__)

DEFAULT_METRICS = [
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

LOWER_IS_BETTER = {
    "circular_error",
    "opposite_value_activation",
    "neighbor_error_rate",
    "opposite_error_rate",
    "confusion_distance_correlation",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run paired bootstrap significance tests over seed-level metrics. "
            "This tests mean method-baseline deltas over the same final seeds."
        )
    )
    parser.add_argument(
        "--seed_level_csv",
        default="results/analysis/paper_tables/seed_level_results.csv",
        help="Aggregated seed-level result CSV from scripts/aggregate_results.py.",
    )
    parser.add_argument(
        "--output",
        default="results/analysis/paper_tables/bootstrap_seed_significance.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--baseline",
        default="bce",
        help="Baseline method to compare against.",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="Methods to compare. Defaults to all methods except baseline.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        help="Metric columns to compare.",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Split to keep from seed-level CSV.",
    )
    parser.add_argument(
        "--threshold_source",
        default="provided",
        help="Threshold source to keep, or 'all'.",
    )
    parser.add_argument(
        "--n_iterations",
        type=int,
        default=2000,
        help="Bootstrap iterations.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Bootstrap RNG seed.")
    parser.add_argument("--alpha", type=float, default=0.05, help="CI alpha.")
    return parser.parse_args()


def _load_rows(path: Path, *, split: str, threshold_source: str) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    filtered = [row for row in rows if str(row.get("split", "")) == split]
    if threshold_source != "all":
        filtered = [
            row
            for row in filtered
            if str(row.get("threshold_source", "")) == threshold_source
        ]
    return filtered


def _metric_value(row: dict[str, str], metric: str) -> float | None:
    raw = row.get(metric, "")
    if raw == "":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _two_sided_p_value(samples: np.ndarray) -> float:
    p = 2.0 * min(float((samples <= 0).mean()), float((samples >= 0).mean()))
    return min(1.0, p)


def bootstrap_metric_delta(
    method_values: np.ndarray,
    baseline_values: np.ndarray,
    *,
    n_iterations: int,
    seed: int,
    alpha: float,
) -> dict[str, float]:
    """Bootstrap the paired mean delta method - baseline."""
    if method_values.shape != baseline_values.shape:
        raise ValueError("method_values and baseline_values must have the same shape")
    if method_values.ndim != 1 or method_values.size == 0:
        raise ValueError("Expected non-empty 1D paired seed values")

    deltas = method_values - baseline_values
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, deltas.size, size=(n_iterations, deltas.size))
    samples = deltas[idx].mean(axis=1)
    return {
        "baseline_mean": float(baseline_values.mean()),
        "method_mean": float(method_values.mean()),
        "delta_mean": float(deltas.mean()),
        "ci_low": float(np.quantile(samples, alpha / 2.0)),
        "ci_high": float(np.quantile(samples, 1.0 - alpha / 2.0)),
        "p_value": _two_sided_p_value(samples),
        "win_count_raw": float((deltas > 0).sum()),
    }


def build_significance_rows(
    rows: list[dict[str, str]],
    *,
    baseline: str,
    methods: list[str] | None,
    metrics: list[str],
    n_iterations: int,
    seed: int,
    alpha: float,
) -> list[dict[str, object]]:
    by_method_seed: dict[str, dict[int, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        method = str(row.get("method", ""))
        if row.get("seed") in {"", None}:
            continue
        by_method_seed[method][int(float(str(row["seed"])))] = row

    if baseline not in by_method_seed:
        raise ValueError(f"Baseline method not found in seed-level rows: {baseline}")

    selected_methods = methods or sorted(
        method for method in by_method_seed if method != baseline
    )

    output_rows: list[dict[str, object]] = []
    for method in selected_methods:
        if method not in by_method_seed:
            LOGGER.warning("Skipping missing method: %s", method)
            continue
        common_seeds = sorted(
            set(by_method_seed[baseline]).intersection(by_method_seed[method])
        )
        if not common_seeds:
            LOGGER.warning("Skipping %s: no seeds in common with %s", method, baseline)
            continue
        for metric in metrics:
            paired_baseline: list[float] = []
            paired_method: list[float] = []
            paired_seeds: list[int] = []
            for run_seed in common_seeds:
                baseline_value = _metric_value(by_method_seed[baseline][run_seed], metric)
                method_value = _metric_value(by_method_seed[method][run_seed], metric)
                if baseline_value is None or method_value is None:
                    continue
                paired_baseline.append(baseline_value)
                paired_method.append(method_value)
                paired_seeds.append(run_seed)
            if not paired_seeds:
                continue

            baseline_arr = np.asarray(paired_baseline, dtype=float)
            method_arr = np.asarray(paired_method, dtype=float)
            stats = bootstrap_metric_delta(
                method_arr,
                baseline_arr,
                n_iterations=n_iterations,
                seed=seed,
                alpha=alpha,
            )

            lower_is_better = metric in LOWER_IS_BETTER
            delta = float(stats["delta_mean"])
            ci_low = float(stats["ci_low"])
            ci_high = float(stats["ci_high"])
            if lower_is_better:
                improvement = -delta
                improvement_ci_low = -ci_high
                improvement_ci_high = -ci_low
                win_count = int((method_arr - baseline_arr < 0).sum())
            else:
                improvement = delta
                improvement_ci_low = ci_low
                improvement_ci_high = ci_high
                win_count = int((method_arr - baseline_arr > 0).sum())

            output_rows.append(
                {
                    "method": method,
                    "baseline": baseline,
                    "metric": metric,
                    "higher_is_better": not lower_is_better,
                    "n_seed_pairs": len(paired_seeds),
                    "seeds": " ".join(str(value) for value in paired_seeds),
                    "baseline_mean": stats["baseline_mean"],
                    "method_mean": stats["method_mean"],
                    "delta_method_minus_baseline": delta,
                    "delta_ci_low": ci_low,
                    "delta_ci_high": ci_high,
                    "improvement_over_baseline": improvement,
                    "improvement_ci_low": improvement_ci_low,
                    "improvement_ci_high": improvement_ci_high,
                    "p_value_two_sided": stats["p_value"],
                    "wins_over_baseline": win_count,
                    "n_iterations": n_iterations,
                    "bootstrap_seed": seed,
                    "significant_0.05": float(stats["p_value"]) < 0.05,
                }
            )
    return output_rows


def main() -> None:
    args = _parse_args()
    input_path = Path(args.seed_level_csv)
    output_path = Path(args.output)
    rows = _load_rows(
        input_path,
        split=str(args.split),
        threshold_source=str(args.threshold_source),
    )
    result_rows = build_significance_rows(
        rows,
        baseline=str(args.baseline),
        methods=args.methods,
        metrics=list(args.metrics),
        n_iterations=int(args.n_iterations),
        seed=int(args.seed),
        alpha=float(args.alpha),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "baseline",
        "metric",
        "higher_is_better",
        "n_seed_pairs",
        "seeds",
        "baseline_mean",
        "method_mean",
        "delta_method_minus_baseline",
        "delta_ci_low",
        "delta_ci_high",
        "improvement_over_baseline",
        "improvement_ci_low",
        "improvement_ci_high",
        "p_value_two_sided",
        "wins_over_baseline",
        "n_iterations",
        "bootstrap_seed",
        "significant_0.05",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result_rows)
    LOGGER.info("Wrote %d bootstrap comparisons to %s", len(result_rows), output_path)


if __name__ == "__main__":
    main()
