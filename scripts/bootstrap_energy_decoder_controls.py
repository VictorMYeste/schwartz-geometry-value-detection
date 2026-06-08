"""Direct sample-bootstrap comparison of Schwartz decoder vs control decoders.

The existing decoder bootstrap tests each decoded output against independent
thresholding. This script tests the paper-critical control contrast directly:

    (Schwartz decoder - standard) - (control decoder - standard)

For any fixed bootstrap sample and metric, the standard term cancels out, so the
contrast is equivalent to Schwartz decoder - control decoder.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from geometry_aware_calibration import (  # noqa: E402
    discover_runs,
    load_prediction_jsonl,
    load_validation_thresholds,
)
from schwartz_energy_decoder import (  # noqa: E402
    BINARY_BOOTSTRAP_METRICS,
    DECODER_FAMILIES,
    DECODER_GEOMETRIES,
    EnergySetting,
    build_geometry_matrices,
    decode_and_evaluate,
    load_empirical_training_labels,
    metric_function,
)
from schwartz_value_geometry.eval.stats import paired_bootstrap_delta  # noqa: E402
from schwartz_value_geometry.utils.logging import get_logger  # noqa: E402

LOGGER = get_logger(__name__)

LOWER_IS_BETTER = {
    "neighbor_error_rate",
    "opposite_error_rate",
    "confusion_distance_correlation",
    "decoder_geometry_cost",
}


def _load_selected_rows(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing selected decoder table: {path}")
    selected = pd.read_csv(path)
    required = {
        "method",
        "seed",
        "model_slug",
        "geometry",
        "family",
        "objective",
        "top_k",
        "max_candidates",
        "max_labels",
        "threshold_factor",
        "min_prob",
        "marginal_temperature",
        "neighbor_steps",
        "random_seed",
        "empirical_metric",
        "alpha_neighbor",
        "beta_opposite",
        "gamma_cardinality",
    }
    missing = sorted(required - set(selected.columns))
    if missing:
        raise ValueError(f"Selected decoder table is missing columns: {missing}")
    return selected


def _setting_from_row(row: pd.Series) -> EnergySetting:
    return EnergySetting(
        alpha_neighbor=float(row["alpha_neighbor"]),
        beta_opposite=float(row["beta_opposite"]),
        gamma_cardinality=float(row["gamma_cardinality"]),
    )


def _unique_value(rows: pd.DataFrame, column: str) -> Any:
    values = rows[column].drop_duplicates().tolist()
    if len(values) != 1:
        raise ValueError(f"Expected one value for {column}; found {values}")
    return values[0]


def _seed_delta_rows(
    *,
    selected: pd.DataFrame,
    controls: list[str],
    metrics: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["method", "seed", "model_slug", "family", "objective"]
    for key, group in selected.groupby(group_cols, dropna=False):
        group_by_geometry = {str(row["geometry"]): row for _, row in group.iterrows()}
        if "schwartz" not in group_by_geometry:
            continue
        schwartz = group_by_geometry["schwartz"]
        for control in controls:
            if control not in group_by_geometry:
                continue
            control_row = group_by_geometry[control]
            out = {
                "method": key[0],
                "seed": int(key[1]),
                "model_slug": key[2],
                "family": key[3],
                "objective": key[4],
                "comparison": f"schwartz_vs_{control}",
                "control_geometry": control,
            }
            for metric in metrics:
                column = f"test_{metric}"
                if column not in group.columns:
                    continue
                delta = float(schwartz[column]) - float(control_row[column])
                out[f"delta_{metric}"] = delta
                out[f"schwartz_improvement_{metric}"] = (
                    -delta if metric in LOWER_IS_BETTER else delta
                )
            rows.append(out)
    return pd.DataFrame(rows)


def _mean_std(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    metric_cols = [
        col
        for col in df.columns
        if (col.startswith("delta_") or col.startswith("schwartz_improvement_"))
        and pd.api.types.is_numeric_dtype(df[col])
    ]
    summary = df.groupby(
        ["method", "model_slug", "family", "objective", "comparison", "control_geometry"],
        dropna=False,
    )[metric_cols].agg(["mean", "std", "count"])
    summary.columns = [
        "_".join(str(part) for part in col if part)
        for col in summary.columns
    ]
    return summary.reset_index()


def run_control_bootstrap(
    *,
    decoder_output_dir: Path,
    predictions_dir: Path,
    logs_dir: Path,
    output_dir: Path,
    methods: set[str] | None,
    seeds: set[int] | None,
    model_slug: str | None,
    family: str,
    objective: str,
    controls: list[str],
    metrics: list[str],
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> dict[str, Path]:
    selected_path = decoder_output_dir / "schwartz_energy_decoder_selected_seed_results.csv"
    selected = _load_selected_rows(selected_path)
    selected = selected[
        (selected["family"] == family)
        & (selected["objective"] == objective)
        & (selected["geometry"].isin(["schwartz", *controls]))
    ].copy()
    if methods:
        selected = selected[selected["method"].isin(methods)]
    if seeds:
        selected = selected[selected["seed"].isin(seeds)]
    if model_slug:
        selected = selected[selected["model_slug"] == model_slug]
    if selected.empty:
        raise ValueError("No selected decoder rows matched the requested filters")

    output_dir.mkdir(parents=True, exist_ok=True)
    seed_deltas = _seed_delta_rows(selected=selected, controls=controls, metrics=metrics)
    seed_summary = _mean_std(seed_deltas)

    paths = {
        "seed_deltas": output_dir / "schwartz_energy_decoder_control_delta_seed_results.csv",
        "seed_summary": output_dir / "schwartz_energy_decoder_control_delta_mean_std.csv",
        "sample_bootstrap": output_dir / "schwartz_energy_decoder_control_sample_bootstrap.csv",
    }
    if bootstrap_iterations <= 0:
        seed_deltas.to_csv(paths["seed_deltas"], index=False)
        seed_summary.to_csv(paths["seed_summary"], index=False)
        pd.DataFrame().to_csv(paths["sample_bootstrap"], index=False)
        return paths

    runs = discover_runs(
        predictions_dir=predictions_dir,
        logs_dir=logs_dir,
        methods=methods,
        seeds=seeds,
        model_slug=model_slug,
    )
    run_map = {(run.method, run.seed, run.model_slug): run for run in runs}

    bootstrap_rows: list[dict[str, Any]] = []
    group_cols = ["method", "seed", "model_slug", "family", "objective"]
    for key, group in selected.groupby(group_cols, dropna=False):
        method, seed, slug, _, _ = key
        run = run_map.get((str(method), int(seed), str(slug)))
        if run is None:
            raise ValueError(f"No prediction run found for method={method} seed={seed}")

        LOGGER.info("Direct control bootstrap method=%s seed=%s", method, seed)
        test = load_prediction_jsonl(run.test_predictions)
        thresholds = load_validation_thresholds(run.validation_metrics, test.label_names)
        empirical_labels = (
            load_empirical_training_labels(test.label_names)
            if "empirical" in set(group["geometry"])
            else None
        )

        decoded_by_geometry: dict[str, np.ndarray] = {}
        rows_by_geometry = {str(row["geometry"]): row for _, row in group.iterrows()}
        for geometry, row in rows_by_geometry.items():
            matrices = build_geometry_matrices(
                test.label_names,
                geometry=geometry,
                neighbor_steps=int(row["neighbor_steps"]),
                random_seed=int(row["random_seed"]),
                empirical_labels=empirical_labels,
                empirical_metric=str(row["empirical_metric"]),
            )
            outputs = decode_and_evaluate(
                test,
                thresholds=thresholds,
                setting=_setting_from_row(row),
                geometry_matrices=matrices,
                top_k=int(row["top_k"]),
                max_candidates=int(row["max_candidates"]),
                max_labels=int(row["max_labels"]),
                threshold_factor=float(row["threshold_factor"]),
                min_prob=float(row["min_prob"]),
                marginal_temperature=float(row["marginal_temperature"]),
                neighbor_steps=int(row["neighbor_steps"]),
            )
            decoded_by_geometry[geometry] = outputs.y_pred

        if "schwartz" not in decoded_by_geometry:
            continue
        for control in controls:
            if control not in decoded_by_geometry:
                continue
            for metric in metrics:
                result = paired_bootstrap_delta(
                    test.y_true,
                    decoded_by_geometry["schwartz"],
                    decoded_by_geometry[control],
                    metric_fn=metric_function(metric, label_names=test.label_names),
                    n_iterations=bootstrap_iterations,
                    seed=bootstrap_seed,
                )
                lower_is_better = metric in LOWER_IS_BETTER
                improvement = -result.delta if lower_is_better else result.delta
                improvement_ci_low = -result.ci_high if lower_is_better else result.ci_low
                improvement_ci_high = -result.ci_low if lower_is_better else result.ci_high
                bootstrap_rows.append(
                    {
                        "method": method,
                        "seed": int(seed),
                        "model_slug": slug,
                        "family": family,
                        "objective": objective,
                        "comparison": f"schwartz_vs_{control}",
                        "control_geometry": control,
                        "metric": metric,
                        "higher_is_better": not lower_is_better,
                        "delta_schwartz_minus_control": result.delta,
                        "delta_ci_low": result.ci_low,
                        "delta_ci_high": result.ci_high,
                        "schwartz_improvement_over_control": improvement,
                        "improvement_ci_low": improvement_ci_low,
                        "improvement_ci_high": improvement_ci_high,
                        "p_value_two_sided": result.p_value,
                        "n_samples": result.n_samples,
                        "n_iterations": result.n_iterations,
                        "bootstrap_seed": bootstrap_seed,
                        "significant_0.05": result.p_value < 0.05,
                    }
                )

    bootstrap_df = pd.DataFrame(bootstrap_rows)
    seed_deltas.to_csv(paths["seed_deltas"], index=False)
    seed_summary.to_csv(paths["seed_summary"], index=False)
    bootstrap_df.to_csv(paths["sample_bootstrap"], index=False)
    return paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Directly bootstrap Schwartz energy decoder vs control decoders."
    )
    parser.add_argument(
        "--decoder_output_dir",
        default="results/analysis/schwartz_energy_decoder_bootstrap_examples_full",
        help="Directory containing schwartz_energy_decoder_selected_seed_results.csv.",
    )
    parser.add_argument("--predictions_dir", default="results/predictions")
    parser.add_argument("--logs_dir", default="results/logs")
    parser.add_argument(
        "--output_dir",
        default="results/analysis/schwartz_energy_decoder_control_bootstrap",
    )
    parser.add_argument("--methods", nargs="*", default=["bce"])
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--model_slug", default="deberta-v3-base")
    parser.add_argument(
        "--family",
        default="full",
        choices=sorted(DECODER_FAMILIES),
    )
    parser.add_argument("--objective", default="pareto_99")
    parser.add_argument(
        "--controls",
        nargs="+",
        default=["random", "empirical"],
        choices=sorted(DECODER_GEOMETRIES - {"schwartz"}),
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=[
            "macro_f1",
            "micro_f1",
            "opposite_error_rate",
            "decoder_geometry_cost",
        ],
        choices=BINARY_BOOTSTRAP_METRICS,
    )
    parser.add_argument("--bootstrap_iterations", type=int, default=2000)
    parser.add_argument("--bootstrap_seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    paths = run_control_bootstrap(
        decoder_output_dir=Path(args.decoder_output_dir),
        predictions_dir=Path(args.predictions_dir),
        logs_dir=Path(args.logs_dir),
        output_dir=Path(args.output_dir),
        methods=set(args.methods) if args.methods else None,
        seeds=set(args.seeds) if args.seeds else None,
        model_slug=str(args.model_slug) if args.model_slug else None,
        family=str(args.family),
        objective=str(args.objective),
        controls=list(args.controls),
        metrics=list(args.metrics),
        bootstrap_iterations=int(args.bootstrap_iterations),
        bootstrap_seed=int(args.bootstrap_seed),
    )
    for name, path in paths.items():
        LOGGER.info("Wrote %s to %s", name, path)


if __name__ == "__main__":
    main()
