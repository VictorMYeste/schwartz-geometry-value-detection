"""Grid search for ASL and geometry-aware loss hyperparameters."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from schwartz_value_geometry.utils.config import load_config  # noqa: E402
from schwartz_value_geometry.utils.logging import (  # noqa: E402
    get_logger,
    silence_transformers_logging,
)
from schwartz_value_geometry.utils.seed import get_tuning_seeds  # noqa: E402

LOGGER = get_logger(__name__)

METHOD_CONFIGS = {
    "asl": "configs/deberta_asl.yaml",
    "random_geoloss": "configs/deberta_random_geoloss.yaml",
    "empirical_structure": "configs/deberta_empirical_structure.yaml",
    "schwartz_geoloss": "configs/deberta_schwartz_geoloss.yaml",
    "schwartz_geosmooth": "configs/deberta_schwartz_geosmooth.yaml",
}

DEFAULT_METHODS = ("asl", "schwartz_geoloss", "schwartz_geosmooth")
ALL_METHODS = tuple(METHOD_CONFIGS)

CSV_FIELDS = [
    "method",
    "seed",
    "base_config",
    "params_json",
    "gamma_pos",
    "gamma_neg",
    "clip",
    "lambda_geo",
    "tau",
    "random_seed",
    "best_macro_f1",
    "collapsed",
    "run_name",
]


@dataclass(frozen=True)
class GridRun:
    method: str
    config_path: str
    seed: int
    params: dict[str, Any]

    @property
    def params_json(self) -> str:
        return json.dumps(self.params, sort_keys=True, separators=(",", ":"))

    @property
    def key(self) -> tuple[str, int, str]:
        return (self.method, self.seed, self.params_json)

    @property
    def run_name(self) -> str:
        parts = [f"tune_deberta_{self.method}", f"seed{self.seed}"]
        parts.extend(f"{key}{_value_slug(value)}" for key, value in self.params.items())
        return "_".join(parts)


def _value_slug(value: Any) -> str:
    text = str(value).replace(".", "p").replace("-", "m")
    return "".join(ch for ch in text if ch.isalnum() or ch in {"_", "p", "m"})


def _method_param_grid(method: str) -> list[dict[str, Any]]:
    if method == "asl":
        return [
            {"gamma_pos": gamma_pos, "gamma_neg": gamma_neg, "clip": clip}
            for gamma_pos, gamma_neg, clip in product(
                [0.0],
                [2.0, 3.0, 4.0, 5.0],
                [0.0, 0.03, 0.05, 0.10],
            )
        ]

    if method in {"random_geoloss", "empirical_structure", "schwartz_geoloss"}:
        return [{"lambda_geo": lambda_geo} for lambda_geo in [0.01, 0.05, 0.1, 0.2]]

    if method == "schwartz_geosmooth":
        return [{"tau": tau} for tau in [0.1, 0.2, 0.5, 1.0]]

    raise ValueError(f"Unknown method: {method}")


def build_grid(methods: list[str], seeds: tuple[int, ...]) -> list[GridRun]:
    """Build the configured tuning grid."""
    runs: list[GridRun] = []
    for method in methods:
        if method not in METHOD_CONFIGS:
            raise ValueError(f"Unknown method {method!r}; choose from {sorted(ALL_METHODS)}")
        for seed in seeds:
            for params in _method_param_grid(method):
                runs.append(
                    GridRun(
                        method=method,
                        config_path=METHOD_CONFIGS[method],
                        seed=int(seed),
                        params=params,
                    )
                )
    return runs


def apply_loss_params(config: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow config copy with loss params applied."""
    tuned = dict(config)
    tuned["loss"] = dict(config.get("loss", {}))
    tuned["loss"].update(params)
    return tuned


def _completed_keys_from_csv(path: Path) -> set[tuple[str, int, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return set()

    completed: set[tuple[str, int, str]] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                completed.add(
                    (
                        row["method"],
                        int(float(row["seed"])),
                        row["params_json"],
                    )
                )
            except Exception:
                LOGGER.warning("Skipping malformed row while resuming: %s", row)
    return completed


def _row_for_run(
    run: GridRun,
    *,
    best_macro_f1: float,
    collapsed: bool,
) -> dict[str, Any]:
    return {
        "method": run.method,
        "seed": run.seed,
        "base_config": run.config_path,
        "params_json": run.params_json,
        "gamma_pos": run.params.get("gamma_pos", ""),
        "gamma_neg": run.params.get("gamma_neg", ""),
        "clip": run.params.get("clip", ""),
        "lambda_geo": run.params.get("lambda_geo", ""),
        "tau": run.params.get("tau", ""),
        "random_seed": run.params.get("random_seed", ""),
        "best_macro_f1": float(best_macro_f1),
        "collapsed": bool(collapsed),
        "run_name": run.run_name,
    }


def _parse_methods(values: list[str]) -> list[str]:
    expanded: list[str] = []
    for value in values:
        if value == "all":
            expanded.extend(ALL_METHODS)
        elif value == "geoloss":
            expanded.extend(["random_geoloss", "empirical_structure", "schwartz_geoloss"])
        else:
            expanded.append(value)
    seen: set[str] = set()
    methods: list[str] = []
    for method in expanded:
        if method not in seen:
            methods.append(method)
            seen.add(method)
    return methods


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune ASL, GeoLoss, and GeoSmooth loss hyperparameters."
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(DEFAULT_METHODS),
        help=(
            "Methods to tune. Use 'geoloss' for random+empirical+schwartz "
            "GeoLoss controls, or 'all' for every supported method."
        ),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(get_tuning_seeds()),
        help="Tuning seeds. Defaults to the project tuning seeds.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional sample limit for smoke/debug tuning runs.",
    )
    parser.add_argument(
        "--output",
        default="results/analysis/grid_loss_hparams.csv",
        help="CSV path to store tuning results.",
    )
    parser.add_argument(
        "--results_dir",
        default=None,
        help=(
            "Optional results directory for tuning artifacts. When omitted, each "
            "base config decides its own results_dir."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of pending runs to execute.",
    )
    parser.add_argument(
        "--retry_collapsed",
        type=int,
        default=1,
        help="Retries for runs flagged as collapsed.",
    )
    parser.add_argument(
        "--save_checkpoints",
        action="store_true",
        help="Persist checkpoints during tuning. Disabled by default to save disk.",
    )
    parser.add_argument(
        "--save_hf_model",
        action="store_true",
        help="Persist Hugging Face model bundles when checkpoints are enabled.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print planned runs without training.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.debug:
        LOGGER.setLevel("DEBUG")

    silence_transformers_logging()

    methods = _parse_methods(args.methods)
    seeds = tuple(int(seed) for seed in args.seeds)
    grid = build_grid(methods, seeds)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed_keys = _completed_keys_from_csv(output_path)
    pending = [run for run in grid if run.key not in completed_keys]
    if args.limit is not None:
        pending = pending[: int(args.limit)]

    LOGGER.info(
        "Prepared %d runs (%d pending, %d completed)",
        len(grid),
        len(pending),
        len(completed_keys),
    )
    LOGGER.info(
        "Tuning output=%s results_dir=%s save_checkpoints=%s save_hf_model=%s",
        output_path,
        args.results_dir or "<config default>",
        bool(args.save_checkpoints),
        bool(args.save_hf_model),
    )

    if args.dry_run:
        for run in pending:
            LOGGER.info(
                "DRY RUN method=%s seed=%d params=%s run_name=%s",
                run.method,
                run.seed,
                run.params_json,
                run.run_name,
            )
        return

    from schwartz_value_geometry.models.training import train_and_eval

    rows_appended = 0
    config_cache: dict[str, dict[str, Any]] = {}
    with output_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if output_path.stat().st_size == 0:
            writer.writeheader()
            handle.flush()
            os.fsync(handle.fileno())

        for idx, run in enumerate(pending, start=1):
            base_config = config_cache.setdefault(
                run.config_path, load_config(run.config_path)
            )
            config = apply_loss_params(base_config, run.params)
            config["seed"] = run.seed
            if args.results_dir is not None:
                config["results_dir"] = args.results_dir
            config["save_checkpoints"] = bool(args.save_checkpoints)
            training_cfg = dict(config.get("training", {}))
            training_cfg["save_hf_model"] = bool(args.save_hf_model)
            config["training"] = training_cfg
            if args.max_samples is not None:
                config["max_samples"] = int(args.max_samples)

            LOGGER.info(
                "[%d/%d] method=%s seed=%d params=%s",
                idx,
                len(pending),
                run.method,
                run.seed,
                run.params_json,
            )

            attempts = max(args.retry_collapsed, 0) + 1
            best_macro_f1 = float("-inf")
            collapsed = True
            for attempt in range(attempts):
                if attempt > 0:
                    LOGGER.warning(
                        "Retrying collapsed run (attempt %d/%d)",
                        attempt + 1,
                        attempts,
                    )
                    config["seed"] = run.seed + attempt
                best_macro_f1, collapsed = train_and_eval(
                    config,
                    run_name=run.run_name,
                    resume_path=None,
                )
                if not collapsed:
                    break

            writer.writerow(
                _row_for_run(
                    run,
                    best_macro_f1=best_macro_f1,
                    collapsed=collapsed,
                )
            )
            handle.flush()
            os.fsync(handle.fileno())
            rows_appended += 1

    LOGGER.info("Saved tuning results to %s", output_path)
    LOGGER.info("Appended %d new rows in this execution", rows_appended)


if __name__ == "__main__":
    main()
