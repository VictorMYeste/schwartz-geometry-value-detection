"""Evaluate a trained DeBERTa checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from schwartz_value_geometry.models.training import run_eval  # noqa: E402
from schwartz_value_geometry.utils.config import load_config  # noqa: E402
from schwartz_value_geometry.utils.logging import (  # noqa: E402
    get_logger,
    silence_transformers_logging,
)
from schwartz_value_geometry.utils.naming import (  # noqa: E402
    artifact_prefix,
    loss_slug,
)

LOGGER = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a DeBERTa checkpoint.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to checkpoint. If omitted, inferred from config/loss/seed.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override config seed when inferring checkpoint and output names.",
    )
    parser.add_argument(
        "--split",
        choices=["validation", "test"],
        default="test",
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional sample limit for quick evaluation runs.",
    )
    parser.add_argument(
        "--tune_threshold",
        action="store_true",
        help="Sweep thresholds on the split to maximize macro-F1.",
    )
    parser.add_argument(
        "--use_validation_thresholds",
        action="store_true",
        help=(
            "Tune thresholds on validation, freeze them, and apply them to the "
            "requested split."
        ),
    )
    parser.add_argument(
        "--threshold_mode",
        choices=["global", "per_label"],
        default="per_label",
        help="Threshold tuning strategy.",
    )
    parser.add_argument(
        "--threshold_start",
        type=float,
        default=0.0,
        help="Threshold sweep start.",
    )
    parser.add_argument(
        "--threshold_stop",
        type=float,
        default=1.0,
        help="Threshold sweep stop.",
    )
    parser.add_argument(
        "--threshold_step",
        type=float,
        default=0.01,
        help="Threshold sweep step.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.debug:
        LOGGER.setLevel("DEBUG")
    if args.tune_threshold and args.use_validation_thresholds:
        raise ValueError(
            "Use either --tune_threshold for same-split diagnostics or "
            "--use_validation_thresholds for protocol-safe evaluation, not both."
        )

    silence_transformers_logging()

    config = load_config(args.config)
    if args.seed is not None:
        config["seed"] = int(args.seed)
    if args.max_samples is not None:
        config["max_samples"] = int(args.max_samples)
    LOGGER.debug("Loaded config keys: %s", list(config.keys()))
    model_name = config.get("model", {}).get("name", "microsoft/deberta-v3-base")
    seed = int(config.get("seed", 42))
    loss_name_slug = loss_slug(config)

    LOGGER.debug(
        "Eval config: loss=%s seed=%d split=%s",
        loss_name_slug,
        seed,
        args.split,
    )

    results_dir = Path(config.get("results_dir", "results"))
    prefix = artifact_prefix(config, seed=seed)
    run_name = f"{prefix}_best"
    ckpt_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else results_dir / "checkpoints" / f"{run_name}.pt"
    )
    LOGGER.debug("Using checkpoint path: %s", ckpt_path)

    predictions_dir = results_dir / "predictions"
    logs_dir = results_dir / "logs"
    pred_path = predictions_dir / f"{prefix}_{args.split}.jsonl"
    metrics_path = logs_dir / f"{prefix}_{args.split}_metrics.json"
    frozen_thresholds = None

    if args.use_validation_thresholds:
        val_pred_path = predictions_dir / f"{prefix}_validation_thresholds.jsonl"
        val_metrics_path = logs_dir / f"{prefix}_validation_thresholds.json"
        LOGGER.info(
            "Tuning %s thresholds on validation split before evaluating %s",
            args.threshold_mode,
            args.split,
        )
        val_metrics = run_eval(
            config,
            checkpoint_path=ckpt_path,
            split="validation",
            output_pred_path=val_pred_path,
            output_metrics_path=val_metrics_path,
            debug=args.debug,
            tune_threshold=True,
            threshold_tuning_mode=args.threshold_mode,
            threshold_start=args.threshold_start,
            threshold_stop=args.threshold_stop,
            threshold_step=args.threshold_step,
        )
        frozen_thresholds = val_metrics["threshold_tuning"]["thresholds"]

    LOGGER.info("=" * 80)
    LOGGER.info(
        "Run: eval model=deberta variant=%s loss=%s seed=%d split=%s",
        model_name,
        loss_name_slug,
        seed,
        args.split,
    )
    LOGGER.info("Evaluating checkpoint %s on %s split", ckpt_path, args.split)
    run_eval(
        config,
        checkpoint_path=ckpt_path,
        split=args.split,
        output_pred_path=pred_path,
        output_metrics_path=metrics_path,
        debug=args.debug,
        tune_threshold=args.tune_threshold,
        threshold_tuning_mode=args.threshold_mode,
        thresholds=frozen_thresholds,
        threshold_start=args.threshold_start,
        threshold_stop=args.threshold_stop,
        threshold_step=args.threshold_step,
    )
    LOGGER.info("=" * 80)


if __name__ == "__main__":
    main()
