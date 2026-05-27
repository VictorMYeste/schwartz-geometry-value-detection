"""Train and optionally evaluate a DeBERTa multi-label baseline."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from schwartz_value_geometry.models.training import run_eval, train_and_eval
from schwartz_value_geometry.utils.config import load_config
from schwartz_value_geometry.utils.logging import (
    get_logger,
    silence_transformers_logging,
)
from schwartz_value_geometry.utils.seed import set_seed

LOGGER = get_logger(__name__)


def _model_slug(model_name: str) -> str:
    base = model_name.split("/")[-1] if model_name else "deberta"
    slug = re.sub(r"[^A-Za-z0-9.-]+", "-", base).strip("-").lower()
    return slug or "deberta"


def _loss_slug(config: dict) -> str:
    loss_name = str(config.get("loss", {}).get("name", "bce")).strip().lower()
    return re.sub(r"[^A-Za-z0-9.-]+", "-", loss_name).strip("-") or "bce"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DeBERTa for one config.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Evaluate and save predictions on test split after training.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional sample limit for quick runs.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Use a temporary results directory and avoid persisting checkpoints.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to a last-checkpoint bundle to resume from.",
    )
    parser.add_argument(
        "--retry_collapsed",
        type=int,
        default=1,
        help="Retries for runs flagged as collapsed.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.debug:
        import logging

        logging.getLogger().setLevel(logging.DEBUG)
        LOGGER.setLevel(logging.DEBUG)

    silence_transformers_logging()

    config = load_config(args.config)
    config["debug"] = bool(args.debug)
    seed = int(args.seed)
    config["seed"] = seed
    set_seed(seed, debug=args.debug)
    if args.max_samples is not None:
        config["max_samples"] = args.max_samples

    model_name = config.get("model", {}).get("name", "microsoft/deberta-v3-base")
    model_slug = _model_slug(model_name)
    loss_slug = _loss_slug(config)

    results_dir = Path(config.get("results_dir", "results"))
    if args.dry_run:
        results_dir = Path(".tmp/schwartz-geometry-value-detection-smoke")
        config["results_dir"] = str(results_dir)
        config["save_checkpoints"] = False

    artifact_prefix = f"deberta_sentence_{loss_slug}_seed{seed}_{model_slug}"
    run_name = f"{artifact_prefix}_best"
    log_dir = results_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{artifact_prefix}.log"

    logger = get_logger(__name__, log_file=str(log_file), overwrite=True)
    get_logger("schwartz_value_geometry.models.training", log_file=str(log_file))
    get_logger("schwartz_value_geometry.data.dataset", log_file=str(log_file))
    get_logger("schwartz_value_geometry.models.deberta", log_file=str(log_file))

    logger.info("=" * 80)
    logger.info("Starting DeBERTa training with config %s", args.config)
    logger.info(
        "Run: model=%s loss=%s seed=%d eval=%s dry_run=%s",
        model_name,
        loss_slug,
        seed,
        args.eval,
        args.dry_run,
    )

    if args.dry_run:
        resume_path = None
    elif args.resume:
        resume_path = Path(args.resume)
    else:
        auto_path = results_dir / "checkpoints" / f"{run_name}_last.pt"
        resume_path = auto_path if auto_path.exists() else None

    attempts = max(args.retry_collapsed, 0) + 1
    best_macro = float("-inf")
    collapsed = True
    for attempt in range(attempts):
        if attempt > 0:
            logger.warning(
                "Retrying collapsed run (attempt %d/%d)", attempt + 1, attempts
            )
        config["seed"] = seed + attempt
        best_macro, collapsed = train_and_eval(
            config,
            run_name=run_name,
            resume_path=resume_path if attempt == 0 else None,
        )
        if not collapsed:
            break

    logger.info("Best validation macro-F1: %.4f", best_macro)

    if args.eval or config.get("eval", False):
        if not config.get("save_checkpoints", True):
            logger.warning("Skipping eval: checkpoints disabled")
            logger.info("=" * 80)
            return
        ckpt_path = results_dir / "checkpoints" / f"{run_name}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Best checkpoint not found at {ckpt_path}")

        predictions_dir = results_dir / "predictions"
        pred_path = predictions_dir / f"{artifact_prefix}_test.jsonl"
        metrics_path = log_dir / f"{artifact_prefix}_test_metrics.json"
        metrics = run_eval(
            config,
            checkpoint_path=ckpt_path,
            split="test",
            output_pred_path=pred_path,
            output_metrics_path=metrics_path,
            debug=args.debug,
        )
        logger.info(
            "Test metrics - macro_f1=%.4f micro_f1=%.4f",
            metrics.get("macro_f1", 0.0),
            metrics.get("micro_f1", 0.0),
        )

    logger.info("=" * 80)


if __name__ == "__main__":
    main()
