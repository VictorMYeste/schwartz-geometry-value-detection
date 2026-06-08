"""Run Qwen2.5-72B-Instruct for the LLM diagnostic."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from schwartz_value_geometry.data.dataset import load_split, get_label_names  # noqa: E402
from schwartz_value_geometry.llm.client import LLMClientConfig, TransformersChatClient  # noqa: E402
from schwartz_value_geometry.llm.parsing import parse_labels  # noqa: E402
from schwartz_value_geometry.llm.prompts import build_prompt  # noqa: E402
from schwartz_value_geometry.utils.config import load_config  # noqa: E402
from schwartz_value_geometry.utils.logging import (  # noqa: E402
    get_logger,
    silence_transformers_logging,
)

LOGGER = get_logger(__name__)


def _model_slug(model_name: str) -> str:
    raw = (model_name or "").strip().rstrip("/")
    tail = raw.split("/")[-1] if raw else "model"
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", tail).strip("-")
    return slug or "model"


def output_path_for(config: dict, *, split: str, output_dir: Path | None = None) -> Path:
    model = config.get("model", {})
    llm = config.get("llm", {})
    model_family = str(model.get("family", "qwen")).strip().lower()
    model_name = str(model.get("name", "Qwen/Qwen2.5-72B-Instruct"))
    model_slug = _model_slug(model_name)
    prompt_type = str(llm.get("prompt_type", "definitions_only")).strip().lower()
    results_dir = output_dir or Path(config.get("results_dir", "results/llm_diagnostic"))
    return (
        results_dir
        / "predictions"
        / f"{model_family}_{model_slug}_{prompt_type}_{split}.jsonl"
    )


def run_llm_diagnostic(
    config: dict,
    *,
    split: str,
    max_samples: int | None = None,
    output_dir: Path | None = None,
) -> Path:
    label_names = get_label_names()
    df = load_split(split)
    if max_samples is not None:
        df = df.head(int(max_samples)).copy()
        LOGGER.info("Limiting LLM diagnostic to %d samples", len(df))

    llm = config.get("llm", {})
    model = config.get("model", {})
    prompt_type = str(llm.get("prompt_type", "definitions_only")).strip().lower()
    model_name = str(model.get("name", "Qwen/Qwen2.5-72B-Instruct"))
    output_path = output_path_for(config, split=split, output_dir=output_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing: set[tuple[str, str]] = set()
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                existing.add((str(record.get("text_id")), str(record.get("sent_id"))))
        LOGGER.info("Resuming from %d existing predictions", len(existing))

    client = TransformersChatClient(
        LLMClientConfig(
            model_name=model_name,
            device=llm.get("device"),
            quantization=llm.get("quantization", "4bit"),
            int8_fp32_cpu_offload=bool(llm.get("int8_fp32_cpu_offload", False)),
            max_new_tokens=int(llm.get("max_tokens", 128)),
            temperature=float(llm.get("temperature", 0.0)),
            top_p=float(llm.get("top_p", 1.0)),
            trust_remote_code=bool(llm.get("trust_remote_code", False)),
        )
    )

    records = df.to_dict(orient="records")
    pending_total = sum(
        1
        for row in records
        if (str(row["text_id"]), str(row["sent_id"])) not in existing
    )
    LOGGER.info(
        "LLM diagnostic queue: total=%d pending=%d prompt=%s",
        len(records),
        pending_total,
        prompt_type,
    )
    log_prompts = bool(llm.get("log_prompts", False))
    log_prompts_n = int(llm.get("log_prompts_n", 1))
    log_prompt_max_chars = int(llm.get("log_prompt_max_chars", 12000))

    mode = "a" if output_path.exists() else "w"
    with output_path.open(mode, encoding="utf-8") as handle:
        processed = 0
        for row in records:
            text_id = str(row["text_id"])
            sent_id = str(row["sent_id"])
            if (text_id, sent_id) in existing:
                continue
            processed += 1
            target_text = str(row["text"])
            progress = f"LLM diagnostic {processed}/{pending_total} text_id={text_id} sent_id={sent_id}"
            sys.stderr.write("\r" + progress)
            sys.stderr.flush()

            prompt = build_prompt(text=target_text, prompt_type=prompt_type)
            if log_prompts and processed <= log_prompts_n:
                rendered = client.preview_model_prompt(prompt)
                if len(rendered) > log_prompt_max_chars:
                    rendered = rendered[:log_prompt_max_chars] + "\n...[prompt truncated]..."
                LOGGER.info("Rendered prompt %d/%d:\n%s", processed, pending_total, rendered)

            raw = client.generate(
                prompt,
                max_tokens=int(llm.get("max_tokens", 128)),
                temperature=float(llm.get("temperature", 0.0)),
                top_p=float(llm.get("top_p", 1.0)),
            )
            parsed = parse_labels(raw, label_names)
            gold_labels = [label for label in label_names if int(float(row.get(label, 0))) == 1]
            record = {
                "model_name": model_name,
                "model_family": str(model.get("family", "qwen")).strip().lower(),
                "model_slug": _model_slug(model_name),
                "prompt_type": prompt_type,
                "split": split,
                "text_id": text_id,
                "sent_id": sent_id,
                "text": target_text,
                "gold_labels": gold_labels,
                "pred_labels": parsed.labels,
                "raw_output": raw,
                "parse_status": parsed.parse_status,
                "invalid_output": parsed.invalid_output,
                "repaired_output": parsed.repaired_output,
                "unknown_labels": parsed.unknown_labels,
                "duplicate_labels": parsed.duplicate_labels,
                "n_pred_labels": len(parsed.labels),
                "prompt_chars": len(prompt),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
        if pending_total > 0:
            sys.stderr.write("\n")
            sys.stderr.flush()

    LOGGER.info("Saved LLM predictions to %s", output_path)
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen LLM diagnostic.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.debug:
        LOGGER.setLevel("DEBUG")
    silence_transformers_logging()
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else None
    pred_path = run_llm_diagnostic(
        config,
        split=args.split,
        max_samples=args.max_samples,
        output_dir=output_dir,
    )
    if args.eval:
        try:
            from scripts.eval_llm_diagnostic import evaluate_predictions
        except ModuleNotFoundError:
            from eval_llm_diagnostic import evaluate_predictions

        evaluate_predictions(pred_path)


if __name__ == "__main__":
    main()

