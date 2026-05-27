"""Training and evaluation loops for sentence-level DeBERTa baselines."""

from __future__ import annotations

import inspect
import json
import math
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from schwartz_value_geometry.data.dataset import get_label_names, load_split
from schwartz_value_geometry.eval.metrics import compute_f1_metrics, sweep_thresholds
from schwartz_value_geometry.models.deberta import build_deberta_model, encode_batch
from schwartz_value_geometry.utils.logging import get_logger
from schwartz_value_geometry.utils.seed import set_seed

LOGGER = get_logger(__name__)


def _resolve_bf16(training_cfg: dict, device: torch.device) -> bool:
    if not bool(training_cfg.get("bf16", False)):
        return False
    if device.type != "cuda":
        LOGGER.warning("bf16 requested but CUDA is unavailable; falling back to fp32")
        return False
    if hasattr(torch.cuda, "is_bf16_supported") and not torch.cuda.is_bf16_supported():
        LOGGER.warning("bf16 requested but CUDA bf16 is unsupported; falling back to fp32")
        return False
    return True


def _build_adamw(
    params,
    *,
    learning_rate: float,
    weight_decay: float,
    training_cfg: dict,
):
    """Build AdamW with conservative defaults to avoid CUDA foreach instability."""
    kwargs: dict[str, object] = {
        "lr": learning_rate,
        "weight_decay": weight_decay,
    }

    adamw_sig = inspect.signature(torch.optim.AdamW)
    if "foreach" in adamw_sig.parameters:
        kwargs["foreach"] = bool(training_cfg.get("adamw_foreach", False))
    if "fused" in adamw_sig.parameters:
        kwargs["fused"] = bool(training_cfg.get("adamw_fused", False))

    LOGGER.info(
        "AdamW settings: lr=%g wd=%g foreach=%s fused=%s",
        learning_rate,
        weight_decay,
        str(kwargs.get("foreach", "n/a")),
        str(kwargs.get("fused", "n/a")),
    )
    return torch.optim.AdamW(params, **kwargs)


def _autocast_context(use_bf16: bool):
    if not use_bf16:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def _save_hf_bundle(
    model,
    tokenizer,
    label_names: list[str],
    output_dir: Path,
    *,
    extra_info: dict | None = None,
) -> None:
    """Save model + tokenizer artifacts in a Hugging Face-friendly folder."""
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        model.save_pretrained(output_dir)
    except Exception:
        torch.save(model.state_dict(), output_dir / "pytorch_model.bin")

    (output_dir / "label_names.json").write_text(
        json.dumps(label_names, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    try:
        tokenizer.save_pretrained(output_dir)
    except Exception:
        LOGGER.warning("Tokenizer could not be saved to %s", output_dir)

    spm_path = output_dir / "spm.model"
    if not spm_path.exists():
        spm_source = getattr(tokenizer, "vocab_file", None)
        if spm_source and Path(spm_source).exists():
            spm_path.write_bytes(Path(spm_source).read_bytes())

    stm_path = output_dir / "special_tokens_map.json"
    if not stm_path.exists():
        try:
            stm_path.write_text(
                json.dumps(tokenizer.special_tokens_map, ensure_ascii=False, indent=2)
                + "\n",
                encoding="utf-8",
            )
        except Exception:
            LOGGER.warning("Could not write special_tokens_map.json to %s", output_dir)

    training_args = extra_info or {}
    torch.save(training_args, output_dir / "training_args.bin")

    model_name = training_args.get("model_name", "microsoft/deberta-v3-base")
    task = training_args.get("task", "multi_label_classification")
    lines = [
        "# DeBERTa-v3 Multi-label Value Classifier",
        "",
        f"- Base model: `{model_name}`",
        f"- Task: `{task}`",
        f"- Labels: `{len(label_names)}`",
        "- Input: `sentence`",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


@dataclass
class TextExample:
    text: str
    labels: torch.Tensor


class TextDataset(Dataset):
    def __init__(self, examples: list[TextExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> TextExample:
        return self.examples[idx]


def _build_dataloader(
    texts: list[str],
    labels: np.ndarray,
    tokenizer,
    *,
    batch_size: int,
    shuffle: bool,
    max_length: int,
) -> DataLoader:
    examples = [
        TextExample(text=text, labels=torch.tensor(label, dtype=torch.float32))
        for text, label in zip(texts, labels, strict=False)
    ]

    def _overflow_flags(batch_texts: list[str]) -> list[bool]:
        encoded = tokenizer(
            batch_texts,
            truncation=True,
            max_length=max_length,
            return_overflowing_tokens=True,
            return_length=True,
            padding=False,
        )
        mapping = encoded.get("overflow_to_sample_mapping", [])
        counts = [0 for _ in batch_texts]
        for idx in mapping:
            if 0 <= idx < len(counts):
                counts[idx] += 1
        return [c > 1 for c in counts]

    def collate(batch: list[TextExample]):
        batch_texts = [ex.text for ex in batch]
        batch_labels = torch.stack([ex.labels for ex in batch])
        encoded = encode_batch(tokenizer, batch_texts, max_length=max_length)
        encoded["labels"] = batch_labels
        encoded["overflowed"] = torch.tensor(
            _overflow_flags(batch_texts), dtype=torch.bool
        )
        return encoded

    return DataLoader(
        TextDataset(examples),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
    )


def _get_logits(outputs):
    if isinstance(outputs, torch.Tensor):
        return outputs
    if isinstance(outputs, dict):
        return outputs["logits"]
    return outputs.logits


def _predict_arrays(
    model,
    dataloader,
    device,
    *,
    threshold: float,
    use_bf16: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_labels: list[np.ndarray] = []
    all_preds: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []

    with torch.no_grad():
        for batch in dataloader:
            labels = batch.pop("labels").to(device)
            batch.pop("overflowed", None)
            batch = {k: v.to(device) for k, v in batch.items()}
            with _autocast_context(use_bf16):
                outputs = model(**batch)
                logits = _get_logits(outputs)
            probs = torch.sigmoid(logits.float()).cpu().numpy()
            preds = (probs >= threshold).astype(int)
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs)
            all_preds.append(preds)

    if not all_labels:
        return np.zeros((0, 0)), np.zeros((0, 0)), np.zeros((0, 0))
    return np.vstack(all_labels), np.vstack(all_preds), np.vstack(all_probs)


def _evaluate(
    model,
    dataloader,
    device,
    *,
    label_names: list[str],
    threshold: float,
    use_bf16: bool = False,
) -> dict[str, object]:
    y_true, y_pred, _ = _predict_arrays(
        model,
        dataloader,
        device,
        threshold=threshold,
        use_bf16=use_bf16,
    )
    return compute_f1_metrics(y_true, y_pred, label_names=label_names)


def _prediction_records(
    df,
    label_names: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
) -> list[dict]:
    records: list[dict] = []
    for row_idx, pred_vec in enumerate(y_pred):
        row = df.iloc[row_idx]
        gold_labels = [label_names[i] for i, val in enumerate(y_true[row_idx]) if val]
        pred_labels = [label_names[i] for i, val in enumerate(pred_vec) if val]
        records.append(
            {
                "text_id": str(row["text_id"]),
                "sent_id": str(row["sent_id"]),
                "gold_labels": gold_labels,
                "pred_labels": pred_labels,
                "probabilities": {
                    label_names[i]: float(y_probs[row_idx, i])
                    for i in range(len(label_names))
                },
            }
        )
    return records


def save_predictions_jsonl(
    model,
    tokenizer,
    df,
    label_names: list[str],
    output_path: Path,
    *,
    max_length: int,
    batch_size: int,
    threshold: float,
    use_bf16: bool = False,
) -> None:
    """Run inference on a dataframe and save predictions to JSONL."""
    device = next(model.parameters()).device
    texts = df["text"].astype(str).tolist()
    labels = df[label_names].to_numpy(dtype=float)
    dataloader = _build_dataloader(
        texts,
        labels,
        tokenizer,
        batch_size=batch_size,
        shuffle=False,
        max_length=max_length,
    )
    y_true, y_pred, y_probs = _predict_arrays(
        model,
        dataloader,
        device,
        threshold=threshold,
        use_bf16=use_bf16,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in _prediction_records(df, label_names, y_true, y_pred, y_probs):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    LOGGER.info("Saved predictions to %s", output_path)


def run_eval(
    config: dict,
    *,
    checkpoint_path: Path,
    split: str,
    output_pred_path: Path,
    output_metrics_path: Path,
    debug: bool = False,
    tune_threshold: bool = False,
    threshold_start: float = 0.0,
    threshold_stop: float = 1.0,
    threshold_step: float = 0.01,
) -> dict[str, object]:
    """Run evaluation for a given split and save predictions + metrics."""
    label_names = get_label_names()
    model_name = config.get("model", {}).get("name", "microsoft/deberta-v3-base")
    model, tokenizer = build_deberta_model(
        num_labels=len(label_names),
        model_name=model_name,
        label_names=label_names,
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    training_cfg = config.get("training", {})
    use_bf16 = _resolve_bf16(training_cfg, device)
    max_length = int(training_cfg.get("max_length", 512))
    batch_size = int(training_cfg.get("batch_size", 16))
    threshold = float(training_cfg.get("pred_threshold", 0.5))

    df = load_split(split)
    if debug:
        LOGGER.debug("Loaded %s split with %d rows", split, len(df))
    max_samples = config.get("max_samples")
    if max_samples is not None:
        df = df.head(int(max_samples))

    texts = df["text"].astype(str).tolist()
    labels = df[label_names].to_numpy(dtype=float)
    dataloader = _build_dataloader(
        texts,
        labels,
        tokenizer,
        batch_size=batch_size,
        shuffle=False,
        max_length=max_length,
    )
    y_true, y_pred, y_probs = _predict_arrays(
        model,
        dataloader,
        device,
        threshold=threshold,
        use_bf16=use_bf16,
    )

    metrics = compute_f1_metrics(y_true, y_pred, label_names=label_names)
    metrics["meta"] = {
        "model_name": model_name,
        "input": "sentence",
        "seed": config.get("seed", 42),
        "split": split,
        "threshold": threshold,
    }

    if tune_threshold:
        sweep = sweep_thresholds(
            y_true,
            y_probs,
            label_names=label_names,
            start=threshold_start,
            stop=threshold_stop,
            step=threshold_step,
        )
        metrics["threshold_sweep"] = {
            "best_threshold": sweep["best_threshold"],
            "best_metrics": sweep["best_metrics"],
        }

    output_pred_path.parent.mkdir(parents=True, exist_ok=True)
    with output_pred_path.open("w", encoding="utf-8") as handle:
        for record in _prediction_records(df, label_names, y_true, y_pred, y_probs):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    output_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    output_metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    LOGGER.info("Saved predictions to %s", output_pred_path)
    LOGGER.info("Saved metrics to %s", output_metrics_path)
    return metrics


def train_and_eval(
    config: dict,
    *,
    run_name: str | None = None,
    resume_path: Path | None = None,
) -> tuple[float, bool]:
    """Train DeBERTa and evaluate on validation."""
    set_seed(config.get("seed", 42))

    training_cfg = config.get("training", {})
    batch_size = int(training_cfg.get("batch_size", 16))
    num_epochs = int(training_cfg.get("num_epochs", 3))
    learning_rate = float(training_cfg.get("learning_rate", 2e-5))
    weight_decay = float(training_cfg.get("weight_decay", 0.01))
    max_length = int(training_cfg.get("max_length", 512))
    grad_accum_steps = int(training_cfg.get("grad_accum_steps", 1))
    max_grad_norm = float(training_cfg.get("max_grad_norm", 1.0))
    early_patience = int(training_cfg.get("early_stopping_patience", 3))
    collapse_threshold = float(training_cfg.get("collapse_threshold", 0.01))
    collapse_min_epochs = int(training_cfg.get("collapse_min_epochs", 3))
    model_name = config.get("model", {}).get("name", "microsoft/deberta-v3-base")

    results_dir = Path(config.get("results_dir", "results"))
    ckpt_dir = results_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading dataset splits")
    train_df = load_split("training")
    val_df = load_split("validation")
    max_samples = config.get("max_samples")
    if max_samples is not None:
        train_df = train_df.head(int(max_samples))
        val_df = val_df.head(int(max_samples))

    label_names = get_label_names()
    train_texts = train_df["text"].astype(str).tolist()
    val_texts = val_df["text"].astype(str).tolist()
    train_labels = train_df[label_names].to_numpy(dtype=float)
    val_labels = val_df[label_names].to_numpy(dtype=float)

    model, tokenizer = build_deberta_model(
        num_labels=len(label_names),
        model_name=model_name,
        label_names=label_names,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = _resolve_bf16(training_cfg, device)
    if training_cfg.get("gradient_checkpointing", False):
        try:
            model.gradient_checkpointing_enable()
            LOGGER.info("Enabled gradient checkpointing")
        except Exception:
            LOGGER.warning("Gradient checkpointing not supported by this model")
    model.to(device)
    if not use_bf16:
        model = model.float()
        model.to(device)
        LOGGER.info("Forced model parameters to fp32")
    else:
        LOGGER.info("Enabled bf16 autocast for training/evaluation")

    train_loader = _build_dataloader(
        train_texts,
        train_labels,
        tokenizer,
        batch_size=batch_size,
        shuffle=True,
        max_length=max_length,
    )
    val_loader = _build_dataloader(
        val_texts,
        val_labels,
        tokenizer,
        batch_size=batch_size,
        shuffle=False,
        max_length=max_length,
    )

    try:
        first_batch = next(iter(train_loader))
        first_labels = first_batch.pop("labels").to(device)
        first_batch.pop("overflowed", None)
        first_batch = {k: v.to(device) for k, v in first_batch.items()}
        with torch.no_grad():
            with _autocast_context(use_bf16):
                first_outputs = model(**first_batch)
                first_logits = _get_logits(first_outputs)
        if torch.isnan(first_logits).any() or torch.isinf(first_logits).any():
            LOGGER.error("NaN/Inf logits detected in sanity check batch")
            LOGGER.debug("Sanity labels mean=%.4f", first_labels.float().mean().item())
            raise RuntimeError("Sanity check failed: NaN/Inf logits in first batch")
    except StopIteration:
        LOGGER.warning("Training dataloader is empty; skipping training")
        return float("-inf"), True

    optimizer = _build_adamw(
        model.parameters(),
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        training_cfg=training_cfg,
    )
    loss_fn = nn.BCEWithLogitsLoss()

    best_metric = -math.inf
    suffix = run_name or "deberta_bce_best"
    best_path = ckpt_dir / f"{suffix}.pt"
    last_path = ckpt_dir / f"{suffix}_last.pt"
    start_epoch = 1
    epochs_no_improve = 0
    save_checkpoints = bool(config.get("save_checkpoints", True))
    save_hf_model = bool(training_cfg.get("save_hf_model", True))
    threshold = float(training_cfg.get("pred_threshold", 0.5))
    checkpoint_every = int(training_cfg.get("checkpoint_every_epochs", 1))
    token_stats = {"max_length": max_length, "train_truncated": 0, "train_total": 0}
    val_history: list[dict[str, float]] = []
    epochs_completed = 0

    if resume_path is not None and resume_path.exists():
        checkpoint = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        best_metric = float(checkpoint.get("best_metric", best_metric))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        LOGGER.info("Resumed training from %s (epoch %d)", resume_path, start_epoch)

    LOGGER.info(
        "Training config: batch=%d lr=%g wd=%g max_len=%d accum=%d save_ckpt=%s",
        batch_size,
        learning_rate,
        weight_decay,
        max_length,
        grad_accum_steps,
        save_checkpoints,
    )

    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        total_loss = 0.0
        batches_used = 0
        LOGGER.info("Starting epoch %d/%d", epoch, num_epochs)
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader, start=1):
            labels = batch.pop("labels").to(device)
            overflowed = batch.pop("overflowed", None)
            batch = {k: v.to(device) for k, v in batch.items()}
            if overflowed is not None and epoch == start_epoch:
                token_stats["train_truncated"] += int(overflowed.sum().item())
                token_stats["train_total"] += int(overflowed.numel())
            with _autocast_context(use_bf16):
                outputs = model(**batch)
                logits = _get_logits(outputs)
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                LOGGER.warning("NaN/Inf logits detected; skipping batch")
                continue
            if torch.isnan(labels).any() or torch.isinf(labels).any():
                LOGGER.warning("NaN/Inf labels detected; skipping batch")
                continue
            loss = loss_fn(logits.float(), labels.float()) / max(grad_accum_steps, 1)
            if torch.isnan(loss):
                LOGGER.warning("NaN loss detected; skipping batch")
                continue
            loss.backward()
            if step % grad_accum_steps == 0:
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
            total_loss += loss.item()
            batches_used += 1

        if batches_used > 0 and (batches_used % grad_accum_steps != 0):
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

        if batches_used == 0:
            LOGGER.warning("No valid batches in epoch %d; stopping early.", epoch)
            break

        avg_loss = total_loss / max(batches_used, 1)
        LOGGER.info("Epoch %d/%d - train loss %.4f", epoch, num_epochs, avg_loss)

        metrics = _evaluate(
            model,
            val_loader,
            device,
            label_names=label_names,
            threshold=threshold,
            use_bf16=use_bf16,
        )
        LOGGER.info(
            "Validation metrics - macro_f1=%.4f micro_f1=%.4f",
            metrics["macro_f1"],
            metrics["micro_f1"],
        )
        val_history.append(
            {
                "epoch": float(epoch),
                "macro_f1": float(metrics["macro_f1"]),
                "micro_f1": float(metrics["micro_f1"]),
            }
        )
        epochs_completed += 1

        if metrics["macro_f1"] > best_metric:
            best_metric = float(metrics["macro_f1"])
            if save_checkpoints:
                torch.save(model.state_dict(), best_path)
                LOGGER.info("Saved best checkpoint to %s", best_path)
                if save_hf_model:
                    hf_dir = results_dir / "hf_models" / suffix
                    hf_meta = {
                        "model_name": model_name,
                        "task": "multi_label_classification",
                        "seed": config.get("seed", 42),
                        "batch_size": batch_size,
                        "learning_rate": learning_rate,
                        "weight_decay": weight_decay,
                        "max_length": max_length,
                        "num_epochs": num_epochs,
                        "grad_accum_steps": grad_accum_steps,
                        "pred_threshold": threshold,
                    }
                    _save_hf_bundle(
                        model, tokenizer, label_names, hf_dir, extra_info=hf_meta
                    )
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if early_patience > 0 and epochs_no_improve >= early_patience:
                LOGGER.info(
                    "Early stopping triggered after %d epochs without improvement",
                    epochs_no_improve,
                )
                break

        if (
            save_checkpoints
            and checkpoint_every > 0
            and (epoch % checkpoint_every == 0)
        ):
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_metric": best_metric,
                },
                last_path,
            )
            LOGGER.info("Saved last checkpoint to %s", last_path)

    stats_path = results_dir / "logs" / f"token_stats_{suffix}.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    if token_stats["train_total"] > 0:
        token_stats["train_truncated_rate"] = (
            token_stats["train_truncated"] / token_stats["train_total"]
        )
    token_stats["truncation_method"] = "overflow_to_sample_mapping"
    stats_path.write_text(
        json.dumps(token_stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if val_history:
        LOGGER.info("Validation summary (per epoch):")
        for row in val_history:
            LOGGER.info(
                "%5d  macro_f1=%8.4f  micro_f1=%8.4f",
                int(row["epoch"]),
                row["macro_f1"],
                row["micro_f1"],
            )

    collapsed = False
    if epochs_completed >= collapse_min_epochs and best_metric < collapse_threshold:
        collapsed = True
        LOGGER.warning(
            "Run flagged as collapsed (best_macro_f1=%.4f < %.4f after %d epochs)",
            best_metric,
            collapse_threshold,
            epochs_completed,
        )
    return float(best_metric), collapsed
