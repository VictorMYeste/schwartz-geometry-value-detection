"""Build the 5-seed BCE probability ensemble from saved predictions (revision).

No model is retrained or re-run. For the five final BCE seeds the script

1. loads the saved validation and test probabilities and checks that every seed has the
   same sentences (Text-ID, Sentence-ID) in the same order with the same gold labels;
2. averages the probabilities over the seeds (arithmetic mean, per sentence and label);
3. retunes per-label thresholds on validation with the same procedure as the single-seed
   pipeline (``sweep_per_label_thresholds``, 0.00-1.00 in steps of 0.01, maximizing each
   label's F1; see ``run_eval(..., tune_threshold=True, threshold_tuning_mode="per_label")``);
4. writes prediction and metrics files in the same format and naming scheme as a seed's, so
   ``load_prediction_jsonl``, ``load_validation_thresholds``, ``discover_runs`` and
   ``decode_predictions`` read them unchanged:

   results/ensemble/predictions/deberta_bce_ensemble5_seed0_deberta-v3-base_validation_thresholds.jsonl(.gz)
   results/ensemble/predictions/deberta_bce_ensemble5_seed0_deberta-v3-base_test.jsonl(.gz)
   results/ensemble/logs/deberta_bce_ensemble5_seed0_deberta-v3-base_validation_thresholds.json
   results/ensemble/logs/deberta_bce_ensemble5_seed0_deberta-v3-base_test_metrics.json

   ("seed0" only marks the ensemble so that the file names match the existing pattern; the
   member seeds are listed in the metrics files.)
5. writes ``results/ensemble/ensemble_summary.csv``: thresholded test metrics of each seed,
   their mean and standard deviation, and the ensemble.

The files hold IDs, gold labels and probabilities only; no sentence text.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from geometry_aware_calibration import (  # noqa: E402
    load_prediction_jsonl,
    load_validation_thresholds,
)
from schwartz_energy_decoder import decoder_geometry_cost  # noqa: E402
from schwartz_value_geometry.eval.metrics import (  # noqa: E402
    binarize_probs,
    compute_all_metrics,
    sweep_per_label_thresholds,
)

SEEDS = [42, 7, 1701, 11, 1984]
MODEL_SLUG = "deberta-v3-base"
PRED_DIR = ROOT / "results" / "predictions"
LOGS_DIR = ROOT / "results" / "logs"
OUT_DIR = ROOT / "results" / "ensemble"
ENSEMBLE_PREFIX = f"deberta_bce_ensemble5_seed0_{MODEL_SLUG}"

# Same sweep as eval_deberta.py defaults and the final Slurm launchers
# (THRESHOLD_MODE=per_label, THRESHOLD_STEP=0.01).
THRESHOLD_START, THRESHOLD_STOP, THRESHOLD_STEP = 0.0, 1.0, 0.01

SUMMARY_METRICS = [
    "macro_f1",
    "micro_f1",
    "decoder_geometry_cost",
    "opposite_error_rate",
    "neighbor_error_rate",
    "confusion_distance_correlation",
    "macro_auprc",
    "micro_auprc",
]


def read_ids(path: Path) -> list[tuple[str, str]]:
    ids: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                row = json.loads(line)
                ids.append((str(row["text_id"]), str(row["sent_id"])))
    return ids


def load_split_across_seeds(split_suffix: str):
    """Return (ids, label_names, y_true, stacked probs [seed, n, labels])."""
    ids_ref = names_ref = gold_ref = None
    probs = []
    for seed in SEEDS:
        path = PRED_DIR / f"deberta_bce_seed{seed}_{MODEL_SLUG}_{split_suffix}.jsonl"
        preds = load_prediction_jsonl(path)
        ids = read_ids(path)
        if ids_ref is None:
            ids_ref, names_ref, gold_ref = ids, preds.label_names, preds.y_true
        else:
            if ids != ids_ref:
                raise ValueError(f"Sentence IDs/order differ for seed {seed} ({split_suffix})")
            if preds.label_names != names_ref:
                raise ValueError(f"Label order differs for seed {seed} ({split_suffix})")
            if not np.array_equal(preds.y_true, gold_ref):
                raise ValueError(f"Gold labels differ for seed {seed} ({split_suffix})")
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate sentence IDs in {path}")
        probs.append(preds.y_probs)
    return ids_ref, names_ref, gold_ref, np.stack(probs)


def write_predictions(path: Path, ids, names, y_true, y_pred, y_probs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for (text_id, sent_id), gold, pred, probs in zip(ids, y_true, y_pred, y_probs, strict=True):
            record = {
                "text_id": text_id,
                "sent_id": sent_id,
                "gold_labels": [names[i] for i in np.flatnonzero(gold)],
                "pred_labels": [names[i] for i in np.flatnonzero(pred)],
                "probabilities": {names[i]: float(probs[i]) for i in range(len(names))},
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with path.open("rb") as src, gzip.open(path.with_name(path.name + ".gz"), "wb") as dst:
        shutil.copyfileobj(src, dst)


def scalar_metrics(metrics: dict) -> dict[str, float]:
    out = {key: float(metrics[key]) for key in SUMMARY_METRICS if key in metrics}
    out["decoder_geometry_cost"] = decoder_geometry_cost(metrics)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    out_dir: Path = args.out_dir
    meta_common = {
        "model_name": "microsoft/deberta-v3-base",
        "loss": {"name": "bce"},
        "input": "sentence",
        "ensemble": {
            "members": [f"deberta_bce_seed{s}_{MODEL_SLUG}" for s in SEEDS],
            "seeds": SEEDS,
            "combination": "arithmetic mean of sigmoid probabilities",
        },
    }

    # Validation: average, retune per-label thresholds exactly as run_eval does.
    val_ids, names, val_true, val_stack = load_split_across_seeds("validation_thresholds")
    val_probs = val_stack.mean(axis=0)
    sweep = sweep_per_label_thresholds(
        val_true, val_probs, label_names=names,
        start=THRESHOLD_START, stop=THRESHOLD_STOP, step=THRESHOLD_STEP,
    )
    thresholds = list(sweep["best_thresholds"])
    val_pred = binarize_probs(val_probs, threshold=thresholds)
    val_metrics = compute_all_metrics(val_true, val_pred, val_probs, label_names=names)
    val_metrics["threshold_tuning"] = {
        "mode": "per_label",
        "thresholds": sweep["best_thresholds"],
        "thresholds_by_label": sweep["best_thresholds_by_label"],
        "best_metrics": sweep["best_metrics"],
    }
    val_metrics["meta"] = {
        **meta_common,
        "seed": 0,
        "split": "validation",
        "threshold": dict(zip(names, thresholds, strict=True)),
        "threshold_source": "tuned_per_label",
    }

    # Test: average, apply the frozen validation thresholds.
    test_ids, test_names, test_true, test_stack = load_split_across_seeds("test")
    if test_names != names:
        raise ValueError("Validation and test label orders differ")
    test_probs = test_stack.mean(axis=0)
    test_pred = binarize_probs(test_probs, threshold=thresholds)
    test_metrics = compute_all_metrics(test_true, test_pred, test_probs, label_names=names)
    test_metrics["meta"] = {
        **meta_common,
        "seed": 0,
        "split": "test",
        "threshold": dict(zip(names, thresholds, strict=True)),
        "threshold_source": "provided",
    }

    pred_dir, logs_dir = out_dir / "predictions", out_dir / "logs"
    val_pred_path = pred_dir / f"{ENSEMBLE_PREFIX}_validation_thresholds.jsonl"
    test_pred_path = pred_dir / f"{ENSEMBLE_PREFIX}_test.jsonl"
    val_metrics_path = logs_dir / f"{ENSEMBLE_PREFIX}_validation_thresholds.json"
    test_metrics_path = logs_dir / f"{ENSEMBLE_PREFIX}_test_metrics.json"
    write_predictions(val_pred_path, val_ids, names, val_true, val_pred, val_probs)
    write_predictions(test_pred_path, test_ids, names, test_true, test_pred, test_probs)
    logs_dir.mkdir(parents=True, exist_ok=True)
    val_metrics_path.write_text(json.dumps(val_metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    test_metrics_path.write_text(json.dumps(test_metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Round-trip check: the standard loaders read the files back unchanged.
    reread = load_prediction_jsonl(test_pred_path)
    reread_thr = load_validation_thresholds(val_metrics_path, reread.label_names)
    assert reread.label_names == names
    assert np.array_equal(reread.y_true, test_true)
    assert np.array_equal(reread.y_probs, test_probs)
    assert np.array_equal(reread_thr, np.asarray(thresholds))

    # Summary: each seed's thresholded test metrics (its own frozen thresholds) and the ensemble.
    rows = []
    for idx, seed in enumerate(SEEDS):
        seed_thr = load_validation_thresholds(
            LOGS_DIR / f"deberta_bce_seed{seed}_{MODEL_SLUG}_validation_thresholds.json", names
        )
        seed_pred = binarize_probs(test_stack[idx], threshold=seed_thr)
        m = compute_all_metrics(test_true, seed_pred, test_stack[idx], label_names=names)
        rows.append({"system": f"seed{seed}", **scalar_metrics(m),
                     "labels_per_sentence": float(seed_pred.sum(1).mean())})
    seed_rows = list(rows)
    for stat, fn in (("seed_mean", np.mean), ("seed_sd", lambda v: np.std(v, ddof=1))):
        rows.append({"system": stat, **{
            key: float(fn([r[key] for r in seed_rows])) for key in seed_rows[0] if key != "system"
        }})
    rows.append({"system": "ensemble5", **scalar_metrics(test_metrics),
                 "labels_per_sentence": float(test_pred.sum(1).mean())})
    summary_path = out_dir / "ensemble_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"validation: {len(val_ids)} sentences, test: {len(test_ids)} sentences, {len(SEEDS)} seeds aligned")
    print("ensemble thresholds:", dict(zip(names, thresholds, strict=True)))
    print(f"{'system':<10} " + " ".join(f"{k:>12}" for k in ("macro_f1", "micro_f1", "geo_cost", "opp_err", "labels/sent")))
    for r in rows:
        print(f"{r['system']:<10} {r['macro_f1']:12.4f} {r['micro_f1']:12.4f} "
              f"{r['decoder_geometry_cost']:12.4f} {r['opposite_error_rate']:12.4f} {r['labels_per_sentence']:12.3f}")
    print(f"wrote {summary_path.relative_to(ROOT)} and the ensemble files under {out_dir.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
