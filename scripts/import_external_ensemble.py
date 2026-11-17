"""Import an external ensemble's saved probabilities so the decoder pipeline can read them (revision).

Used for the near-state-of-the-art check (Reviewer E): the soft-voting DeBERTa ensemble of
Yeste & Rosso (2026, arXiv 2601.14172), Macro-F1 0.332 on the official test split. Its
validation and test probabilities are TSV files (Text-ID, Sentence-ID, one column per value)
released in that paper's public repository (https://github.com/VictorMYeste/human-value-detection,
approaches/moral-values/output/). No model is retrained or re-run.

The script
1. reads the two TSVs and aligns them, by (Text-ID, Sentence-ID), to the sentence order and
   label order of this repository's own prediction files, failing on any missing, extra or
   duplicate sentence and on any label mismatch;
2. takes the gold labels from this repository's prediction files (value presence, attained and
   constrained merged), so the external system is scored against exactly our gold;
3. reproduces the external system's own decision rule: one global threshold tuned on validation
   (0.01-0.99 in steps of 0.01, maximizing Macro-F1). With ``--expect-test-macro-f1`` it checks
   that the resulting test Macro-F1 matches the published value;
4. writes prediction and metrics files in the format of a seed's, so ``load_prediction_jsonl``,
   ``load_validation_thresholds`` and ``decode_predictions`` read them unchanged:

   results/external/predictions/<name>_seed0_external_{validation_thresholds,test}.jsonl(.gz)
   results/external/logs/<name>_seed0_external_{validation_thresholds,test_metrics}.json
   results/external/external_summary.csv

The files hold IDs, gold labels and probabilities only; no sentence text.

Example (the external repository sits next to this one):
  python3 scripts/import_external_ensemble.py \
    --val-tsv  ../../Paper-1/human-value-detection/approaches/moral-values/output/direct_champion-tuned-soft-champion-val.tsv \
    --test-tsv ../../Paper-1/human-value-detection/approaches/moral-values/output/direct_champion-tuned-soft-champion-test.tsv \
    --expect-test-macro-f1 0.33187
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_bce_ensemble import read_ids, scalar_metrics, write_predictions  # noqa: E402
from geometry_aware_calibration import load_prediction_jsonl, load_validation_thresholds  # noqa: E402
from schwartz_value_geometry.eval.metrics import (  # noqa: E402
    binarize_probs,
    compute_all_metrics,
    compute_global_metrics,
)

REFERENCE = "deberta_bce_seed42_deberta-v3-base"  # any of our runs: only IDs, label order and gold are used
PRED_DIR = ROOT / "results" / "predictions"
OUT_DIR = ROOT / "results" / "external"
GRID = np.round(np.arange(0.01, 1.00, 0.01), 2)
ID_COLS = ["Text-ID", "Sentence-ID"]


def load_external(tsv: Path, ids: list[tuple[str, str]], names: list[str]) -> np.ndarray:
    df = pd.read_csv(tsv, sep="\t", dtype={c: str for c in ID_COLS})
    values = [c for c in df.columns if c not in ID_COLS]
    if set(values) != set(names) or len(values) != len(names):
        raise ValueError(f"{tsv.name}: label columns differ from ours: {sorted(set(values) ^ set(names))}")
    df = df.set_index(ID_COLS)
    if not df.index.is_unique:
        raise ValueError(f"{tsv.name}: duplicate sentence IDs")
    if set(df.index) != set(ids):
        raise ValueError(f"{tsv.name}: sentences differ from our split ({len(df)} vs {len(ids)})")
    probs = df.loc[ids, names].to_numpy(dtype=float)
    if not np.isfinite(probs).all() or probs.min() < 0 or probs.max() > 1:
        raise ValueError(f"{tsv.name}: values are not probabilities")
    return probs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--val-tsv", type=Path, required=True)
    parser.add_argument("--test-tsv", type=Path, required=True)
    parser.add_argument("--name", default="yeste2026sentence_softvote",
                        help="file prefix; the default names the soft-voting ensemble of arXiv 2601.14172")
    parser.add_argument("--source", default="Yeste & Rosso (2026), arXiv 2601.14172: soft-voting DeBERTa ensemble "
                                            "(direct_champion-tuned-soft-champion)")
    parser.add_argument("--expect-test-macro-f1", type=float, default=None,
                        help="fail unless the reproduced test Macro-F1 matches this value within 5e-6")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    splits = {}
    for split, tsv in (("validation_thresholds", args.val_tsv), ("test", args.test_tsv)):
        ref_path = PRED_DIR / f"{REFERENCE}_{split}.jsonl"
        ref = load_prediction_jsonl(ref_path)
        ids = read_ids(ref_path)
        probs = load_external(tsv, ids, ref.label_names)
        splits[split] = {"ids": ids, "names": ref.label_names, "gold": ref.y_true.astype(int), "probs": probs}
    names = splits["test"]["names"]
    if splits["validation_thresholds"]["names"] != names:
        raise ValueError("validation and test label orders differ")

    val = splits["validation_thresholds"]
    val_scores = np.array([compute_global_metrics(val["gold"], (val["probs"] >= t).astype(int))["macro_f1"]
                           for t in GRID])
    t_star = float(GRID[int(np.argmax(val_scores))])  # first maximum, as np.argmax
    thresholds = [t_star] * len(names)

    prefix = f"{args.name}_seed0_external"
    meta = {"source": args.source, "input_files": [args.val_tsv.name, args.test_tsv.name],
            "threshold_source": "tuned_global (0.01-0.99, step 0.01, validation Macro-F1)",
            "reference_ids_and_gold": REFERENCE}
    pred_dir, logs_dir = args.out_dir / "predictions", args.out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    metrics = {}
    for split, data in splits.items():
        pred = binarize_probs(data["probs"], threshold=thresholds)
        m = compute_all_metrics(data["gold"], pred, data["probs"], label_names=names)
        m["meta"] = {**meta, "seed": 0, "split": "validation" if split != "test" else "test",
                     "threshold": dict(zip(names, thresholds, strict=True))}
        write_predictions(pred_dir / f"{prefix}_{split}.jsonl", data["ids"], names, data["gold"], pred, data["probs"])
        out_name = f"{prefix}_validation_thresholds.json" if split != "test" else f"{prefix}_test_metrics.json"
        (logs_dir / out_name).write_text(json.dumps(m, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        metrics[split] = (m, pred)

    # Round trip: the standard loaders read the files back unchanged.
    reread = load_prediction_jsonl(pred_dir / f"{prefix}_test.jsonl")
    reread_thr = load_validation_thresholds(logs_dir / f"{prefix}_validation_thresholds.json", reread.label_names)
    assert reread.label_names == names
    assert np.array_equal(reread.y_true, splits["test"]["gold"])
    assert np.allclose(reread.y_probs, splits["test"]["probs"], rtol=0, atol=1e-12)
    assert np.array_equal(reread_thr, np.asarray(thresholds))

    test_m, test_pred = metrics["test"]
    macro = float(test_m["macro_f1"])
    if args.expect_test_macro_f1 is not None and abs(macro - args.expect_test_macro_f1) > 5e-6:
        raise SystemExit(f"reproduced test Macro-F1 {macro:.5f} != expected {args.expect_test_macro_f1:.5f}")
    row = {"system": args.name, "threshold_global": t_star, "validation_macro_f1": float(val_scores.max()),
           **scalar_metrics(test_m), "labels_per_sentence": float(test_pred.sum(1).mean())}
    summary = args.out_dir / "external_summary.csv"
    with summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    print(f"aligned {len(splits['validation_thresholds']['ids'])} validation and {len(splits['test']['ids'])} "
          f"test sentences; global threshold {t_star:.2f} (validation Macro-F1 {val_scores.max():.5f})")
    print("test: " + ", ".join(f"{k} {v:.4f}" for k, v in row.items() if isinstance(v, float)))
    print(f"wrote {summary.relative_to(ROOT)} and files under {args.out_dir.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
