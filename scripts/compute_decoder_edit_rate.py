"""Compute the Schwartz energy decoder's edit rate on the test split.

Reuses the frozen decoder (schwartz / full / pareto_99) over the saved BCE
probabilities and validation-frozen per-label thresholds. Reports, per seed and
averaged over the five final seeds:

- edit rate: fraction of test sentences whose decoded label set differs from the
  independent-thresholding label set;
- average |y_hat| before/after decoding (over all sentences and over edited ones);
- additions/removals among edited sentences.

This is a post-hoc analysis on saved predictions; no model weights are touched.
"""

from __future__ import annotations

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
from schwartz_value_geometry.eval.metrics import binarize_probs  # noqa: E402
from schwartz_energy_decoder import (  # noqa: E402
    EnergySetting,
    build_geometry_matrices,
    decode_predictions,
)

SEEDS = [42, 7, 1701, 11, 1984]
PRED_DIR = ROOT / "results" / "predictions"
LOGS_DIR = ROOT / "results" / "logs"
MODEL_SLUG = "deberta-v3-base"

# Frozen schwartz / full / pareto_99 selection (identical across all five seeds).
SETTING = EnergySetting(alpha_neighbor=0.1, beta_opposite=0.2, gamma_cardinality=0.0)
DECODE_KW = dict(
    top_k=8,
    max_candidates=8,
    max_labels=5,
    threshold_factor=0.5,
    min_prob=0.01,
    marginal_temperature=1.0,
    neighbor_steps=2,
)


def main() -> None:
    rows = []
    for seed in SEEDS:
        test_path = PRED_DIR / f"deberta_bce_seed{seed}_{MODEL_SLUG}_test.jsonl"
        thr_path = LOGS_DIR / f"deberta_bce_seed{seed}_{MODEL_SLUG}_validation_thresholds.json"
        preds = load_prediction_jsonl(test_path)
        thresholds = load_validation_thresholds(thr_path, preds.label_names)

        geometry = build_geometry_matrices(
            preds.label_names, geometry="schwartz",
            neighbor_steps=DECODE_KW["neighbor_steps"], random_seed=42,
        )
        y_thr = binarize_probs(preds.y_probs, threshold=thresholds)
        y_dec, _ = decode_predictions(
            preds, thresholds=thresholds, setting=SETTING,
            geometry_matrices=geometry, **DECODE_KW,
        )

        n = y_thr.shape[0]
        before = y_thr.sum(axis=1)
        after = y_dec.sum(axis=1)
        changed = np.any(y_thr != y_dec, axis=1)
        n_changed = int(changed.sum())

        added = int(((y_dec == 1) & (y_thr == 0)).sum())
        removed = int(((y_dec == 0) & (y_thr == 1)).sum())

        rows.append(dict(
            seed=seed,
            n=n,
            edit_rate=n_changed / n,
            n_changed=n_changed,
            mean_before_all=before.mean(),
            mean_after_all=after.mean(),
            mean_before_changed=before[changed].mean() if n_changed else 0.0,
            mean_after_changed=after[changed].mean() if n_changed else 0.0,
            labels_added=added,
            labels_removed=removed,
        ))
        r = rows[-1]
        print(
            f"seed {seed:>4}: edit_rate={r['edit_rate']*100:5.2f}%  "
            f"changed={n_changed:>4}/{n}  "
            f"|y| all {r['mean_before_all']:.4f}->{r['mean_after_all']:.4f}  "
            f"|y| edited {r['mean_before_changed']:.3f}->{r['mean_after_changed']:.3f}  "
            f"+{added} / -{removed}"
        )

    def ms(key):
        vals = np.array([r[key] for r in rows], dtype=float)
        return vals.mean(), vals.std()

    print("\n=== mean +/- std over 5 seeds ===")
    er_m, er_s = ms("edit_rate")
    print(f"edit rate            : {er_m*100:.2f}% +/- {er_s*100:.2f}%")
    ba_m, ba_s = ms("mean_before_all")
    aa_m, aa_s = ms("mean_after_all")
    print(f"avg |y_hat| (all)    : {ba_m:.4f} -> {aa_m:.4f}")
    bc_m, _ = ms("mean_before_changed")
    ac_m, _ = ms("mean_after_changed")
    print(f"avg |y_hat| (edited) : {bc_m:.3f} -> {ac_m:.3f}")
    add_m, _ = ms("labels_added")
    rem_m, _ = ms("labels_removed")
    print(f"labels added (mean)  : {add_m:.1f}")
    print(f"labels removed (mean): {rem_m:.1f}")
    nc_m, nc_s = ms("n_changed")
    print(f"changed sentences    : {nc_m:.1f} +/- {nc_s:.1f} of {rows[0]['n']}")


if __name__ == "__main__":
    main()
