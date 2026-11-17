"""Draw the decoder-edit examples for the paper by a fixed rule.

Pool: test sentences that the Schwartz decoder edits on seed 42 (frozen schwartz /
full / pareto_99 setting, the same as the human evaluation), minus every sentence
used in the human evaluation (so no judge sees an example's labels before judging).

Each edited sentence falls into one cell: edit type (addition only / removal only /
mixed) x outcome against the gold labels (sentence-level F1 better / worse / same).
The script writes the count of every cell, computed on all edited sentences, and
draws one sentence at random from each addition and removal cell (fixed seed).

Outputs (no dataset text):
- results/paper_examples/edit_outcome_cells.csv: cell counts, seed 42.
- results/paper_examples/examples.csv: the drawn sentences (IDs, cell, label sets,
  F1 before and after).
With --text-out FILE (outside the repository) it also writes the original sentences,
for paraphrasing; the paper shows paraphrases only (data usage agreement).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from compute_decoder_edit_rate import DECODE_KW, SETTING  # noqa: E402
from eval_decoder_edits_external import TIE, sample_f1  # noqa: E402
from geometry_aware_calibration import load_prediction_jsonl, load_validation_thresholds  # noqa: E402
from schwartz_energy_decoder import build_geometry_matrices, decode_predictions  # noqa: E402
from schwartz_value_geometry.data.dataset import load_split  # noqa: E402
from schwartz_value_geometry.eval.metrics import binarize_probs  # noqa: E402

REFERENCE_SEED = 42
EXAMPLE_SEED = 2026
EDIT_TYPES = ("addition", "removal", "mixed")
OUTCOMES = ("better", "worse", "same")
OUT_DIR = ROOT / "results" / "paper_examples"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--human-key", type=Path, default=ROOT.parent / "human-eval" / "key.csv",
                    help="human-evaluation key (only its text_id / sent_id columns are read)")
    ap.add_argument("--text-out", type=Path, default=None,
                    help="write the original sentences here (must be outside the repository)")
    args = ap.parse_args()
    if args.text_out is not None and ROOT in args.text_out.resolve().parents:
        sys.exit("--text-out must be outside the repository (dataset text)")

    pred_path = ROOT / "results" / "predictions" / f"deberta_bce_seed{REFERENCE_SEED}_deberta-v3-base_test.jsonl"
    thr_path = ROOT / "results" / "logs" / f"deberta_bce_seed{REFERENCE_SEED}_deberta-v3-base_validation_thresholds.json"
    preds = load_prediction_jsonl(pred_path)
    names = preds.label_names
    thresholds = load_validation_thresholds(thr_path, names)
    y_thr = binarize_probs(preds.y_probs, threshold=thresholds).astype(int)
    geometry = build_geometry_matrices(names, geometry="schwartz",
                                       neighbor_steps=DECODE_KW["neighbor_steps"], random_seed=42)
    y_dec, _ = decode_predictions(preds, thresholds=thresholds, setting=SETTING,
                                  geometry_matrices=geometry, **DECODE_KW)
    y_dec = y_dec.astype(int)
    gold = preds.y_true.astype(int)
    ids = pd.read_json(pred_path, lines=True, dtype=False)[["text_id", "sent_id"]].astype(str)

    added = ((y_dec == 1) & (y_thr == 0)).any(1)
    removed = ((y_dec == 0) & (y_thr == 1)).any(1)
    edited = added | removed
    edit_type = np.where(added & removed, "mixed", np.where(added, "addition", "removal"))
    f1_thr, f1_dec = sample_f1(gold, y_thr), sample_f1(gold, y_dec)
    delta = f1_dec - f1_thr
    outcome = np.where(delta > TIE, "better", np.where(delta < -TIE, "worse", "same"))

    cells = pd.crosstab(pd.Categorical(edit_type[edited], EDIT_TYPES),
                        pd.Categorical(outcome[edited], OUTCOMES), dropna=False)
    cells.index.name, cells.columns.name = "edit_type", "outcome"
    print(f"edited sentences (seed {REFERENCE_SEED}): {int(edited.sum())}\n{cells}")

    key = pd.read_csv(args.human_key, dtype=str, usecols=["text_id", "sent_id"])
    in_study = set(zip(key["text_id"], key["sent_id"]))
    eligible = edited & ~np.array([(t, s) in in_study for t, s in zip(ids["text_id"], ids["sent_id"])])
    print(f"eligible after removing human-study sentences: {int(eligible.sum())}")

    def labels(row: np.ndarray) -> str:
        return "; ".join(names[j] for j in np.flatnonzero(row))

    rng = np.random.default_rng(EXAMPLE_SEED)
    rows = []
    for t in ("addition", "removal"):
        for o in OUTCOMES:
            pool = np.flatnonzero(eligible & (edit_type == t) & (outcome == o))
            i = int(rng.choice(pool))
            rows.append({"edit_type": t, "outcome": o, "cell_count": int(cells.loc[t, o]),
                         "eligible_in_cell": len(pool), "text_id": ids.at[i, "text_id"],
                         "sent_id": ids.at[i, "sent_id"], "gold": labels(gold[i]),
                         "thresholded": labels(y_thr[i]), "decoded": labels(y_dec[i]),
                         "added": labels((y_dec[i] == 1) & (y_thr[i] == 0)),
                         "removed": labels((y_dec[i] == 0) & (y_thr[i] == 1)),
                         "f1_thresholded": round(float(f1_thr[i]), 4),
                         "f1_decoded": round(float(f1_dec[i]), 4)})
    examples = pd.DataFrame(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cells.reset_index().to_csv(OUT_DIR / "edit_outcome_cells.csv", index=False)
    examples.to_csv(OUT_DIR / "examples.csv", index=False)
    print(f"wrote {(OUT_DIR / 'examples.csv').relative_to(ROOT)} and edit_outcome_cells.csv")

    if args.text_out is not None:
        test = load_split("test")
        text = {(str(t), str(s)): str(x) for t, s, x in zip(test["text_id"], test["sent_id"], test["text"])}
        out = examples[["edit_type", "outcome", "text_id", "sent_id"]].copy()
        out["text"] = [text[(t, s)] for t, s in zip(out["text_id"], out["sent_id"])]
        out.to_csv(args.text_out, index=False)
        print(f"wrote original sentences to {args.text_out.name} (outside the repository)")


if __name__ == "__main__":
    main()
