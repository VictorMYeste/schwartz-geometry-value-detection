"""Gold-based evaluation of the Schwartz decoder's edits, with a matched-weight random-order null.

Read-only on saved predictions; no weights are re-selected.

Systems: the five final BCE seeds and the 5-seed BCE ensemble (``build_bce_ensemble.py``),
each with its validation-frozen per-label thresholds. Decoder: the frozen setting of the
paper (``SETTING``, ``DECODE_KW`` in ``compute_decoder_edit_rate.py``: alpha = 0.1,
beta = 0.2, gamma = 0, neighbor_steps = 2) on the circular order of ``geometry.py``.

Per system, decoded vs thresholded label sets:
- edited sentences, split by edit type (additions only / removals only / both);
- improved / worsened / unchanged sample-level F1 among edited sentences;
- labels added (share in gold) and removed (share not in gold);
- delta Macro-F1, delta Micro-F1, and delta of the Schwartz geometry cost
  (opposite-error rate + neighbor-error rate + confusion-distance correlation, as in the
  paper), on all sentences and on value-positive sentences (gold set non-empty).

Random-order null at matched weights (P1): ``--n-orders`` random circular orders, order i
drawn with the repository's ``random_circular_order(label_names, seed=order_seed_base + i)``
(the label column order is never changed). Each order is decoded at exactly the frozen weights.
The same order is applied to all five seeds and its deltas are averaged over the seeds; the
ensemble is tested against the same orders separately. Empirical one-sided p-values:
p = (1 + #{null >= observed}) / (1 + N) for delta Macro-F1 and delta Micro-F1, and
p = (1 + #{null <= observed}) / (1 + N) for delta geometry cost (a larger reduction).

Null geometry construction (``--constructions``):
- ``matched`` (primary): the random order's matrices are built exactly as
  ``schwartz_geometry_matrices`` builds the theory order's (neighbor weight 1 at one step,
  0.5 at two steps; opposite mask distance > 0.75), so only the order differs.
- ``repo``: the construction of the submitted random control (``random_geometry_matrices``
  before the revision fixed it), whose neighbor term comes from
  ``_neighbor_similarity_from_distances`` and is weaker (0.5 at one step, ~0 at two steps).
  Rebuilt here explicitly so it stays reproducible. Reported for comparison only.

Plus a sample-level paired bootstrap (decoded - thresholded) for the ensemble, with the
repository's ``paired_bootstrap_delta``.

Outputs (``--out-dir``, default results/decoder_external/): observed_by_system.csv,
null_by_order.csv (every order x system, so p-values can be recomputed), null_orders.csv,
null_summary.csv, ensemble_bootstrap.csv, run_info.json. IDs and counts only; no sentence text.
"""

from __future__ import annotations

import os

# One BLAS thread per process: the decoder does many tiny matrix products, and the null runs
# in a process pool (set before numpy is imported).
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from compute_decoder_edit_rate import DECODE_KW, SETTING  # noqa: E402
from geometry_aware_calibration import (  # noqa: E402
    load_prediction_jsonl,
    load_validation_thresholds,
)
from schwartz_energy_decoder import (  # noqa: E402
    GeometryMatrices,
    LOWER_IS_BETTER,
    build_geometry_matrices,
    decode_predictions,
    _neighbor_similarity_from_distances,
)
from schwartz_value_geometry.eval.metrics import (  # noqa: E402
    binarize_probs,
    compute_confusion_distance_correlation,
    compute_global_metrics,
    compute_neighbor_error_rate,
    compute_opposite_error_rate,
)
from schwartz_value_geometry.eval.stats import paired_bootstrap_delta  # noqa: E402
from schwartz_value_geometry.geometry import (  # noqa: E402
    NEIGHBOR_STEP_DISTANCE,
    OPPOSITE_DISTANCE_THRESHOLD,
    circular_distance_matrix,
    circular_step_distance_matrix,
    distance_matrix_for_order,
    random_circular_order,
)

SEEDS = [42, 7, 1701, 11, 1984]
MODEL_SLUG = "deberta-v3-base"
PRED_DIR = ROOT / "results" / "predictions"
LOGS_DIR = ROOT / "results" / "logs"
ENSEMBLE_PREFIX = f"deberta_bce_ensemble5_seed0_{MODEL_SLUG}"
ENSEMBLE_PRED = ROOT / "results" / "ensemble" / "predictions" / f"{ENSEMBLE_PREFIX}_test.jsonl"
ENSEMBLE_THR = ROOT / "results" / "ensemble" / "logs" / f"{ENSEMBLE_PREFIX}_validation_thresholds.json"
SEED_SYSTEMS = [f"seed{seed}" for seed in SEEDS]
ORDER_SEED_BASE = 20261116
TIE = 1e-12

# Statistics averaged over seeds for the seed-mean null (counts are averaged too).
NULL_STATS = [
    "n_edited", "n_improved", "n_worsened", "n_unchanged", "n_added", "n_added_in_gold",
    "n_removed", "n_removed_not_in_gold", "share_added_in_gold", "share_removed_not_in_gold",
    "win_rate", "delta_macro_f1", "delta_micro_f1", "delta_geometry_cost",
    "delta_geometry_cost_value_positive", "delta_opposite_error_rate",
    "delta_opposite_error_rate_value_positive",
]
# (statistic, direction): "ge" counts null >= observed, "le" counts null <= observed.
TESTED = [
    ("delta_macro_f1", "ge"),
    ("delta_micro_f1", "ge"),
    ("delta_geometry_cost", "le"),
    ("delta_geometry_cost_value_positive", "le"),
    ("win_rate", "ge"),
    ("share_added_in_gold", "ge"),
]


# --------------------------------------------------------------------------- edit outcomes
def sample_f1(gold: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Per-sentence F1; 1.0 when gold and prediction are both empty (as ``_sample_f1``)."""
    gold = np.asarray(gold, dtype=int)
    pred = np.asarray(pred, dtype=int)
    tp = (gold & pred).sum(axis=1).astype(float)
    denom = (gold.sum(axis=1) + pred.sum(axis=1)).astype(float)
    return np.where(denom > 0, 2.0 * tp / np.maximum(denom, 1.0), 1.0)


def edit_outcomes(gold: np.ndarray, y_thr: np.ndarray, y_dec: np.ndarray) -> dict[str, float]:
    """Count what the decoder changed relative to thresholding, judged against gold."""
    gold = np.asarray(gold, dtype=int)
    y_thr = np.asarray(y_thr, dtype=int)
    y_dec = np.asarray(y_dec, dtype=int)
    added = (y_dec == 1) & (y_thr == 0)
    removed = (y_dec == 0) & (y_thr == 1)
    has_add = added.any(axis=1)
    has_rem = removed.any(axis=1)
    edited = has_add | has_rem
    delta = sample_f1(gold, y_dec) - sample_f1(gold, y_thr)
    improved = edited & (delta > TIE)
    worsened = edited & (delta < -TIE)
    n_added, n_removed = int(added.sum()), int(removed.sum())
    n_added_in_gold = int((added & (gold == 1)).sum())
    n_removed_not_in_gold = int((removed & (gold == 0)).sum())
    n_imp, n_wor = int(improved.sum()), int(worsened.sum())
    return {
        "n_sentences": int(gold.shape[0]),
        "n_edited": int(edited.sum()),
        "n_edited_addition_only": int((has_add & ~has_rem).sum()),
        "n_edited_removal_only": int((has_rem & ~has_add).sum()),
        "n_edited_mixed": int((has_add & has_rem).sum()),
        "n_edited_gold_empty": int((edited & (gold.sum(axis=1) == 0)).sum()),
        "n_improved": n_imp,
        "n_worsened": n_wor,
        "n_unchanged": int(edited.sum()) - n_imp - n_wor,
        "mean_delta_sample_f1_edited": float(delta[edited].mean()) if edited.any() else 0.0,
        "n_added": n_added,
        "n_added_in_gold": n_added_in_gold,
        "n_removed": n_removed,
        "n_removed_not_in_gold": n_removed_not_in_gold,
        "share_added_in_gold": n_added_in_gold / n_added if n_added else float("nan"),
        "share_removed_not_in_gold": n_removed_not_in_gold / n_removed if n_removed else float("nan"),
        "win_rate": n_imp / (n_imp + n_wor) if (n_imp + n_wor) else float("nan"),
    }


# --------------------------------------------------------------- vectorized paper metrics
class SchwartzMetrics:
    """Vectorized versions of the repository's geometry metrics (Schwartz order).

    Same definitions as ``compute_neighbor_error_rate``, ``compute_opposite_error_rate`` and
    ``compute_confusion_distance_correlation``; ``verify`` checks equality on real outputs.
    """

    def __init__(self, label_names: list[str]):
        self.label_names = list(label_names)
        self.dist = circular_distance_matrix(label_names)
        steps = circular_step_distance_matrix(label_names)
        self.near = (steps <= NEIGHBOR_STEP_DISTANCE).astype(np.int64)
        self.close = (self.dist <= OPPOSITE_DISTANCE_THRESHOLD).astype(np.int64)
        self.offdiag = ~np.eye(len(label_names), dtype=bool)

    def terms(self, gold: np.ndarray, pred: np.ndarray) -> dict[str, float]:
        gold = np.asarray(gold, dtype=np.int64)
        pred = np.asarray(pred, dtype=np.int64)
        fp = (pred == 1) & (gold == 0)
        n_fp = int(fp.sum())
        near = (gold @ self.near) > 0
        close = (gold @ self.close) > 0
        neighbor = float((fp & near).sum() / n_fp) if n_fp else 0.0
        opposite = float((fp & ~close).sum() / n_fp) if n_fp else 0.0
        confusion = gold.T.astype(float) @ fp.astype(float)
        dv, cv = self.dist[self.offdiag], confusion[self.offdiag]
        if np.all(cv == cv[0]):
            corr = 0.0
        else:
            corr = float(np.corrcoef(dv, cv)[0, 1])
            corr = 0.0 if np.isnan(corr) else corr
        return {
            "neighbor_error_rate": neighbor,
            "opposite_error_rate": opposite,
            "confusion_distance_correlation": corr,
            "geometry_cost": neighbor + opposite + corr,
        }

    def all_metrics(self, gold: np.ndarray, pred: np.ndarray, positive: np.ndarray) -> dict[str, float]:
        glob = compute_global_metrics(gold, pred)
        out = {"macro_f1": float(glob["macro_f1"]), "micro_f1": float(glob["micro_f1"])}
        out.update(self.terms(gold, pred))
        vp = self.terms(gold[positive], pred[positive])
        out["geometry_cost_value_positive"] = vp["geometry_cost"]
        out["opposite_error_rate_value_positive"] = vp["opposite_error_rate"]
        out["labels_per_sentence"] = float(np.asarray(pred).sum(axis=1).mean())
        return out

    def verify(self, gold: np.ndarray, pred: np.ndarray) -> None:
        mine = self.terms(gold, pred)
        ref = {
            "neighbor_error_rate": compute_neighbor_error_rate(gold, pred, label_names=self.label_names),
            "opposite_error_rate": compute_opposite_error_rate(gold, pred, label_names=self.label_names),
            "confusion_distance_correlation": compute_confusion_distance_correlation(
                gold, pred, label_names=self.label_names),
        }
        for key, value in ref.items():
            if abs(mine[key] - value) > 1e-12:
                raise AssertionError(f"vectorized {key} differs: {mine[key]} vs {value}")


# ----------------------------------------------------------------------- geometries
def matched_geometry_for_order(label_names: list[str], order, *, neighbor_steps: int) -> GeometryMatrices:
    """Matrices for ``order`` built exactly like ``schwartz_geometry_matrices`` builds the theory order's."""
    distances = distance_matrix_for_order(label_names, order)
    steps = np.rint(distances * len(order) / 2.0).astype(int)
    neighbor = np.zeros_like(distances, dtype=float)
    if neighbor_steps > 0:
        mask = (steps > 0) & (steps <= neighbor_steps)
        neighbor[mask] = (neighbor_steps + 1 - steps[mask]) / neighbor_steps
    opposite = (distances > OPPOSITE_DISTANCE_THRESHOLD).astype(float)
    np.fill_diagonal(neighbor, 0.0)
    np.fill_diagonal(opposite, 0.0)
    return GeometryMatrices(name="random_matched", neighbor_similarity=neighbor,
                            opposite_mask=opposite, distance_matrix=distances)


def null_geometry(construction: str, label_names: list[str], order_seed: int) -> GeometryMatrices:
    if construction == "matched":
        order = random_circular_order(label_names, seed=order_seed)
        return matched_geometry_for_order(label_names, order, neighbor_steps=DECODE_KW["neighbor_steps"])
    if construction == "repo":  # submitted (pre-fix) random control construction
        order = random_circular_order(label_names, seed=order_seed)
        distances = distance_matrix_for_order(label_names, order)
        neighbor = _neighbor_similarity_from_distances(distances, neighbor_steps=DECODE_KW["neighbor_steps"])
        opposite = (distances > OPPOSITE_DISTANCE_THRESHOLD).astype(float)
        np.fill_diagonal(opposite, 0.0)
        return GeometryMatrices(name="random", neighbor_similarity=neighbor,
                                opposite_mask=opposite, distance_matrix=distances)
    raise ValueError(construction)


# ------------------------------------------------------------------------- systems
def system_files(include_seeds: bool, include_ensemble: bool) -> dict[str, tuple[Path, Path]]:
    files: dict[str, tuple[Path, Path]] = {}
    if include_seeds:
        for seed in SEEDS:
            files[f"seed{seed}"] = (
                PRED_DIR / f"deberta_bce_seed{seed}_{MODEL_SLUG}_test.jsonl",
                LOGS_DIR / f"deberta_bce_seed{seed}_{MODEL_SLUG}_validation_thresholds.json",
            )
    if include_ensemble:
        if not ENSEMBLE_PRED.exists():
            raise FileNotFoundError(f"{ENSEMBLE_PRED} missing: run scripts/build_bce_ensemble.py first")
        files["ensemble5"] = (ENSEMBLE_PRED, ENSEMBLE_THR)
    return files


_STATE: dict = {}


def _init_worker(files: dict[str, tuple[Path, Path]]) -> None:
    logging.disable(logging.WARNING)
    systems = {}
    names = None
    for name, (pred_path, thr_path) in files.items():
        preds = load_prediction_jsonl(pred_path)
        if names is None:
            names = preds.label_names
        elif preds.label_names != names:
            raise ValueError(f"Label order differs for {name}")
        thresholds = load_validation_thresholds(thr_path, preds.label_names)
        gold = preds.y_true.astype(int)
        y_thr = binarize_probs(preds.y_probs, threshold=thresholds).astype(int)
        systems[name] = {"preds": preds, "thresholds": thresholds, "gold": gold, "y_thr": y_thr,
                         "positive": gold.sum(axis=1) > 0}
    metrics = SchwartzMetrics(names)
    for system in systems.values():
        system["thr_metrics"] = metrics.all_metrics(system["gold"], system["y_thr"], system["positive"])
    _STATE.update(systems=systems, names=names, metrics=metrics)


def evaluate_decode(system: dict, y_dec: np.ndarray, metrics: SchwartzMetrics) -> dict[str, float]:
    row = edit_outcomes(system["gold"], system["y_thr"], y_dec)
    dec = metrics.all_metrics(system["gold"], y_dec, system["positive"])
    thr = system["thr_metrics"]
    for key in ("macro_f1", "micro_f1", "geometry_cost", "geometry_cost_value_positive",
                "opposite_error_rate", "opposite_error_rate_value_positive", "labels_per_sentence"):
        row[f"thresholded_{key}"] = thr[key]
        row[f"decoded_{key}"] = dec[key]
        row[f"delta_{key}"] = dec[key] - thr[key]
    return row


def decode(system: dict, geometry: GeometryMatrices) -> np.ndarray:
    y_dec, _ = decode_predictions(system["preds"], thresholds=system["thresholds"], setting=SETTING,
                                  geometry_matrices=geometry, **DECODE_KW)
    return y_dec.astype(int)


def _null_task(task: tuple[str, int, int]) -> list[dict]:
    construction, order_idx, order_seed = task
    names, metrics = _STATE["names"], _STATE["metrics"]
    geometry = null_geometry(construction, names, order_seed)
    rows = []
    for name, system in _STATE["systems"].items():
        row = evaluate_decode(system, decode(system, geometry), metrics)
        rows.append({"construction": construction, "order_idx": order_idx, "order_seed": order_seed,
                     "system": name, **row})
    return rows


# ---------------------------------------------------------------------- aggregation
def add_seed_mean(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Append, per key group, a 'seed_mean' row averaging the five seeds (counts included)."""
    seeds = df[df["system"].isin(SEED_SYSTEMS)]
    if seeds.empty:
        return df
    numeric = [c for c in seeds.columns if c not in keys + ["system"] and pd.api.types.is_numeric_dtype(seeds[c])]
    mean = seeds.groupby(keys, as_index=False)[numeric].mean() if keys else seeds[numeric].mean().to_frame().T
    mean["system"] = "seed_mean"
    return pd.concat([df, mean], ignore_index=True)


def pooled_seed_row(observed: pd.DataFrame) -> dict:
    seeds = observed[observed["system"].isin(SEED_SYSTEMS)]
    tot = {k: int(seeds[k].sum()) for k in ("n_edited", "n_improved", "n_worsened", "n_unchanged",
                                           "n_added", "n_added_in_gold", "n_removed", "n_removed_not_in_gold")}
    tot["share_added_in_gold"] = tot["n_added_in_gold"] / tot["n_added"]
    tot["share_removed_not_in_gold"] = tot["n_removed_not_in_gold"] / tot["n_removed"]
    tot["win_rate"] = tot["n_improved"] / (tot["n_improved"] + tot["n_worsened"])
    return {"system": "seed_pooled", **tot}


def summarize_null(observed: pd.DataFrame, null: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (construction, system), group in null.groupby(["construction", "system"], sort=False):
        obs_row = observed[observed["system"] == system].iloc[0]
        n = len(group)
        for stat, direction in TESTED:
            values = group[stat].to_numpy(dtype=float)
            values = values[~np.isnan(values)]
            obs = float(obs_row[stat])
            count = int((values >= obs - TIE).sum()) if direction == "ge" else int((values <= obs + TIE).sum())
            rows.append({
                "construction": construction, "system": system, "statistic": stat,
                "direction": "null >= observed" if direction == "ge" else "null <= observed",
                "observed": obs, "null_mean": float(values.mean()), "null_sd": float(values.std(ddof=1)),
                "null_q025": float(np.quantile(values, 0.025)), "null_median": float(np.median(values)),
                "null_q975": float(np.quantile(values, 0.975)), "null_min": float(values.min()),
                "null_max": float(values.max()), "n_orders": n, "n_valid": int(values.size),
                "n_as_extreme": count, "p_one_sided": (1 + count) / (1 + values.size),
            })
        for stat in ("n_edited", "n_added", "n_removed", "share_removed_not_in_gold"):
            values = group[stat].to_numpy(dtype=float)
            rows.append({"construction": construction, "system": system, "statistic": stat,
                         "direction": "descriptive", "observed": float(obs_row[stat]),
                         "null_mean": float(np.nanmean(values)), "null_sd": float(np.nanstd(values, ddof=1)),
                         "null_q025": float(np.nanquantile(values, 0.025)),
                         "null_median": float(np.nanmedian(values)),
                         "null_q975": float(np.nanquantile(values, 0.975)), "null_min": float(np.nanmin(values)),
                         "null_max": float(np.nanmax(values)), "n_orders": n, "n_valid": n,
                         "n_as_extreme": np.nan, "p_one_sided": np.nan})
    return pd.DataFrame(rows)


def ensemble_bootstrap(system: dict, y_dec: np.ndarray, metrics: SchwartzMetrics,
                       n_iterations: int, seed: int) -> pd.DataFrame:
    gold, y_thr = system["gold"], system["y_thr"]
    fns = {
        "macro_f1": lambda t, p: float(compute_global_metrics(t, p)["macro_f1"]),
        "micro_f1": lambda t, p: float(compute_global_metrics(t, p)["micro_f1"]),
        "decoder_geometry_cost": lambda t, p: metrics.terms(t, p)["geometry_cost"],
        "opposite_error_rate": lambda t, p: metrics.terms(t, p)["opposite_error_rate"],
        "decoder_geometry_cost_value_positive":
            lambda t, p: metrics.terms(t[t.sum(1) > 0], p[t.sum(1) > 0])["geometry_cost"],
        "opposite_error_rate_value_positive":
            lambda t, p: metrics.terms(t[t.sum(1) > 0], p[t.sum(1) > 0])["opposite_error_rate"],
    }
    lower = LOWER_IS_BETTER | {"decoder_geometry_cost_value_positive", "opposite_error_rate_value_positive"}
    rows = []
    for metric, fn in fns.items():
        res = paired_bootstrap_delta(gold, y_dec, y_thr, metric_fn=fn, n_iterations=n_iterations, seed=seed)
        low = metric in lower
        rows.append({
            "system": "ensemble5", "metric": metric, "higher_is_better": not low,
            "delta_decoded_minus_standard": res.delta, "delta_ci_low": res.ci_low, "delta_ci_high": res.ci_high,
            "improvement_over_standard": -res.delta if low else res.delta,
            "p_value_two_sided": res.p_value, "n_samples": res.n_samples, "n_iterations": res.n_iterations,
            "bootstrap_seed": seed, "significant_0.05": res.p_value < 0.05,
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n-orders", type=int, default=1000)
    parser.add_argument("--order-seed-base", type=int, default=ORDER_SEED_BASE)
    parser.add_argument("--constructions", default="matched,repo",
                        help="comma-separated subset of {matched, repo}; 'matched' is the primary null")
    parser.add_argument("--systems", default="seeds,ensemble", help="comma-separated subset of {seeds, ensemble}")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "decoder_external")
    args = parser.parse_args()
    logging.disable(logging.WARNING)
    t0 = time.time()

    systems_arg = {s.strip() for s in args.systems.split(",") if s.strip()}
    files = system_files("seeds" in systems_arg, "ensemble" in systems_arg)
    constructions = [c.strip() for c in args.constructions.split(",") if c.strip()]
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Observed: Schwartz order, frozen weights, checked against the repository's metric code.
    _init_worker(files)
    names, metrics, systems = _STATE["names"], _STATE["metrics"], _STATE["systems"]
    schwartz = build_geometry_matrices(names, geometry="schwartz",
                                       neighbor_steps=DECODE_KW["neighbor_steps"], random_seed=42)
    from schwartz_value_geometry.geometry import SCHWARTZ_VALUE_ORDER
    check = matched_geometry_for_order(names, SCHWARTZ_VALUE_ORDER, neighbor_steps=DECODE_KW["neighbor_steps"])
    for attr in ("neighbor_similarity", "opposite_mask", "distance_matrix"):
        if not np.array_equal(getattr(check, attr), getattr(schwartz, attr)):
            raise AssertionError(f"matched construction does not reproduce the Schwartz {attr}")
    observed_rows, decoded = [], {}
    for name, system in systems.items():
        y_dec = decode(system, schwartz)
        decoded[name] = y_dec
        metrics.verify(system["gold"], system["y_thr"])
        metrics.verify(system["gold"], y_dec)
        observed_rows.append({"system": name, **evaluate_decode(system, y_dec, metrics)})
    observed = pd.DataFrame(observed_rows)
    observed = add_seed_mean(observed, keys=[])
    if "seed_mean" in set(observed["system"]):
        observed = pd.concat([observed, pd.DataFrame([pooled_seed_row(observed)])], ignore_index=True)
    observed.to_csv(out_dir / "observed_by_system.csv", index=False)
    t_obs = time.time() - t0
    print(f"observed decodes done in {t_obs:.1f}s")
    print(observed[["system", "n_edited", "n_improved", "n_worsened", "n_unchanged", "n_added",
                    "share_added_in_gold", "n_removed", "share_removed_not_in_gold", "delta_macro_f1",
                    "delta_micro_f1", "delta_geometry_cost", "delta_geometry_cost_value_positive"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # Ensemble bootstrap.
    if "ensemble5" in systems:
        boot = ensemble_bootstrap(systems["ensemble5"], decoded["ensemble5"], metrics,
                                  args.bootstrap_iterations, args.bootstrap_seed)
        boot.to_csv(out_dir / "ensemble_bootstrap.csv", index=False)
        print(boot[["metric", "delta_decoded_minus_standard", "delta_ci_low", "delta_ci_high",
                    "p_value_two_sided"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # Orders (written once, for reproducibility).
    order_seeds = [args.order_seed_base + i for i in range(args.n_orders)]
    pd.DataFrame({"order_idx": range(args.n_orders), "order_seed": order_seeds,
                  "order": [" | ".join(random_circular_order(names, seed=s)) for s in order_seeds]}
                 ).to_csv(out_dir / "null_orders.csv", index=False)

    # Null.
    tasks = [(c, i, s) for c in constructions for i, s in enumerate(order_seeds)]
    t1 = time.time()
    null_rows: list[dict] = []
    with Pool(args.workers, initializer=_init_worker, initargs=(files,)) as pool:
        for k, rows in enumerate(pool.imap_unordered(_null_task, tasks, chunksize=4), start=1):
            null_rows.extend(rows)
            if k % 100 == 0 or k == len(tasks):
                rate = (time.time() - t1) / k
                print(f"  null {k}/{len(tasks)} orders ({rate:.2f}s/order, ~{rate * (len(tasks) - k) / 60:.1f} min left)",
                      flush=True)
    null = pd.DataFrame(null_rows).sort_values(["construction", "order_idx", "system"], ignore_index=True)
    null = add_seed_mean(null, keys=["construction", "order_idx", "order_seed"])
    null.to_csv(out_dir / "null_by_order.csv", index=False)
    summary = summarize_null(observed, null)
    summary.to_csv(out_dir / "null_summary.csv", index=False)
    t_null = time.time() - t1

    info = {
        "script": "scripts/eval_decoder_edits_external.py",
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "systems": {k: [str(p.relative_to(ROOT)) for p in v] for k, v in files.items()},
        "setting": {"alpha_neighbor": SETTING.alpha_neighbor, "beta_opposite": SETTING.beta_opposite,
                    "gamma_cardinality": SETTING.gamma_cardinality},
        "decode_kw": DECODE_KW,
        "n_orders": args.n_orders, "order_seed_base": args.order_seed_base, "constructions": constructions,
        "p_value": "(1 + #{null as or more extreme}) / (1 + N), one-sided; seed_mean = per-order mean over the five seeds",
        "bootstrap": {"iterations": args.bootstrap_iterations, "seed": args.bootstrap_seed},
        "workers": args.workers, "seconds_observed": round(t_obs, 1), "seconds_null": round(t_null, 1),
        "seconds_total": round(time.time() - t0, 1),
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
    }
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")

    show = summary[summary["direction"] != "descriptive"]
    print(show[["construction", "system", "statistic", "observed", "null_mean", "null_q975", "n_as_extreme",
                "p_one_sided"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    shown = out_dir.resolve().relative_to(ROOT) if ROOT in out_dir.resolve().parents else out_dir
    print(f"done in {(time.time() - t0) / 60:.1f} min; outputs in {shown}/")  # no absolute paths in logs


if __name__ == "__main__":
    main()
