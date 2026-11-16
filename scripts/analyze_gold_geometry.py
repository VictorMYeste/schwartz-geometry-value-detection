"""Is the Schwartz circle visible in the gold annotations? Presence vs stance.

Sentence-level analysis of the Touche24-ValueEval gold labels (all splits pooled),
answering two questions against the same null of random circular orders:

(a) Presence -- do values that annotators mark together in a sentence lie close on the
    circle? Co-occurring value pairs (presence = attained or constrained, as in the
    task labels) by circular step 1..9: observed vs expected under independence, and
    Spearman(step, association) across the 171 value pairs (lift primary; phi and
    Jaccard secondary). Theory predicts a negative correlation.

(b) Stance -- when two values co-occur with a clear stance each, are they attained or
    constrained together (same stance) more often when close, and does one get
    attained while the other is constrained (opposite stance) more often when far
    apart? Label instances with an unclear stance (0.5 / 0.5 in the raw files) are
    excluded. Primary statistic: mean step of opposite-stance pairs minus mean step of
    same-stance pairs (theory: positive). Secondary: Spearman(step, opposite-stance
    share) across value pairs with at least 5 / 10 / 20 clear-stance co-occurrences.

Null: N random circular orders over the same 19 labels, each drawn with the repo's own
``random_circular_order(label_names, seed=seed + i)`` and turned into steps with
``distance_matrix_for_order``; the label column order is never changed, only the
positions on the circle. Empirical one-sided p = (1 + #{null at least as extreme as
observed, in the theory-predicted direction}) / (1 + N).

Read-only on the raw label files; reads no sentence text and writes only aggregate
counts and statistics.

Usage:
    python scripts/analyze_gold_geometry.py [--n-orders 10000] [--seed 0]
        [--out-dir results/gold_geometry]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from schwartz_value_geometry.data.dataset import RAW_DATA_DIR  # noqa: E402
from schwartz_value_geometry.geometry import (  # noqa: E402
    SCHWARTZ_VALUE_ORDER,
    circular_step_distance_matrix,
    distance_matrix_for_order,
    random_circular_order,
)

SPLITS = ("training", "validation", "test")
MAX_STEP = len(SCHWARTZ_VALUE_ORDER) // 2  # 9 for 19 values
NEAR_MAX_STEP = 2  # "neighbors": within 2 steps
FAR_MIN_STEP = 8  # "far side": 8 or 9 steps apart
STANCE_MIN_COUNTS = (5, 10, 20)
DEFAULT_OUT_DIR = ROOT / "results" / "gold_geometry"


# --------------------------------------------------------------------------- data


def label_names_from_columns(columns: list[str]) -> list[str]:
    """Return the value names in file column order (from the '<value> attained' columns)."""
    return [c[: -len(" attained")] for c in columns if c.endswith(" attained")]


def load_stance_matrices(raw_dir: Path = RAW_DATA_DIR) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return raw attained / constrained matrices (sentences x values) for all splits."""
    attained, constrained, names = [], [], None
    for split in SPLITS:
        df = pd.read_csv(raw_dir / split / "labels.tsv", sep="\t", dtype=str)
        split_names = label_names_from_columns(list(df.columns))
        if names is None:
            names = split_names
        elif split_names != names:
            raise ValueError(f"Label columns differ in split {split}")
        attained.append(df[[f"{n} attained" for n in names]].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy())
        constrained.append(df[[f"{n} constrained" for n in names]].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy())
    return np.vstack(attained), np.vstack(constrained), list(names)


def stance_indicators(attained: np.ndarray, constrained: np.ndarray) -> dict[str, np.ndarray]:
    """Split label instances into presence, clear attained, clear constrained and unclear.

    In the raw files a value is attained (1, 0), constrained (0, 1), unclear (0.5, 0.5)
    or absent (0, 0). Presence is attained or constrained, as in the task labels.
    """
    a = np.asarray(attained) > 0
    c = np.asarray(constrained) > 0
    return {
        "presence": a | c,
        "attained": a & ~c,
        "constrained": c & ~a,
        "unclear": a & c,
    }


def pair_cooccurrence(presence: np.ndarray) -> np.ndarray:
    """Symmetric value x value co-occurrence counts (diagonal = support)."""
    y = np.asarray(presence, dtype=float)
    return y.T @ y


def stance_pair_counts(attained: np.ndarray, constrained: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Same-stance and opposite-stance co-occurrence counts for every value pair.

    same[i, j]: sentences where i and j are both attained or both constrained.
    opp[i, j]:  sentences where one is attained and the other constrained.
    Both are symmetric; only clear-stance instances count.
    """
    a = np.asarray(attained, dtype=float)
    c = np.asarray(constrained, dtype=float)
    same = a.T @ a + c.T @ c
    opp = a.T @ c + c.T @ a
    return same, opp


# ------------------------------------------------------------------------- nulls


def steps_for_order(label_names: list[str], order: tuple[str, ...]) -> np.ndarray:
    """Circular step matrix for a custom order, in the label column order given."""
    distances = distance_matrix_for_order(label_names, order)
    return np.rint(distances * len(order) / 2.0).astype(int)


def random_order_steps(label_names: list[str], n_orders: int, seed: int) -> np.ndarray:
    """(n_orders x n_pairs) circular steps of the upper-triangle pairs under random orders."""
    iu = np.triu_indices(len(label_names), 1)
    out = np.empty((n_orders, len(iu[0])), dtype=np.int8)
    for i in range(n_orders):
        order = random_circular_order(label_names, seed=seed + i)
        out[i] = steps_for_order(label_names, order)[iu]
    return out


def rowwise_spearman(x_rows: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Spearman correlation of each row of x_rows with y (average ranks for ties)."""
    rx = rankdata(np.atleast_2d(x_rows), axis=1)
    ry = rankdata(y)
    rx = rx - rx.mean(axis=1, keepdims=True)
    ry = ry - ry.mean()
    denom = np.linalg.norm(rx, axis=1) * np.linalg.norm(ry)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (rx @ ry) / denom


def empirical_p(observed: float, null: np.ndarray, direction: str) -> float:
    """One-sided empirical p in the theory-predicted direction ('greater' or 'less')."""
    null = np.asarray(null, dtype=float)
    if direction == "greater":
        extreme = np.sum(null >= observed - 1e-12)
    elif direction == "less":
        extreme = np.sum(null <= observed + 1e-12)
    else:
        raise ValueError(direction)
    return float((1 + extreme) / (1 + len(null)))


# -------------------------------------------------------------------- statistics


def presence_statistics(co_u, exp_u, lift, phi, jac, steps):
    """Presence statistics for one or many step assignments (steps: n_pairs or N x n_pairs)."""
    s = np.atleast_2d(steps)
    total, exp_total = co_u.sum(), exp_u.sum()
    near, far = s <= NEAR_MAX_STEP, s >= FAR_MIN_STEP
    return {
        "presence_spearman_lift": rowwise_spearman(s, lift),
        "presence_spearman_phi": rowwise_spearman(s, phi),
        "presence_spearman_jaccard": rowwise_spearman(s, jac),
        "presence_mean_step_obs_minus_exp": (s @ co_u) / total - (s @ exp_u) / exp_total,
        "presence_near_share_obs_minus_exp": (near @ co_u) / total - (near @ exp_u) / exp_total,
        "presence_far_share_obs_minus_exp": (far @ co_u) / total - (far @ exp_u) / exp_total,
    }


def stance_statistics(same_u, opp_u, steps):
    """Stance statistics for one or many step assignments."""
    s = np.atleast_2d(steps)
    tot = same_u + opp_u
    near, far = s <= NEAR_MAX_STEP, s >= FAR_MIN_STEP
    stats = {
        "stance_mean_step_opp_minus_same": (s @ opp_u) / opp_u.sum() - (s @ same_u) / same_u.sum(),
        "stance_opp_share_far_minus_near": (far @ opp_u) / (far @ tot) - (near @ opp_u) / (near @ tot),
    }
    for min_n in STANCE_MIN_COUNTS:
        m = tot >= min_n
        stats[f"stance_spearman_opp_share_min{min_n}"] = rowwise_spearman(s[:, m], opp_u[m] / tot[m])
    return stats


STAT_META = {
    # name: (question, direction predicted by the theory, primary?, description)
    "presence_spearman_lift": ("presence", "less", True,
                               "Spearman(circular step, lift) over the 171 value pairs"),
    "presence_spearman_phi": ("presence", "less", False, "Spearman(circular step, phi)"),
    "presence_spearman_jaccard": ("presence", "less", False, "Spearman(circular step, Jaccard)"),
    "presence_mean_step_obs_minus_exp": ("presence", "less", False,
                                         "mean step of co-occurring pairs, observed minus expected under independence"),
    "presence_near_share_obs_minus_exp": ("presence", "greater", False,
                                          "share of co-occurring pairs within 2 steps, observed minus expected"),
    "presence_far_share_obs_minus_exp": ("presence", "less", False,
                                         "share of co-occurring pairs 8-9 steps apart, observed minus expected"),
    "stance_mean_step_opp_minus_same": ("stance", "greater", True,
                                        "mean step of opposite-stance pairs minus mean step of same-stance pairs"),
    "stance_opp_share_far_minus_near": ("stance", "greater", False,
                                        "opposite-stance share among pairs 8-9 steps apart minus among pairs within 2 steps"),
    **{
        f"stance_spearman_opp_share_min{n}": ("stance", "greater", False,
                                              f"Spearman(circular step, opposite-stance share) over value pairs "
                                              f"with >= {n} clear-stance co-occurrences")
        for n in STANCE_MIN_COUNTS
    },
}


def by_step_table(co_u, exp_u, same_u, opp_u, exp_same_u, exp_opp_u, st, null_steps):
    """Per-step observed values with 95% random-order bands (computed on the same null)."""
    rows = []
    total, exp_total = co_u.sum(), exp_u.sum()
    tot = same_u + opp_u
    for d in range(1, MAX_STEP + 1):
        m = st == d
        nm = null_steps == d
        oe = (co_u[m].sum() / total) / (exp_u[m].sum() / exp_total)
        null_oe = ((nm @ co_u) / total) / ((nm @ exp_u) / exp_total)
        opp_share = opp_u[m].sum() / tot[m].sum()
        null_opp = (nm @ opp_u) / (nm @ tot)
        rows.append({
            "step": d,
            "n_value_pairs": int(m.sum()),
            "presence_cooccurrences": int(co_u[m].sum()),
            "presence_expected": float(exp_u[m].sum()),
            "presence_share_observed": float(co_u[m].sum() / total),
            "presence_share_expected": float(exp_u[m].sum() / exp_total),
            "presence_obs_over_exp": float(oe),
            "presence_obs_over_exp_null_lo95": float(np.quantile(null_oe, 0.025)),
            "presence_obs_over_exp_null_hi95": float(np.quantile(null_oe, 0.975)),
            "same_stance_pairs": int(same_u[m].sum()),
            "opposite_stance_pairs": int(opp_u[m].sum()),
            "same_stance_share_of_type": float(same_u[m].sum() / same_u.sum()),
            "opposite_stance_share_of_type": float(opp_u[m].sum() / opp_u.sum()),
            "opposite_stance_share": float(opp_share),
            "opposite_stance_share_null_lo95": float(np.quantile(null_opp, 0.025)),
            "opposite_stance_share_null_hi95": float(np.quantile(null_opp, 0.975)),
            "same_stance_obs_over_exp": float(same_u[m].sum() / exp_same_u[m].sum()),
            "opposite_stance_obs_over_exp": float(opp_u[m].sum() / exp_opp_u[m].sum()),
            "mean_pairwise_opposite_share_min20": float(np.mean((opp_u / np.maximum(tot, 1))[m & (tot >= 20)]))
            if (m & (tot >= 20)).any() else float("nan"),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------ figure


def make_figure(by_step: pd.DataFrame, overall_opp_share: float, out_base: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.5,
        "axes.titlesize": 7.5,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
    })
    steps = by_step["step"].to_numpy()
    band = "#d9d9d9"
    line = "#1f4e79"
    fig, axes = plt.subplots(1, 2, figsize=(3.3, 1.9), constrained_layout=True)

    ax = axes[0]
    ax.fill_between(steps, by_step["presence_obs_over_exp_null_lo95"], by_step["presence_obs_over_exp_null_hi95"],
                    color=band, lw=0, label="random orders (95%)")
    ax.axhline(1.0, color="#7f7f7f", lw=0.6, ls=":")
    ax.plot(steps, by_step["presence_obs_over_exp"], "-o", color=line, ms=2.6, lw=1.0, label="Schwartz order")
    ax.set_title("(a) Presence: co-occurrence", loc="left")
    ax.set_ylabel("observed / expected")

    ax = axes[1]
    ax.fill_between(steps, by_step["opposite_stance_share_null_lo95"], by_step["opposite_stance_share_null_hi95"],
                    color=band, lw=0)
    ax.axhline(overall_opp_share, color="#7f7f7f", lw=0.6, ls=":")
    ax.plot(steps, by_step["opposite_stance_share"], "-o", color=line, ms=2.6, lw=1.0)
    ax.set_title("(b) Stance: trade-offs", loc="left")
    ax.set_ylabel("opposite-stance share")

    for ax in axes:
        ax.set_xticks(range(1, MAX_STEP + 1))
        ax.set_xlabel("circular step")
        ax.tick_params(width=0.6, length=2.5)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=2, frameon=False,
               fontsize=6.5, handlelength=1.6, columnspacing=1.5)
    fig.savefig(out_base.with_suffix(".pdf"))
    fig.savefig(out_base.with_suffix(".png"), dpi=300)
    plt.close(fig)


# -------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n-orders", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0,
                        help="random order i uses random_circular_order(..., seed=seed + i)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    t0 = time.time()

    attained, constrained, names = load_stance_matrices()
    ind = stance_indicators(attained, constrained)
    n_sent, k = ind["presence"].shape
    iu = np.triu_indices(k, 1)
    st = circular_step_distance_matrix(names)[iu]  # Schwartz order (geometry.py), file column order

    # (a) presence
    co = pair_cooccurrence(ind["presence"])
    support = np.diag(co).copy()
    co_u = co[iu]
    exp_u = (np.outer(support, support) / n_sent)[iu]
    lift = co_u / exp_u
    union = (support[:, None] + support[None, :] - co)[iu]
    jac = co_u / union
    p = support / n_sent
    phi = ((co / n_sent - np.outer(p, p)) / np.sqrt(np.outer(p * (1 - p), p * (1 - p))))[iu]

    # (b) stance
    same, opp = stance_pair_counts(ind["attained"], ind["constrained"])
    same_u, opp_u = same[iu], opp[iu]
    pa, pc = ind["attained"].mean(0), ind["constrained"].mean(0)
    exp_same_u = ((np.outer(pa, pa) + np.outer(pc, pc)) * n_sent)[iu]
    exp_opp_u = ((np.outer(pa, pc) + np.outer(pc, pa)) * n_sent)[iu]

    observed = {**presence_statistics(co_u, exp_u, lift, phi, jac, st), **stance_statistics(same_u, opp_u, st)}
    observed = {name: float(v[0]) for name, v in observed.items()}

    null_steps = random_order_steps(names, args.n_orders, args.seed)
    t_null = time.time()
    null = {**presence_statistics(co_u, exp_u, lift, phi, jac, null_steps),
            **stance_statistics(same_u, opp_u, null_steps)}

    rows = []
    for name, (question, direction, primary, desc) in STAT_META.items():
        nl = null[name]
        rows.append({
            "question": question,
            "statistic": name,
            "primary": primary,
            "observed": observed[name],
            "direction_predicted": "observed > null" if direction == "greater" else "observed < null",
            "p_one_sided": empirical_p(observed[name], nl, direction),
            "null_mean": float(np.mean(nl)),
            "null_sd": float(np.std(nl)),
            "null_q025": float(np.quantile(nl, 0.025)),
            "null_q975": float(np.quantile(nl, 0.975)),
            "n_value_pairs": int(np.sum((same_u + opp_u) >= int(min_n.group(1))))
            if (min_n := re.search(r"_min(\d+)$", name)) else len(co_u),
            "description": desc,
        })
    stats_df = pd.DataFrame(rows)
    by_step = by_step_table(co_u, exp_u, same_u, opp_u, exp_same_u, exp_opp_u, st, null_steps)

    pairs_df = pd.DataFrame({
        "value_i": [names[i] for i in iu[0]],
        "value_j": [names[j] for j in iu[1]],
        "step": st,
        "cooccurrences": co_u.astype(int),
        "expected": exp_u,
        "lift": lift,
        "phi": phi,
        "jaccard": jac,
        "same_stance": same_u.astype(int),
        "opposite_stance": opp_u.astype(int),
        "opposite_stance_share": np.divide(opp_u, same_u + opp_u, out=np.full_like(opp_u, np.nan),
                                           where=(same_u + opp_u) > 0),
    })

    cardinality = ind["presence"].sum(1)
    counts = {
        "sentences": int(n_sent),
        "splits": list(SPLITS),
        "sentences_with_value": int((cardinality > 0).sum()),
        "multi_label_sentences": int((cardinality >= 2).sum()),
        "label_instances_present": int(ind["presence"].sum()),
        "label_instances_attained": int(ind["attained"].sum()),
        "label_instances_constrained": int(ind["constrained"].sum()),
        "label_instances_unclear_excluded_from_stance": int(ind["unclear"].sum()),
        "cooccurring_pair_instances": int(co_u.sum()),
        "same_stance_pair_instances": int(same_u.sum()),
        "opposite_stance_pair_instances": int(opp_u.sum()),
        "overall_opposite_stance_share": float(opp_u.sum() / (same_u + opp_u).sum()),
        "presence_mean_step_observed": float((co_u * st).sum() / co_u.sum()),
        "presence_mean_step_expected": float((exp_u * st).sum() / exp_u.sum()),
        "presence_near_share_observed": float(co_u[st <= NEAR_MAX_STEP].sum() / co_u.sum()),
        "presence_near_share_expected": float(exp_u[st <= NEAR_MAX_STEP].sum() / exp_u.sum()),
        "presence_far_share_observed": float(co_u[st >= FAR_MIN_STEP].sum() / co_u.sum()),
        "presence_far_share_expected": float(exp_u[st >= FAR_MIN_STEP].sum() / exp_u.sum()),
        "pair_type_share_near": float(np.mean(st <= NEAR_MAX_STEP)),
        "pair_type_share_far": float(np.mean(st >= FAR_MIN_STEP)),
        "same_stance_mean_step": float((same_u * st).sum() / same_u.sum()),
        "opposite_stance_mean_step": float((opp_u * st).sum() / opp_u.sum()),
        "same_stance_near_share": float(same_u[st <= NEAR_MAX_STEP].sum() / same_u.sum()),
        "same_stance_far_share": float(same_u[st >= FAR_MIN_STEP].sum() / same_u.sum()),
        "opposite_stance_near_share": float(opp_u[st <= NEAR_MAX_STEP].sum() / opp_u.sum()),
        "opposite_stance_far_share": float(opp_u[st >= FAR_MIN_STEP].sum() / opp_u.sum()),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(args.out_dir / "gold_geometry_stats.csv", index=False)
    by_step.to_csv(args.out_dir / "gold_geometry_by_step.csv", index=False)
    pairs_df.to_csv(args.out_dir / "gold_geometry_pairs.csv", index=False)
    summary = {
        "description": "Sentence-level gold co-occurrence vs circular distance: presence and stance.",
        "circular_order": "SCHWARTZ_VALUE_ORDER in geometry.py (Schwartz 2012 chapter, Fig. 1)",
        "null": {
            "n_orders": args.n_orders,
            "seed": args.seed,
            "generator": "random_circular_order(label_names, seed=seed + i) + distance_matrix_for_order",
            "p_value": "(1 + #{null at least as extreme in the predicted direction}) / (1 + n_orders), one-sided",
        },
        "near_definition": f"step <= {NEAR_MAX_STEP}",
        "far_definition": f"step >= {FAR_MIN_STEP}",
        "presence_obs_over_exp_definition": "share of co-occurring pairs at the step divided by the share "
                                            "expected under independence of the value marginals",
        "by_step_null_bands": "pointwise 2.5-97.5% quantiles per step over the same random orders",
        "figure": "gold_geometry.pdf/png: (a) presence observed/expected by step; (b) opposite-stance share "
                  "by step (pooled over value pairs); grey = pointwise 95% random-order band",
        "counts": counts,
        "statistics": {r["statistic"]: {k_: v for k_, v in r.items() if k_ != "statistic"} for r in rows},
        "runtime_seconds": None,
    }
    make_figure(by_step, counts["overall_opposite_stance_share"], args.out_dir / "gold_geometry")
    summary["runtime_seconds"] = round(time.time() - t0, 2)
    summary["null_runtime_seconds"] = round(time.time() - t_null, 2)
    (args.out_dir / "gold_geometry_summary.json").write_text(json.dumps(summary, indent=2))

    # console report
    print(f"sentences {n_sent:,} | multi-label {counts['multi_label_sentences']:,} | "
          f"co-occurring pairs {counts['cooccurring_pair_instances']:,}")
    print(f"stance instances: attained {counts['label_instances_attained']:,}, constrained "
          f"{counts['label_instances_constrained']:,}, unclear (excluded) "
          f"{counts['label_instances_unclear_excluded_from_stance']:,}; pairs same "
          f"{counts['same_stance_pair_instances']:,}, opposite {counts['opposite_stance_pair_instances']:,}")
    print("presence O/E by step 1..9: " + " ".join(f"{v:.2f}" for v in by_step["presence_obs_over_exp"]))
    print("opposite-stance share by step 1..9: " + " ".join(f"{v:.3f}" for v in by_step["opposite_stance_share"]))
    print(f"\n{'statistic':42s} {'observed':>9s} {'null mean':>9s} {'null sd':>8s} {'p':>8s}  direction")
    for r in rows:
        flag = "*" if r["primary"] else " "
        print(f"{flag}{r['statistic']:41s} {r['observed']:+9.4f} {r['null_mean']:+9.4f} {r['null_sd']:8.4f} "
              f"{r['p_one_sided']:8.4f}  {r['direction_predicted']}")
    print(f"\n(* primary) {args.n_orders:,} random orders, seed {args.seed}; "
          f"runtime {summary['runtime_seconds']:.1f} s; outputs in {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
