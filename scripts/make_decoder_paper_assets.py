"""Create paper-ready decoder tables, figures, and qualitative examples."""

from __future__ import annotations

import argparse
import csv
import html
import textwrap
from pathlib import Path
from typing import Any

import pandas as pd


SELECTED_EXAMPLES = [
    ("11", "TR_032", "6", "adds_two_nearby_true_values"),
    ("1701", "EL_298", "7", "adds_true_value_and_removes_opposite_false_positive"),
    ("11", "HE_194", "26", "removes_two_opposite_false_positives"),
    ("11", "EL_075", "2", "removes_opposite_false_positive"),
    ("42", "BG_039", "18", "removes_opposite_false_positive"),
    ("7", "BG_123", "23", "adds_nearby_true_value"),
    ("7", "DE_122", "7", "adds_nearby_true_value"),
    ("7", "HE_177", "22", "adds_nearby_true_value"),
    ("11", "IT_141", "29", "adds_nearby_true_value"),
    ("42", "TR_235", "18", "adds_nearby_true_value"),
]


def _require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _fmt_mean_std(mean: float, std: float, digits: int = 4) -> str:
    return f"{mean:.{digits}f} +/- {std:.{digits}f}"


def _p_label(value: float) -> str:
    if value == 0.0:
        return "<0.001"
    return f"{value:.3f}"


def _short_text(value: str, width: int = 170) -> str:
    clean = html.unescape(str(value)).replace("\n", " ").strip()
    return textwrap.shorten(clean, width=width, placeholder="...")


def build_decoder_main_table(
    *,
    selected_mean_std: pd.DataFrame,
    delta_vs_standard: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    standard = selected_mean_std[
        (selected_mean_std["geometry"] == "schwartz")
        & (selected_mean_std["objective"] == "standard")
    ].iloc[0]
    delta_lookup = {
        row["geometry"]: row
        for _, row in delta_vs_standard.iterrows()
        if row["objective"] == "pareto_99"
    }

    rows.append(
        {
            "condition": "BCE thresholding",
            "geometry": "none",
            "macro_f1": _fmt_mean_std(
                standard["test_macro_f1_mean"],
                standard["test_macro_f1_std"],
            ),
            "micro_f1": _fmt_mean_std(
                standard["test_micro_f1_mean"],
                standard["test_micro_f1_std"],
            ),
            "opposite_error_rate": _fmt_mean_std(
                standard["test_opposite_error_rate_mean"],
                standard["test_opposite_error_rate_std"],
            ),
            "decoder_geometry_cost": _fmt_mean_std(
                standard["test_decoder_geometry_cost_mean"],
                standard["test_decoder_geometry_cost_std"],
            ),
            "delta_macro_f1_vs_threshold": "0.0000",
            "delta_decoder_geometry_cost_vs_threshold": "0.0000",
        }
    )

    labels = {
        "empirical": "Empirical decoder",
        "random": "Random decoder",
        "schwartz": "Schwartz decoder",
    }
    for geometry in ["empirical", "random", "schwartz"]:
        row = selected_mean_std[
            (selected_mean_std["geometry"] == geometry)
            & (selected_mean_std["objective"] == "pareto_99")
        ].iloc[0]
        delta = delta_lookup[geometry]
        rows.append(
            {
                "condition": labels[geometry],
                "geometry": geometry,
                "macro_f1": _fmt_mean_std(
                    row["test_macro_f1_mean"],
                    row["test_macro_f1_std"],
                ),
                "micro_f1": _fmt_mean_std(
                    row["test_micro_f1_mean"],
                    row["test_micro_f1_std"],
                ),
                "opposite_error_rate": _fmt_mean_std(
                    row["test_opposite_error_rate_mean"],
                    row["test_opposite_error_rate_std"],
                ),
                "decoder_geometry_cost": _fmt_mean_std(
                    row["test_decoder_geometry_cost_mean"],
                    row["test_decoder_geometry_cost_std"],
                ),
                "delta_macro_f1_vs_threshold": (
                    f"{float(delta['delta_macro_f1_mean']):+.4f}"
                ),
                "delta_decoder_geometry_cost_vs_threshold": (
                    f"{float(delta['delta_decoder_geometry_cost_mean']):+.4f}"
                ),
            }
        )
    return pd.DataFrame(rows)


def build_control_bootstrap_table(bootstrap: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (comparison, metric), group in bootstrap.groupby(["comparison", "metric"]):
        mean_delta = float(group["delta_schwartz_minus_control"].mean())
        mean_improvement = float(group["schwartz_improvement_over_control"].mean())
        sig = int(group["significant_0.05"].sum())
        n = int(group["seed"].nunique())
        rows.append(
            {
                "comparison": comparison,
                "metric": metric,
                "mean_delta_schwartz_minus_control": f"{mean_delta:+.4f}",
                "mean_schwartz_improvement": f"{mean_improvement:+.4f}",
                "significant_seeds": f"{sig}/{n}",
                "max_p_value": _p_label(float(group["p_value_two_sided"].max())),
                "min_improvement_ci_low": f"{float(group['improvement_ci_low'].min()):+.4f}",
                "max_improvement_ci_high": f"{float(group['improvement_ci_high'].max()):+.4f}",
            }
        )
    order = {
        "decoder_geometry_cost": 0,
        "opposite_error_rate": 1,
        "macro_f1": 2,
        "micro_f1": 3,
    }
    out = pd.DataFrame(rows)
    out["_order"] = out["metric"].map(order)
    out = out.sort_values(["comparison", "_order"]).drop(columns=["_order"])
    return out


def build_selected_examples(error_examples: pd.DataFrame) -> pd.DataFrame:
    indexed = {
        (str(row["seed"]), str(row["text_id"]), str(row["sent_id"])): row
        for _, row in error_examples.iterrows()
    }
    rows: list[dict[str, Any]] = []
    for rank, (seed, text_id, sent_id, reason) in enumerate(SELECTED_EXAMPLES, start=1):
        key = (seed, text_id, sent_id)
        if key not in indexed:
            raise KeyError(f"Selected qualitative example not found: {key}")
        row = indexed[key]
        rows.append(
            {
                "rank": rank,
                "selection_reason": reason,
                "seed": seed,
                "text_id": text_id,
                "sent_id": sent_id,
                "excerpt": _short_text(row["text"], width=220),
                "gold_labels": row["gold_labels"],
                "standard_labels": row["standard_labels"],
                "decoded_labels": row["decoded_labels"],
                "added_labels": row["added_labels"],
                "removed_labels": row["removed_labels"],
                "removed_opposite_false_positives": row[
                    "removed_opposite_false_positives"
                ],
                "gained_true_positives": row["gained_true_positives"],
                "sample_f1_standard": float(row["sample_f1_standard"]),
                "sample_f1_decoded": float(row["sample_f1_decoded"]),
                "sample_f1_delta": float(row["sample_f1_delta"]),
                "top_probabilities": row["top_probabilities"],
            }
        )
    return pd.DataFrame(rows)


def write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    columns = [
        "rank",
        "selection_reason",
        "excerpt",
        "gold_labels",
        "standard_labels",
        "decoded_labels",
        "added_labels",
        "removed_labels",
        "sample_f1_delta",
    ]
    with path.open("w", encoding="utf-8") as f:
        f.write("# Selected Qualitative Decoder Examples\n\n")
        f.write(
            "Selected from cases where the Schwartz decoder changed the final "
            "label set and improved sample-level F1.\n\n"
        )
        f.write("| " + " | ".join(columns) + " |\n")
        f.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for _, row in df.iterrows():
            values = []
            for column in columns:
                value = str(row[column]).replace("\n", " ").replace("|", "\\|")
                values.append(value)
            f.write("| " + " | ".join(values) + " |\n")


def _svg_escape(value: str) -> str:
    return html.escape(str(value), quote=True)


def _write_grouped_svg_bar_chart(
    *,
    panels: list[dict[str, Any]],
    path: Path,
    width: int = 900,
    height: int = 360,
) -> None:
    """Write a compact SVG bar-chart figure without plotting dependencies."""
    margin_left = 70
    margin_right = 30
    margin_top = 42
    margin_bottom = 68
    panel_gap = 46
    panel_width = int((width - margin_left - margin_right - panel_gap) / len(panels))
    plot_height = height - margin_top - margin_bottom
    colors = ["#7a7f87", "#b07d2f", "#1f77b4"]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;font-size:13px;}'
        '.small{font-size:11px;} .title{font-size:15px;font-weight:bold;}</style>',
    ]
    for panel_idx, panel in enumerate(panels):
        x0 = margin_left + panel_idx * (panel_width + panel_gap)
        y0 = margin_top
        values = [float(v) for v in panel["values"]]
        lows = [float(v) for v in panel.get("ci_low", values)]
        highs = [float(v) for v in panel.get("ci_high", values)]
        labels = panel["labels"]
        ymin = min(0.0, min(lows))
        ymax = max(0.0, max(highs))
        if ymax == ymin:
            ymax = ymin + 1.0
        pad = (ymax - ymin) * 0.12
        ymin -= pad
        ymax += pad

        def y(value: float) -> float:
            return y0 + (ymax - value) / (ymax - ymin) * plot_height

        baseline = y(0.0)
        parts.append(
            f'<text class="title" x="{x0 + panel_width / 2:.1f}" y="22" '
            f'text-anchor="middle">{_svg_escape(panel["title"])}</text>'
        )
        parts.append(
            f'<line x1="{x0}" y1="{baseline:.1f}" x2="{x0 + panel_width}" '
            f'y2="{baseline:.1f}" stroke="#222" stroke-width="1"/>'
        )
        parts.append(
            f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + plot_height}" '
            f'stroke="#333" stroke-width="1"/>'
        )
        parts.append(
            f'<text class="small" x="{x0 - 8}" y="{y(ymax - pad):.1f}" '
            f'text-anchor="end">{ymax - pad:.3f}</text>'
        )
        parts.append(
            f'<text class="small" x="{x0 - 8}" y="{y(ymin + pad):.1f}" '
            f'text-anchor="end">{ymin + pad:.3f}</text>'
        )

        bar_gap = 16
        bar_width = (panel_width - bar_gap * (len(labels) + 1)) / len(labels)
        for idx, (label, value) in enumerate(zip(labels, values)):
            x = x0 + bar_gap + idx * (bar_width + bar_gap)
            y_value = y(value)
            rect_y = min(y_value, baseline)
            rect_h = abs(baseline - y_value)
            parts.append(
                f'<rect x="{x:.1f}" y="{rect_y:.1f}" width="{bar_width:.1f}" '
                f'height="{rect_h:.1f}" fill="{colors[idx % len(colors)]}"/>'
            )
            if "ci_low" in panel and "ci_high" in panel:
                cy_low = y(float(panel["ci_low"][idx]))
                cy_high = y(float(panel["ci_high"][idx]))
                cx = x + bar_width / 2
                parts.append(
                    f'<line x1="{cx:.1f}" y1="{cy_low:.1f}" x2="{cx:.1f}" '
                    f'y2="{cy_high:.1f}" stroke="#111" stroke-width="1.2"/>'
                )
                parts.append(
                    f'<line x1="{cx - 5:.1f}" y1="{cy_low:.1f}" '
                    f'x2="{cx + 5:.1f}" y2="{cy_low:.1f}" '
                    f'stroke="#111" stroke-width="1.2"/>'
                )
                parts.append(
                    f'<line x1="{cx - 5:.1f}" y1="{cy_high:.1f}" '
                    f'x2="{cx + 5:.1f}" y2="{cy_high:.1f}" '
                    f'stroke="#111" stroke-width="1.2"/>'
                )
            parts.append(
                f'<text class="small" x="{x + bar_width / 2:.1f}" '
                f'y="{height - 42}" text-anchor="middle" '
                f'transform="rotate(-20 {x + bar_width / 2:.1f} {height - 42})">'
                f'{_svg_escape(label)}</text>'
            )
            value_y = rect_y - 6 if value >= 0 else rect_y + rect_h + 14
            parts.append(
                f'<text class="small" x="{x + bar_width / 2:.1f}" '
                f'y="{value_y:.1f}" text-anchor="middle">{value:+.3f}</text>'
            )
        if panel_idx == 0 and panel.get("ylabel"):
            parts.append(
                f'<text x="18" y="{y0 + plot_height / 2:.1f}" '
                f'text-anchor="middle" transform="rotate(-90 18 {y0 + plot_height / 2:.1f})">'
                f'{_svg_escape(panel["ylabel"])}</text>'
            )
    parts.append("</svg>\n")
    path.write_text("\n".join(parts), encoding="utf-8")


def plot_decoder_deltas(delta_vs_standard: pd.DataFrame, figures_dir: Path) -> None:
    df = delta_vs_standard[delta_vs_standard["objective"] == "pareto_99"].copy()
    df["geometry"] = pd.Categorical(
        df["geometry"],
        categories=["empirical", "random", "schwartz"],
        ordered=True,
    )
    df = df.sort_values("geometry")
    labels = ["Empirical", "Random", "Schwartz"]
    _write_grouped_svg_bar_chart(
        panels=[
            {
                "title": "Macro-F1",
                "ylabel": "Delta vs BCE thresholding",
                "labels": labels,
                "values": df["delta_macro_f1_mean"].tolist(),
            },
            {
                "title": "Geometry-cost reduction",
                "labels": labels,
                "values": (-df["delta_decoder_geometry_cost_mean"]).tolist(),
            },
        ],
        path=figures_dir / "figure_decoder_delta_vs_threshold.svg",
    )


def plot_control_bootstrap(control_bootstrap: pd.DataFrame, figures_dir: Path) -> None:
    df = control_bootstrap[
        control_bootstrap["metric"].isin(
            ["decoder_geometry_cost", "opposite_error_rate"]
        )
    ].copy()
    df["metric_label"] = df["metric"].map(
        {
            "decoder_geometry_cost": "Geometry-cost reduction",
            "opposite_error_rate": "Opposite-error reduction",
        }
    )
    df["control_label"] = df["control_geometry"].map(
        {"random": "vs Random", "empirical": "vs Empirical"}
    )
    summary = (
        df.groupby(["metric_label", "control_label"], sort=False)
        .agg(
            mean_improvement=("schwartz_improvement_over_control", "mean"),
            ci_low=("improvement_ci_low", "min"),
            ci_high=("improvement_ci_high", "max"),
        )
        .reset_index()
    )
    panels = []
    for metric in summary["metric_label"].drop_duplicates():
        sub = summary[summary["metric_label"] == metric]
        panels.append(
            {
                "title": metric,
                "ylabel": "Schwartz improvement over control",
                "labels": sub["control_label"].tolist(),
                "values": sub["mean_improvement"].tolist(),
                "ci_low": sub["ci_low"].tolist(),
                "ci_high": sub["ci_high"].tolist(),
            }
        )
    _write_grouped_svg_bar_chart(
        panels=panels,
        path=figures_dir / "figure_schwartz_vs_controls.svg",
    )


def run(
    *,
    analysis_dir: Path,
    paper_tables_dir: Path,
    figures_dir: Path,
) -> dict[str, Path]:
    decoder_dir = analysis_dir / "schwartz_energy_decoder_bootstrap_examples_full"
    control_dir = analysis_dir / "schwartz_energy_decoder_control_bootstrap"
    paper_tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    selected_mean_std = pd.read_csv(
        _require_file(decoder_dir / "schwartz_energy_decoder_selected_mean_std.csv")
    )
    delta_vs_standard = pd.read_csv(
        _require_file(decoder_dir / "schwartz_energy_decoder_delta_vs_standard.csv")
    )
    control_bootstrap = pd.read_csv(
        _require_file(control_dir / "schwartz_energy_decoder_control_sample_bootstrap.csv")
    )
    control_delta = pd.read_csv(
        _require_file(control_dir / "schwartz_energy_decoder_control_delta_mean_std.csv")
    )
    error_examples = pd.read_csv(
        _require_file(decoder_dir / "schwartz_energy_decoder_error_examples.csv"),
        keep_default_na=False,
    )

    decoder_main = build_decoder_main_table(
        selected_mean_std=selected_mean_std,
        delta_vs_standard=delta_vs_standard,
    )
    control_compact = build_control_bootstrap_table(control_bootstrap)
    selected_examples = build_selected_examples(error_examples)

    paths = {
        "decoder_main": paper_tables_dir / "decoder_main_results.csv",
        "control_bootstrap": paper_tables_dir
        / "decoder_control_bootstrap_summary.csv",
        "control_delta": paper_tables_dir
        / "schwartz_energy_decoder_control_delta_mean_std.csv",
        "selected_examples_csv": paper_tables_dir
        / "qualitative_decoder_examples_selected.csv",
        "selected_examples_md": paper_tables_dir
        / "qualitative_decoder_examples_selected.md",
    }
    decoder_main.to_csv(paths["decoder_main"], index=False, quoting=csv.QUOTE_MINIMAL)
    control_compact.to_csv(
        paths["control_bootstrap"],
        index=False,
        quoting=csv.QUOTE_MINIMAL,
    )
    control_delta.to_csv(paths["control_delta"], index=False, quoting=csv.QUOTE_MINIMAL)
    selected_examples.to_csv(
        paths["selected_examples_csv"],
        index=False,
        quoting=csv.QUOTE_MINIMAL,
    )
    write_markdown_table(selected_examples, paths["selected_examples_md"])

    plot_decoder_deltas(delta_vs_standard, figures_dir)
    plot_control_bootstrap(control_bootstrap, figures_dir)
    paths["figure_decoder_delta_svg"] = figures_dir / "figure_decoder_delta_vs_threshold.svg"
    paths["figure_controls_svg"] = figures_dir / "figure_schwartz_vs_controls.svg"
    return paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build final paper assets for the Schwartz decoder analysis."
    )
    parser.add_argument("--analysis_dir", default="results/analysis")
    parser.add_argument("--paper_tables_dir", default="results/analysis/paper_tables")
    parser.add_argument("--figures_dir", default="results/analysis/paper_figures")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    paths = run(
        analysis_dir=Path(args.analysis_dir),
        paper_tables_dir=Path(args.paper_tables_dir),
        figures_dir=Path(args.figures_dir),
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
