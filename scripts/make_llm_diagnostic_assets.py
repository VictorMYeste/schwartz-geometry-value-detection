"""Create paper-ready assets for the LLM diagnostic."""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


def _first_float(value: str) -> float:
    match = re.search(r"[-+]?\d+(?:\.\d+)?", str(value))
    if not match:
        raise ValueError(f"Could not parse numeric value from {value!r}")
    return float(match.group(0))


def _fmt(value: float | None, digits: int = 4) -> str:
    return "-" if value is None else f"{float(value):.{digits}f}"


def _load_llm_metrics(logs_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(logs_dir.glob("*_metrics.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        meta = data.get("meta", {})
        if not meta.get("prompt_type"):
            continue
        prompt_type = str(meta.get("prompt_type"))
        prompt_label = {
            "definitions_only": "Qwen definitions",
            "schwartz_continuum": "Qwen continuum",
        }.get(prompt_type, prompt_type)
        rows.append(
            {
                "method": prompt_label,
                "supervision": "none",
                "schwartz_theory": "prompt" if prompt_type == "schwartz_continuum" else "no",
                "macro_f1": float(data["macro_f1"]),
                "micro_f1": float(data["micro_f1"]),
                "opposite_error_rate": float(data["opposite_error_rate"]),
                "decoder_geometry_cost": float(data["decoder_geometry_cost"]),
                "invalid_output_rate": float(data.get("invalid_output_rate", 0.0)),
                "repaired_output_rate": float(data.get("repaired_output_rate", 0.0)),
                "empty_prediction_rate": float(data.get("empty_prediction_rate", 0.0)),
                "avg_predicted_labels": float(data.get("avg_predicted_labels", 0.0)),
                "n_samples": int(data.get("n_samples", 0)),
                "source": str(path),
            }
        )
    return rows


def _load_supervised_reference(decoder_main_path: Path) -> list[dict[str, Any]]:
    if not decoder_main_path.exists():
        return []
    table = pd.read_csv(decoder_main_path)
    rows: list[dict[str, Any]] = []
    labels = {
        "BCE thresholding": ("BCE thresholding", "fine-tuned", "no"),
        "Schwartz decoder": ("BCE + Schwartz decoder", "fine-tuned", "decoder"),
    }
    for _, row in table.iterrows():
        condition = str(row["condition"])
        if condition not in labels:
            continue
        method, supervision, theory = labels[condition]
        rows.append(
            {
                "method": method,
                "supervision": supervision,
                "schwartz_theory": theory,
                "macro_f1": _first_float(row["macro_f1"]),
                "micro_f1": _first_float(row["micro_f1"]),
                "opposite_error_rate": _first_float(row["opposite_error_rate"]),
                "decoder_geometry_cost": _first_float(row["decoder_geometry_cost"]),
                "invalid_output_rate": None,
                "repaired_output_rate": None,
                "empty_prediction_rate": None,
                "avg_predicted_labels": None,
                "n_samples": None,
                "source": str(decoder_main_path),
            }
        )
    return rows


def build_llm_table(*, logs_dir: Path, decoder_main_path: Path) -> pd.DataFrame:
    rows = _load_llm_metrics(logs_dir)
    rows.extend(_load_supervised_reference(decoder_main_path))
    order = {
        "Qwen definitions": 0,
        "Qwen continuum": 1,
        "BCE thresholding": 2,
        "BCE + Schwartz decoder": 3,
    }
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["_order"] = table["method"].map(order).fillna(99)
    table = table.sort_values(["_order", "method"]).drop(columns=["_order"])
    output = table.copy()
    for column in [
        "macro_f1",
        "micro_f1",
        "opposite_error_rate",
        "decoder_geometry_cost",
        "invalid_output_rate",
        "repaired_output_rate",
        "empty_prediction_rate",
        "avg_predicted_labels",
    ]:
        output[column] = output[column].map(lambda value: _fmt(value) if pd.notna(value) else "-")
    return output


def _svg_escape(value: str) -> str:
    return html.escape(str(value), quote=True)


def _write_svg_bars(table: pd.DataFrame, path: Path) -> None:
    if table.empty:
        return
    numeric = table.copy()
    for column in ["macro_f1", "decoder_geometry_cost", "invalid_output_rate"]:
        numeric[column] = pd.to_numeric(numeric[column].replace("-", "nan"), errors="coerce")

    labels = numeric["method"].tolist()
    panels = [
        ("Macro-F1", "macro_f1", True),
        ("Geometry cost", "decoder_geometry_cost", False),
        ("Invalid output", "invalid_output_rate", False),
    ]
    width = 980
    height = 360
    margin_left = 64
    margin_top = 42
    margin_bottom = 88
    gap = 34
    panel_width = int((width - margin_left - gap * 2 - 20) / 3)
    plot_height = height - margin_top - margin_bottom
    colors = ["#5b6f95", "#1f77b4", "#7a7f87", "#2b6f4e"]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;font-size:12px}.title{font-size:15px;font-weight:bold}</style>',
    ]
    for panel_idx, (title, column, higher) in enumerate(panels):
        x0 = margin_left + panel_idx * (panel_width + gap)
        values = numeric[column].fillna(0.0).tolist()
        ymax = max(values + [0.001])
        if column == "decoder_geometry_cost":
            ymax = max(ymax, 0.60)
        elif column == "macro_f1":
            ymax = max(ymax, 0.35)
        else:
            ymax = max(ymax, 0.10)

        def y(value: float) -> float:
            return margin_top + (ymax - value) / ymax * plot_height

        parts.append(f'<text class="title" x="{x0 + panel_width / 2:.1f}" y="22" text-anchor="middle">{_svg_escape(title)}</text>')
        parts.append(f'<line x1="{x0}" y1="{margin_top + plot_height:.1f}" x2="{x0 + panel_width}" y2="{margin_top + plot_height:.1f}" stroke="#222"/>')
        parts.append(f'<line x1="{x0}" y1="{margin_top}" x2="{x0}" y2="{margin_top + plot_height}" stroke="#222"/>')
        parts.append(f'<text x="{x0 - 7}" y="{margin_top + 4}" text-anchor="end">{ymax:.2f}</text>')
        bar_gap = 8
        bar_w = (panel_width - bar_gap * (len(labels) + 1)) / len(labels)
        for idx, (label, value) in enumerate(zip(labels, values, strict=False)):
            x = x0 + bar_gap + idx * (bar_w + bar_gap)
            yv = y(value)
            h = margin_top + plot_height - yv
            parts.append(f'<rect x="{x:.1f}" y="{yv:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{colors[idx % len(colors)]}"/>')
            parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{yv - 5:.1f}" text-anchor="middle">{value:.3f}</text>')
            parts.append(
                f'<text x="{x + bar_w / 2:.1f}" y="{height - 50}" text-anchor="middle" '
                f'transform="rotate(-25 {x + bar_w / 2:.1f} {height - 50})">{_svg_escape(label)}</text>'
            )
        arrow = "higher is better" if higher else "lower is better"
        parts.append(f'<text x="{x0 + panel_width / 2:.1f}" y="{height - 10}" text-anchor="middle">{arrow}</text>')
    parts.append("</svg>\n")
    path.write_text("\n".join(parts), encoding="utf-8")


def run(*, logs_dir: Path, paper_tables_dir: Path, figures_dir: Path) -> dict[str, Path]:
    paper_tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    table = build_llm_table(
        logs_dir=logs_dir,
        decoder_main_path=paper_tables_dir / "decoder_main_results.csv",
    )
    table_path = paper_tables_dir / "llm_diagnostic_results.csv"
    table.to_csv(table_path, index=False)
    figure_path = figures_dir / "figure_llm_diagnostic.svg"
    _write_svg_bars(table, figure_path)
    return {"table": table_path, "figure": figure_path}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build LLM diagnostic paper assets.")
    parser.add_argument("--logs_dir", default="results/llm_diagnostic/logs")
    parser.add_argument("--paper_tables_dir", default="results/analysis/paper_tables")
    parser.add_argument("--figures_dir", default="results/analysis/paper_figures")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    paths = run(
        logs_dir=Path(args.logs_dir),
        paper_tables_dir=Path(args.paper_tables_dir),
        figures_dir=Path(args.figures_dir),
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
