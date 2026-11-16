"""Analyse the blind pairwise human evaluation (TACL revision, WP2).

Written on 2026-11-16, before any answer was returned; the analysis below is fixed in
advance (see ``HUMAN_EVAL_PROTOCOL.md``). Inputs are the returned annotation workbooks
written by ``sample_human_eval_items.py`` and its private ``key.csv``. Nothing here reads
or writes sentence text: answers are matched to the key by row ID and label sets.

Two modes:
  --check  For each returned file: which file number it is, how many items are answered,
           invalid entries, and the attention-check score with the exclusion verdict.
           Shows no preference results, so it can be run as files come in.
  default  The full analysis on exactly three files (the analysed annotators).

Analysis (fixed in advance):
  * Exclusion: an annotator who chooses the gold side on fewer than 10 of the 15
    attention checks is replaced by a reserve (files 4, 5); the analysis refuses to run
    with an excluded file unless --allow-excluded is given.
  * Each answer is mapped to the system it favours: decoded, thresholded, or none
    ("No clear difference"). Practice items are ignored.
  * Per item (150 edited sentences), majority vote of the three annotators; three
    different answers = no preference. With a missing answer, two agreeing answers
    decide; otherwise no preference.
  * Primary: share of decisive items (majority decoded or thresholded) that favour the
    decoded set, with a 95% percentile bootstrap CI over items (10,000 resamples) and an
    exact two-sided sign test against 0.5.
  * Secondary, descriptive: the same by edit type (addition / removal / mixed); the
    pooled share over all individual judgments; per-annotator shares; agreement as
    Krippendorff's alpha (nominal over decoded / none / thresholded; ordinal as a
    sensitivity check); attention-check accuracy; and the majority preference against
    the gold-based outcome of the edit (sample F1 of the decoded vs the thresholded set
    against the gold labels: improved / worsened / unchanged).

Outputs (``--out-dir``, default ``results/human_eval``): ``judgments.csv`` (one row per
annotator x item: anonymous annotator slot, IDs, sides, answer; no text, no comments),
``items.csv`` (majority per item), ``summary.json`` and ``by_edit_type.csv``. Free-text
comments, if any, go to ``--comments-out`` (default ``../human-eval/comments.csv``, outside
the repository) because they may quote the sentences.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KEY = ROOT.parent / "human-eval" / "key.csv"
DEFAULT_COMMENTS = ROOT.parent / "human-eval" / "comments.csv"
EMPTY = "(no value)"
MIN_ATTENTION_CORRECT = 10
N_BOOT = 10_000
BOOT_SEED = 11406
CATEGORIES = ("thresholded", "none", "decoded")  # ordinal order for the sensitivity alpha


# ---------------------------------------------------------------------------- reading
def normalize_answer(value) -> str | None:
    """Map a cell to 'A', 'B', 'none', None (blank) or 'invalid:<text>'."""
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value)).strip().casefold()
    if not text:
        return None
    if text in {"a fits better", "a", "option a", "a better"}:
        return "A"
    if text in {"b fits better", "b", "option b", "b better"}:
        return "B"
    if text in {"no clear difference", "no difference", "none", "equal", "same"}:
        return "none"
    return f"invalid:{value}"


def label_set(text) -> frozenset[str]:
    """Label set from a workbook cell (newline-separated) or a key cell ('; '-separated)."""
    if text is None:
        return frozenset()
    text = str(text).strip()
    if not text or text == EMPTY or text.lower() == "nan":
        return frozenset()
    parts = re.split(r"\n|;", text)
    return frozenset(p.strip() for p in parts if p.strip())


def read_workbook(path: Path) -> pd.DataFrame:
    """Row ID, option label sets, answer and comment from the Items sheet (no sentence text)."""
    wb = load_workbook(path, read_only=True, data_only=True)
    if "Items" not in wb.sheetnames:
        raise ValueError(f"{path.name}: no 'Items' sheet")
    ws = wb["Items"]
    header = [c for c in next(ws.iter_rows(min_row=1, max_row=1, values_only=True))]
    expected = ["#", "Sentence", "Option A", "Option B", "Your answer", "Comment (optional)"]
    if [str(h).strip() if h is not None else None for h in header[:6]] != expected:
        raise ValueError(f"{path.name}: unexpected Items header {header[:6]}")
    rows = []
    for values in ws.iter_rows(min_row=2, values_only=True):
        row_id = values[0]
        if row_id is None or str(row_id).strip() == "":
            continue
        rows.append(
            {
                "row": str(row_id).strip(),
                "set_a": label_set(values[2]),
                "set_b": label_set(values[3]),
                "raw_answer": values[4],
                "answer": normalize_answer(values[4]),
                "comment": values[5] if len(values) > 5 else None,
            }
        )
    wb.close()
    return pd.DataFrame(rows)


def load_key(path: Path) -> pd.DataFrame:
    key = pd.read_csv(path, dtype=str, keep_default_na=False)
    key["annotator"] = key["annotator"].astype(int)
    key["set_a"] = key["labels_a"].map(label_set)
    key["set_b"] = key["labels_b"].map(label_set)
    return key


def identify_file(sheet: pd.DataFrame, key: pd.DataFrame, path: Path) -> int:
    """Which file number a returned workbook is, from its row IDs and label sets."""
    matches = []
    for k, block in key.groupby("annotator"):
        merged = block.merge(sheet, on="row", how="inner", suffixes=("_key", "_wb"))
        if len(merged) != len(block) or len(sheet) != len(block):
            continue
        same = (merged["set_a_key"] == merged["set_a_wb"]) & (merged["set_b_key"] == merged["set_b_wb"])
        if bool(same.all()):
            matches.append(int(k))
    if len(matches) != 1:
        raise ValueError(f"{path.name}: matches key files {matches}; expected exactly one")
    return matches[0]


def score_file(path: Path, key: pd.DataFrame) -> tuple[int, pd.DataFrame]:
    sheet = read_workbook(path)
    k = identify_file(sheet, key, path)
    block = key[key["annotator"] == k].drop(columns=["set_a", "set_b"])
    merged = block.merge(sheet[["row", "answer", "raw_answer", "comment"]], on="row", how="left")

    def favoured(row) -> str | None:
        if row["answer"] == "A":
            return row["option_a"]
        if row["answer"] == "B":
            return row["option_b"]
        if row["answer"] == "none":
            return "none"
        return None

    merged["choice"] = merged.apply(favoured, axis=1)
    merged["file"] = k
    return k, merged


def file_report(k: int, scored: pd.DataFrame, path: Path) -> dict:
    counted = scored[scored["kind"] != "practice"]
    att = scored[scored["kind"] == "attention"]
    invalid = counted["answer"].fillna("").str.startswith("invalid:")
    n_att_correct = int((att["choice"] == "gold").sum())
    return {
        "file": k,
        "path": path.name,
        "answered": int(counted["answer"].notna().sum() - invalid.sum()),
        "of": int(len(counted)),
        "blank": int(counted["answer"].isna().sum()),
        "invalid": int(invalid.sum()),
        "invalid_examples": counted.loc[invalid, "raw_answer"].astype(str).head(3).tolist(),
        "practice_answered": int(scored.loc[scored["kind"] == "practice", "answer"].notna().sum()),
        "attention_correct": n_att_correct,
        "attention_of": int(len(att)),
        "attention_answered": int(att["choice"].notna().sum()),
        "excluded": n_att_correct < MIN_ATTENTION_CORRECT
        and int(att["choice"].notna().sum()) == len(att),
        "attention_below_bar_so_far": n_att_correct + int(att["choice"].isna().sum())
        < MIN_ATTENTION_CORRECT,
    }


# ---------------------------------------------------------------------------- statistics
def majority(choices: list[str | None]) -> str:
    """Majority of the given answers; 'no preference' when no answer has two votes."""
    present = [c for c in choices if isinstance(c, str)]
    for c in set(present):
        if present.count(c) >= 2:
            return "no preference" if c == "none" else c
    return "no preference"


def decoded_share(outcomes: np.ndarray) -> tuple[float, int, int]:
    n_dec = int((outcomes == "decoded").sum())
    n_thr = int((outcomes == "thresholded").sum())
    n = n_dec + n_thr
    return (n_dec / n if n else float("nan")), n_dec, n


def bootstrap_ci(outcomes: np.ndarray, *, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    dec = (outcomes == "decoded").astype(float)
    thr = (outcomes == "thresholded").astype(float)
    idx = rng.integers(0, len(outcomes), size=(n_boot, len(outcomes)))
    d, t = dec[idx].sum(1), thr[idx].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = d / (d + t)
    share = share[np.isfinite(share)]
    return float(np.quantile(share, 0.025)), float(np.quantile(share, 0.975))


def krippendorff_alpha(data: np.ndarray, *, level: str = "nominal") -> float:
    """Krippendorff's alpha for a coders x units matrix of integer codes (NaN = missing)."""
    data = np.asarray(data, dtype=float)
    values = np.unique(data[np.isfinite(data)])
    index = {v: i for i, v in enumerate(values)}
    o = np.zeros((len(values), len(values)))
    for u in range(data.shape[1]):
        col = data[np.isfinite(data[:, u]), u]
        m = len(col)
        if m < 2:
            continue
        for i in range(m):
            for j in range(m):
                if i != j:
                    o[index[col[i]], index[col[j]]] += 1.0 / (m - 1)
    n_c = o.sum(1)
    n = n_c.sum()
    if level == "nominal":
        delta = 1.0 - np.eye(len(values))
    elif level == "ordinal":
        cum = np.cumsum(n_c)
        delta = np.zeros_like(o)
        for c in range(len(values)):
            for k in range(len(values)):
                lo, hi = min(c, k), max(c, k)
                s = cum[hi] - (cum[lo - 1] if lo > 0 else 0.0)
                delta[c, k] = (s - (n_c[c] + n_c[k]) / 2.0) ** 2
    elif level == "interval":
        delta = (values[:, None] - values[None, :]) ** 2
    else:
        raise ValueError(level)
    d_o = (o * delta).sum()
    d_e = (np.outer(n_c, n_c) * delta).sum() / (n - 1)
    return float(1.0 - d_o / d_e) if d_e > 0 else float("nan")


def sample_f1(pred: frozenset, gold: frozenset) -> float:
    if not pred and not gold:
        return 1.0
    return 2.0 * len(pred & gold) / (len(pred) + len(gold))


def gold_outcome(row) -> str:
    sets = {row["option_a"]: label_set(row["labels_a"]), row["option_b"]: label_set(row["labels_b"])}
    gold = label_set(row["gold"])
    diff = sample_f1(sets["decoded"], gold) - sample_f1(sets["thresholded"], gold)
    return "improved" if diff > 1e-12 else "worsened" if diff < -1e-12 else "unchanged"


# ---------------------------------------------------------------------------- analysis
def analyse(scored: list[pd.DataFrame], out_dir: Path, comments_out: Path | None) -> dict:
    slots = {int(s["file"].iloc[0]): i + 1 for i, s in enumerate(scored)}
    allj = pd.concat(scored, ignore_index=True)
    allj["annotator"] = allj["file"].map(slots)

    edits = allj[allj["kind"] == "edit"]
    # unstack (not pivot_table) keeps items that some annotator left blank
    wide = (
        edits.set_index(["text_id", "sent_id", "edit_type", "annotator"])["choice"]
        .unstack("annotator")
        .reindex(columns=sorted(slots.values()))
    )
    items = wide.reset_index()
    items["majority"] = [majority(list(r)) for r in wide.itertuples(index=False)]
    first = edits.drop_duplicates(["text_id", "sent_id"]).set_index(["text_id", "sent_id"])
    items["gold_outcome"] = [
        gold_outcome(first.loc[(t, s)]) for t, s in zip(items["text_id"], items["sent_id"])
    ]
    outcomes = items["majority"].to_numpy()

    share, n_dec, n_decisive = decoded_share(outcomes)
    ci = bootstrap_ci(outcomes) if n_decisive else (float("nan"), float("nan"))
    sign_p = binomtest(n_dec, n_decisive, 0.5).pvalue if n_decisive else float("nan")

    by_type = []
    for et, grp in items.groupby("edit_type"):
        s, d, n = decoded_share(grp["majority"].to_numpy())
        by_type.append(
            {"edit_type": et, "items": len(grp), "decisive": n, "prefer_decoded": d,
             "prefer_thresholded": n - d, "no_preference": len(grp) - n, "share_decoded": s}
        )
    by_type = pd.DataFrame(by_type)

    pooled_share, pooled_dec, pooled_n = decoded_share(edits["choice"].to_numpy())
    per_annotator = {}
    for a, grp in edits.groupby("annotator"):
        s, d, n = decoded_share(grp["choice"].to_numpy())
        per_annotator[int(a)] = {
            "share_decoded": s, "prefer_decoded": d, "decisive": n,
            "no_difference": int((grp["choice"] == "none").sum()),
            "blank_or_invalid": int(grp["choice"].isna().sum()),
        }

    codes = {c: i for i, c in enumerate(CATEGORIES)}
    mat = np.array(
        [[codes.get(v, np.nan) if isinstance(v, str) else np.nan for v in wide[a]] for a in wide.columns]
    )
    att = allj[allj["kind"] == "attention"]
    gold_x = pd.crosstab(items["gold_outcome"], items["majority"])

    summary = {
        "n_annotators": len(slots),
        "files_analysed": sorted(slots),
        "n_items": int(len(items)),
        "majority_counts": {k: int(v) for k, v in pd.Series(outcomes).value_counts().items()},
        "primary": {
            "share_decoded_among_decisive": share,
            "prefer_decoded": n_dec,
            "decisive": n_decisive,
            "bootstrap_ci95": ci,
            "sign_test_p_two_sided": sign_p,
        },
        "pooled_judgments": {"share_decoded": pooled_share, "prefer_decoded": pooled_dec, "decisive": pooled_n},
        "per_annotator": per_annotator,
        "krippendorff_alpha_nominal": krippendorff_alpha(mat, level="nominal"),
        "krippendorff_alpha_ordinal": krippendorff_alpha(mat, level="ordinal"),
        "attention_accuracy": {
            int(a): float((g["choice"] == "gold").mean()) for a, g in att.groupby("annotator")
        },
        "majority_vs_gold_outcome": {r: {c: int(v) for c, v in row.items()} for r, row in gold_x.iterrows()},
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    release = allj[allj["kind"] != "practice"][
        ["annotator", "kind", "edit_type", "text_id", "sent_id", "option_a", "option_b", "answer", "choice"]
    ].rename(columns={"answer": "answer_side"})
    release.sort_values(["annotator", "kind", "text_id", "sent_id"]).to_csv(out_dir / "judgments.csv", index=False)
    items.rename(columns={a: f"annotator_{a}" for a in slots.values()}).to_csv(out_dir / "items.csv", index=False)
    by_type.to_csv(out_dir / "by_edit_type.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n", encoding="utf-8")

    comments = allj[allj["comment"].notna() & (allj["comment"].astype(str).str.strip() != "")]
    if comments_out is not None and len(comments):
        if ROOT in comments_out.resolve().parents:
            raise SystemExit("--comments-out must be outside the repository (comments may quote sentences)")
        comments_out.parent.mkdir(parents=True, exist_ok=True)
        comments[["annotator", "file", "row", "kind", "text_id", "sent_id", "choice", "comment"]].to_csv(
            comments_out, index=False
        )
    summary["comments_written"] = int(len(comments)) if comments_out is not None else 0
    return summary


def print_summary(s: dict) -> None:
    p = s["primary"]
    print(f"\nItems: {s['n_items']}  majority: {s['majority_counts']}")
    print(
        f"PRIMARY  decoded preferred in {p['prefer_decoded']}/{p['decisive']} decisive items "
        f"= {p['share_decoded_among_decisive']:.3f}  95% CI [{p['bootstrap_ci95'][0]:.3f}, "
        f"{p['bootstrap_ci95'][1]:.3f}]  sign test p = {p['sign_test_p_two_sided']:.4f}"
    )
    q = s["pooled_judgments"]
    print(f"Pooled judgments: {q['prefer_decoded']}/{q['decisive']} = {q['share_decoded']:.3f}")
    print(f"Krippendorff alpha: nominal {s['krippendorff_alpha_nominal']:.3f}, ordinal {s['krippendorff_alpha_ordinal']:.3f}")
    print(f"Attention accuracy: {s['attention_accuracy']}")
    print(f"Majority vs gold outcome: {s['majority_vs_gold_outcome']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="+", type=Path, help="returned workbooks (any file names)")
    ap.add_argument("--key", type=Path, default=DEFAULT_KEY)
    ap.add_argument("--check", action="store_true", help="completeness and attention checks only")
    ap.add_argument("--allow-excluded", action="store_true", help="analyse even if a file fails the attention bar")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results" / "human_eval")
    ap.add_argument("--comments-out", type=Path, default=DEFAULT_COMMENTS)
    args = ap.parse_args()

    key = load_key(args.key)
    scored, reports = [], []
    for path in args.files:
        k, df = score_file(path, key)
        scored.append(df)
        reports.append(file_report(k, df, path))
    for r in reports:
        verdict = (
            "REPLACE with a reserve (attention bar failed)" if r["excluded"]
            else "at risk: cannot reach the attention bar" if r["attention_below_bar_so_far"]
            else "complete" if r["answered"] == r["of"] else "incomplete"
        )
        print(
            f"File {r['file']} ({r['path']}): answered {r['answered']}/{r['of']}, blank {r['blank']}, "
            f"invalid {r['invalid']} {r['invalid_examples'] or ''}; practice {r['practice_answered']}/10; "
            f"attention {r['attention_correct']}/{r['attention_of']} (answered {r['attention_answered']}) -> {verdict}"
        )
    files = [r["file"] for r in reports]
    if len(set(files)) != len(files):
        sys.exit(f"The same file number appears twice: {files}")
    if args.check:
        return
    if len(files) != 3:
        sys.exit(f"The analysis uses exactly three annotators; got files {files}")
    if any(r["excluded"] for r in reports) and not args.allow_excluded:
        sys.exit("An analysed file fails the attention bar; replace it with a reserve (or --allow-excluded)")
    summary = analyse(scored, args.out_dir, args.comments_out)
    print_summary(summary)
    print(f"\nWrote {args.out_dir}/judgments.csv, items.csv, by_edit_type.csv, summary.json")
    if summary["comments_written"]:
        print(f"Wrote {summary['comments_written']} comments to {args.comments_out} (private; may quote sentences)")


if __name__ == "__main__":
    main()
