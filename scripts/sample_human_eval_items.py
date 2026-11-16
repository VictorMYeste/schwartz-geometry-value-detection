"""Sample the human-evaluation items and write one annotation workbook per annotator.

Blind pairwise preference study (TACL revision, WP2). Read-only on saved predictions.

Items, all from the test split, drawn with a fixed seed:
- 150 sentences the Schwartz decoder edits on seed 42 (frozen schwartz / full /
  pareto_99 setting), stratified proportionally by edit type (addition only /
  removal only / mixed). The two options are the thresholded and the decoded set.
- 15 attention checks: sentences the decoder does not edit, with at least one gold
  value. The options are the gold set and a random set of the same size that shares
  no value with it.
- 10 practice items (shown first, not analysed): other decoder-edited sentences.

Item order and the left/right side are shuffled separately for every annotator.
Values inside an option are listed alphabetically, never in circular order.

Outputs in --out-dir. Keep that folder OUTSIDE the repository: the workbooks contain
dataset sentences, which the data usage agreement does not allow to redistribute.
- values_annotation_<k>.xlsx: sheets Instructions, Values, Items.
- key.csv: which option is which, per annotator and row (IDs and labels, no text).
  Never give it to the annotators.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.worksheet.datavalidation import DataValidation

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from compute_decoder_edit_rate import DECODE_KW, SETTING  # noqa: E402
from geometry_aware_calibration import (  # noqa: E402
    load_prediction_jsonl,
    load_validation_thresholds,
)
from schwartz_energy_decoder import build_geometry_matrices, decode_predictions  # noqa: E402
from schwartz_value_geometry.data.dataset import load_split  # noqa: E402
from schwartz_value_geometry.eval.metrics import binarize_probs  # noqa: E402
from schwartz_value_geometry.llm.prompts import VALUE_DEFINITIONS  # noqa: E402

SAMPLING_SEED = 11406
REFERENCE_SEED = 42
N_EDIT_ITEMS = 150
N_ATTENTION = 15
N_PRACTICE = 10
ANSWERS = ("A fits better", "B fits better", "No clear difference")
EMPTY = "(no value)"

# Plain-language examples shown next to each definition. The definitions themselves
# are the ones used in the LLM prompts (Schwartz 2012). Examples are ours.
VALUE_EXAMPLES = {
    "Achievement": "being ambitious, competent and successful; being recognised for what one achieves",
    "Benevolence: caring": "helping and looking after family, friends and one's own group",
    "Benevolence: dependability": "being loyal and responsible; someone the people close to you can count on",
    "Conformity: interpersonal": "being polite and tactful; not upsetting, annoying or hurting others",
    "Conformity: rules": "obeying laws and rules; fulfilling formal obligations",
    "Face": "protecting one's reputation and public image; avoiding being shamed",
    "Hedonism": "enjoying life, having fun, comfort and pleasure",
    "Humility": "being modest, not seeking attention, accepting one's limits",
    "Power: dominance": "having authority, being in charge, controlling what others do",
    "Power: resources": "wealth, property, economic strength and control over resources",
    "Security: personal": "personal and family safety and health; avoiding danger",
    "Security: societal": "public safety, national security, social order and stability",
    "Self-direction: action": "choosing one's own actions; independence and autonomy",
    "Self-direction: thought": "having one's own ideas and opinions; creativity; learning and understanding",
    "Stimulation": "excitement, adventure, new experiences, change",
    "Tradition": "customs, religion and heritage; keeping traditions alive",
    "Universalism: concern": "equal rights, justice and fairness for everyone; protecting the weak; peace",
    "Universalism: nature": "protecting the environment, the climate and animals",
    "Universalism: tolerance": "accepting and listening to people who are different or think differently",
}

INSTRUCTIONS = [
    ("title", "Which set of values fits the sentence better?"),
    ("text", "Thank you very much for helping! This takes about 1.5-2 hours. You can split it into two sittings."),
    ("head", "What this is"),
    ("text", "We are testing computer programs that read a sentence and tag which human values it refers to, "
             "for example security, tradition or caring for others. For each sentence you see two sets of "
             "values, A and B, produced in different ways. Your job is to say which set fits the sentence better."),
    ("head", "What to do"),
    ("text", "1. Read the 'Values' sheet first (about 5 minutes). Keep it open in another window, or print it: it fits on one page."),
    ("text", "2. Go to the 'Items' sheet. Start with the 10 grey practice rows (P1-P10), then do items 1-165."),
    ("text", "3. For each row, read the sentence, then Option A and Option B, and choose in the 'Your answer' column: "
             "'A fits better', 'B fits better' or 'No clear difference'. The comment column is optional."),
    ("head", "How to judge"),
    ("text", "- A value fits when the sentence refers to it or appeals to it, either in its favour or as something "
             "threatened or missing. 'Crime is rising in our cities' refers to Security: societal, even though it "
             "says security is lacking."),
    ("text", "- Judge the whole set. A set can be worse because it misses a value that is clearly there, or because "
             "it includes one that is not. There is no rule that shorter or longer sets are better."),
    ("text", f"- '{EMPTY}' means the set is empty. It is the right answer when the sentence refers to none of the values."),
    ("text", "- Judge each sentence on its own. The sentences come from longer texts, but only the sentence counts."),
    ("text", "- If both sets are equally good or equally bad, choose 'No clear difference'."),
    ("text", "- Go with your considered first judgment; most items take less than a minute. Please do not look "
             "things up, use AI tools, or discuss the items with anyone else doing this task."),
    ("head", "Two made-up examples"),
    ("example", ("The new law guarantees equal pay for women and men, and every company must now publish its salaries.",
                 "Universalism: concern\nConformity: rules", "Universalism: concern",
                 "A fits better: the sentence is about equality and also about a new obligation for companies.")),
    ("example", ("The meeting has been moved to Thursday at nine.", EMPTY, "Conformity: rules",
                 "A fits better: the sentence states a fact and refers to no value.")),
    ("head", "About the sentences"),
    ("text", "The sentences come from news articles and political texts in a research dataset; some touch on "
             "sensitive political topics. The dataset may only be used for research, so please do not copy or "
             "forward the sentences, and delete any copy of this file once you have sent it back."),
    ("head", "Your participation"),
    ("text", "Taking part is voluntary and unpaid, and you can stop at any time without giving a reason. Your "
             "answers are analysed anonymously; only anonymous judgments (no names) will be published. With your "
             "permission we would like to thank you by name in the paper's acknowledgments. Just tell us if you "
             "prefer not to be named."),
    ("due", None),
]


def label_text(labels: list[str]) -> str:
    return "\n".join(sorted(labels)) if labels else EMPTY


def proportional_counts(sizes: dict[str, int], total: int) -> dict[str, int]:
    """Largest-remainder allocation of `total` proportional to `sizes`."""
    n = sum(sizes.values())
    raw = {k: total * v / n for k, v in sizes.items()}
    out = {k: math.floor(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - out[k], reverse=True)[: total - sum(out.values())]:
        out[k] += 1
    return out


def build_items(rng: np.random.Generator) -> tuple[list[dict], list[dict]]:
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
    test = load_split("test")
    text_by_id = {(str(t), str(s)): str(x) for t, s, x in zip(test["text_id"], test["sent_id"], test["text"])}

    def labels(row: np.ndarray) -> list[str]:
        return [names[j] for j in np.flatnonzero(row)]

    def base(i: int) -> dict:
        key = (ids.at[i, "text_id"], ids.at[i, "sent_id"])
        return {"text_id": key[0], "sent_id": key[1], "text": text_by_id[key], "gold": labels(gold[i])}

    added = ((y_dec == 1) & (y_thr == 0)).any(1)
    removed = ((y_dec == 0) & (y_thr == 1)).any(1)
    edited = np.flatnonzero(added | removed)
    edit_type = np.where(added & removed, "mixed", np.where(added, "addition", "removal"))
    sizes = {t: int((edit_type[edited] == t).sum()) for t in ("addition", "removal", "mixed")}
    counts = proportional_counts(sizes, N_EDIT_ITEMS)
    print(f"edited sentences (seed {REFERENCE_SEED}): {len(edited)} {sizes} -> sampled {counts}")

    chosen = []
    for t, k in counts.items():
        pool = edited[edit_type[edited] == t]
        chosen += [int(i) for i in rng.choice(pool, size=k, replace=False)]
    rest = np.setdiff1d(edited, chosen)
    practice = [int(i) for i in rng.choice(rest, size=N_PRACTICE, replace=False)]

    def edit_item(i: int, kind: str) -> dict:
        return {**base(i), "kind": kind, "edit_type": str(edit_type[i]),
                "system": labels(y_dec[i]), "other": labels(y_thr[i]),
                "system_name": "decoded", "other_name": "thresholded"}

    main = [edit_item(i, "edit") for i in chosen]
    # Attention checks must test attention, not the noise in the gold labels (the pilot
    # showed obscure gold sets): use unedited sentences whose thresholded prediction equals
    # the gold set with every gold value at probability >= 0.5, and a random set drawn from
    # other broad values (no shared prefix, e.g. never Security: personal vs Security: societal).
    gold_prob = np.where(gold == 1, preds.y_probs, 1.0).min(1)
    clear = (gold.sum(1) > 0) & (y_thr == gold).all(1) & (gold_prob >= 0.5)
    candidates = np.setdiff1d(np.flatnonzero(clear), edited)
    print(f"attention-check candidates: {len(candidates)}")
    broad = {n: n.split(":")[0] for n in names}
    for i in rng.choice(candidates, size=N_ATTENTION, replace=False):
        g = labels(gold[i])
        pool = [n for n in names if broad[n] not in {broad[x] for x in g}]
        rand = sorted(rng.choice(pool, size=len(g), replace=False).tolist())
        main.append({**base(int(i)), "kind": "attention", "edit_type": "",
                     "system": g, "other": rand, "system_name": "gold", "other_name": "random"})
    return [edit_item(i, "practice") for i in practice], main


def style_instructions(ws, annotator: int, due: str) -> None:
    ws.column_dimensions["A"].width = 60
    ws.column_dimensions["B"].width = 26
    ws.column_dimensions["C"].width = 26
    ws.column_dimensions["D"].width = 50
    wrap = Alignment(wrap_text=True, vertical="top")
    row = 1
    example_header_done = False
    for kind, content in INSTRUCTIONS:
        if kind == "example":
            if not example_header_done:
                for col, h in enumerate(("Sentence", "Option A", "Option B", "Answer"), 1):
                    ws.cell(row, col, h).font = Font(bold=True)
                row += 1
                example_header_done = True
            for col, val in enumerate(content, 1):
                ws.cell(row, col, val).alignment = wrap
            ws.row_dimensions[row].height = 48
            row += 1
            continue
        if kind == "due":
            content = (f"Please send the file back by {due}. Questions at any time: just ask Víctor. "
                       f"(File {annotator})")
            kind = "head"
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=4)
        cell = ws.cell(row, 1, content)
        cell.alignment = wrap
        if kind == "title":
            cell.font = Font(bold=True, size=14)
            ws.row_dimensions[row].height = 24
        elif kind == "head":
            cell.font = Font(bold=True)
        else:
            ws.row_dimensions[row].height = 15 * max(1, math.ceil(len(content) / 150))
        row += 1


def style_values(ws) -> None:
    ws.append(["Value", "What it means", "For example"])
    for c in ws[1]:
        c.font = Font(bold=True)
    for name in sorted(VALUE_DEFINITIONS):
        d = VALUE_DEFINITIONS[name]
        ws.append([name, d[0].upper() + d[1:], VALUE_EXAMPLES[name]])
    for col, w in zip("ABC", (27, 48, 62)):
        ws.column_dimensions[col].width = w
    for r in ws.iter_rows(min_row=2):
        for c in r:
            c.alignment = Alignment(wrap_text=True, vertical="top")
        ws.row_dimensions[r[0].row].height = 30
    ws.page_setup.orientation = "landscape"
    ws.page_setup.fitToWidth = 1
    ws.page_setup.fitToHeight = 1
    ws.sheet_properties.pageSetUpPr.fitToPage = True


def style_items(ws, rows: list[tuple[str, str, str, str]], n_practice: int) -> None:
    headers = ["#", "Sentence", "Option A", "Option B", "Your answer", "Comment (optional)"]
    ws.append(headers)
    for c in ws[1]:
        c.font = Font(bold=True)
        c.fill = PatternFill("solid", fgColor="DDE7F0")
    for row in rows:
        ws.append(list(row) + ["", ""])
    last = len(rows) + 1
    widths = {"A": 6, "B": 70, "C": 28, "D": 28, "E": 20, "F": 30}
    for col, w in widths.items():
        ws.column_dimensions[col].width = w
    grey = PatternFill("solid", fgColor="EEEEEE")
    thin = Side(style="thin", color="BBBBBB")
    for r in ws.iter_rows(min_row=2, max_row=last):
        n_lines = max(math.ceil(len(r[1].value) / 80), r[2].value.count("\n") + 1, r[3].value.count("\n") + 1)
        ws.row_dimensions[r[0].row].height = 15 * n_lines + 6
        for c in r:
            c.alignment = Alignment(wrap_text=True, vertical="top")
            c.border = Border(bottom=thin)
            if r[0].row <= n_practice + 1:
                c.fill = grey
    dv = DataValidation(type="list", formula1='"' + ",".join(ANSWERS) + '"', allow_blank=True,
                        showErrorMessage=True, errorTitle="Please choose from the list",
                        error="Choose: " + " / ".join(ANSWERS))
    ws.add_data_validation(dv)
    dv.add(f"E2:E{last}")
    ws.freeze_panes = "B2"
    ws["H1"] = f'="Answered: "&COUNTA(E{n_practice + 2}:E{last})&" of {last - n_practice - 1}"'
    ws["H1"].font = Font(bold=True)
    ws.column_dimensions["H"].width = 22


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", type=Path, default=ROOT.parent / "human-eval",
                    help="output folder, outside the repository (default: ../human-eval)")
    ap.add_argument("--annotators", type=int, default=5,
                    help="number of workbooks: three annotators plus two reserves (files 4 and 5)")
    ap.add_argument("--due", default="Friday 11 December", help="return date shown in the instructions")
    args = ap.parse_args()
    if ROOT in args.out_dir.resolve().parents or args.out_dir.resolve() == ROOT:
        sys.exit("--out-dir must be outside the repository (the workbooks contain dataset text)")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    practice, main_items = build_items(np.random.default_rng(SAMPLING_SEED))
    key_rows = []
    for k in range(1, args.annotators + 1):
        rng = np.random.default_rng([SAMPLING_SEED, k])
        sheet_rows = []
        blocks = [("P", [practice[i] for i in rng.permutation(len(practice))]),
                  ("", [main_items[i] for i in rng.permutation(len(main_items))])]
        for prefix, items in blocks:
            for n, it in enumerate(items, 1):
                row_id = f"{prefix}{n}"
                system_on_a = bool(rng.random() < 0.5)
                a, b = (it["system"], it["other"]) if system_on_a else (it["other"], it["system"])
                sheet_rows.append((row_id, it["text"], label_text(a), label_text(b)))
                key_rows.append({
                    "annotator": k, "row": row_id, "kind": it["kind"], "edit_type": it["edit_type"],
                    "text_id": it["text_id"], "sent_id": it["sent_id"],
                    "option_a": it["system_name"] if system_on_a else it["other_name"],
                    "option_b": it["other_name"] if system_on_a else it["system_name"],
                    "labels_a": "; ".join(sorted(a)), "labels_b": "; ".join(sorted(b)),
                    "gold": "; ".join(sorted(it["gold"])),
                })
        wb = Workbook()
        ws_i = wb.active
        ws_i.title = "Instructions"
        style_instructions(ws_i, k, args.due)
        style_values(wb.create_sheet("Values"))
        style_items(wb.create_sheet("Items"), sheet_rows, len(practice))
        path = args.out_dir / f"values_annotation_{k}.xlsx"
        wb.save(path)
        print("wrote", path)

    key_path = args.out_dir / "key.csv"
    with key_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(key_rows[0]))
        writer.writeheader()
        writer.writerows(key_rows)
    print("wrote", key_path, f"({len(key_rows)} rows; never share with annotators)")


if __name__ == "__main__":
    main()
