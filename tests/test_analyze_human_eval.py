import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from openpyxl import Workbook


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_human_eval.py"
    spec = importlib.util.spec_from_file_location("analyze_human_eval", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


he = _load()


def test_krippendorff_alpha_matches_published_example():
    # Krippendorff (2011), "Computing Krippendorff's alpha-reliability", 4 coders x 12 units.
    n = np.nan
    data = np.array(
        [
            [1, 2, 3, 3, 2, 1, 4, 1, 2, n, n, n],
            [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, n, 3],
            [n, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, n],
            [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, n],
        ]
    )
    assert he.krippendorff_alpha(data, level="nominal") == pytest.approx(0.743, abs=5e-4)
    assert he.krippendorff_alpha(data, level="ordinal") == pytest.approx(0.815, abs=5e-4)
    assert he.krippendorff_alpha(data, level="interval") == pytest.approx(0.849, abs=5e-4)


def test_majority_rule():
    assert he.majority(["decoded", "decoded", "thresholded"]) == "decoded"
    assert he.majority(["none", "none", "decoded"]) == "no preference"
    assert he.majority(["decoded", "thresholded", "none"]) == "no preference"
    assert he.majority(["thresholded", np.nan, "thresholded"]) == "thresholded"
    assert he.majority(["decoded", None, "thresholded"]) == "no preference"


def test_normalize_answer_and_sets():
    assert he.normalize_answer(" A fits better ") == "A"
    assert he.normalize_answer("b fits  better") == "B"
    assert he.normalize_answer("No clear difference") == "none"
    assert he.normalize_answer(None) is None
    assert he.normalize_answer("maybe").startswith("invalid:")
    assert he.label_set("(no value)") == frozenset()
    assert he.label_set("Face\nHumility") == he.label_set("Face; Humility")


def _toy(tmp_path):
    """Key + three workbooks: 1 practice, 2 attention and 4 edit items per annotator."""
    items = [
        ("practice", "addition", "P1", "T_1", "1", "Face", "", "Face"),
        ("attention", "", "1", "T_2", "1", "Tradition", "Hedonism", "Tradition"),
        ("attention", "", "2", "T_3", "1", "Stimulation", "Humility", "Stimulation"),
        ("edit", "addition", "3", "T_4", "1", "Face; Humility", "Face", "Face; Humility"),
        ("edit", "removal", "4", "T_5", "1", "Achievement", "Achievement; Hedonism", "Hedonism"),
        ("edit", "addition", "5", "T_6", "1", "Tradition; Conformity: rules", "Tradition", "Tradition"),
        ("edit", "removal", "6", "T_7", "1", "", "Stimulation", ""),
    ]
    # system (decoded or gold) on side A, except: all flipped for annotator 2, two edits for annotator 3
    names = {"edit": ("decoded", "thresholded"), "attention": ("gold", "random"), "practice": ("decoded", "thresholded")}
    # answers (as the favoured system) per annotator for rows 1..6
    favour = {
        1: ["gold", "gold", "decoded", "thresholded", "decoded", "decoded"],
        2: ["gold", "random", "decoded", "none", "thresholded", "decoded"],
        3: ["gold", "gold", "none", "thresholded", "decoded", "thresholded"],
    }
    key_rows, files = [], []
    for k in (1, 2, 3):
        wb = Workbook()
        wb.active.title = "Instructions"
        ws = wb.create_sheet("Items")
        ws.append(["#", "Sentence", "Option A", "Option B", "Your answer", "Comment (optional)"])
        for i, (kind, et, row, tid, sid, sys_set, other_set, gold) in enumerate(items):
            sys_name, other_name = names[kind]
            flip = k == 2 or (k == 3 and row in {"3", "5"})
            a_name, b_name = (other_name, sys_name) if flip else (sys_name, other_name)
            a_set, b_set = (other_set, sys_set) if flip else (sys_set, other_set)
            key_rows.append(
                {"annotator": k, "row": row, "kind": kind, "edit_type": et, "text_id": tid, "sent_id": sid,
                 "option_a": a_name, "option_b": b_name, "labels_a": a_set, "labels_b": b_set, "gold": gold}
            )
            if kind == "practice":
                answer = "A fits better"
            else:
                want = favour[k][i - 1]
                answer = "No clear difference" if want == "none" else "A fits better" if want == a_name else "B fits better"
            cell = lambda s: "\n".join(x.strip() for x in s.split(";")) if s else "(no value)"
            ws.append([row, "s", cell(a_set), cell(b_set), answer, "hm" if (k, row) == (3, "4") else None])
        path = tmp_path / f"returned_{4 - k}.xlsx"  # names do not match file numbers on purpose
        wb.save(path)
        files.append(path)
    key = tmp_path / "key.csv"
    with key.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(key_rows[0]))
        writer.writeheader()
        writer.writerows(key_rows)
    return key, files


def test_end_to_end_on_toy_workbooks(tmp_path):
    key_path, files = _toy(tmp_path)
    key = he.load_key(key_path)
    scored = []
    for f in files:
        k, df = he.score_file(f, key)
        assert f.name == f"returned_{4 - k}.xlsx"  # identified from content, not from the name
        scored.append(df)
        rep = he.file_report(k, df, f)
        assert rep["answered"] == 6 and rep["blank"] == 0 and rep["invalid"] == 0
        assert rep["attention_correct"] == (1 if k == 2 else 2)
    out = tmp_path / "out"
    comments = tmp_path / "private" / "comments.csv"
    summary = he.analyse(scored, out, comments)
    # majorities: T_4 (d, d, none) -> decoded; T_5 (t, none, t) -> thresholded;
    # T_6 (d, t, d) -> decoded; T_7 (d, d, t) -> decoded
    assert summary["majority_counts"] == {"decoded": 3, "thresholded": 1}
    assert summary["primary"]["prefer_decoded"] == 3 and summary["primary"]["decisive"] == 4
    assert summary["attention_accuracy"] == {1: 1.0, 2: 0.5, 3: 1.0}
    # gold outcome: T_4 improved, T_5 worsened, T_6 worsened, T_7 improved
    assert summary["majority_vs_gold_outcome"]["improved"]["decoded"] == 2
    written = json.loads((out / "summary.json").read_text())
    assert written["n_items"] == 4
    released = (out / "judgments.csv").read_text()
    assert "hm" not in released and ",s," not in released  # no comments, no sentence text
    assert comments.exists()
