import importlib.util
import json
import sys
from pathlib import Path

from schwartz_value_geometry.geometry import SCHWARTZ_VALUE_ORDER
from schwartz_value_geometry.llm.parsing import parse_labels
from schwartz_value_geometry.llm.prompts import build_prompt


def _load_eval_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "eval_llm_diagnostic.py"
    spec = importlib.util.spec_from_file_location("eval_llm_diagnostic", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_prompt_templates_use_strict_json_and_continuum():
    prompt = build_prompt(text="The city adopted new safety rules.", prompt_type="schwartz_continuum")
    assert '{"labels": ["Label name", "..."]}' in prompt
    assert "Schwartz-continuum structure" in prompt
    assert "Self-direction: thought" in prompt
    assert "Universalism: tolerance" in prompt


def test_parse_labels_valid_json():
    parsed = parse_labels(
        '{"labels": ["Achievement", "Power: resources"]}',
        list(SCHWARTZ_VALUE_ORDER),
    )
    assert parsed.labels == ["Achievement", "Power: resources"]
    assert parsed.parse_status == "valid_json"
    assert not parsed.invalid_output


def test_parse_labels_repairs_markdown_and_unknown_label():
    parsed = parse_labels(
        '```json\n{"labels": ["Achievement", "Unknown value"]}\n```',
        list(SCHWARTZ_VALUE_ORDER),
    )
    assert parsed.labels == ["Achievement"]
    assert parsed.invalid_output
    assert parsed.repaired_output
    assert parsed.unknown_labels == ["Unknown value"]


def test_eval_llm_diagnostic_writes_metrics(tmp_path, monkeypatch):
    module = _load_eval_module()
    monkeypatch.setattr(module, "get_label_names", lambda: list(SCHWARTZ_VALUE_ORDER))
    pred_dir = tmp_path / "results" / "llm_diagnostic" / "predictions"
    pred_dir.mkdir(parents=True)
    pred_path = pred_dir / "qwen_Qwen2.5-72B-Instruct_definitions_only_test.jsonl"
    rows = [
        {
            "model_name": "Qwen/Qwen2.5-72B-Instruct",
            "model_slug": "Qwen2.5-72B-Instruct",
            "prompt_type": "definitions_only",
            "split": "test",
            "text_id": "a",
            "sent_id": "1",
            "gold_labels": ["Achievement"],
            "pred_labels": ["Achievement"],
            "invalid_output": False,
            "repaired_output": False,
        },
        {
            "model_name": "Qwen/Qwen2.5-72B-Instruct",
            "model_slug": "Qwen2.5-72B-Instruct",
            "prompt_type": "definitions_only",
            "split": "test",
            "text_id": "a",
            "sent_id": "2",
            "gold_labels": ["Power: resources"],
            "pred_labels": [],
            "invalid_output": True,
            "repaired_output": False,
        },
    ]
    pred_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    metrics = module.evaluate_predictions(pred_path)
    assert metrics["n_samples"] == 2
    assert metrics["invalid_output_rate"] == 0.5
    assert metrics["macro_auprc"] is None
    assert (tmp_path / "results" / "llm_diagnostic" / "logs").exists()
