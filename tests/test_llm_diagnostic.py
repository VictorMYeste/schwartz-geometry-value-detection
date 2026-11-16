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


# SHA-256 of the prompts used in the submitted paper, for text="Example sentence.".
SUBMITTED_PROMPT_SHA256 = {
    "definitions_only": "b08ae4e2292b4b15851c963c975833a272d85aed7bac1d2752e4f5a5b0bca1fb",
    "schwartz_continuum": "62091b3d3bc08c3324227c87a0daabf2be69b74794369024c2b279998b533d32",
}


def test_original_wording_reproduces_submitted_prompts():
    import hashlib

    for prompt_type, digest in SUBMITTED_PROMPT_SHA256.items():
        for variant in (None, "original"):
            prompt = build_prompt(
                text="Example sentence.", prompt_type=prompt_type, prompt_variant=variant
            )
            assert hashlib.sha256(prompt.encode("utf-8")).hexdigest() == digest


def test_paraphrases_keep_labels_order_schema_and_structure():
    from schwartz_value_geometry.llm.prompts import PROMPT_VARIANTS

    order = " -> ".join(SCHWARTZ_VALUE_ORDER)
    for variant, wording in PROMPT_VARIANTS.items():
        assert set(wording.definitions) == set(SCHWARTZ_VALUE_ORDER), variant
        assert len(wording.continuum_lines) == 5 and len(wording.output_rules) == 4
        defs_only = build_prompt(text="X.", prompt_type="definitions_only", prompt_variant=variant)
        continuum = build_prompt(text="X.", prompt_type="schwartz_continuum", prompt_variant=variant)
        for prompt in (defs_only, continuum):
            assert '{"labels": ["Label name", "..."]}' in prompt
            assert '{"labels": []}' in prompt
            assert prompt.endswith("\n\nX.")
            for label in SCHWARTZ_VALUE_ORDER:
                assert f"- {label}: " in prompt
        assert order not in defs_only and order in continuum
        if variant != "original":
            # every sentence is reworded, none copied from the original
            original = PROMPT_VARIANTS["original"]
            assert wording.role != original.role and wording.task != original.task
            assert all(
                wording.definitions[k] != original.definitions[k] for k in SCHWARTZ_VALUE_ORDER
            )
            assert not set(wording.continuum_lines) & set(original.continuum_lines)
            assert not set(wording.output_rules) & set(original.output_rules)


def test_output_path_keeps_original_names_and_tags_variants():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_qwen_llm_diagnostic.py"
    spec = importlib.util.spec_from_file_location("run_qwen_llm_diagnostic", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    base = {"model": {"family": "qwen", "name": "Qwen/Qwen2.5-72B-Instruct"}}
    cfg = {**base, "llm": {"prompt_type": "schwartz_continuum"}}
    assert module.output_path_for(cfg, split="test", output_dir=Path("r")).name == (
        "qwen_Qwen2.5-72B-Instruct_schwartz_continuum_test.jsonl"
    )
    cfg = {**base, "llm": {"prompt_type": "definitions_only", "prompt_variant": "paraphrase2"}}
    assert module.output_path_for(cfg, split="test", output_dir=Path("r")).name == (
        "qwen_Qwen2.5-72B-Instruct_definitions_only_paraphrase2_test.jsonl"
    )
    llama = {
        "model": {"family": "llama", "name": "meta-llama/Llama-3.3-70B-Instruct"},
        "llm": {"prompt_type": "definitions_only", "prompt_variant": "original"},
    }
    assert module.output_path_for(llama, split="test", output_dir=Path("r")).name == (
        "llama_Llama-3.3-70B-Instruct_definitions_only_test.jsonl"
    )


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
