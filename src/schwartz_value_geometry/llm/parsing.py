"""Strict parsing and validation for LLM diagnostic outputs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ParsedLabels:
    labels: list[str]
    parse_status: str
    invalid_output: bool
    repaired_output: bool
    unknown_labels: list[str]
    duplicate_labels: list[str]


def _json_candidate(raw: str) -> tuple[str | None, bool]:
    text = raw.strip()
    if not text:
        return None, False
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    if text.startswith("{") or text.startswith("["):
        return text, False

    start_positions = [pos for pos in [text.find("{"), text.find("[")] if pos >= 0]
    if not start_positions:
        return None, False
    start = min(start_positions)
    end_obj = text.rfind("}")
    end_arr = text.rfind("]")
    end = max(end_obj, end_arr)
    if end <= start:
        return None, False
    return text[start : end + 1], True


def _labels_from_json(obj: Any) -> tuple[list[str], bool]:
    if isinstance(obj, dict) and isinstance(obj.get("labels"), list):
        return [str(item).strip() for item in obj["labels"]], True
    if isinstance(obj, list):
        return [str(item).strip() for item in obj], False
    return [], False


def _fallback_labels(raw: str) -> list[str]:
    normalized = raw.strip()
    if normalized.upper() in {"NONE", "NO VALUES", "NO VALUE", "[]"}:
        return []
    chunks = re.split(r"[,;\n]+", normalized)
    labels: list[str] = []
    for chunk in chunks:
        cleaned = chunk.strip().strip("\"'`*_- ")
        cleaned = re.sub(r"^[\-\d.)\s]+", "", cleaned)
        cleaned = re.sub(r"[.!?]+$", "", cleaned)
        if cleaned:
            labels.append(cleaned)
    return labels


def parse_labels(raw_output: str, label_names: list[str]) -> ParsedLabels:
    """Parse output, return valid labels, and expose format violations."""
    raw = raw_output or ""
    canonical = {label.lower(): label for label in label_names}
    candidates: list[str] = []
    parse_status = "invalid_empty" if not raw.strip() else "invalid"
    invalid_output = True
    repaired_output = False
    unknown_labels: list[str] = []
    duplicate_labels: list[str] = []

    json_text, extracted = _json_candidate(raw)
    if json_text is not None:
        try:
            obj = json.loads(json_text)
            candidates, strict_schema = _labels_from_json(obj)
            repaired_output = extracted or not strict_schema or json_text.strip() != raw.strip()
            parse_status = "valid_json" if not repaired_output else "repaired_json"
            invalid_output = repaired_output
        except Exception:
            candidates = _fallback_labels(raw)
            parse_status = "fallback_list" if candidates else "invalid_json"
            repaired_output = bool(candidates)
            invalid_output = True
    else:
        candidates = _fallback_labels(raw)
        parse_status = "fallback_list" if candidates or raw.strip().upper() == "NONE" else parse_status
        repaired_output = bool(raw.strip())
        invalid_output = True

    labels: list[str] = []
    seen_raw: set[str] = set()
    for candidate in candidates:
        cleaned = candidate.strip().strip("\"'` ")
        key = cleaned.lower()
        if key in seen_raw:
            duplicate_labels.append(cleaned)
            invalid_output = True
            continue
        seen_raw.add(key)
        if key in canonical:
            label = canonical[key]
            if label in labels:
                duplicate_labels.append(cleaned)
                invalid_output = True
            else:
                labels.append(label)
        elif cleaned:
            unknown_labels.append(cleaned)
            invalid_output = True

    if unknown_labels:
        repaired_output = True
    if parse_status == "valid_json" and not unknown_labels and not duplicate_labels:
        invalid_output = False
        repaired_output = False

    return ParsedLabels(
        labels=labels,
        parse_status=parse_status,
        invalid_output=invalid_output,
        repaired_output=repaired_output,
        unknown_labels=unknown_labels,
        duplicate_labels=duplicate_labels,
    )

