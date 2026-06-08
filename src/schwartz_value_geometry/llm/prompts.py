"""Prompt templates for the LLM diagnostic."""

from __future__ import annotations

from schwartz_value_geometry.geometry import SCHWARTZ_VALUE_ORDER

VALUE_DEFINITIONS: dict[str, str] = {
    "Self-direction: thought": "freedom to cultivate one's own ideas and abilities",
    "Self-direction: action": "freedom to determine one's own actions",
    "Stimulation": "excitement, novelty, and change",
    "Hedonism": "pleasure and sensuous gratification",
    "Achievement": "success according to social standards",
    "Power: dominance": "power through exercising control over people",
    "Power: resources": "power through control of material and social resources",
    "Face": "maintaining one's public image and avoiding humiliation",
    "Security: personal": "safety in one's immediate environment",
    "Security: societal": "safety and stability in the wider society",
    "Tradition": "maintaining and preserving cultural, family, or religious traditions",
    "Conformity: rules": "compliance with rules, laws, and formal obligations",
    "Conformity: interpersonal": "avoidance of upsetting or harming other people",
    "Humility": "recognising one's insignificance in the larger scheme of things",
    "Benevolence: dependability": "being a reliable and trustworthy member of the in-group",
    "Benevolence: caring": "devotion to the welfare of in-group members",
    "Universalism: concern": "commitment to equality, justice, and protection for all people",
    "Universalism: nature": "preservation of the natural environment",
    "Universalism: tolerance": "acceptance and understanding of those who are different from oneself",
}

PROMPT_TYPES = {"definitions_only", "schwartz_continuum"}


def _format_definitions() -> str:
    lines = ["Allowed labels and definitions:"]
    for label in SCHWARTZ_VALUE_ORDER:
        lines.append(f"- {label}: {VALUE_DEFINITIONS[label]}.")
    return "\n".join(lines)


def _format_continuum() -> str:
    order = " -> ".join(SCHWARTZ_VALUE_ORDER)
    return (
        "Schwartz-continuum structure:\n"
        f"- The refined values are arranged around this motivational circle: {order}.\n"
        "- Nearby labels on the circle usually express compatible motivations.\n"
        "- Labels on the opposite side of the circle usually express motivational conflict.\n"
        "- Multi-label outputs are allowed, especially for nearby or conceptually compatible values.\n"
        "- Avoid predicting distant/opposing values together unless the sentence clearly expresses both."
    )


def build_prompt(*, text: str, prompt_type: str) -> str:
    """Build a deterministic classification prompt for one sentence."""
    prompt_type = prompt_type.strip().lower()
    if prompt_type not in PROMPT_TYPES:
        raise ValueError(f"Unknown prompt_type={prompt_type!r}")

    parts = [
        "You are a sentence-level classifier for human values.",
        "Task: identify which of the 19 refined Schwartz values are expressed in the target sentence.",
        "The labels collapse attained and constrained cases into value presence: predict a label if the sentence expresses that value in either form.",
        _format_definitions(),
    ]
    if prompt_type == "schwartz_continuum":
        parts.append(_format_continuum())
    parts.extend(
        [
            "Output rules:",
            '- Return exactly one JSON object with this schema: {"labels": ["Label name", "..."]}.',
            '- Use only exact labels from the allowed list. Do not invent labels.',
            '- If no listed value is expressed, return {"labels": []}.',
            "- Do not include explanations, markdown, comments, or text outside the JSON object.",
            "Target sentence:",
            text,
        ]
    )
    return "\n\n".join(parts)

