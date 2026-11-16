"""Prompt templates for the LLM diagnostic.

Two prompt types (``definitions_only`` and ``schwartz_continuum``) and three
wordings (``prompt_variant``): the ``original`` wording used in the submitted
paper and two meaning-preserving paraphrases (``paraphrase1``,
``paraphrase2``) used to measure prompt brittleness. The paraphrases keep the
section structure, the content of every instruction, the 19 label names, the
value order and the JSON schema; only the wording changes. The ``original``
wording reproduces the submitted prompts byte for byte (see
``tests/test_llm_diagnostic.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

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
DEFAULT_PROMPT_VARIANT = "original"


@dataclass(frozen=True)
class PromptWording:
    """All wording of one prompt variant. ``{order}`` is filled with the value order."""

    role: str
    task: str
    presence: str
    definitions_header: str
    definitions: dict[str, str]
    continuum_header: str
    continuum_lines: tuple[str, ...]
    output_header: str
    output_rules: tuple[str, ...]
    target_header: str


ORIGINAL = PromptWording(
    role="You are a sentence-level classifier for human values.",
    task="Task: identify which of the 19 refined Schwartz values are expressed in the target sentence.",
    presence=(
        "The labels collapse attained and constrained cases into value presence: "
        "predict a label if the sentence expresses that value in either form."
    ),
    definitions_header="Allowed labels and definitions:",
    definitions=VALUE_DEFINITIONS,
    continuum_header="Schwartz-continuum structure:",
    continuum_lines=(
        "The refined values are arranged around this motivational circle: {order}.",
        "Nearby labels on the circle usually express compatible motivations.",
        "Labels on the opposite side of the circle usually express motivational conflict.",
        "Multi-label outputs are allowed, especially for nearby or conceptually compatible values.",
        "Avoid predicting distant/opposing values together unless the sentence clearly expresses both.",
    ),
    output_header="Output rules:",
    output_rules=(
        'Return exactly one JSON object with this schema: {"labels": ["Label name", "..."]}.',
        "Use only exact labels from the allowed list. Do not invent labels.",
        'If no listed value is expressed, return {"labels": []}.',
        "Do not include explanations, markdown, comments, or text outside the JSON object.",
    ),
    target_header="Target sentence:",
)

PARAPHRASE1 = PromptWording(
    role="Your job is to classify individual sentences by the human values they express.",
    task="Task: decide which of the 19 values of Schwartz's refined theory the target sentence expresses.",
    presence=(
        "Attained and constrained cases are not told apart; only whether a value is present counts: "
        "assign a label whenever the sentence expresses that value in either form."
    ),
    definitions_header="Labels you may use, with their definitions:",
    definitions={
        "Self-direction: thought": "the freedom to develop one's own ideas and abilities",
        "Self-direction: action": "the freedom to decide one's own actions",
        "Stimulation": "novelty, change, and excitement",
        "Hedonism": "pleasure and gratification of the senses",
        "Achievement": "success as judged by the standards of society",
        "Power: dominance": "power that comes from controlling other people",
        "Power: resources": "power that comes from controlling material and social resources",
        "Face": "protecting one's public image and avoiding being humiliated",
        "Security: personal": "safety in one's immediate surroundings",
        "Security: societal": "safety and stability in society at large",
        "Tradition": "keeping and preserving cultural, family, or religious traditions",
        "Conformity: rules": "complying with rules, laws, and formal obligations",
        "Conformity: interpersonal": "avoiding upsetting or harming others",
        "Humility": "acknowledging one's insignificance in the bigger picture",
        "Benevolence: dependability": "being a member of the in-group whom others can rely on and trust",
        "Benevolence: caring": "dedication to the well-being of in-group members",
        "Universalism: concern": "dedication to equality, justice, and protection for everyone",
        "Universalism: nature": "protecting the natural environment",
        "Universalism: tolerance": "accepting and understanding people who differ from oneself",
    },
    continuum_header="Structure of the Schwartz continuum:",
    continuum_lines=(
        "The refined values form a motivational circle in this order: {order}.",
        "Labels close to each other on the circle usually reflect compatible motivations.",
        "Labels on opposite sides of the circle usually reflect conflicting motivations.",
        "You may output several labels, particularly when they are close on the circle or conceptually compatible.",
        "Refrain from outputting distant or opposing values together unless the sentence clearly expresses both of them.",
    ),
    output_header="Output format:",
    output_rules=(
        'Answer with a single JSON object that follows this schema: {"labels": ["Label name", "..."]}.',
        "Only use labels exactly as they appear in the list above; do not make up new labels.",
        'If the sentence expresses none of the listed values, answer {"labels": []}.',
        "Write nothing outside the JSON object: no explanations, markdown, or comments.",
    ),
    target_header="Sentence to classify:",
)

PARAPHRASE2 = PromptWording(
    role="Act as a classifier that labels single sentences with human values.",
    task="Task: for the target sentence, determine which of the 19 refined Schwartz values it expresses.",
    presence=(
        "What matters here is value presence, so attained and constrained cases are merged: "
        "give a label if the sentence expresses the value in either of these forms."
    ),
    definitions_header="Allowed labels, each followed by its definition:",
    definitions={
        "Self-direction: thought": "being free to develop one's own ideas and skills",
        "Self-direction: action": "being free to determine what one does",
        "Stimulation": "excitement, new experiences, and change",
        "Hedonism": "pleasure and gratifying one's senses",
        "Achievement": "being successful by society's standards",
        "Power: dominance": "power in the form of control over people",
        "Power: resources": "power in the form of control over material and social resources",
        "Face": "keeping up one's public image and not being humiliated",
        "Security: personal": "being safe in one's immediate environment",
        "Security: societal": "safety and stability of society as a whole",
        "Tradition": "upholding and preserving cultural, family, or religious traditions",
        "Conformity: rules": "following rules, laws, and formal obligations",
        "Conformity: interpersonal": "not upsetting or harming other people",
        "Humility": "being aware of one's insignificance in the larger scheme of things",
        "Benevolence: dependability": "being an in-group member others can count on and trust",
        "Benevolence: caring": "devoting oneself to the welfare of the in-group's members",
        "Universalism: concern": "being committed to equality, justice, and protection for every person",
        "Universalism: nature": "preserving the natural environment",
        "Universalism: tolerance": "accepting and understanding those unlike oneself",
    },
    continuum_header="How the values are structured (Schwartz continuum):",
    continuum_lines=(
        "The refined values lie on a motivational circle, in this order: {order}.",
        "Values that are close together on the circle tend to express motivations that are compatible.",
        "Values on opposite sides of the circle tend to express motivations that conflict.",
        "More than one label may be returned, in particular for values that are close on the circle or conceptually compatible.",
        "Unless the sentence clearly expresses both, values that are far apart or opposed on the circle should not be predicted together.",
    ),
    output_header="How to answer:",
    output_rules=(
        'Reply with exactly one JSON object in this format: {"labels": ["Label name", "..."]}.',
        "Copy every label exactly from the allowed list; never create new labels.",
        'When the sentence expresses none of the listed values, reply {"labels": []}.',
        "Output only the JSON object, with no explanation, markdown, or comment before or after it.",
    ),
    target_header="The sentence:",
)

PROMPT_VARIANTS: dict[str, PromptWording] = {
    "original": ORIGINAL,
    "paraphrase1": PARAPHRASE1,
    "paraphrase2": PARAPHRASE2,
}


def normalize_prompt_variant(prompt_variant: str | None) -> str:
    variant = (prompt_variant or DEFAULT_PROMPT_VARIANT).strip().lower()
    if variant not in PROMPT_VARIANTS:
        raise ValueError(
            f"Unknown prompt_variant={prompt_variant!r}; expected one of {sorted(PROMPT_VARIANTS)}"
        )
    return variant


def _bullets(lines) -> list[str]:
    return [f"- {line}" for line in lines]


def _format_definitions(wording: PromptWording) -> str:
    lines = [wording.definitions_header]
    for label in SCHWARTZ_VALUE_ORDER:
        lines.append(f"- {label}: {wording.definitions[label]}.")
    return "\n".join(lines)


def _format_continuum(wording: PromptWording) -> str:
    order = " -> ".join(SCHWARTZ_VALUE_ORDER)
    lines = [line.format(order=order) for line in wording.continuum_lines]
    return "\n".join([wording.continuum_header, *_bullets(lines)])


def build_prompt(*, text: str, prompt_type: str, prompt_variant: str | None = None) -> str:
    """Build a deterministic classification prompt for one sentence."""
    prompt_type = prompt_type.strip().lower()
    if prompt_type not in PROMPT_TYPES:
        raise ValueError(f"Unknown prompt_type={prompt_type!r}")
    wording = PROMPT_VARIANTS[normalize_prompt_variant(prompt_variant)]

    parts = [wording.role, wording.task, wording.presence, _format_definitions(wording)]
    if prompt_type == "schwartz_continuum":
        parts.append(_format_continuum(wording))
    # As in the submitted prompt, the output header and each output rule are separate
    # paragraphs (blank line between them), unlike the definition and continuum lists.
    parts.extend(
        [wording.output_header, *_bullets(wording.output_rules), wording.target_header, text]
    )
    return "\n\n".join(parts)
