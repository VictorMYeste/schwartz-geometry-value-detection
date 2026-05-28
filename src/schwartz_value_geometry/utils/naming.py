"""Stable naming helpers for experiment artifacts."""

from __future__ import annotations

import re


def slugify(value: object, *, default: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-").lower()
    return slug or default


def model_slug(model_name: str) -> str:
    base = model_name.split("/")[-1] if model_name else "deberta"
    return slugify(base, default="deberta")


def loss_slug(config: dict) -> str:
    loss_name = str(config.get("loss", {}).get("name", "bce")).strip().lower()
    aliases = {
        "asymmetric": "asl",
        "asymmetric_loss": "asl",
        "geo_loss": "geoloss",
        "geo_smooth": "geosmooth",
    }
    return slugify(aliases.get(loss_name, loss_name), default="bce")


def artifact_prefix(config: dict, *, seed: int | None = None) -> str:
    model_name = config.get("model", {}).get("name", "microsoft/deberta-v3-base")
    run_seed = int(config.get("seed", 42) if seed is None else seed)
    return f"deberta_{loss_slug(config)}_seed{run_seed}_{model_slug(model_name)}"


def run_name(config: dict, *, seed: int | None = None, suffix: str = "best") -> str:
    prefix = artifact_prefix(config, seed=seed)
    return f"{prefix}_{suffix}" if suffix else prefix
