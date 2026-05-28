"""Loss functions for multi-label value detection."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from schwartz_value_geometry.geometry import (
    circular_distance_matrix,
    distance_matrix_for_order,
    empirical_cooccurrence_distance_matrix,
    random_circular_order,
)

LOSS_ALIASES = {
    "asymmetric": "asl",
    "asymmetric_loss": "asl",
    "geo_loss": "geoloss",
    "geo_smooth": "geosmooth",
}


STRUCTURED_GEOLOSS_NAMES = {
    "geoloss",
    "schwartz_geoloss",
    "random_geoloss",
    "empirical_structure",
    "empirical_geoloss",
}

GEOSMOOTH_NAMES = {"geosmooth", "schwartz_geosmooth"}


def normalize_loss_name(name: object) -> str:
    """Normalize supported loss aliases to stable names."""
    raw_name = str(name).strip().lower() or "bce"
    return LOSS_ALIASES.get(raw_name, raw_name)


def _base_loss_name(loss_cfg: dict[str, Any], *, default: str = "bce") -> str:
    return normalize_loss_name(loss_cfg.get("base", loss_cfg.get("base_loss", default)))


class AsymmetricLoss(nn.Module):
    """Asymmetric loss for imbalanced multi-label classification.

    ASL is BCE with two additions: easy negative labels can be clipped, and
    positive/negative labels can receive different focal-style downweighting.
    """

    def __init__(
        self,
        *,
        gamma_pos: float = 0.0,
        gamma_neg: float = 4.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be one of {'mean', 'sum', 'none'}")
        self.gamma_pos = float(gamma_pos)
        self.gamma_neg = float(gamma_neg)
        self.clip = float(clip)
        self.eps = float(eps)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        probs_pos = torch.sigmoid(logits)
        probs_neg = 1.0 - probs_pos

        if self.clip > 0:
            probs_neg = torch.clamp(probs_neg + self.clip, max=1.0)

        log_pos = torch.log(torch.clamp(probs_pos, min=self.eps))
        log_neg = torch.log(torch.clamp(probs_neg, min=self.eps))
        loss = targets * log_pos + (1.0 - targets) * log_neg

        if self.gamma_pos > 0 or self.gamma_neg > 0:
            pt = probs_pos * targets + probs_neg * (1.0 - targets)
            gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
            loss = loss * torch.pow(1.0 - pt, gamma)

        loss = -loss
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


class GeometryAwareLoss(nn.Module):
    """Base multi-label loss plus a distance-weighted geometry penalty."""

    def __init__(
        self,
        base_loss: nn.Module,
        distance_matrix: np.ndarray,
        *,
        lambda_geo: float = 0.05,
        lambda_conflict: float = 0.0,
        distance_power: float = 1.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        distances = np.asarray(distance_matrix, dtype=float)
        if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
            raise ValueError("distance_matrix must be square")
        if distance_power != 1.0:
            distances = np.power(distances, float(distance_power))
        self.base_loss = base_loss
        self.lambda_geo = float(lambda_geo)
        self.lambda_conflict = float(lambda_conflict)
        self.eps = float(eps)
        self.register_buffer(
            "distance_matrix",
            torch.tensor(distances, dtype=torch.float32),
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        probs = torch.sigmoid(logits.float())
        distances = self.distance_matrix.to(device=logits.device, dtype=probs.dtype)

        base = self.base_loss(logits.float(), targets)
        weighted_distances = torch.einsum("bi,ij,bj->b", targets, distances, probs)
        gold_counts = targets.sum(dim=1).clamp_min(self.eps)
        geo_loss = (weighted_distances / gold_counts).mean()
        total = base + self.lambda_geo * geo_loss

        if self.lambda_conflict > 0:
            conflict = torch.einsum("bi,ij,bj->b", probs, distances, probs).mean()
            total = total + self.lambda_conflict * conflict

        return total


class GeometrySmoothedLoss(nn.Module):
    """Base loss trained against geometry-smoothed soft targets."""

    def __init__(
        self,
        base_loss: nn.Module,
        distance_matrix: np.ndarray,
        *,
        tau: float = 0.2,
    ) -> None:
        super().__init__()
        if tau <= 0:
            raise ValueError("tau must be positive")
        distances = np.asarray(distance_matrix, dtype=float)
        if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
            raise ValueError("distance_matrix must be square")
        kernel = np.exp(-(distances**2) / float(tau))
        self.base_loss = base_loss
        self.tau = float(tau)
        self.register_buffer("smoothing_kernel", torch.tensor(kernel, dtype=torch.float32))

    def smooth_targets(self, targets: torch.Tensor) -> torch.Tensor:
        """Apply Schwartz-style max-kernel smoothing to binary targets."""
        targets = targets.float()
        kernel = self.smoothing_kernel.to(device=targets.device, dtype=targets.dtype)
        expanded = targets.unsqueeze(2) * kernel.unsqueeze(0)
        smoothed = expanded.max(dim=1).values
        smoothed = torch.maximum(smoothed, targets)
        return torch.clamp(smoothed, 0.0, 1.0)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        smoothed_targets = self.smooth_targets(targets)
        return self.base_loss(logits.float(), smoothed_targets)


def _build_plain_loss(loss_cfg: dict[str, Any], *, name: str) -> nn.Module:
    reduction = str(loss_cfg.get("reduction", "mean"))

    if name == "bce":
        return nn.BCEWithLogitsLoss(reduction=reduction)

    if name == "asl":
        return AsymmetricLoss(
            gamma_pos=float(loss_cfg.get("gamma_pos", 0.0)),
            gamma_neg=float(loss_cfg.get("gamma_neg", 4.0)),
            clip=float(loss_cfg.get("clip", 0.05)),
            eps=float(loss_cfg.get("eps", 1e-8)),
            reduction=reduction,
        )

    raise ValueError(f"Unsupported loss.name={name!r}")


def _distance_matrix_from_config(
    loss_cfg: dict[str, Any],
    *,
    name: str,
    label_names: list[str] | None,
    train_labels: np.ndarray | None,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if label_names is None:
        raise ValueError(f"label_names are required for loss.name={name!r}")

    geometry = str(loss_cfg.get("geometry", "")).strip().lower()
    if name == "schwartz_geoloss" or name == "schwartz_geosmooth":
        geometry = "schwartz"
    elif name == "random_geoloss":
        geometry = "random"
    elif name in {"empirical_structure", "empirical_geoloss"}:
        geometry = "empirical"
    elif not geometry:
        geometry = "schwartz"

    if geometry == "schwartz":
        return circular_distance_matrix(label_names), {"geometry": "schwartz"}

    if geometry == "random":
        random_seed = int(loss_cfg.get("random_seed", seed))
        order = random_circular_order(label_names, seed=random_seed)
        distances = distance_matrix_for_order(label_names, order)
        return distances, {
            "geometry": "random_circular",
            "random_seed": random_seed,
            "random_order": list(order),
        }

    if geometry == "empirical":
        if train_labels is None:
            raise ValueError("train_labels are required for empirical structure loss")
        metric = str(loss_cfg.get("empirical_metric", "jaccard")).strip().lower()
        distances = empirical_cooccurrence_distance_matrix(
            np.asarray(train_labels),
            metric=metric,
        )
        return distances, {"geometry": "empirical_cooccurrence", "metric": metric}

    raise ValueError("geometry must be one of {'schwartz', 'random', 'empirical'}")


def build_loss_from_config(
    config: dict[str, Any],
    *,
    label_names: list[str] | None = None,
    train_labels: np.ndarray | None = None,
    seed: int | None = None,
) -> nn.Module:
    """Build a training loss from a project config."""
    loss_cfg = dict(config.get("loss", {}))
    name = normalize_loss_name(loss_cfg.get("name", "bce"))
    run_seed = int(config.get("seed", 42) if seed is None else seed)

    if name in {"bce", "asl"}:
        return _build_plain_loss(loss_cfg, name=name)

    if name in STRUCTURED_GEOLOSS_NAMES:
        base_name = _base_loss_name(loss_cfg, default="bce")
        base_loss = _build_plain_loss(loss_cfg, name=base_name)
        distances, _ = _distance_matrix_from_config(
            loss_cfg,
            name=name,
            label_names=label_names,
            train_labels=train_labels,
            seed=run_seed,
        )
        return GeometryAwareLoss(
            base_loss,
            distances,
            lambda_geo=float(loss_cfg.get("lambda_geo", 0.05)),
            lambda_conflict=float(loss_cfg.get("lambda_conflict", 0.0)),
            distance_power=float(loss_cfg.get("distance_power", 1.0)),
            eps=float(loss_cfg.get("eps", 1e-8)),
        )

    if name in GEOSMOOTH_NAMES:
        base_name = _base_loss_name(loss_cfg, default="bce")
        base_loss = _build_plain_loss(loss_cfg, name=base_name)
        distances, _ = _distance_matrix_from_config(
            loss_cfg,
            name=name,
            label_names=label_names,
            train_labels=train_labels,
            seed=run_seed,
        )
        return GeometrySmoothedLoss(
            base_loss,
            distances,
            tau=float(loss_cfg.get("tau", 0.2)),
        )

    raise ValueError(f"Unsupported loss.name={name!r}")


def _base_loss_metadata(loss_cfg: dict[str, Any], *, name: str) -> dict[str, Any]:
    if name == "asl":
        return {
            "name": "asl",
            "gamma_pos": float(loss_cfg.get("gamma_pos", 0.0)),
            "gamma_neg": float(loss_cfg.get("gamma_neg", 4.0)),
            "clip": float(loss_cfg.get("clip", 0.05)),
            "eps": float(loss_cfg.get("eps", 1e-8)),
        }
    return {"name": name}


def loss_config_for_metadata(
    config: dict[str, Any],
    *,
    label_names: list[str] | None = None,
) -> dict[str, Any]:
    """Return a JSON-friendly normalized loss config."""
    loss_cfg = dict(config.get("loss", {}))
    name = normalize_loss_name(loss_cfg.get("name", "bce"))
    normalized: dict[str, Any] = {"name": name}
    if name == "asl":
        normalized.update(_base_loss_metadata(loss_cfg, name="asl"))
    elif name in STRUCTURED_GEOLOSS_NAMES or name in GEOSMOOTH_NAMES:
        base_name = _base_loss_name(loss_cfg, default="bce")
        normalized["base"] = _base_loss_metadata(loss_cfg, name=base_name)
        if name in STRUCTURED_GEOLOSS_NAMES:
            normalized["lambda_geo"] = float(loss_cfg.get("lambda_geo", 0.05))
            normalized["lambda_conflict"] = float(loss_cfg.get("lambda_conflict", 0.0))
            normalized["distance_power"] = float(loss_cfg.get("distance_power", 1.0))
        if name in GEOSMOOTH_NAMES:
            normalized["tau"] = float(loss_cfg.get("tau", 0.2))
        geometry = str(loss_cfg.get("geometry", "")).strip().lower()
        if name == "random_geoloss":
            random_seed = int(loss_cfg.get("random_seed", config.get("seed", 42)))
            normalized["geometry"] = "random_circular"
            normalized["random_seed"] = random_seed
            if label_names is not None:
                normalized["random_order"] = list(
                    random_circular_order(
                        label_names,
                        seed=random_seed,
                    )
                )
        elif name in {"empirical_structure", "empirical_geoloss"}:
            normalized["geometry"] = "empirical_cooccurrence"
            normalized["metric"] = str(
                loss_cfg.get("empirical_metric", "jaccard")
            ).strip().lower()
        else:
            normalized["geometry"] = geometry or "schwartz"
    if "reduction" in loss_cfg:
        normalized["reduction"] = str(loss_cfg["reduction"])
    return normalized
