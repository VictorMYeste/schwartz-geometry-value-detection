import importlib

import numpy as np
import pytest
from schwartz_value_geometry.geometry import SCHWARTZ_VALUE_ORDER
from schwartz_value_geometry.utils.naming import artifact_prefix, loss_slug

torch = pytest.importorskip("torch")
losses = importlib.import_module("schwartz_value_geometry.models.losses")
AsymmetricLoss = losses.AsymmetricLoss
GeometryAwareLoss = losses.GeometryAwareLoss
GeometrySmoothedLoss = losses.GeometrySmoothedLoss
build_loss_from_config = losses.build_loss_from_config
loss_config_for_metadata = losses.loss_config_for_metadata
nn = torch.nn


def test_build_loss_defaults_to_bce():
    loss_fn = build_loss_from_config({})
    assert isinstance(loss_fn, nn.BCEWithLogitsLoss)


def test_asymmetric_loss_is_finite_and_differentiable():
    logits = torch.tensor([[2.0, -2.0], [-1.0, 1.0]], requires_grad=True)
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    loss_fn = AsymmetricLoss(gamma_pos=0.0, gamma_neg=4.0, clip=0.05)

    loss = loss_fn(logits, targets)

    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_build_loss_from_asl_config():
    config = {"loss": {"name": "asl", "gamma_pos": 0, "gamma_neg": 4, "clip": 0.05}}
    loss_fn = build_loss_from_config(config)
    meta = loss_config_for_metadata(config)

    assert isinstance(loss_fn, AsymmetricLoss)
    assert meta["name"] == "asl"
    assert meta["gamma_neg"] == 4.0


def test_build_loss_rejects_unknown_loss():
    with pytest.raises(ValueError):
        build_loss_from_config({"loss": {"name": "unknown"}})


def test_run_naming_uses_loss_slug():
    config = {
        "loss": {"name": "asymmetric_loss"},
        "model": {"name": "microsoft/deberta-v3-base"},
    }

    assert loss_slug(config) == "asl"
    assert artifact_prefix(config, seed=1984) == "deberta_asl_seed1984_deberta-v3-base"


def test_schwartz_geoloss_is_finite_and_differentiable():
    label_names = list(SCHWARTZ_VALUE_ORDER)
    config = {"loss": {"name": "schwartz_geoloss", "base": "bce", "lambda_geo": 0.1}}
    loss_fn = build_loss_from_config(config, label_names=label_names)
    logits = torch.zeros((2, len(label_names)), requires_grad=True)
    targets = torch.zeros_like(logits)
    targets[0, 0] = 1.0
    targets[1, 9] = 1.0

    loss = loss_fn(logits, targets)

    assert isinstance(loss_fn, GeometryAwareLoss)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None


def test_geosmooth_assigns_more_mass_to_neighbors_than_opposites():
    label_names = list(SCHWARTZ_VALUE_ORDER)
    config = {"loss": {"name": "schwartz_geosmooth", "base": "bce", "tau": 0.2}}
    loss_fn = build_loss_from_config(config, label_names=label_names)
    targets = torch.zeros((1, len(label_names)))
    targets[0, 0] = 1.0

    smoothed = loss_fn.smooth_targets(targets)

    assert isinstance(loss_fn, GeometrySmoothedLoss)
    assert smoothed[0, 0] == 1.0
    assert smoothed[0, 1] > smoothed[0, 9]


def test_empirical_structure_loss_uses_training_labels():
    label_names = list(SCHWARTZ_VALUE_ORDER)
    train_labels = np.zeros((4, len(label_names)), dtype=float)
    train_labels[0, [0, 1]] = 1.0
    train_labels[1, [0, 1]] = 1.0
    train_labels[2, 9] = 1.0
    train_labels[3, 10] = 1.0
    config = {
        "loss": {
            "name": "empirical_structure",
            "base": "bce",
            "lambda_geo": 0.05,
            "empirical_metric": "jaccard",
        }
    }

    loss_fn = build_loss_from_config(
        config,
        label_names=label_names,
        train_labels=train_labels,
    )
    meta = loss_config_for_metadata(config, label_names=label_names)

    assert isinstance(loss_fn, GeometryAwareLoss)
    assert meta["geometry"] == "empirical_cooccurrence"
    assert meta["metric"] == "jaccard"


def test_random_geoloss_metadata_records_order():
    label_names = list(SCHWARTZ_VALUE_ORDER)
    config = {"loss": {"name": "random_geoloss", "base": "bce", "random_seed": 42}}

    meta = loss_config_for_metadata(config, label_names=label_names)

    assert loss_slug(config) == "random_geoloss"
    assert meta["geometry"] == "random_circular"
    assert len(meta["random_order"]) == len(label_names)
