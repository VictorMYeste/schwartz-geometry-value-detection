import importlib.util
import sys
from pathlib import Path


def _load_grid_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "grid_loss_hparams.py"
    spec = importlib.util.spec_from_file_location("grid_loss_hparams", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_grid_for_asl():
    module = _load_grid_module()

    grid = module.build_grid(["asl"], (42,))

    assert len(grid) == 16
    assert grid[0].method == "asl"
    assert grid[0].config_path == "configs/deberta_asl.yaml"
    assert "gamma_neg" in grid[0].params


def test_build_grid_for_structured_methods():
    module = _load_grid_module()

    grid = module.build_grid(["schwartz_geoloss", "schwartz_geosmooth"], (42, 7))

    assert len(grid) == 16
    assert {run.method for run in grid} == {"schwartz_geoloss", "schwartz_geosmooth"}


def test_apply_loss_params_only_touches_loss_mapping():
    module = _load_grid_module()
    config = {
        "loss": {"name": "schwartz_geoloss", "base": "asl", "lambda_geo": 0.05},
        "training": {"batch_size": 8},
    }

    tuned = module.apply_loss_params(config, {"lambda_geo": 0.2})

    assert tuned["loss"]["lambda_geo"] == 0.2
    assert tuned["training"] == {"batch_size": 8}
    assert config["loss"]["lambda_geo"] == 0.05
