"""Pytest configuration: put the src layout on the import path so plain
``pytest`` passes from the project root, with or without an install.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from calibplat.data import generate_dataset  # noqa: E402
from calibplat.pipeline import (RunConfig, _fit_base, _scores,  # noqa: E402
                                run_calibration_pipeline)
from calibplat.registry import CalibratorRegistry  # noqa: E402
from calibplat.serving import ServingConfig, ServingLayer  # noqa: E402

SEED = 42


@pytest.fixture(scope="session")
def data():
    return generate_dataset(n_train=600, n_cal=400, n_test=1000, seed=SEED)


@pytest.fixture(scope="session")
def fitted(data):
    """Scaler + deliberately overconfident logistic model."""
    scaler, model = _fit_base(RunConfig(), data)
    return scaler, model


@pytest.fixture(scope="session")
def cal_scores(fitted, data):
    scaler, model = fitted
    return _scores(scaler, model, data.X_cal)


@pytest.fixture(scope="session")
def test_scores(fitted, data):
    scaler, model = fitted
    return _scores(scaler, model, data.X_test)


@pytest.fixture(scope="session")
def registry():
    reg = CalibratorRegistry()
    run_calibration_pipeline(RunConfig(), registry=reg)
    return reg


@pytest.fixture(scope="session")
def serving(registry):
    return ServingLayer(registry, ServingConfig())
