"""Tests for the end-to-end pipeline and the drift scenario."""
from calibplat.pipeline import (RunConfig, run_calibration_pipeline,
                                run_drift_recalibration)
from calibplat.registry import CalibratorRegistry


def _small_config(**overrides):
    cfg = {"stability_seeds": (43,)}
    cfg.update(overrides)
    return RunConfig(**cfg)


def test_pipeline_promotes_platt_champion():
    registry = CalibratorRegistry()
    report = run_calibration_pipeline(_small_config(), registry=registry)
    assert report["raw_metrics"]["ece"] >= 0.10
    assert report["calibrated_metrics"]["ece"] < 0.05
    assert report["calibrated_metrics"]["brier"] < \
        report["raw_metrics"]["brier"]
    assert report["promoted"] is True
    assert registry.champion_name == report["version"]
    assert report["manifest_valid"] is True
    assert len(report["seed_eces"]) == 2


def test_pipeline_knn_base_model():
    registry = CalibratorRegistry()
    report = run_calibration_pipeline(
        _small_config(model_kind="knn", method="isotonic"),
        registry=registry)
    assert report["calibrated_metrics"]["ece"] < 0.10
    assert report["manifest_valid"] is True


def test_pipeline_temperature_method():
    registry = CalibratorRegistry()
    report = run_calibration_pipeline(
        _small_config(method="temperature"), registry=registry)
    assert report["calibrated_metrics"]["ece"] < 0.05


def test_drift_recalibration_promotes_new_version():
    registry = CalibratorRegistry()
    run_calibration_pipeline(_small_config(), registry=registry)
    report = run_drift_recalibration(registry, shift=0.8)
    assert report["drift"]["drifted"] is True
    assert report["drift"]["psi"] > 0.2 or report["drift"]["ks"] > 0.15
    assert report["recalibrated"] is True
    assert report["new_version"] is not None
    assert registry.champion_name == report["new_version"]
    assert report["recalibrated_metrics"]["ece"] < 0.05
    assert report["significance"]["observed_improvement"] > 0


def test_drift_recalibration_without_drift_is_noop():
    registry = CalibratorRegistry()
    run_calibration_pipeline(_small_config(), registry=registry)
    report = run_drift_recalibration(registry, shift=0.0)
    assert report["drift"]["drifted"] is False
    assert report["recalibrated"] is False
    assert report["new_version"] is None


def test_drift_recalibration_requires_champion():
    import pytest
    with pytest.raises(ValueError, match="no champion"):
        run_drift_recalibration(CalibratorRegistry(), shift=0.8)
