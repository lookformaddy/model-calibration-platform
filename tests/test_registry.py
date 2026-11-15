"""Tests for the calibrator registry, promotion gates and rollback."""
import pytest

from calibplat.manifest import ReproducibilityManifest
from calibplat.registry import (CalibratorRegistry, CalibratorVersion,
                                PromotionGates)


def _version(name, ece=0.02, brier=0.15, base_brier=0.20,
             max_slice_ece=0.03, seed_eces=(0.02, 0.025, 0.018)):
    manifest = ReproducibilityManifest.build(
        version_name=name, dataset_fingerprint="abc",
        config={"method": "platt"}, model_config={"model_kind": "logistic"},
        calibrator_config={"method": "platt", "a": 1.0, "b": 0.0},
        seeds={"data": 42})
    return CalibratorVersion(
        name=name, model_version="logistic-seed42", method="platt",
        metrics={"ece": ece, "brier": brier, "max_slice_ece": max_slice_ece},
        baseline_metrics={"brier": base_brier}, seed_eces=list(seed_eces),
        model_state={"kind": "logistic", "weights": [0.1], "intercept": 0.0,
                     "n_features": 1},
        scaler_state={"mean": [0.0], "std": [1.0]},
        calibrator_state={"method": "platt", "a": 1.0, "b": 0.0},
        manifest=manifest)


def test_all_four_gates_pass_for_good_version():
    reg = CalibratorRegistry()
    version = reg.register(_version("v1"))
    assert version.all_gates_passed
    assert [g.name for g in version.gates] == [
        "ece_floor", "brier_beats_baseline", "slice_gates",
        "stability_across_seeds"]


def test_gate_fails_on_high_ece():
    gates = PromotionGates()
    results = gates.evaluate({"ece": 0.20, "brier": 0.15,
                              "max_slice_ece": 0.03},
                             {"brier": 0.20}, [0.02, 0.02])
    failed = [r for r in results if not r.passed]
    assert [r.name for r in failed] == ["ece_floor"]


def test_gate_fails_when_brier_does_not_beat_baseline():
    gates = PromotionGates()
    results = gates.evaluate({"ece": 0.02, "brier": 0.25,
                              "max_slice_ece": 0.03},
                             {"brier": 0.20}, [0.02, 0.02])
    assert [r.name for r in results if not r.passed] == [
        "brier_beats_baseline"]


def test_gate_fails_on_slice_and_stability():
    gates = PromotionGates()
    results = gates.evaluate({"ece": 0.02, "brier": 0.15,
                              "max_slice_ece": 0.12},
                             {"brier": 0.20}, [0.01, 0.05])
    assert {r.name for r in results if not r.passed} == {
        "slice_gates", "stability_across_seeds"}


def test_promote_requires_passing_gates():
    reg = CalibratorRegistry()
    reg.register(_version("bad", ece=0.30))
    with pytest.raises(ValueError, match="gates failed"):
        reg.promote("bad")
    assert reg.champion is None


def test_promote_and_rollback():
    reg = CalibratorRegistry()
    reg.register(_version("v1"))
    reg.register(_version("v2"))
    reg.promote("v1")
    assert reg.champion_name == "v1"
    reg.promote("v2")
    assert reg.champion_name == "v2"
    assert reg.versions["v1"].status == "candidate"
    restored = reg.rollback()
    assert restored is not None
    assert restored.name == "v1"
    assert reg.champion_name == "v1"
    assert reg.versions["v2"].status == "rolled_back"


def test_rollback_with_no_champion_returns_none():
    reg = CalibratorRegistry()
    assert reg.rollback() is None


def test_duplicate_registration_rejected():
    reg = CalibratorRegistry()
    reg.register(_version("v1"))
    with pytest.raises(ValueError, match="already registered"):
        reg.register(_version("v1"))


def test_audit_log_records_promotion(registry):
    # The session registry was built by the pipeline: register +
    # promote events must be in the audit log and the chain valid.
    events = [e.event for e in registry.audit.entries]
    assert "register" in events
    assert "promote" in events
    ok, broken = registry.audit.verify()
    assert ok and broken is None
