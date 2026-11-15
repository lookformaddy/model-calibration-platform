"""Tests for the reproducibility manifest."""
from calibplat.manifest import ReproducibilityManifest


def _manifest(**overrides):
    data = {"version_name": "platt-logistic-v1-seed42",
            "dataset_fingerprint": "abc123",
            "config": {"method": "platt", "n_folds": 5},
            "model_config": {"model_kind": "logistic"},
            "calibrator_config": {"method": "platt", "a": 0.33,
                                  "b": 0.01},
            "seeds": {"data": 42, "run": 42}}
    data.update(overrides)
    return ReproducibilityManifest.build(**data)


def test_manifest_verifies_when_untouched():
    manifest = _manifest()
    assert manifest.verify()
    assert len(manifest.content_hash) == 64


def test_manifest_detects_tampering():
    manifest = _manifest()
    tampered = ReproducibilityManifest.from_dict(manifest.to_dict())
    tampered.config["n_folds"] = 99
    assert not tampered.verify()
    tampered2 = ReproducibilityManifest.from_dict(manifest.to_dict())
    tampered2.seeds["data"] = 7
    assert not tampered2.verify()
    tampered3 = ReproducibilityManifest.from_dict(manifest.to_dict())
    tampered3.calibrator_config["a"] = 9.9
    assert not tampered3.verify()


def test_manifest_roundtrip():
    manifest = _manifest()
    restored = ReproducibilityManifest.from_dict(manifest.to_dict())
    assert restored == manifest
    assert restored.verify()
    assert restored.environment["python"]
