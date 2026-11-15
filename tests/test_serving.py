"""Tests for latency-budgeted serving and the stale fallback."""
from calibplat.serving import ServingConfig, ServingLayer, wilson_interval


def test_predict_returns_calibrated_probability(serving, data):
    pred = serving.predict(data.X_test[0])
    assert pred.calibrated is True
    assert pred.stale is False
    assert pred.fallback is False
    assert 0.0 <= pred.probability <= 1.0
    assert pred.method == "platt"
    assert pred.version is not None
    assert "Platt" in pred.explanation or "platt" in pred.explanation


def test_confidence_interval_contains_probability(serving, data):
    pred = serving.predict(data.X_test[0])
    lo, hi = pred.confidence_interval
    assert 0.0 <= lo <= pred.probability <= hi <= 1.0


def test_wilson_interval_widens_with_small_n():
    narrow = wilson_interval(0.7, 10_000)
    wide = wilson_interval(0.7, 10)
    assert narrow[1] - narrow[0] < wide[1] - wide[0]
    assert wilson_interval(0.5, 0) == (0.0, 1.0)


def test_stale_calibrator_falls_back_uncalibrated(serving, data):
    serving.mark_stale(True)
    pred = serving.predict(data.X_test[0])
    assert pred.calibrated is False
    assert pred.stale is True
    assert "UNCALIBRATED" in pred.explanation
    serving.mark_stale(False)
    pred2 = serving.predict(data.X_test[0])
    assert pred2.calibrated is True


def test_tiny_latency_budget_forces_fallback(registry, data):
    serving = ServingLayer(registry, ServingConfig(latency_budget_ms=1e-9))
    pred = serving.predict(data.X_test[0])
    assert pred.fallback is True
    assert pred.probability == 0.5


def test_no_champion_falls_back(registry, data):
    from calibplat.registry import CalibratorRegistry
    serving = ServingLayer(CalibratorRegistry(), ServingConfig())
    pred = serving.predict(data.X_test[0])
    assert pred.fallback is True
    assert "no champion" in pred.explanation


def test_stats_tracks_requests(serving, data):
    before = serving.stats()["n_requests"]
    serving.predict(data.X_test[0])
    after = serving.stats()
    assert after["n_requests"] == before + 1
    assert after["champion"] is not None
    assert "p95_latency_ms" in after
