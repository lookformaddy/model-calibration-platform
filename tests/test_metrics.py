"""Tests for reliability metrics."""
import pytest

from calibplat.metrics import (brier_score, evaluate_calibration,
                               expected_calibration_error, log_loss,
                               maximum_calibration_error,
                               per_slice_calibration,
                               reliability_diagram)


def test_perfect_probabilities_have_zero_error():
    y = [0, 0, 1, 1, 0, 1]
    p = [0.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    assert expected_calibration_error(y, p) == 0.0
    assert maximum_calibration_error(y, p) == 0.0
    assert brier_score(y, p) == 0.0
    assert log_loss(y, p) == pytest.approx(0.0, abs=1e-9)


def test_constant_half_probability():
    y = [0, 1, 0, 1]
    p = [0.5, 0.5, 0.5, 0.5]
    assert expected_calibration_error(y, p) == 0.0
    assert brier_score(y, p) == 0.25


def test_ece_detects_overconfidence():
    # Predicts 0.9 everywhere but only 50% are positive.
    y = [0, 1] * 50
    p = [0.9] * 100
    assert expected_calibration_error(y, p) == pytest.approx(0.4, abs=1e-9)


def test_reliability_diagram_counts_sum():
    y = [0, 1, 1, 0, 1, 0, 1, 1, 0, 0]
    p = [0.1, 0.2, 0.8, 0.4, 0.9, 0.3, 0.6, 0.7, 0.05, 0.95]
    bins = reliability_diagram(y, p, n_bins=10)
    assert len(bins) == 10
    assert sum(b["count"] for b in bins) == len(y)


def test_mce_is_worst_bin():
    y = [0, 1] * 50
    p = [0.9] * 100
    assert maximum_calibration_error(y, p) == pytest.approx(0.4, abs=1e-9)


def test_per_slice_covers_all_segments():
    y = [0, 1, 1, 0, 1, 0]
    p = [0.2, 0.8, 0.7, 0.3, 0.9, 0.1]
    segments = [0, 0, 1, 1, 2, 2]
    table = per_slice_calibration(y, p, segments)
    assert set(table) == {0, 1, 2}
    assert all(t["n"] == 2 for t in table.values())


def test_evaluate_calibration_summary(data, test_scores):
    from calibplat.data import sigmoid
    probs = [sigmoid(s) for s in test_scores]
    summary = evaluate_calibration(data.y_test, probs,
                                   data.segments_test)
    assert summary["n"] == len(data.y_test)
    assert 0.0 <= summary["ece"] <= 1.0
    assert "max_slice_ece" in summary
    assert len(summary["reliability_diagram"]) == 10
