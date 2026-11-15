"""Tests for the four calibration methods and cross-fitting."""
import pytest

from calibplat.calibration import (build_calibrator, cross_fit_calibrate,
                                   load_calibrator)
from calibplat.data import sigmoid
from calibplat.metrics import (brier_score, evaluate_calibration,
                               expected_calibration_error)


def _raw_metrics(data, test_scores):
    raw_probs = [sigmoid(s) for s in test_scores]
    return evaluate_calibration(data.y_test, raw_probs)


def test_raw_model_is_miscalibrated(data, test_scores):
    """The x3-logit production model must actually be miscalibrated —
    otherwise the whole demo is vacuous."""
    raw = _raw_metrics(data, test_scores)
    assert raw["ece"] >= 0.10


def test_platt_fixes_calibration(data, cal_scores, test_scores):
    result = cross_fit_calibrate("platt", cal_scores, data.y_cal,
                                 n_folds=5, seed=42)
    probs = result.calibrator.calibrate(test_scores)
    assert expected_calibration_error(data.y_test, probs) < 0.05
    assert (brier_score(data.y_test, probs)
            < brier_score(data.y_test, [sigmoid(s) for s in test_scores]))


def test_isotonic_fixes_calibration(data, cal_scores, test_scores):
    result = cross_fit_calibrate("isotonic", cal_scores, data.y_cal,
                                 n_folds=5, seed=42)
    probs = result.calibrator.calibrate(test_scores)
    assert expected_calibration_error(data.y_test, probs) < 0.08


def test_histogram_fixes_calibration(data, cal_scores, test_scores):
    result = cross_fit_calibrate("histogram", cal_scores, data.y_cal,
                                 n_folds=5, seed=42)
    probs = result.calibrator.calibrate(test_scores)
    assert expected_calibration_error(data.y_test, probs) < 0.10


def test_temperature_fixes_calibration(data, cal_scores, test_scores):
    result = cross_fit_calibrate("temperature", cal_scores, data.y_cal,
                                 n_folds=5, seed=42)
    assert result.calibrator.temperature > 1.0  # softens overconfidence
    probs = result.calibrator.calibrate(test_scores)
    assert expected_calibration_error(data.y_test, probs) < 0.05


def test_isotonic_is_monotone(data, cal_scores):
    from calibplat.calibration import IsotonicRegression
    cal = IsotonicRegression().fit(cal_scores, data.y_cal)
    grid = [x * 0.5 - 5 for x in range(41)]
    out = cal.calibrate(grid)
    assert all(b >= a for a, b in zip(out, out[1:]))


def test_calibrated_probabilities_bounded(data, cal_scores):
    for method in ("platt", "isotonic", "histogram", "temperature"):
        cal = build_calibrator(method).fit(cal_scores, data.y_cal)
        probs = cal.calibrate([-100.0, -1.0, 0.0, 1.0, 100.0])
        assert all(0.0 <= p <= 1.0 for p in probs)


def test_calibrator_serialization_roundtrip(data, cal_scores):
    for method in ("platt", "isotonic", "histogram", "temperature"):
        cal = build_calibrator(method).fit(cal_scores, data.y_cal)
        restored = load_calibrator(cal.to_dict())
        assert restored.calibrate(cal_scores[:20]) == cal.calibrate(
            cal_scores[:20])


def test_cross_fit_oof_is_unbiased_estimate(data, cal_scores):
    result = cross_fit_calibrate("platt", cal_scores, data.y_cal,
                                 n_folds=5, seed=42)
    assert len(result.oof_probs) == len(data.y_cal)
    assert result.n_folds == 5
    oof_ece = expected_calibration_error(data.y_cal, result.oof_probs)
    assert oof_ece < 0.08


def test_cross_fit_rejects_one_fold(data, cal_scores):
    with pytest.raises(ValueError):
        cross_fit_calibrate("platt", cal_scores, data.y_cal, n_folds=1)


def test_unknown_method_rejected():
    with pytest.raises(ValueError):
        build_calibrator("nope")
