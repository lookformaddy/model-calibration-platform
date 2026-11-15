"""Tests for drift detection and the recalibration policy."""
from calibplat.data import generate_dataset
from calibplat.drift import (RecalibrationPolicy, detect_drift,
                             ks_statistic,
                             paired_bootstrap_brier_improvement, psi)
from calibplat.pipeline import _fit_base, _scores, RunConfig


def test_psi_zero_for_identical_distributions(cal_scores):
    assert psi(cal_scores, list(cal_scores)) == 0.0


def test_psi_high_for_shifted_distribution(data, cal_scores):
    shifted = generate_dataset(n_train=300, n_cal=300, n_test=300,
                               seed=42, shift=1.5)
    scaler, model = _fit_base(RunConfig(), data)
    shifted_scores = _scores(scaler, model, shifted.X_cal)
    assert psi(list(cal_scores), shifted_scores) > 0.2


def test_ks_zero_for_identical(cal_scores):
    assert ks_statistic(cal_scores, list(cal_scores)) == 0.0


def test_ks_high_for_shifted(cal_scores, data):
    shifted = generate_dataset(n_train=300, n_cal=300, n_test=300,
                               seed=42, shift=1.5)
    scaler, model = _fit_base(RunConfig(), data)
    shifted_scores = _scores(scaler, model, shifted.X_cal)
    assert ks_statistic(cal_scores, shifted_scores) > 0.15


def test_detect_drift_stable_case(cal_scores):
    report = detect_drift(cal_scores, list(cal_scores))
    assert not report.drifted
    assert report.reason == "stable"


def test_detect_drift_flags_shift(cal_scores, data):
    shifted = generate_dataset(n_train=300, n_cal=300, n_test=300,
                               seed=42, shift=1.5)
    scaler, model = _fit_base(RunConfig(), data)
    shifted_scores = _scores(scaler, model, shifted.X_cal)
    report = detect_drift(cal_scores, shifted_scores)
    assert report.drifted
    assert "PSI" in report.reason or "KS" in report.reason


def test_policy_scheduled_recalibration():
    policy = RecalibrationPolicy(max_age_predictions=100)
    report = detect_drift([0.1, 0.2], [0.1, 0.2])
    needed, reason = policy.needs_recalibration(report, 150)
    assert needed
    assert "scheduled" in reason
    needed, _ = policy.needs_recalibration(report, 50)
    assert not needed


def test_policy_drift_overrides_schedule():
    policy = RecalibrationPolicy(max_age_predictions=10_000)
    report = detect_drift([0.0] * 50, [5.0] * 50)
    needed, reason = policy.needs_recalibration(report, 0)
    assert needed
    assert "drift" in reason


def test_bootstrap_improvement_significant():
    # "Before" is confidently wrong; "after" is modestly right.
    y = [0, 1] * 100
    before = [0.95 if v == 0 else 0.05 for v in y]
    after = [0.45 if v == 0 else 0.55 for v in y]
    result = paired_bootstrap_brier_improvement(y, before, after,
                                                n_resamples=100, seed=1)
    assert result["observed_improvement"] > 0
    assert 0.0 <= result["p_value"] <= 1.0
    assert set(result) == {"observed_improvement", "p_value",
                           "n_resamples", "significant"}


def test_bootstrap_no_improvement_when_identical():
    y = [0, 1] * 50
    p = [0.4] * 100
    result = paired_bootstrap_brier_improvement(y, p, p, n_resamples=50,
                                                seed=1)
    assert result["observed_improvement"] == 0.0
    assert not result["significant"]
