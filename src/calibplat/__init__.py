"""Model Calibration Platform.

Probability calibration for production classifiers, built from scratch
on pure Python stdlib: from-scratch logistic regression and k-NN
classifiers, four calibration methods (Platt scaling, isotonic
regression, histogram binning, temperature scaling) with cross-fitted
fitting, reliability metrics (ECE, MCE, Brier, log-loss), per-slice
calibration, drift-triggered recalibration (PSI / KS), a versioned
calibrator registry with four promotion gates and rollback, a
hash-chained audit log, reproducibility manifests and latency-budgeted
serving. All data is seeded synthetic; nothing here comes from any
real system or employer.
"""

from calibplat.calibration import (HistogramBinning, IsotonicRegression,
                                   PlattScaling, TemperatureScaling,
                                   cross_fit_calibrate)
from calibplat.drift import detect_drift
from calibplat.metrics import (brier_score, expected_calibration_error,
                               log_loss, maximum_calibration_error)
from calibplat.pipeline import run_calibration_pipeline

__all__ = [
    "HistogramBinning",
    "IsotonicRegression",
    "PlattScaling",
    "TemperatureScaling",
    "cross_fit_calibrate",
    "detect_drift",
    "brier_score",
    "expected_calibration_error",
    "log_loss",
    "maximum_calibration_error",
    "run_calibration_pipeline",
]
