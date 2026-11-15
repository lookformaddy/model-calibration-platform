"""Latency-budgeted serving of calibrated probabilities.

Each prediction returns the calibrated probability, a Wilson-score
confidence interval for that probability, the calibrator version and
method that produced it, and a plain-language explanation of the
calibration applied. Two fail-closed behaviours:

* Latency budget — if a prediction would exceed the budget, the
  serving layer records the majority-baseline fallback instead of
  serving a late answer.
* Stale calibrator — when drift has been detected (or the calibrator
  is older than its scheduled window), predictions fall back to the
  uncalibrated score and say so explicitly (``calibrated: false``,
  ``stale: true``) instead of silently serving stale probabilities.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from calibplat.calibration import load_calibrator
from calibplat.data import sigmoid
from calibplat.models import StandardScaler, load_model
from calibplat.registry import CalibratorRegistry, CalibratorVersion

# z for a 95% Wilson interval.
_Z95 = 1.96


def wilson_interval(p: float, n: int, z: float = _Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return 0.0, 1.0
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


@dataclass
class ServingConfig:
    latency_budget_ms: float = 100.0
    # Effective calibration-set size used for the Wilson interval when
    # the calibrator does not carry per-bin counts.
    calibration_n: int = 400
    max_age_predictions: int = 10_000


@dataclass
class Prediction:
    probability: float
    calibrated: bool
    stale: bool
    method: str | None
    version: str | None
    confidence_interval: tuple[float, float]
    explanation: str
    latency_ms: float
    fallback: bool = False

    def to_dict(self) -> dict:
        return {"probability": self.probability,
                "calibrated": self.calibrated, "stale": self.stale,
                "method": self.method, "version": self.version,
                "confidence_interval": list(self.confidence_interval),
                "explanation": self.explanation,
                "latency_ms": self.latency_ms,
                "fallback": self.fallback}


@dataclass
class ServingLayer:
    registry: CalibratorRegistry
    config: ServingConfig = field(default_factory=ServingConfig)
    n_requests: int = 0
    n_fallbacks: int = 0
    n_stale: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    _stale_override: bool = False

    def mark_stale(self, stale: bool = True) -> None:
        """Called by the drift monitor when recalibration is due."""
        self._stale_override = stale

    @property
    def stale(self) -> bool:
        return (self._stale_override
                or self.n_requests >= self.config.max_age_predictions)

    def _load(self, version: CalibratorVersion):
        scaler = StandardScaler.from_dict(version.scaler_state)
        model = load_model(version.model_state)
        calibrator = load_calibrator(version.calibrator_state)
        return scaler, model, calibrator

    def predict(self, features: list[float],
                request_id: str = "") -> Prediction:
        started = time.perf_counter()
        self.n_requests += 1
        champion = self.registry.champion
        if champion is None:
            return self._fallback(started, "no champion calibrator "
                                  "registered")
        scaler, model, calibrator = self._load(champion)
        scaled = scaler.transform([features])[0]
        raw_score = model.decision_scores([scaled])[0]
        raw_prob = sigmoid(raw_score)
        if self.stale:
            self.n_stale += 1
            latency = (time.perf_counter() - started) * 1000.0
            self.latencies_ms.append(latency)
            cal_n = (len(champion.reference_scores)
                     if champion.reference_scores
                     else self.config.calibration_n)
            return Prediction(
                probability=raw_prob, calibrated=False, stale=True,
                method=None, version=champion.name,
                confidence_interval=wilson_interval(raw_prob, cal_n),
                explanation=(
                    f"Calibrator {champion.name} ({champion.method}) is "
                    "stale — drift detected or scheduled window expired — "
                    "so this prediction uses the UNCALIBRATED model score. "
                    "Recalibration is required before calibrated "
                    "probabilities resume."),
                latency_ms=latency)
        calibrated = calibrator.calibrate([raw_score])[0]
        latency = (time.perf_counter() - started) * 1000.0
        self.latencies_ms.append(latency)
        if latency > self.config.latency_budget_ms:
            return self._fallback(
                started,
                f"latency {latency:.2f} ms exceeded budget "
                f"{self.config.latency_budget_ms} ms")
        # Effective calibration-set size: the champion's own reference
        # scores when available, else the configured default.
        cal_n = (len(champion.reference_scores)
                 if champion.reference_scores
                 else self.config.calibration_n)
        lo, hi = wilson_interval(calibrated, cal_n)
        return Prediction(
            probability=calibrated, calibrated=True, stale=False,
            method=champion.method, version=champion.name,
            confidence_interval=(lo, hi),
            explanation=(
                f"Raw model score {raw_score:.3f} (probability "
                f"{raw_prob:.3f}) was mapped by {champion.method} "
                f"calibration, version {champion.name}, to {calibrated:.3f}. "
                f"95% Wilson interval [{lo:.3f}, {hi:.3f}] from "
                f"{cal_n} calibration samples."),
            latency_ms=latency)

    def _fallback(self, started: float, reason: str) -> Prediction:
        self.n_fallbacks += 1
        latency = (time.perf_counter() - started) * 1000.0
        self.latencies_ms.append(latency)
        return Prediction(
            probability=0.5, calibrated=False, stale=False,
            method=None, version=None,
            confidence_interval=(0.0, 1.0),
            explanation=f"Fallback (majority baseline 0.5): {reason}.",
            latency_ms=latency, fallback=True)

    def stats(self) -> dict:
        lat = sorted(self.latencies_ms)
        p95 = lat[int(0.95 * (len(lat) - 1))] if lat else 0.0
        return {"n_requests": self.n_requests,
                "n_fallbacks": self.n_fallbacks,
                "n_stale": self.n_stale,
                "p95_latency_ms": round(p95, 3),
                "champion": (self.registry.champion.name
                             if self.registry.champion else None),
                "stale": self.stale}
