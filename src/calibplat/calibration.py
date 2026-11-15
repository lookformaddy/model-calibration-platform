"""Calibration methods (pure stdlib).

Four calibrators, all mapping a raw model score to a calibrated
probability in [0, 1]:

* Platt scaling — logistic regression on the raw score.
* Isotonic regression — pool-adjacent-violators (PAVA), a monotone
  step function fit on the calibration set.
* Histogram / binning — equal-width bins, each mapped to its empirical
  positive rate.
* Temperature scaling — a single temperature T applied to the logit,
  fitted by minimising log-loss (golden-section search).

Plus cross-fitted calibration: K-fold out-of-fold predictions give an
unbiased estimate of post-calibration reliability, and the shipped
calibrator is refit on the full calibration set.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from calibplat.data import sigmoid
from calibplat.metrics import log_loss


def _clip01(p: float) -> float:
    return min(max(p, 0.0), 1.0)


class Calibrator:
    """Common interface for all calibrators."""

    method: str = "base"

    def fit(self, scores: list[float], y: list[int]) -> "Calibrator":
        raise NotImplementedError

    def calibrate(self, scores: list[float]) -> list[float]:
        raise NotImplementedError

    def to_dict(self) -> dict:
        raise NotImplementedError

    @classmethod
    def from_dict(cls, d: dict) -> "Calibrator":
        raise NotImplementedError


@dataclass
class PlattScaling(Calibrator):
    """Logistic regression on the raw score: p = sigmoid(a * s + b)."""

    method: str = "platt"
    a: float = 1.0
    b: float = 0.0

    def fit(self, scores: list[float], y: list[int],
            epochs: int = 500, lr: float = 0.5) -> "PlattScaling":
        # Standardise the score for stable optimisation, then fold the
        # standardisation back into (a, b) so predict is a single line.
        n = len(scores)
        mean = sum(scores) / n
        var = sum((s - mean) ** 2 for s in scores) / n
        std = math.sqrt(var) if var > 1e-12 else 1.0
        a, b = 0.0, 0.0
        for _ in range(epochs):
            ga = gb = 0.0
            for s, target in zip(scores, y):
                z = a * (s - mean) / std + b
                err = sigmoid(z) - target
                ga += err * (s - mean) / std
                gb += err
            a -= lr * ga / n
            b -= lr * gb / n
        self.a = a / std
        self.b = b - a * mean / std
        return self

    def calibrate(self, scores: list[float]) -> list[float]:
        return [_clip01(sigmoid(self.a * s + self.b)) for s in scores]

    def to_dict(self) -> dict:
        return {"method": "platt", "a": self.a, "b": self.b}

    @classmethod
    def from_dict(cls, d: dict) -> "PlattScaling":
        return cls(a=d["a"], b=d["b"])


@dataclass
class IsotonicRegression(Calibrator):
    """PAVA isotonic regression: a monotone non-decreasing step
    function from score to calibrated probability."""

    method: str = "isotonic"
    # Sorted breakpoints: (score_threshold, calibrated_value).
    points: list[tuple[float, float]] = field(default_factory=list)

    def fit(self, scores: list[float], y: list[int]) -> "IsotonicRegression":
        order = sorted(range(len(scores)), key=lambda i: scores[i])
        xs = [scores[i] for i in order]
        ys = [float(y[i]) for i in order]
        # PAVA: blocks of (value, weight, first_x, last_x).
        blocks: list[list[float]] = []
        for x, target in zip(xs, ys):
            blocks.append([target, 1.0, x, x])
            while len(blocks) >= 2 and blocks[-2][0] > blocks[-1][0]:
                v2, w2, _, x2 = blocks.pop()
                v1, w1, x1, _ = blocks.pop()
                blocks.append([(v1 * w1 + v2 * w2) / (w1 + w2),
                               w1 + w2, x1, x2])
        self.points = [(b[3], _clip01(b[0])) for b in blocks]
        return self

    def calibrate(self, scores: list[float]) -> list[float]:
        out = []
        for s in scores:
            if not self.points:
                out.append(0.5)
                continue
            if s <= self.points[0][0]:
                out.append(self.points[0][1])
                continue
            value = self.points[-1][1]
            for threshold, val in self.points:
                if s <= threshold:
                    value = val
                    break
            out.append(value)
        return out

    def to_dict(self) -> dict:
        return {"method": "isotonic",
                "points": [[x, v] for x, v in self.points]}

    @classmethod
    def from_dict(cls, d: dict) -> "IsotonicRegression":
        return cls(points=[(x, v) for x, v in d["points"]])


@dataclass
class HistogramBinning(Calibrator):
    """Equal-width bins over the score range; each bin maps to its
    empirical positive rate on the calibration set."""

    method: str = "histogram"
    n_bins: int = 10
    edges: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    global_rate: float = 0.5

    def fit(self, scores: list[float], y: list[int]) -> "HistogramBinning":
        self.global_rate = sum(y) / max(1, len(y))
        lo, hi = min(scores), max(scores)
        if hi - lo < 1e-12:
            hi = lo + 1.0
        self.edges = [lo + (hi - lo) * i / self.n_bins
                      for i in range(self.n_bins + 1)]
        self.values = []
        for b in range(self.n_bins):
            idx = [i for i, s in enumerate(scores)
                   if self.edges[b] <= s < self.edges[b + 1]
                   or (b == self.n_bins - 1 and s == self.edges[-1])]
            if idx:
                self.values.append(sum(y[i] for i in idx) / len(idx))
            else:
                self.values.append(self.global_rate)
        return self

    def _bin(self, s: float) -> int:
        if s <= self.edges[0]:
            return 0
        if s >= self.edges[-1]:
            return self.n_bins - 1
        for b in range(self.n_bins):
            if self.edges[b] <= s < self.edges[b + 1]:
                return b
        return self.n_bins - 1

    def calibrate(self, scores: list[float]) -> list[float]:
        return [_clip01(self.values[self._bin(s)]) for s in scores]

    def to_dict(self) -> dict:
        return {"method": "histogram", "n_bins": self.n_bins,
                "edges": self.edges, "values": self.values,
                "global_rate": self.global_rate}

    @classmethod
    def from_dict(cls, d: dict) -> "HistogramBinning":
        return cls(n_bins=d["n_bins"], edges=list(d["edges"]),
                   values=list(d["values"]),
                   global_rate=d["global_rate"])


@dataclass
class TemperatureScaling(Calibrator):
    """p = sigmoid(logit / T), T fitted by golden-section search on
    log-loss. T > 1 softens overconfident scores; T < 1 sharpens."""

    method: str = "temperature"
    temperature: float = 1.0

    def _loss(self, scores: list[float], y: list[int], T: float) -> float:
        probs = [sigmoid(s / T) for s in scores]
        return log_loss(y, probs)

    def fit(self, scores: list[float], y: list[int]) -> "TemperatureScaling":
        lo, hi = 0.05, 20.0
        gr = (math.sqrt(5.0) - 1.0) / 2.0
        c = hi - gr * (hi - lo)
        d = lo + gr * (hi - lo)
        for _ in range(80):
            if self._loss(scores, y, c) < self._loss(scores, y, d):
                hi = d
            else:
                lo = c
            c = hi - gr * (hi - lo)
            d = lo + gr * (hi - lo)
        self.temperature = (lo + hi) / 2.0
        return self

    def calibrate(self, scores: list[float]) -> list[float]:
        return [_clip01(sigmoid(s / self.temperature)) for s in scores]

    def to_dict(self) -> dict:
        return {"method": "temperature",
                "temperature": self.temperature}

    @classmethod
    def from_dict(cls, d: dict) -> "TemperatureScaling":
        return cls(temperature=d["temperature"])


CALIBRATOR_CLASSES: dict[str, type[Calibrator]] = {
    "platt": PlattScaling,
    "isotonic": IsotonicRegression,
    "histogram": HistogramBinning,
    "temperature": TemperatureScaling,
}


def build_calibrator(method: str, **kwargs) -> Calibrator:
    """Factory used by the pipeline, registry and CLI."""
    if method not in CALIBRATOR_CLASSES:
        raise ValueError(f"unknown calibration method: {method!r} "
                         f"(expected one of {sorted(CALIBRATOR_CLASSES)})")
    return CALIBRATOR_CLASSES[method](**kwargs)


def load_calibrator(state: dict) -> Calibrator:
    return CALIBRATOR_CLASSES[state["method"]].from_dict(state)


@dataclass
class CrossFitResult:
    """Out-of-fold calibrated probabilities plus the calibrator refit
    on the full calibration set (the one that ships)."""
    oof_probs: list[float]
    calibrator: Calibrator
    n_folds: int


def cross_fit_calibrate(method: str, scores: list[float], y: list[int],
                        n_folds: int = 5, seed: int = 42,
                        **kwargs) -> CrossFitResult:
    """K-fold cross-fitted calibration.

    Each fold's probabilities come from a calibrator that never saw
    that fold, so the OOF ECE is an unbiased estimate of what the
    shipped calibrator will do on fresh data. The shipped calibrator
    is then refit on the full calibration set.
    """
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    rng = random.Random(seed)
    idx = list(range(len(scores)))
    rng.shuffle(idx)
    folds: list[list[int]] = [[] for _ in range(n_folds)]
    for pos, i in enumerate(idx):
        folds[pos % n_folds].append(i)
    oof = [0.5] * len(scores)
    for f in range(n_folds):
        test_idx = set(folds[f])
        train_idx = [i for i in idx if i not in test_idx]
        cal = build_calibrator(method, **kwargs)
        cal.fit([scores[i] for i in train_idx],
                [y[i] for i in train_idx])
        for i in folds[f]:
            oof[i] = cal.calibrate([scores[i]])[0]
    final = build_calibrator(method, **kwargs)
    final.fit(scores, y)
    return CrossFitResult(oof_probs=oof, calibrator=final, n_folds=n_folds)
