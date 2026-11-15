"""Seeded synthetic data for the calibration platform.

Binary classification in a generic 6-dimensional feature space. Labels
come from a known logistic ground truth, so the *true* probabilities are
available for sanity checks and the base classifier can be made
deliberately miscalibrated (overconfident logits) in a controlled way —
exactly the failure mode calibration exists to fix. A categorical
segment (derived from one feature) supports per-slice calibration.
A shifted variant of the generator produces the drift scenario used by
the recalibration demo. Nothing here comes from any real system or
employer.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass, field

# Ground-truth weights: fixed, documented, and independent of any real
# system. The intercept makes the positive rate roughly 45%.
TRUE_WEIGHTS = [0.9, -0.7, 0.6, 0.4, -0.5, 0.3]
TRUE_INTERCEPT = -0.15
N_SEGMENTS = 3


def sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


@dataclass
class Dataset:
    """A labelled synthetic dataset with calibration and test splits."""

    X: list[list[float]]
    y: list[int]
    segments: list[int]
    X_cal: list[list[float]]
    y_cal: list[int]
    segments_cal: list[int]
    X_test: list[list[float]]
    y_test: list[int]
    segments_test: list[int]
    n_features: int = 6
    seed: int = 42
    shift: float = 0.0

    def fingerprint(self) -> str:
        """Stable SHA-256 over the data — recorded in manifests."""
        payload = json.dumps(
            {"X": self.X, "y": self.y, "X_cal": self.X_cal,
             "y_cal": self.y_cal, "X_test": self.X_test,
             "y_test": self.y_test, "seed": self.seed,
             "shift": self.shift},
            sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()

    @property
    def n_train(self) -> int:
        return len(self.X)

    @property
    def positive_rate(self) -> float:
        return sum(self.y) / max(1, len(self.y))


def _draw_row(rng: random.Random, n_features: int,
              shift: float) -> list[float]:
    # ``shift`` moves the mean of EVERY feature: the ground-truth
    # logit moves by sum(weights) * shift, so the score distribution
    # moves decisively and PSI/KS fire. (Shifting only a subset
    # cancels out when the shifted weights have opposite signs.)
    return [rng.gauss(shift, 1.0) for _ in range(n_features)]


def _segment(row: list[float]) -> int:
    """Segment from a feature value: three roughly equal slices."""
    v = row[5]
    if v < -0.43:
        return 0
    if v > 0.43:
        return 2
    return 1


def _label(rng: random.Random, row: list[float]) -> int:
    logit = TRUE_INTERCEPT + sum(w * x for w, x in zip(TRUE_WEIGHTS, row))
    return 1 if rng.random() < sigmoid(logit) else 0


def _draw_split(rng: random.Random, n: int, n_features: int,
                shift: float) -> tuple[list[list[float]], list[int],
                                       list[int]]:
    X, y, seg = [], [], []
    for _ in range(n):
        row = _draw_row(rng, n_features, shift)
        X.append(row)
        y.append(_label(rng, row))
        seg.append(_segment(row))
    return X, y, seg


def generate_dataset(n_train: int = 600, n_cal: int = 400,
                     n_test: int = 600, n_features: int = 6,
                     seed: int = 42, shift: float = 0.0) -> Dataset:
    """Generate train / calibration / test splits from one seed.

    ``shift`` moves the mean of every feature and is the drift lever:
    a dataset generated with ``shift > 0`` comes from a different
    input distribution than the model was trained on.
    """
    rng = random.Random(seed)
    X, y, seg = _draw_split(rng, n_train, n_features, shift)
    X_cal, y_cal, seg_cal = _draw_split(rng, n_cal, n_features, shift)
    X_test, y_test, seg_test = _draw_split(rng, n_test, n_features, shift)
    return Dataset(X=X, y=y, segments=seg, X_cal=X_cal, y_cal=y_cal,
                   segments_cal=seg_cal, X_test=X_test, y_test=y_test,
                   segments_test=seg_test, n_features=n_features,
                   seed=seed, shift=shift)


def true_probability(row: list[float]) -> float:
    """The ground-truth probability for a feature row (diagnostics)."""
    logit = TRUE_INTERCEPT + sum(w * x for w, x in zip(TRUE_WEIGHTS, row))
    return sigmoid(logit)
