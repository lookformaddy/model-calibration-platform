"""From-scratch classifiers (pure stdlib).

Logistic regression trained by batch gradient descent and a
distance-weighted k-NN classifier. Both expose raw *scores* (logits for
logistic, positive-neighbour fraction for k-NN) as well as
probabilities; the calibration layer works on scores, never on labels.
Models serialize to plain dicts so the registry can store them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from calibplat.data import sigmoid


@dataclass
class StandardScaler:
    mean: list[float] = field(default_factory=list)
    std: list[float] = field(default_factory=list)

    def fit(self, X: list[list[float]]) -> "StandardScaler":
        n = len(X)
        n_feat = len(X[0])
        self.mean = [sum(row[j] for row in X) / n for j in range(n_feat)]
        self.std = []
        for j in range(n_feat):
            var = sum((row[j] - self.mean[j]) ** 2 for row in X) / n
            self.std.append(math.sqrt(var) if var > 1e-12 else 1.0)
        return self

    def transform(self, X: list[list[float]]) -> list[list[float]]:
        return [[(x - m) / s for x, m, s in zip(row, self.mean, self.std)]
                for row in X]

    def to_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_dict(cls, d: dict) -> "StandardScaler":
        return cls(mean=list(d["mean"]), std=list(d["std"]))


@dataclass
class LogisticRegression:
    """Binary logistic regression, batch gradient descent, L2 penalty."""

    weights: list[float] = field(default_factory=list)
    intercept: float = 0.0
    n_features: int = 0

    def fit(self, X: list[list[float]], y: list[int], epochs: int = 300,
            lr: float = 0.1, l2: float = 1e-3) -> "LogisticRegression":
        n = len(X)
        self.n_features = len(X[0])
        self.weights = [0.0] * self.n_features
        self.intercept = 0.0
        for _ in range(epochs):
            gw = [0.0] * self.n_features
            gb = 0.0
            for row, target in zip(X, y):
                z = self.intercept + sum(w * x for w, x
                                         in zip(self.weights, row))
                p = sigmoid(z)
                err = p - target
                for j in range(self.n_features):
                    gw[j] += err * row[j]
                gb += err
            for j in range(self.n_features):
                self.weights[j] -= lr * (gw[j] / n + l2 * self.weights[j])
            self.intercept -= lr * gb / n
        return self

    def decision_scores(self, X: list[list[float]]) -> list[float]:
        """Raw logits — the input to the calibration layer."""
        return [self.intercept + sum(w * x for w, x
                                     in zip(self.weights, row))
                for row in X]

    def predict_proba(self, X: list[list[float]]) -> list[float]:
        return [sigmoid(z) for z in self.decision_scores(X)]

    def predict(self, X: list[list[float]]) -> list[int]:
        return [1 if p >= 0.5 else 0 for p in self.predict_proba(X)]

    def to_dict(self) -> dict:
        return {"kind": "logistic", "weights": self.weights,
                "intercept": self.intercept,
                "n_features": self.n_features}

    @classmethod
    def from_dict(cls, d: dict) -> "LogisticRegression":
        return cls(weights=list(d["weights"]), intercept=d["intercept"],
                   n_features=d["n_features"])


@dataclass
class KNNClassifier:
    """k-NN with inverse-distance weighting; score is the weighted
    positive-neighbour fraction (notoriously miscalibrated, which is
    exactly why it is a useful second base model here)."""

    k: int = 15
    X_train: list[list[float]] = field(default_factory=list)
    y_train: list[int] = field(default_factory=list)

    def fit(self, X: list[list[float]], y: list[int]) -> "KNNClassifier":
        self.X_train = [list(row) for row in X]
        self.y_train = list(y)
        return self

    def decision_scores(self, X: list[list[float]]) -> list[float]:
        """Logit of the (smoothed) positive fraction, so calibrators
        that expect an unbounded score work unchanged."""
        probs = self.predict_proba(X)
        return [math.log(min(max(p, 1e-6), 1 - 1e-6) /
                         (1 - min(max(p, 1e-6), 1 - 1e-6)))
                for p in probs]

    def predict_proba(self, X: list[list[float]]) -> list[float]:
        out = []
        for row in X:
            dists = []
            for train_row, label in zip(self.X_train, self.y_train):
                d = math.sqrt(sum((a - b) ** 2 for a, b
                                  in zip(row, train_row)))
                dists.append((d, label))
            dists.sort(key=lambda t: t[0])
            neigh = dists[: self.k]
            # Laplace smoothing: a lone neighbour never yields 0 or 1.
            num = 1.0
            den = 2.0
            for d, label in neigh:
                w = 1.0 / (d + 1e-6)
                num += w * label
                den += w
            out.append(num / den)
        return out

    def predict(self, X: list[list[float]]) -> list[int]:
        return [1 if p >= 0.5 else 0 for p in self.predict_proba(X)]

    def to_dict(self) -> dict:
        return {"kind": "knn", "k": self.k, "X_train": self.X_train,
                "y_train": self.y_train}

    @classmethod
    def from_dict(cls, d: dict) -> "KNNClassifier":
        return cls(k=d["k"], X_train=d["X_train"], y_train=d["y_train"])


def load_model(state: dict):
    """Rebuild a classifier from its serialized state dict."""
    if state["kind"] == "logistic":
        return LogisticRegression.from_dict(state)
    if state["kind"] == "knn":
        return KNNClassifier.from_dict(state)
    raise ValueError(f"unknown model kind: {state['kind']!r}")
