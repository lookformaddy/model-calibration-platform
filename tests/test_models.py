"""Tests for the from-scratch classifiers."""
import math

from calibplat.data import sigmoid
from calibplat.models import (KNNClassifier, LogisticRegression,
                              StandardScaler, load_model)


def test_scaler_centres_and_scales(data):
    scaler = StandardScaler().fit(data.X)
    scaled = scaler.transform(data.X)
    n = len(scaled)
    for j in range(data.n_features):
        mean = sum(row[j] for row in scaled) / n
        assert abs(mean) < 1e-9
        var = sum(row[j] ** 2 for row in scaled) / n
        assert abs(var - 1.0) < 1e-9


def test_scaler_roundtrip(data):
    scaler = StandardScaler().fit(data.X)
    restored = StandardScaler.from_dict(scaler.to_dict())
    assert restored.mean == scaler.mean
    assert restored.std == scaler.std


def test_logistic_learns(data):
    scaler = StandardScaler().fit(data.X)
    model = LogisticRegression().fit(scaler.transform(data.X), data.y,
                                     epochs=300)
    acc = sum(int(p >= 0.5) == y for p, y in
              zip(model.predict_proba(scaler.transform(data.X_test)),
                  data.y_test)) / len(data.y_test)
    assert acc > 0.70


def test_logistic_scores_are_logits(data):
    scaler = StandardScaler().fit(data.X)
    model = LogisticRegression().fit(scaler.transform(data.X), data.y,
                                     epochs=50)
    scores = model.decision_scores(scaler.transform(data.X_test[:5]))
    probs = model.predict_proba(scaler.transform(data.X_test[:5]))
    for s, p in zip(scores, probs):
        assert abs(sigmoid(s) - p) < 1e-9


def test_logistic_serialization_roundtrip(data):
    scaler = StandardScaler().fit(data.X)
    model = LogisticRegression().fit(scaler.transform(data.X), data.y,
                                     epochs=50)
    restored = load_model(model.to_dict())
    assert restored.decision_scores(
        scaler.transform(data.X_test[:10])) == model.decision_scores(
        scaler.transform(data.X_test[:10]))


def test_knn_accuracy(data):
    scaler = StandardScaler().fit(data.X)
    model = KNNClassifier(k=15).fit(scaler.transform(data.X), data.y)
    acc = sum(int(p >= 0.5) == y for p, y in
              zip(model.predict_proba(scaler.transform(data.X_test)),
                  data.y_test)) / len(data.y_test)
    assert acc > 0.65


def test_knn_probabilities_bounded(data):
    scaler = StandardScaler().fit(data.X)
    model = KNNClassifier(k=5).fit(scaler.transform(data.X), data.y)
    probs = model.predict_proba(scaler.transform(data.X_test))
    assert all(0.0 <= p <= 1.0 for p in probs)
    # Laplace smoothing: never exactly 0 or 1.
    assert all(0.0 < p < 1.0 for p in probs)


def test_knn_serialization_roundtrip(data):
    scaler = StandardScaler().fit(data.X)
    model = KNNClassifier(k=5).fit(scaler.transform(data.X), data.y)
    restored = load_model(model.to_dict())
    assert restored.predict_proba(
        scaler.transform(data.X_test[:5])) == model.predict_proba(
        scaler.transform(data.X_test[:5]))
