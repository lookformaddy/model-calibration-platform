"""API tests (FastAPI TestClient). These run only when fastapi and
httpx are installed; otherwise they are skipped, not failed.
"""
import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from calibplat.service import app  # noqa: E402

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["champion"] == "platt-logistic-v1-seed42"
    assert data["ece"] is not None
    assert data["ece"] < 0.05


def test_predict():
    resp = client.post("/predict", json={
        "features": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "request_id": "api-1",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert 0.0 <= data["probability"] <= 1.0
    assert data["calibrated"] is True
    assert data["method"] == "platt"
    assert data["confidence_interval"][0] <= data["probability"] \
        <= data["confidence_interval"][1]
    assert "platt" in data["explanation"]


def test_predict_empty_features_rejected():
    resp = client.post("/predict", json={"features": []})
    assert resp.status_code == 422


def test_calibrators_endpoint():
    resp = client.get("/calibrators")
    assert resp.status_code == 200
    data = resp.json()
    assert data["champion"] == "platt-logistic-v1-seed42"
    assert any(v["name"] == "platt-logistic-v1-seed42"
               for v in data["versions"])
    version = data["versions"][0]
    assert all(g["passed"] for g in version["gates"])


def test_stats_endpoint():
    client.post("/predict", json={"features": [0.1] * 6})
    resp = client.get("/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["n_requests"] >= 1
    assert "p95_latency_ms" in data


def test_promote_unknown_version_rejected():
    resp = client.post("/promote", json={"version": "does-not-exist"})
    assert resp.status_code == 400


def test_drift_check_stable_then_drifted():
    # Scores drawn from the champion's own reference distribution:
    # stable. Wildly shifted scores: drifted.
    versions = client.get("/calibrators").json()["versions"]
    ref_scores = versions[0]["reference_scores"]
    resp = client.post("/drift-check",
                       json={"scores": ref_scores[::4]})
    assert resp.status_code == 200
    assert resp.json()["drifted"] is False
    resp = client.post("/drift-check",
                       json={"scores": [25.0] * 50})
    assert resp.status_code == 200
    assert resp.json()["drifted"] is True
