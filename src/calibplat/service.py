"""FastAPI service for the calibration platform.

Builds a champion calibrator at import (seeded, deterministic, fast)
and serves calibrated probabilities with confidence intervals and an
explanation of the calibration applied. Endpoints:

* GET  /health — status + champion version
* POST /predict — calibrated probability for one feature vector
* GET  /calibrators — registered versions + gate results
* POST /promote — promote a candidate version (gates enforced)
* POST /rollback — restore the previous champion
* POST /drift-check — PSI/KS against the champion's reference scores
* GET  /stats — serving counters and p95 latency
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from calibplat.drift import detect_drift
from calibplat.pipeline import (RunConfig, run_calibration_pipeline,
                                run_drift_recalibration)
from calibplat.registry import CalibratorRegistry
from calibplat.serving import ServingConfig, ServingLayer

# Small, fast build so importing the app stays cheap in tests.
_REGISTRY = CalibratorRegistry()
_REPORT = run_calibration_pipeline(RunConfig(), registry=_REGISTRY)
_SERVING = ServingLayer(_REGISTRY, ServingConfig())

app = FastAPI(title="Model Calibration Platform",
              version="1.0.0")


class PredictRequest(BaseModel):
    features: list[float] = Field(..., min_length=1)
    request_id: str = ""


class DriftCheckRequest(BaseModel):
    scores: list[float] = Field(..., min_length=1)


@app.get("/health")
def health() -> dict:
    champion = _REGISTRY.champion
    return {"status": "ok",
            "champion": champion.name if champion else None,
            "method": champion.method if champion else None,
            "ece": champion.metrics["ece"] if champion else None}


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    if not req.features:
        raise HTTPException(status_code=422, detail="features must be non-empty")
    return _SERVING.predict(req.features, req.request_id).to_dict()


@app.get("/calibrators")
def calibrators() -> dict:
    return {"champion": _REGISTRY.champion_name,
            "versions": _REGISTRY.list_versions()}


@app.post("/promote")
def promote(payload: dict) -> dict:
    name = payload.get("version")
    if not name:
        raise HTTPException(status_code=400,
                            detail="version is required")
    try:
        version = _REGISTRY.promote(name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"promoted": version.name, "champion": _REGISTRY.champion_name}


@app.post("/rollback")
def rollback() -> dict:
    restored = _REGISTRY.rollback()
    if restored is None:
        raise HTTPException(status_code=400,
                            detail="nothing to roll back to")
    return {"restored": restored.name,
            "champion": _REGISTRY.champion_name}


@app.post("/drift-check")
def drift_check(req: DriftCheckRequest) -> dict:
    champion = _REGISTRY.champion
    if champion is None:
        raise HTTPException(status_code=400,
                            detail="no champion calibrator registered")
    report = detect_drift(champion.reference_scores, req.scores)
    _SERVING.mark_stale(report.drifted)
    return report.to_dict()


@app.get("/stats")
def stats() -> dict:
    return _SERVING.stats()
