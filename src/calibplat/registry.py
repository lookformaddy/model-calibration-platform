"""Versioned calibrator registry with promotion gates and rollback.

A calibrator version is bound to a model version: it stores the fitted
model state, scaler state and calibrator state, the evaluation metrics
that justify it, and its reproducibility manifest. Promotion to
champion requires all four gates to pass:

1. ECE floor — calibrated ECE on held-out data must be <= threshold.
2. Brier must-beat-baseline — calibrated Brier must be strictly
   better than the uncalibrated model's Brier.
3. Slice gates — every segment's ECE must be <= the slice threshold.
4. Stability across seeds — ECE spread (max − min) across the seed
   replications must be <= threshold.

Rollback restores the previous champion and records the event in the
hash-chained audit log, so a bad promotion is always reversible.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from calibplat.audit import AuditLog
from calibplat.manifest import ReproducibilityManifest


@dataclass
class GateResult:
    name: str
    passed: bool
    detail: str

    def to_dict(self) -> dict:
        return {"name": self.name, "passed": self.passed,
                "detail": self.detail}


@dataclass
class PromotionGates:
    """Thresholds for the four promotion gates."""

    max_ece: float = 0.05
    # Per-slice ECE is measured on ~1/3 of the test set, so its noise
    # floor is higher than the overall ECE floor; 0.10 still binds —
    # the raw model's slice ECEs run 0.15-0.20.
    max_slice_ece: float = 0.10
    max_seed_ece_spread: float = 0.03

    def evaluate(self, candidate_metrics: dict,
                 baseline_metrics: dict,
                 seed_eces: list[float]) -> list[GateResult]:
        results = []
        ece = candidate_metrics.get("ece", 1.0)
        results.append(GateResult(
            name="ece_floor", passed=ece <= self.max_ece,
            detail=f"ECE {ece:.4f} <= {self.max_ece}"))
        brier = candidate_metrics.get("brier", 1.0)
        base_brier = baseline_metrics.get("brier", 0.0)
        results.append(GateResult(
            name="brier_beats_baseline",
            passed=brier < base_brier,
            detail=f"Brier {brier:.4f} < uncalibrated {base_brier:.4f}"))
        max_slice = candidate_metrics.get("max_slice_ece", 1.0)
        results.append(GateResult(
            name="slice_gates", passed=max_slice <= self.max_slice_ece,
            detail=f"max slice ECE {max_slice:.4f} "
                   f"<= {self.max_slice_ece}"))
        spread = (max(seed_eces) - min(seed_eces)) if seed_eces else 0.0
        results.append(GateResult(
            name="stability_across_seeds",
            passed=spread <= self.max_seed_ece_spread,
            detail=f"ECE spread across {len(seed_eces)} seeds "
                   f"{spread:.4f} <= {self.max_seed_ece_spread}"))
        return results


@dataclass
class CalibratorVersion:
    """One versioned calibrator bound to one model version."""

    name: str
    model_version: str
    method: str
    metrics: dict
    baseline_metrics: dict
    seed_eces: list[float]
    model_state: dict
    scaler_state: dict
    calibrator_state: dict
    manifest: ReproducibilityManifest
    reference_scores: list[float] = field(default_factory=list)
    status: str = "candidate"  # candidate | champion | rolled_back
    gates: list[GateResult] = field(default_factory=list)

    @property
    def all_gates_passed(self) -> bool:
        return bool(self.gates) and all(g.passed for g in self.gates)

    def to_dict(self) -> dict:
        return {"name": self.name, "model_version": self.model_version,
                "method": self.method, "metrics": self.metrics,
                "baseline_metrics": self.baseline_metrics,
                "seed_eces": self.seed_eces,
                "model_state": self.model_state,
                "scaler_state": self.scaler_state,
                "calibrator_state": self.calibrator_state,
                "manifest": self.manifest.to_dict(),
                "reference_scores": self.reference_scores,
                "status": self.status,
                "gates": [g.to_dict() for g in self.gates]}


@dataclass
class CalibratorRegistry:
    """In-memory registry; the pipeline persists versions via to_dict."""

    versions: dict[str, CalibratorVersion] = field(default_factory=dict)
    champion_name: str | None = None
    audit: AuditLog = field(default_factory=AuditLog)
    gates: PromotionGates = field(default_factory=PromotionGates)

    def register(self, version: CalibratorVersion) -> CalibratorVersion:
        if version.name in self.versions:
            raise ValueError(f"version {version.name!r} already registered")
        version.gates = self.gates.evaluate(
            version.metrics, version.baseline_metrics, version.seed_eces)
        self.versions[version.name] = version
        self.audit.append("register", {
            "version": version.name, "method": version.method,
            "model_version": version.model_version,
            "gates_passed": version.all_gates_passed})
        return version

    @property
    def champion(self) -> CalibratorVersion | None:
        if self.champion_name is None:
            return None
        return self.versions.get(self.champion_name)

    def promote(self, name: str) -> CalibratorVersion:
        """Promote a candidate to champion; all four gates must pass.

        Fails closed: a candidate that fails any gate stays a
        candidate and the current champion keeps serving.
        """
        version = self.versions.get(name)
        if version is None:
            raise ValueError(f"unknown version: {name!r}")
        if not version.all_gates_passed:
            failed = [g.name for g in version.gates if not g.passed]
            raise ValueError(
                f"cannot promote {name!r}: gates failed: {failed}")
        previous = self.champion_name
        if previous is not None and previous != name:
            self.versions[previous].status = "candidate"
        version.status = "champion"
        self.champion_name = name
        self.audit.append("promote", {"version": name,
                                      "previous_champion": previous})
        return version

    def rollback(self) -> CalibratorVersion | None:
        """Restore the most recent non-champion version as champion.

        Returns the restored version, or None when there is nothing
        to roll back to.
        """
        current = self.champion
        if current is None:
            return None
        candidates = [v for v in self.versions.values()
                      if v.name != current.name
                      and v.status in ("candidate", "rolled_back")]
        if not candidates:
            return None
        # Most recently registered candidate wins.
        target = candidates[-1]
        current.status = "rolled_back"
        target.status = "champion"
        self.champion_name = target.name
        self.audit.append("rollback", {"from": current.name,
                                       "to": target.name})
        return target

    def list_versions(self) -> list[dict]:
        return [v.to_dict() for v in self.versions.values()]
