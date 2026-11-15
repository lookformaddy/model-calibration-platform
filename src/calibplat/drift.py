"""Drift monitoring and the recalibration policy.

Population Stability Index (PSI) and the two-sample Kolmogorov–Smirnov
statistic compare the live score distribution against the reference
distribution the calibrator was fitted on. When either crosses its
threshold — or the calibrator ages past its scheduled window — the
policy fires and the pipeline recalibrates on fresh labelled data.
A paired bootstrap compares Brier score before vs after recalibration
so a recalibration is only claimed to help when the improvement is
real, not noise.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from calibplat.metrics import brier_score

# Conventional PSI bands: < 0.1 stable, 0.1–0.2 watch, > 0.2 drift.
PSI_DRIFT_THRESHOLD = 0.2
KS_DRIFT_THRESHOLD = 0.15


def psi(reference: list[float], current: list[float],
        n_bins: int = 10) -> float:
    """Population Stability Index over quantile bins of the reference."""
    if not reference or not current:
        return 0.0
    ref = sorted(reference)
    edges = [ref[int(len(ref) * i / n_bins)] for i in range(n_bins)]
    edges.append(ref[-1] + 1e-9)
    total = 0.0
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        ref_frac = sum(1 for v in reference if lo <= v < hi) / len(reference)
        cur_frac = sum(1 for v in current if lo <= v < hi) / len(current)
        ref_frac = max(ref_frac, 1e-4)
        cur_frac = max(cur_frac, 1e-4)
        total += (cur_frac - ref_frac) * math.log(cur_frac / ref_frac)
    return total


def ks_statistic(reference: list[float], current: list[float]) -> float:
    """Two-sample KS statistic: max gap between the empirical CDFs."""
    if not reference or not current:
        return 0.0
    points = sorted(set(reference) | set(current))
    worst = 0.0
    for v in points:
        cdf_ref = sum(1 for x in reference if x <= v) / len(reference)
        cdf_cur = sum(1 for x in current if x <= v) / len(current)
        worst = max(worst, abs(cdf_ref - cdf_cur))
    return worst


@dataclass
class DriftReport:
    psi: float
    ks: float
    n_reference: int
    n_current: int
    drifted: bool
    reason: str = ""

    def to_dict(self) -> dict:
        return {"psi": self.psi, "ks": self.ks,
                "n_reference": self.n_reference,
                "n_current": self.n_current, "drifted": self.drifted,
                "reason": self.reason}


def detect_drift(reference_scores: list[float],
                 current_scores: list[float],
                 psi_threshold: float = PSI_DRIFT_THRESHOLD,
                 ks_threshold: float = KS_DRIFT_THRESHOLD) -> DriftReport:
    """Compare live scores against the reference distribution."""
    p = psi(reference_scores, current_scores)
    k = ks_statistic(reference_scores, current_scores)
    reasons = []
    if p > psi_threshold:
        reasons.append(f"PSI {p:.3f} > {psi_threshold}")
    if k > ks_threshold:
        reasons.append(f"KS {k:.3f} > {ks_threshold}")
    return DriftReport(psi=p, ks=k, n_reference=len(reference_scores),
                       n_current=len(current_scores),
                       drifted=bool(reasons),
                       reason="; ".join(reasons) or "stable")


@dataclass
class RecalibrationPolicy:
    """When to recalibrate: drift thresholds plus a scheduled window.

    ``max_age_predictions`` is the scheduled arm — even a stable
    calibrator is refit after this many served predictions, because
    silent drift is the failure mode that hurts most.
    """

    psi_threshold: float = PSI_DRIFT_THRESHOLD
    ks_threshold: float = KS_DRIFT_THRESHOLD
    max_age_predictions: int = 10_000

    def needs_recalibration(self, report: DriftReport,
                            predictions_since_fit: int) -> tuple[bool, str]:
        if report.drifted:
            return True, f"drift detected: {report.reason}"
        if predictions_since_fit >= self.max_age_predictions:
            return True, (f"scheduled: {predictions_since_fit} predictions "
                          f"since fit >= {self.max_age_predictions}")
        return False, "stable"


def paired_bootstrap_brier_improvement(
        y: list[int], p_before: list[float], p_after: list[float],
        n_resamples: int = 200, seed: int = 42) -> dict:
    """Paired bootstrap on per-sample Brier differences.

    Returns the observed improvement (Brier before − after, positive is
    better) and the fraction of resamples where the improvement was
    <= 0 — a one-sided bootstrap p-value for "recalibration did not
    help". Seeded, so the significance check is reproducible.
    """
    diffs = [(pb - yi) ** 2 - (pa - yi) ** 2
             for yi, pb, pa in zip(y, p_before, p_after)]
    observed = sum(diffs) / len(diffs)
    rng = random.Random(seed)
    worse_or_equal = 0
    for _ in range(n_resamples):
        sample = [rng.choice(diffs) for _ in diffs]
        if sum(sample) / len(sample) <= 0:
            worse_or_equal += 1
    return {"observed_improvement": observed,
            "p_value": worse_or_equal / n_resamples,
            "n_resamples": n_resamples,
            "significant": worse_or_equal / n_resamples < 0.05}
