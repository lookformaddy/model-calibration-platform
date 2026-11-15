"""Reliability metrics for calibrated probabilities.

Expected calibration error (ECE), maximum calibration error (MCE),
Brier score and log-loss, plus reliability-diagram bins and per-slice
calibration. All functions take plain lists — no third-party code.
"""

from __future__ import annotations

import math


def _clip(p: float, eps: float = 1e-15) -> float:
    return min(max(p, eps), 1.0 - eps)


def brier_score(y: list[int], p: list[float]) -> float:
    """Mean squared error between predicted probability and outcome."""
    if not y:
        return 0.0
    return sum((pi - yi) ** 2 for pi, yi in zip(p, y)) / len(y)


def log_loss(y: list[int], p: list[float]) -> float:
    """Mean negative log-likelihood of the true class."""
    if not y:
        return 0.0
    total = 0.0
    for pi, yi in zip(p, y):
        total += -math.log(_clip(pi)) if yi == 1 else -math.log(1 - _clip(pi))
    return total / len(y)


def reliability_diagram(y: list[int], p: list[float],
                        n_bins: int = 10) -> list[dict]:
    """Binned observed-vs-predicted table for a reliability diagram.

    Equal-width bins over [0, 1]. Empty bins are kept (count 0) so the
    diagram always has ``n_bins`` rows and plots line up.
    """
    bins: list[dict] = []
    for b in range(n_bins):
        lo = b / n_bins
        hi = (b + 1) / n_bins
        idx = [i for i, pi in enumerate(p)
               if (lo <= pi < hi) or (b == n_bins - 1 and pi == 1.0)]
        if idx:
            mean_pred = sum(p[i] for i in idx) / len(idx)
            frac_pos = sum(y[i] for i in idx) / len(idx)
        else:
            mean_pred = (lo + hi) / 2
            frac_pos = 0.0
        bins.append({"bin": b, "bin_low": lo, "bin_high": hi,
                     "count": len(idx), "mean_predicted": mean_pred,
                     "fraction_positive": frac_pos,
                     "gap": abs(mean_pred - frac_pos) if idx else 0.0})
    return bins


def expected_calibration_error(y: list[int], p: list[float],
                               n_bins: int = 10) -> float:
    """ECE: count-weighted mean |mean predicted - fraction positive|."""
    if not y:
        return 0.0
    bins = reliability_diagram(y, p, n_bins)
    return sum(b["count"] / len(y) * b["gap"] for b in bins)


def maximum_calibration_error(y: list[int], p: list[float],
                              n_bins: int = 10) -> float:
    """MCE: the worst single-bin calibration gap."""
    bins = reliability_diagram(y, p, n_bins)
    populated = [b for b in bins if b["count"] > 0]
    return max((b["gap"] for b in populated), default=0.0)


def per_slice_calibration(y: list[int], p: list[float],
                          segments: list[int],
                          n_bins: int = 10) -> dict[int, dict]:
    """ECE / Brier / log-loss per segment value.

    A calibrator that is fine on average but badly wrong for one
    segment is not shippable; the slice gate in the registry uses
    exactly this table.
    """
    out: dict[int, dict] = {}
    for seg in sorted(set(segments)):
        idx = [i for i, s in enumerate(segments) if s == seg]
        ys = [y[i] for i in idx]
        ps = [p[i] for i in idx]
        out[seg] = {"n": len(idx),
                    "ece": expected_calibration_error(ys, ps, n_bins),
                    "mce": maximum_calibration_error(ys, ps, n_bins),
                    "brier": brier_score(ys, ps),
                    "log_loss": log_loss(ys, ps),
                    "positive_rate": sum(ys) / max(1, len(ys))}
    return out


def evaluate_calibration(y: list[int], p: list[float],
                         segments: list[int] | None = None,
                         n_bins: int = 10) -> dict:
    """One-shot summary dict used by the pipeline, registry and API."""
    result = {"ece": expected_calibration_error(y, p, n_bins),
              "mce": maximum_calibration_error(y, p, n_bins),
              "brier": brier_score(y, p),
              "log_loss": log_loss(y, p),
              "n": len(y),
              "reliability_diagram": reliability_diagram(y, p, n_bins)}
    if segments is not None:
        result["per_slice"] = per_slice_calibration(y, p, segments, n_bins)
        result["max_slice_ece"] = max(
            (s["ece"] for s in result["per_slice"].values()), default=0.0)
    return result
