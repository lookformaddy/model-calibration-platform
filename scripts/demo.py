"""End-to-end demo: a miscalibrated model, corrected.

1. A deliberately overconfident logistic model (logits x3) scores
   ECE ~0.15+ — probabilities that cannot be trusted.
2. Four calibration methods are compared; Platt scaling is promoted
   through all four gates and becomes the champion.
3. A drift event (shifted input distribution) degrades the champion;
   PSI/KS fire, recalibration restores calibration, and a paired
   bootstrap confirms the Brier improvement is significant.
4. Serving returns calibrated probabilities with Wilson intervals
   and an explanation; a stale calibrator falls back to uncalibrated
   scores with an explicit flag.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from calibplat.calibration import cross_fit_calibrate  # noqa: E402
from calibplat.data import generate_dataset, sigmoid  # noqa: E402
from calibplat.metrics import evaluate_calibration  # noqa: E402
from calibplat.pipeline import (RunConfig, _fit_base, _scores,  # noqa: E402
                                run_calibration_pipeline,
                                run_drift_recalibration)
from calibplat.registry import CalibratorRegistry  # noqa: E402
from calibplat.serving import ServingConfig, ServingLayer  # noqa: E402


def _row(label: str, m: dict) -> None:
    print(f"  {label:<12} ECE {m['ece']:.4f}  MCE {m['mce']:.4f}  "
          f"Brier {m['brier']:.4f}  log-loss {m['log_loss']:.4f}")


def main() -> int:
    print("=== 1. The miscalibrated model ===")
    # Larger test set (2,000) so the reported ECE is stable, not
    # sampling noise: with 10 bins, ECE on a few hundred rows wobbles
    # by ~0.02 even for a well-calibrated model.
    config = RunConfig(n_train=800, n_cal=600, n_test=2000)
    data = generate_dataset(n_train=800, n_cal=600, n_test=2000,
                            seed=config.seed)
    scaler, model = _fit_base(config, data)
    test_scores = _scores(scaler, model, data.X_test)
    raw_probs = [sigmoid(s) for s in test_scores]
    raw = evaluate_calibration(data.y_test, raw_probs, data.segments_test)
    _row("raw (x3 logits)", raw)
    print("  The model is systematically overconfident: probabilities "
          "near 0/1 far exceed the observed rates.")
    print()

    print("=== 2. Calibration methods (cross-fitted, 5-fold) ===")
    cal_scores = _scores(scaler, model, data.X_cal)
    for method in ("platt", "isotonic", "histogram", "temperature"):
        result = cross_fit_calibrate(method, cal_scores, data.y_cal,
                                     n_folds=5, seed=config.seed)
        oof = evaluate_calibration(data.y_cal, result.oof_probs,
                                   data.segments_cal)
        test_probs = result.calibrator.calibrate(test_scores)
        test_m = evaluate_calibration(data.y_test, test_probs,
                                      data.segments_test)
        print(f"  {method:<12} OOF ECE {oof['ece']:.4f}  "
              f"test ECE {test_m['ece']:.4f}  "
              f"test Brier {test_m['brier']:.4f}")
    print()

    print("=== 3. Promotion through the four gates (Platt) ===")
    registry = CalibratorRegistry()
    report = run_calibration_pipeline(config, registry=registry)
    _row("raw", report["raw_metrics"])
    _row("calibrated", report["calibrated_metrics"])
    print(f"  cross-fitted OOF ECE: "
          f"{report['oof_metrics']['ece']:.4f}")
    print(f"  seed ECEs (42/43/44): "
          f"{[round(e, 4) for e in report['seed_eces']]}")
    for gate in report["gates"]:
        mark = "PASS" if gate["passed"] else "FAIL"
        print(f"  gate {gate['name']}: {mark} — {gate['detail']}")
    print(f"  promoted: {report['promoted']}  "
          f"champion: {registry.champion_name}")
    print()

    print("=== 4. Reliability diagram (raw vs calibrated) ===")
    cal_probs = report["calibrator"].calibrate(test_scores)
    print("  bin   raw_pred  raw_obs   cal_pred  cal_obs   n")
    raw_bins = raw["reliability_diagram"]
    cal_bins = report["calibrated_metrics"]["reliability_diagram"]
    for rb, cb in zip(raw_bins, cal_bins):
        if rb["count"] == 0:
            continue
        print(f"  {rb['bin_low']:.1f}-{rb['bin_high']:.1f}  "
              f"{rb['mean_predicted']:.3f}    {rb['fraction_positive']:.3f}   "
              f"{cb['mean_predicted']:.3f}    {cb['fraction_positive']:.3f}   "
              f"{rb['count']}")
    print()

    print("=== 5. Drift event triggers recalibration ===")
    drift_report = run_drift_recalibration(registry, shift=1.0)
    d = drift_report["drift"]
    print(f"  PSI {d['psi']:.3f}  KS {d['ks']:.3f}  "
          f"drifted={d['drifted']} ({d['reason']})")
    _row("stale champion", drift_report["stale_metrics"])
    _row("recalibrated", drift_report["recalibrated_metrics"])
    sig = drift_report["significance"]
    print(f"  Brier improvement {sig['observed_improvement']:.4f}, "
          f"bootstrap p={sig['p_value']:.3f} "
          f"(significant={sig['significant']})")
    print(f"  new version {drift_report['new_version']} promoted: "
          f"{drift_report['recalibrated']}")
    print()

    print("=== 6. Serving (calibrated, with intervals) ===")
    serving = ServingLayer(registry, ServingConfig())
    for features in (data.X_test[0], data.X_test[1], data.X_test[2]):
        pred = serving.predict(features)
        lo, hi = pred.confidence_interval
        print(f"  p={pred.probability:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  "
              f"method={pred.method}  version={pred.version}  "
              f"latency={pred.latency_ms:.2f} ms")
    print(f"  explanation: {serving.predict(data.X_test[0]).explanation}")
    print()

    print("=== 7. Stale calibrator falls back, explicitly ===")
    serving.mark_stale(True)
    stale_pred = serving.predict(data.X_test[0])
    print(f"  calibrated={stale_pred.calibrated}  stale={stale_pred.stale}  "
          f"p={stale_pred.probability:.3f} (uncalibrated score)")
    serving.mark_stale(False)
    print()

    print("=== 8. Rollback ===")
    restored = registry.rollback()
    print(f"  rolled back to: {restored.name if restored else None}  "
          f"champion now: {registry.champion_name}")
    ok, broken = registry.audit.verify()
    print(f"  audit log: {len(registry.audit.entries)} entries, "
          f"chain valid: {ok}")
    print()
    print("Demo complete: miscalibration corrected, drift handled, "
          "promotion gated, serving explained.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
