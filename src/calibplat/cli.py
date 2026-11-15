"""Command-line interface for the calibration platform.

Subcommands:
  calibrate  — run the full pipeline and print the report
  predict    — one calibrated prediction for a comma-separated
               feature vector, using a freshly built champion
  drift      — run the drift scenario end to end
  demo       — run scripts/demo.py
"""

from __future__ import annotations

import argparse
import json
import sys

from calibplat.pipeline import (RunConfig, run_calibration_pipeline,
                                run_drift_recalibration)
from calibplat.registry import CalibratorRegistry
from calibplat.serving import ServingConfig, ServingLayer


def _print_report(report: dict) -> None:
    raw = report["raw_metrics"]
    cal = report["calibrated_metrics"]
    print(f"version:   {report['version']} ({report['method']} / "
          f"{report['model_kind']})")
    print(f"raw:       ECE {raw['ece']:.4f}  MCE {raw['mce']:.4f}  "
          f"Brier {raw['brier']:.4f}  log-loss {raw['log_loss']:.4f}")
    print(f"calibrated: ECE {cal['ece']:.4f}  MCE {cal['mce']:.4f}  "
          f"Brier {cal['brier']:.4f}  log-loss {cal['log_loss']:.4f}")
    print(f"seed ECEs: {[round(e, 4) for e in report['seed_eces']]}")
    for gate in report["gates"]:
        mark = "PASS" if gate["passed"] else "FAIL"
        print(f"  gate {gate['name']}: {mark} — {gate['detail']}")
    print(f"promoted:  {report['promoted']}")
    print(f"manifest valid: {report['manifest_valid']}")


def cmd_calibrate(args: argparse.Namespace) -> int:
    config = RunConfig(method=args.method, model_kind=args.model,
                       seed=args.seed)
    report = run_calibration_pipeline(config)
    _print_report(report)
    if args.json:
        print(json.dumps(
            {"version": report["version"],
             "raw": report["raw_metrics"],
             "calibrated": report["calibrated_metrics"],
             "gates": report["gates"],
             "promoted": report["promoted"]}, indent=2))
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    features = [float(v) for v in args.features.split(",")]
    registry = CalibratorRegistry()
    run_calibration_pipeline(RunConfig(), registry=registry)
    serving = ServingLayer(registry, ServingConfig())
    prediction = serving.predict(features)
    print(json.dumps(prediction.to_dict(), indent=2))
    return 0


def cmd_drift(args: argparse.Namespace) -> int:
    registry = CalibratorRegistry()
    run_calibration_pipeline(RunConfig(), registry=registry)
    report = run_drift_recalibration(registry, shift=args.shift)
    drift = report["drift"]
    print(f"drift: PSI {drift['psi']:.3f}  KS {drift['ks']:.3f}  "
          f"drifted={drift['drifted']} ({drift['reason']})")
    stale = report["stale_metrics"]
    fresh = report["recalibrated_metrics"]
    print(f"stale champion on shifted data: ECE {stale['ece']:.4f}  "
          f"Brier {stale['brier']:.4f}")
    print(f"after recalibration:             ECE {fresh['ece']:.4f}  "
          f"Brier {fresh['brier']:.4f}")
    sig = report["significance"]
    print(f"Brier improvement {sig['observed_improvement']:.4f}, "
          f"bootstrap p={sig['p_value']:.3f} "
          f"(significant={sig['significant']})")
    print(f"new version: {report['new_version']}  "
          f"promoted: {report['recalibrated']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="calibplat", description="Model Calibration Platform")
    sub = parser.add_subparsers(dest="command", required=True)

    p_cal = sub.add_parser("calibrate", help="run the pipeline")
    p_cal.add_argument("--method", default="platt",
                       choices=["platt", "isotonic", "histogram",
                                "temperature"])
    p_cal.add_argument("--model", default="logistic",
                       choices=["logistic", "knn"])
    p_cal.add_argument("--seed", type=int, default=42)
    p_cal.add_argument("--json", action="store_true")
    p_cal.set_defaults(func=cmd_calibrate)

    p_pred = sub.add_parser("predict", help="one calibrated prediction")
    p_pred.add_argument("features",
                        help="comma-separated feature values, e.g. 0.1,0.2")
    p_pred.set_defaults(func=cmd_predict)

    p_drift = sub.add_parser("drift", help="run the drift scenario")
    p_drift.add_argument("--shift", type=float, default=0.8)
    p_drift.set_defaults(func=cmd_drift)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
