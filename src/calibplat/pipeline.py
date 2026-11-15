"""End-to-end calibration pipeline.

One run: generate seeded synthetic data, train a base classifier,
deliberately overconfident (logits scaled by ``overconfidence`` — the
controlled stand-in for a production model that drifted or overfit),
calibrate it cross-fitted on a held-out calibration set, evaluate the
calibrated probabilities on a test set, check stability across seeds,
and register the version with the promotion gates applied.

A second entry point, :func:`run_drift_recalibration`, replays the
drift scenario: fresh data from a shifted distribution, drift
detection against the champion's reference scores, recalibration on
the shifted calibration set, and a paired-bootstrap significance
check on the Brier improvement.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from calibplat.calibration import cross_fit_calibrate
from calibplat.data import Dataset, generate_dataset, sigmoid
from calibplat.drift import (detect_drift,
                             paired_bootstrap_brier_improvement)
from calibplat.manifest import ReproducibilityManifest
from calibplat.metrics import evaluate_calibration
from calibplat.models import KNNClassifier, LogisticRegression, StandardScaler
from calibplat.registry import (CalibratorRegistry, CalibratorVersion,
                                PromotionGates)


@dataclass
class RunConfig:
    method: str = "platt"
    model_kind: str = "logistic"
    seed: int = 42
    n_train: int = 600
    n_cal: int = 400
    n_test: int = 600
    # Controlled miscalibration: the production model's logits are
    # scaled by this factor, simulating an overconfident model.
    overconfidence: float = 3.0
    epochs: int = 300
    knn_k: int = 15
    n_folds: int = 5
    stability_seeds: tuple[int, ...] = (43, 44)
    gates: PromotionGates = field(default_factory=PromotionGates)


def _fit_base(config: RunConfig, data: Dataset):
    """Fit scaler + base classifier; apply the overconfidence scale."""
    scaler = StandardScaler().fit(data.X)
    X_train = scaler.transform(data.X)
    if config.model_kind == "logistic":
        model = LogisticRegression().fit(X_train, data.y,
                                         epochs=config.epochs)
        if config.overconfidence != 1.0:
            model.weights = [w * config.overconfidence
                             for w in model.weights]
            model.intercept *= config.overconfidence
    elif config.model_kind == "knn":
        model = KNNClassifier(k=config.knn_k).fit(X_train, data.y)
    else:
        raise ValueError(f"unknown model_kind: {config.model_kind!r}")
    return scaler, model


def _scores(scaler: StandardScaler, model, X) -> list[float]:
    return model.decision_scores(scaler.transform(X))


def _seed_ece(config: RunConfig, seed: int) -> float:
    """ECE of a full train+calibrate replication under another seed."""
    data = generate_dataset(n_train=config.n_train, n_cal=config.n_cal,
                            n_test=config.n_test, seed=seed)
    scaler, model = _fit_base(config, data)
    cal_scores = _scores(scaler, model, data.X_cal)
    result = cross_fit_calibrate(config.method, cal_scores, data.y_cal,
                                 n_folds=config.n_folds, seed=seed)
    test_scores = _scores(scaler, model, data.X_test)
    test_probs = result.calibrator.calibrate(test_scores)
    return evaluate_calibration(data.y_test, test_probs)["ece"]


def run_calibration_pipeline(
        config: RunConfig | None = None,
        registry: CalibratorRegistry | None = None,
        data: Dataset | None = None) -> dict:
    """Run one full calibration cycle and register the version.

    Returns a report dict with raw vs calibrated metrics, gate
    results, the version name and whether it was promoted.
    """
    config = config or RunConfig()
    registry = registry or CalibratorRegistry(gates=config.gates)
    data = data or generate_dataset(
        n_train=config.n_train, n_cal=config.n_cal, n_test=config.n_test,
        seed=config.seed)

    scaler, model = _fit_base(config, data)

    cal_scores = _scores(scaler, model, data.X_cal)
    cross_fit = cross_fit_calibrate(config.method, cal_scores, data.y_cal,
                                    n_folds=config.n_folds,
                                    seed=config.seed)
    # Unbiased (out-of-fold) estimate of post-calibration reliability.
    oof_metrics = evaluate_calibration(data.y_cal, cross_fit.oof_probs,
                                       data.segments_cal)

    test_scores = _scores(scaler, model, data.X_test)
    raw_probs = [sigmoid(s) for s in test_scores]
    raw_metrics = evaluate_calibration(data.y_test, raw_probs,
                                       data.segments_test)
    cal_probs = cross_fit.calibrator.calibrate(test_scores)
    cal_metrics = evaluate_calibration(data.y_test, cal_probs,
                                       data.segments_test)

    seed_eces = [cal_metrics["ece"]]
    for seed in config.stability_seeds:
        seed_eces.append(_seed_ece(config, seed))

    version_name = (f"{config.method}-{config.model_kind}-"
                    f"v1-seed{config.seed}")
    manifest = ReproducibilityManifest.build(
        version_name=version_name,
        dataset_fingerprint=data.fingerprint(),
        config={"method": config.method, "n_folds": config.n_folds,
                "overconfidence": config.overconfidence},
        model_config={"model_kind": config.model_kind,
                      "epochs": config.epochs,
                      "overconfidence_factor": config.overconfidence},
        calibrator_config=cross_fit.calibrator.to_dict(),
        seeds={"data": config.seed, "run": config.seed,
               "stability": list(config.stability_seeds)})
    version = CalibratorVersion(
        name=version_name, model_version=f"{config.model_kind}-seed{config.seed}",
        method=config.method, metrics=cal_metrics,
        baseline_metrics=raw_metrics, seed_eces=seed_eces,
        model_state=model.to_dict(), scaler_state=scaler.to_dict(),
        calibrator_state=cross_fit.calibrator.to_dict(),
        manifest=manifest, reference_scores=cal_scores)
    registry.register(version)
    promoted = False
    promotion_error = None
    if version.all_gates_passed:
        registry.promote(version_name)
        promoted = True
    else:
        promotion_error = [g.name for g in version.gates if not g.passed]

    return {"version": version_name, "method": config.method,
            "model_kind": config.model_kind, "seed": config.seed,
            "raw_metrics": raw_metrics, "calibrated_metrics": cal_metrics,
            "oof_metrics": oof_metrics, "seed_eces": seed_eces,
            "gates": [g.to_dict() for g in version.gates],
            "promoted": promoted, "promotion_error": promotion_error,
            "manifest_valid": manifest.verify(),
            "registry": registry, "version_obj": version,
            "data": data, "scaler": scaler, "model": model,
            "calibrator": cross_fit.calibrator}


def run_drift_recalibration(registry: CalibratorRegistry,
                            config: RunConfig | None = None,
                            shift: float = 0.8,
                            seed: int = 123) -> dict:
    """Drift scenario: detect, recalibrate, verify the improvement.

    Fresh data arrives from a shifted distribution. The champion's
    reference scores are compared with the new scores (PSI / KS); if
    drift fires, a new calibrator is fitted on the shifted calibration
    set and the Brier improvement is significance-checked with a
    paired bootstrap before the new version is registered.
    """
    config = config or RunConfig()
    champion = registry.champion
    if champion is None:
        raise ValueError("no champion calibrator to drift-check against")
    # Recalibration needs enough fresh labels for a stable fit and a
    # stable gate evaluation: ECE measured on a few hundred rows
    # wobbles by ~0.02-0.05 even for a well-calibrated model, so the
    # shifted sets are deliberately larger than the pipeline defaults.
    shifted = generate_dataset(n_train=config.n_train,
                               n_cal=max(config.n_cal, 1000),
                               n_test=max(config.n_test, 3000),
                               seed=seed, shift=shift)
    # The champion's model stays FIXED — only the data distribution
    # moves. That is the production reality: the model keeps serving
    # while its input distribution drifts, and calibration is what
    # has to be refreshed.
    from calibplat.calibration import load_calibrator
    from calibplat.models import StandardScaler, load_model
    scaler = StandardScaler.from_dict(champion.scaler_state)
    model = load_model(champion.model_state)
    new_cal_scores = _scores(scaler, model, shifted.X_cal)
    drift = detect_drift(champion.reference_scores, new_cal_scores)

    # What the stale champion would have scored on the shifted data.
    old_calibrator = load_calibrator(champion.calibrator_state)
    test_scores = _scores(scaler, model, shifted.X_test)
    stale_probs = old_calibrator.calibrate(test_scores)
    stale_metrics = evaluate_calibration(shifted.y_test, stale_probs,
                                         shifted.segments_test)

    result = cross_fit_calibrate(config.method, new_cal_scores,
                                 shifted.y_cal,
                                 n_folds=config.n_folds, seed=seed)
    fresh_probs = result.calibrator.calibrate(test_scores)
    fresh_metrics = evaluate_calibration(shifted.y_test, fresh_probs,
                                         shifted.segments_test)
    significance = paired_bootstrap_brier_improvement(
        shifted.y_test, stale_probs, fresh_probs)

    report: dict = {"drift": drift.to_dict(),
                    "stale_metrics": stale_metrics,
                    "recalibrated_metrics": fresh_metrics,
                    "significance": significance,
                    "recalibrated": False, "new_version": None}
    if not drift.drifted:
        return report
    version_name = (f"{config.method}-{config.model_kind}-"
                    f"v2-drift-seed{seed}")
    manifest = ReproducibilityManifest.build(
        version_name=version_name,
        dataset_fingerprint=shifted.fingerprint(),
        config={"method": config.method, "n_folds": config.n_folds,
                "drift_shift": shift, "recalibration": True},
        model_config={"model_kind": config.model_kind,
                      "epochs": config.epochs,
                      "overconfidence_factor": config.overconfidence},
        calibrator_config=result.calibrator.to_dict(),
        seeds={"data": seed, "run": seed})
    raw_probs = [sigmoid(s) for s in test_scores]
    raw_metrics = evaluate_calibration(shifted.y_test, raw_probs,
                                       shifted.segments_test)
    version = CalibratorVersion(
        name=version_name,
        model_version=f"{config.model_kind}-drift-seed{seed}",
        method=config.method, metrics=fresh_metrics,
        baseline_metrics=raw_metrics, seed_eces=[fresh_metrics["ece"]],
        model_state=model.to_dict(), scaler_state=scaler.to_dict(),
        calibrator_state=result.calibrator.to_dict(),
        manifest=manifest, reference_scores=new_cal_scores)
    registry.register(version)
    report["new_version"] = version_name
    report["gates"] = [g.to_dict() for g in version.gates]
    if version.all_gates_passed:
        registry.promote(version_name)
        report["recalibrated"] = True
    registry.audit.append("drift_recalibration", {
        "drift": drift.to_dict(), "new_version": version_name,
        "promoted": report["recalibrated"],
        "brier_improvement": significance["observed_improvement"]})
    return report
