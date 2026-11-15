# Architecture — Model Calibration Platform

## Module map

```
src/calibplat/
├── __init__.py      public re-exports
├── data.py          seeded synthetic data (train / calibration / test),
│                    segment derivation, dataset fingerprint
├── models.py        StandardScaler, LogisticRegression (gradient
│                    descent), KNNClassifier; dict serialization
├── calibration.py   PlattScaling, IsotonicRegression (PAVA),
│                    HistogramBinning, TemperatureScaling,
│                    cross_fit_calibrate (K-fold out-of-fold)
├── metrics.py       ECE, MCE, Brier, log-loss, reliability diagrams,
│                    per-slice calibration, evaluate_calibration
├── drift.py         PSI, two-sample KS, DriftReport, RecalibrationPolicy,
│                    paired-bootstrap Brier significance check
├── registry.py      CalibratorVersion, PromotionGates (4 gates),
│                    CalibratorRegistry (register / promote / rollback)
├── audit.py         hash-chained, tamper-evident audit log
├── manifest.py      reproducibility manifest (seed + config hash +
│                    dataset fingerprint), verify()
├── serving.py       latency-budgeted ServingLayer, Wilson intervals,
│                    stale-calibrator fallback, stats
├── pipeline.py      run_calibration_pipeline, run_drift_recalibration
├── service.py       FastAPI app (health / predict / calibrators /
│                    promote / rollback / drift-check / stats)
└── cli.py           calibrate / predict / drift subcommands

tests/               pytest, one file per module + API + pipeline
scripts/demo.py      the end-to-end story with live numbers
docs/github-actions-ci.yml   CI definition (move to .github/workflows/)
```

## Data flow

1. **Generate.** `generate_dataset` draws three splits from one
   seed (default 600 train / 400 calibration / 600 test, 6 generic
   features, binary label from a known logistic ground truth). A
   three-way segment is derived from one feature for per-slice
   checks. `shift` moves every feature's mean and is the drift
   lever. Nothing is real data.
2. **Fit the base model.** Scaler + logistic regression (or k-NN).
   The production model is deliberately overconfident: its logits
   are scaled ×3, the controlled stand-in for a model that drifted
   or overfit. The scale factor is recorded in the manifest's
   `model_config`, so the simulation is transparent and
   reproducible.
3. **Calibrate, cross-fitted.** The calibration set is split into
   5 folds; each fold's probabilities come from a calibrator that
   never saw that fold, giving an unbiased OOF ECE. The shipped
   calibrator is refit on the full calibration set.
4. **Evaluate.** Raw vs calibrated ECE / MCE / Brier / log-loss on
   the test set, plus per-slice tables and the reliability diagram.
   Seed replications (42/43/44) measure stability.
5. **Gate and register.** The version (model state + scaler state +
   calibrator state + metrics + manifest) is registered; the four
   gates are evaluated at registration. Promotion requires all
   four to pass; otherwise the candidate waits and the champion
   keeps serving.
6. **Monitor and recalibrate.** Live scores are compared with the
   champion's reference scores (PSI > 0.2 or KS > 0.15 fires, or
   the scheduled prediction-count window expires). Recalibration
   fits a new calibrator on fresh labelled data from the shifted
   distribution; a paired bootstrap on Brier score must show a
   significant improvement before the new version is even
   registered as a promotion candidate.
7. **Serve.** The champion's model + calibrator answer predictions
   with a Wilson 95% interval (effective n = the champion's
   calibration-set size) and a plain-language explanation. A stale
   calibrator falls back to the uncalibrated score, flagged; a
   latency breach falls back to the 0.5 baseline, flagged.

## Component contracts

### Calibrators (`calibration.py`)

All implement `fit(scores, y) -> self`, `calibrate(scores) ->
list[float] in [0, 1]`, `to_dict() / from_dict()`. Scores are raw
model outputs (logits for logistic, logit of the smoothed positive
fraction for k-NN) — calibrators never see labels at predict time.

- **Platt:** logistic regression on the score, fitted by gradient
  descent on the standardised score and folded back to `(a, b)`.
  Two parameters, stable on small calibration sets, exact inverse
  of a linear logit distortion.
- **Isotonic:** PAVA over score-sorted calibration rows; a monotone
  step function. Most flexible, but each step is only as good as
  the rows behind it — noisy below ~1,000 calibration rows.
- **Histogram:** equal-width bins over the score range, each mapped
  to its empirical positive rate; empty bins fall back to the
  global rate. The most explainable method; the coarsest.
- **Temperature:** single temperature `T` on the logit, fitted by
  golden-section search on log-loss. `T > 1` softens overconfident
  scores. One parameter — the safest choice when calibration data
  is scarce.

### Metrics (`metrics.py`)

- **ECE:** count-weighted mean `|mean predicted − fraction
  positive|` over 10 equal-width bins. The headline number.
- **MCE:** the worst single-bin gap. Catches a calibrator that is
  fine on average and terrible in one region.
- **Brier / log-loss:** proper scoring rules; Brier is also the
  must-beat-baseline gate and the recalibration significance
  metric.
- **Per-slice:** the same table per segment value. The slice gate
  uses max slice ECE.

### Drift (`drift.py`)

- **PSI** over reference-quantile bins (bands: < 0.1 stable,
  0.1–0.2 watch, > 0.2 drift). **KS** is the max CDF gap.
  Either crossing its threshold fires.
- **RecalibrationPolicy** adds the scheduled arm: even a stable
  calibrator is refit after `max_age_predictions` served
  predictions.
- **Significance:** paired bootstrap (200 resamples, seeded) on
  per-sample Brier differences; improvement is claimed only when
  the one-sided bootstrap p-value is < 0.05.

### Registry (`registry.py`)

A version binds a model version to a calibrator: `model_state`,
`scaler_state`, `calibrator_state`, metrics, seed ECEs, manifest
and reference scores. Four gates:

1. `ece_floor` — calibrated ECE ≤ 0.05.
2. `brier_beats_baseline` — calibrated Brier strictly below the
   uncalibrated model's Brier.
3. `slice_gates` — every segment's ECE ≤ 0.10 (looser than the
   overall floor because per-slice ECE is measured on ~1/3 of the
   rows and its noise floor is higher).
4. `stability_across_seeds` — ECE spread across seed replications
   ≤ 0.03.

Promotion fails closed. Rollback restores the most recently
registered non-champion version. Every register / promote /
rollback / drift-recalibration event is appended to the
hash-chained audit log.

### Serving (`serving.py`)

- Latency budget (default 100 ms; predictions run in ~0.01–0.08 ms
  in the demo). Breach → 0.5 baseline fallback, flagged.
- Stale (drift detected or prediction-count window expired) →
  uncalibrated score, flagged `calibrated: false, stale: true`.
- Wilson 95% interval around the calibrated probability, using the
  champion's calibration-set size as the effective n.
- Every prediction carries an explanation naming the method,
  version, raw score and interval.

## Trade-offs (deliberate)

- **Cross-fitting costs K fits.** Five Platt fits are trivial;
  five isotonic fits on large calibration sets are not free, but
  the unbiased OOF estimate is worth more than the compute.
- **ECE's binning is arbitrary.** 10 equal-width bins is the
  convention; the demo prints the full reliability diagram so the
  bins are visible rather than hidden inside a single number.
- **Drift detection is univariate on scores.** PSI/KS on the score
  distribution catch the shifts that matter for calibration, but
  they do not localise *which* feature moved. Feature-level drift
  attribution is a separate concern and is intentionally out of
  scope.
- **Recalibration assumes fresh labels exist.** In production the
  shifted calibration set comes from delayed ground truth; the
  demo generates it directly. The pipeline does not invent labels.
- **No model retraining on drift.** When ranking itself degrades,
  recalibration is the wrong fix and the model must be retrained.
  This platform detects that case indirectly — recalibration that
  fails its gates leaves the old champion in place — but it does
  not retrain.

## Failure modes and how they surface

| Failure | Surface |
|---|---|
| Calibrator fitted on training data | OOF ECE looks perfect; test ECE does not — cross-fitting prevents this by construction |
| Calibration set too small | OOF ECE and test ECE disagree; gates catch it via the seed-spread and ECE floor |
| Drift with no fresh labels | Policy fires, recalibration cannot run — the serving layer keeps flagging predictions stale |
| Recalibration does not help | Bootstrap p-value ≥ 0.05; the new version is not promoted |
| Bad promotion | Rollback restores the previous champion; the audit log shows the full history |
| Audit tampering | `AuditLog.verify()` reports the first broken index |
| Manifest tampering | `ReproducibilityManifest.verify()` fails on any changed field |

## Running the pieces

```bash
make test     # 73 tests, incl. 7 API tests (need fastapi + httpx)
make demo     # full story with live numbers
PYTHONPATH=src python -m calibplat.cli calibrate --method isotonic
PYTHONPATH=src python -m calibplat.cli drift --shift 1.0
uvicorn calibplat.service:app --port 8000
```
