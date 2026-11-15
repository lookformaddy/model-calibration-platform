# Model Calibration Platform

A probability-calibration platform built entirely from scratch (pure
Python stdlib — no sklearn, numpy or torch): from-scratch logistic
regression and k-NN classifiers, four calibration methods (Platt
scaling, isotonic regression, histogram binning, temperature scaling)
with cross-fitted fitting, reliability metrics (ECE, MCE, Brier,
log-loss), per-slice calibration, PSI/KS drift-triggered
recalibration, a versioned calibrator registry with four promotion
gates and rollback, a hash-chained audit log, reproducibility
manifests and latency-budgeted serving — all on seeded synthetic
data.

## The enterprise problem

A classifier that says "90% confident" should be right about 90% of
the time. In production it usually isn't: models trained on stale
data, overfit models and models served under distribution shift are
systematically over- or under-confident, and every downstream
decision — thresholds, ranking, human review queues, risk scores —
inherits that error. Calibration fixes the probabilities without
retraining the model, but only if it is governed: the calibrator
must be fitted on held-out data (never the training set), its
reliability must be measured out-of-fold, it must hold per segment
as well as on average, it must be re-fitted when the input
distribution moves, and a new version must prove itself against
explicit gates before it serves traffic. This project is that loop,
end to end.

## Demo results (computed live, `scripts/demo.py`)

A deliberately overconfident logistic model (logits ×3, simulating a
production model that drifted) on 800 train / 600 calibration /
2,000 test rows, seed 42:

| Model | ECE | MCE | Brier | Log-loss |
|---|---|---|---|---|
| Raw (overconfident) | **0.1576** | 0.2101 | 0.2093 | 0.7319 |
| Platt scaling | **0.0166** | 0.0495 | 0.1814 | 0.5381 |
| Isotonic regression | 0.0503 | — | 0.1856 | — |
| Histogram binning | 0.0414 | — | 0.1852 | — |
| Temperature scaling (T=2.84) | 0.0167 | — | 0.1812 | — |

Cross-fitted (5-fold) out-of-fold ECE for Platt is 0.0446 — the
honest estimate, not the in-sample one. The reliability diagram
tells the story in one row: in the 0.9–1.0 bin the raw model predicts
0.973 but is right only 0.808 of the time; after Platt it predicts
0.937 and is right 0.959.

**Promotion:** Platt passes all four gates (ECE 0.0166 ≤ 0.05; Brier
0.1814 < uncalibrated 0.2093; max slice ECE 0.0321 ≤ 0.10; ECE spread
across seeds 42/43/44 is 0.0290 ≤ 0.03) and is promoted to champion.

**Drift:** fresh data from a shifted distribution (shift 1.0) trips
the monitor — PSI 0.147, **KS 0.159 > 0.15** — and the stale champion
degrades to ECE 0.0782. Recalibration on 1,000 fresh labelled rows
restores ECE 0.0243; the paired bootstrap confirms the Brier
improvement (+0.0073) is significant (p = 0.000), and the new
version `platt-logistic-v2-drift-seed123` is promoted through the
gates. Rollback restores the previous champion, and the hash-chained
audit log (6 entries) verifies clean.

**Serving:** predictions return in ~0.01–0.08 ms with a 95% Wilson
interval and an explanation of the calibration applied; a stale
calibrator falls back to the uncalibrated score with an explicit
`calibrated: false, stale: true` flag instead of silently serving
stale probabilities.

## Architecture overview

```
seeded synthetic data ──► base classifier (logistic / k-NN, x3 logits)
                                  │
                                  ▼
                    calibration set ──► cross-fitted calibrator
                                  │        (Platt / isotonic /
                                  │         histogram / temperature)
                                  ▼
                    reliability metrics (ECE / MCE / Brier / log-loss,
                                         per-slice, reliability diagram)
                                  │
                                  ▼
              versioned registry ──► 4 promotion gates ──► champion
                                  │                              │
                                  ▼                              ▼
                    drift monitor (PSI / KS) ──► recalibration
                                                              │
                                                              ▼
                              latency-budgeted serving ──► fallback
                              (Wilson CI + explanation)    (uncalibrated,
                                                            flagged)
```

See [`docs/architecture.md`](docs/architecture.md) for the full design,
including the module map, the per-component contracts and the
trade-offs.

## How to run

```bash
pip install -r requirements.txt
make test        # full pytest suite
make demo        # the end-to-end story above
make cli         # CLI help

# Full pipeline (Platt on logistic, default seed 42)
PYTHONPATH=src python -m calibplat.cli calibrate

# One calibrated prediction
PYTHONPATH=src python -m calibplat.cli predict 0.1,-0.2,0.3,0.0,0.5,-0.1

# Drift scenario end to end
PYTHONPATH=src python -m calibplat.cli drift --shift 1.0

# API service
uvicorn calibplat.service:app --host 0.0.0.0 --port 8000
```

API endpoints: `GET /health`, `POST /predict`, `GET /calibrators`,
`POST /promote`, `POST /rollback`, `POST /drift-check`, `GET /stats`.

## Key design decisions and trade-offs

- **Cross-fitted calibration, not in-sample.** A calibrator fitted
  and evaluated on the same rows looks perfect and is lying.
  Out-of-fold probabilities are the honest reliability estimate;
  the shipped calibrator is then refit on the full calibration set.
- **Platt as the default, isotonic as the check.** Platt is a
  two-parameter logistic fit on the score — it inverts a linear
  logit distortion almost exactly and is stable on a few hundred
  rows. Isotonic (PAVA) is more flexible but noisier; histogram
  binning is the most interpretable and the coarsest. The demo
  compares all four on the same data instead of asserting one.
- **ECE is noisy on small samples.** ECE on a few hundred rows
  wobbles by ~0.02–0.05 even for a well-calibrated model, and
  per-slice ECE more so. The pipeline uses 600/400/600 splits and
  the drift scenario uses 1,000 calibration / 3,000 test rows
  precisely so the gates measure signal, not noise. The slice gate
  (0.10) is looser than the overall ECE floor (0.05) for the same
  reason.
- **Drift is a recalibration trigger, not a retraining trigger.**
  The base model stays fixed while the distribution moves; only
  the calibration mapping is refreshed on fresh labels. That is
  cheaper than retraining and is the correct fix when the model's
  ranking is intact but its probabilities have shifted.
- **Stale calibrators fail loudly.** Serving never silently serves
  a stale calibrator's probabilities: it falls back to the
  uncalibrated score and flags the prediction. A wrong-but-flagged
  probability is operable; a wrong-and-silent one is not.
- **Gates fail closed.** A candidate that fails any of the four
  gates stays a candidate and the current champion keeps serving.
  Rollback is always available and every event is hash-chained in
  the audit log.
- **Pure stdlib core.** The calibration, metrics, drift and
  registry logic has no third-party dependencies; FastAPI is only
  the service layer. The whole platform is auditable by reading
  it.

## What it demonstrates

End-to-end probability calibration as a production concern:
miscalibration diagnosis (ECE/MCE/reliability diagrams), four
calibration methods with cross-fitting, per-slice fairness-style
checks, drift monitoring (PSI/KS) with significance-checked
recalibration, gated promotion and rollback of versioned
calibrators, tamper-evident audit logging, reproducibility
manifests, and latency-budgeted serving with confidence intervals
and explanations — all from scratch, on synthetic data.
```

