# Data

This project ships no data files. All data is synthetic and generated
in-process by `calibplat.data.generate_dataset` with a fixed seed:

- Binary labels from a known logistic ground truth in a generic
  6-dimensional feature space, with a three-way segment derived from
  one feature for per-slice calibration.
- Three splits per seed: train (600), calibration (400), test (600).
- `shift` moves the mean of every feature and is the drift
  lever: data generated with `shift > 0` comes from a different input
  distribution than the model was trained on.

The base classifier is deliberately overconfident (logits scaled x3)
to simulate a miscalibrated production model in a controlled way.
Nothing here comes from any real system or employer.
