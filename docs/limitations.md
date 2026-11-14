# Limitations and scientific interpretation

## What a detection means

A **model-detected anomaly** is an interval where a model's smoothed error crossed its threshold. It
is an unusual pattern that warrants investigation. It is **not** a diagnosis, and it is not proof of a
hardware malfunction. The application never claims that a detection is a confirmed failure or a
previously unknown spacecraft failure.

The app distinguishes:

* **model-detected anomaly** (violet),
* **expert-labelled anomaly** (amber, hatched) — a benchmark reference interval,
* **normal telemetry** — neither flagged nor labelled (unlabelled ≠ guaranteed normal),
* **uncertain or unexplained behaviour** — a detection without label overlap. It counts as a false
  positive in benchmark metrics, but may still be interesting.

## Dataset

* Anonymised channels, anonymised time, no units. Channel semantics cannot be interpreted beyond the
  documented type letter.
* The stated test-set-based scaling is a look-ahead built into the benchmark. Several channels
  nevertheless exceed (−1, 1), and 14 channels have constant training data.
* Labels: interval annotations from incident reports. Their boundaries are approximate, some are
  long, and labelled points make up ~13 % (SMAP) / ~12 % (MSL) of the evaluated region, which is
  dense for an anomaly benchmark.
* The benchmark has been criticised for triviality and possible mislabelling (Wu & Keogh, 2023).
* Training splits are not guaranteed anomaly-free.
* `P-2` is duplicated in the label file; this project merges the rows.

## Models and thresholds

* Per-channel models with a univariate target ignore relationships between channels.
* A one-step forecaster can adapt to slow drift and miss it; very short training splits (312 points
  for `D-12`) give weak models.
* The autoencoder may learn to reconstruct anomalies present in the training split.
* NDT has hand-set constants (window, buffer, p, guards) inherited from Telemanom; using published
  defaults on the same benchmark is itself mild indirect tuning.
* The interval score is a margin, not a probability; no calibrated confidence is shown.

## Computation and deployment

* The reference run used the "standard" LSTM profile, not the full "paper" profile, to keep CPU time
  reasonable. Users can run the paper profile from the laboratory.
* On a laptop CPU, LSTM training takes minutes per channel. An all-channel LSTM run can take hours.
* Floating-point results differ slightly across platforms and PyTorch builds.
* The Windows scripts were syntax-checked with PowerShell 7.4 on Linux but could not be executed on
  Windows in the build environment. See the acceptance report in the README.

## Comparisons

Metrics are not directly comparable with the paper (different training profile, early stopping
with restored weights, thresholding deviations, `P-2` handling). Comparisons *between* methods inside
this application are fair: same channels, same evaluation region, same smoothing, thresholding and
evaluation code.
