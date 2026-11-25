# Scientific methodology

This is an independent implementation inspired by Hundman et al. (KDD 2018). It does **not** claim to
reproduce the paper's results.

## 1. Problem setting

For each telemetry channel there is a training split (no labels) and a test split with expert
reference intervals. Models learn "normal" behaviour from the training split only. Detection on the
test split must not use labels. Labels are used only for evaluation afterwards.

## 2. Methods

All methods turn the test stream into a per-timestep **raw score**, which then goes through exactly
the same smoothing, thresholding and evaluation.

| Method | Raw score | Why it is included |
|---|---|---|
| **LSTM forecaster** (principal) | \|y_t − ŷ_t\|, ŷ_t from the window [t−w, t) | the approach of the paper |
| LSTM autoencoder (experimental) | mean \|x − x̂\| over the window ending at t | complementary reconstruction view |
| Ridge autoregression | \|y_t − ŷ_t\|, linear on 30 lags + commands | isolates the value of non-linearity |
| Persistence | \|y_t − y_{t−1}\| | floor any learned forecaster must beat |
| Rolling z-score | \|y_t − mean\| / sd over the previous 100 points | classic statistical thresholding, no model |
| Isolation Forest | −score_samples on 16-lag features (fit on train) | standard non-temporal detector |
| Random scores | uniform noise (seeded) | control that exposes metric inflation |

Random Forest is deliberately **not** included: the training split carries no labels, so any
supervised classifier would have to be trained on test labels, which is leakage.

### LSTM forecaster

Two stacked LSTM layers, dropout, and a linear head predicting the next `horizon` values. Inputs are
all features (telemetry + commands). Training: MSE, Adam, gradient clipping 1.0, shuffled
mini-batches, early stopping on a chronological validation split (last 20 % of training windows,
with a gap so training targets do not overlap validation inputs), and the best validation weights
are restored. Profiles:

| profile | window | hidden | layers | dropout | max epochs | patience |
|---|---|---|---|---|---|---|
| quick (UI default) | 100 | 48 | 2 | 0.3 | 8 | 3 |
| standard (reference run) | 250 | 64 | 2 | 0.3 | 20 | 5 |
| paper (Telemanom defaults) | 250 | 80 | 2 | 0.3 | 35 | 10 |

All use horizon 10, batch 64, learning rate 1e-3, min Δ 3e-4. The first predicted step is scored.

### LSTM autoencoder

Encoder LSTM (all features) → last hidden state → tanh linear bottleneck (16) → repeated over the
window → decoder LSTM → linear output reconstructing the telemetry column. Window 100, stride 1,
MSE, same early stopping. The window-mean error is causal and therefore replayable.

## 3. Common evaluation region

Every method is scored on test indices **[250, N)**. 250 is the paper's history length; no
benchmark label starts before index 290, so no labelled sequence is excluded. Labelled points before
250 would be reported as "unscoreable" (there are none in this dataset).

## 4. Smoothing and nonparametric dynamic thresholding (NDT)

Implemented in `backend/app/ml/scoring/` following `telemanom/errors.py`:

1. **EWMA smoothing**, span = batch × window × smoothing = 70 × 30 × 0.05 = 105, identical to
   `pandas.Series.ewm(span, adjust=True).mean()` and computed incrementally.
2. **Windows** of `window_size` × `batch_size` = 30 × 70 points, advancing one batch at a time. Each
   window reports only its newest batch (the first window reports all of it). Short channels shrink
   the number of batches per window, as in Telemanom.
3. **ε selection**: for z ∈ {2.5, 3.0, …, 11.5}, ε = μ + zσ (window statistics). Points ≥ ε are
   buffered by 99 points; z maximises (Δμ/μ + Δσ/σ) / (n_sequences² + n_flagged) subject to ≤ 5
   sequences and < 50 % flagged. Fallback ε = μ + 12σ.
4. **Inverse pass**: the same search on errors reflected about the window mean (2μ − e) flags
   unusually low errors.
5. **Scale guard**: a window can flag only if (σ_e > 5 % of σ_y or max e > 5 % of the 95–5
   percentile range of y) and max e > 0.05. A flagged point must also exceed 5 % of that range.
6. **Buffer** flags by 100 points and keep only the reporting zone.
7. **Pruning**: sort the peaks of the flagged sequences, append the highest non-flagged error;
   walk the chain and drop sequences whose relative drop to the next peak is < p = 0.13 (the
   last run of close peaks). Sequences of length 1 are discarded.
8. **Score** of an interval = max over its points of (e − ε) / (μ + σ) of the flagging window. It is a
   severity margin, **not** a calibrated probability.

Deliberate deviations from Telemanom:

* no initial errors are ignored (Telemanom ignores up to 2·l_s = 500 points and special-cases
  channel C-2, whose anomaly starts at index 290);
* the EWMA warm-up is not overwritten with a mean;
* the final window reports every point after the previous window's zone (Telemanom's final window
  can leave up to batch−1 points unevaluated).

The detector is streaming-capable: windows are processed as soon as complete, so the historical
replay issues verdicts with realistic latency and ends with **exactly** the offline result
(asserted by `tests/unit/test_scoring.py` and `tests/integration/test_api.py`).

An alternative **static threshold** (ε = μ + kσ of the smoothed *training-split* scores, k = 3)
is available for comparison; it uses no test information.

## 5. Evaluation

Three views, always labelled separately:

**Event-wise** (paper protocol). A labelled sequence overlapped by ≥ 1 detected sequence is a TP,
otherwise a FN; a detected sequence overlapping no labelled sequence is a FP. Precision, recall, F1,
F0.5. True negatives are undefined, so instead of a false-positive *rate* the app reports false
alarms per 10,000 evaluated points.

**Point-wise** (strict). Each evaluated timestep is a sample: TP/FP/FN/TN, precision, recall, F1,
F0.5, TPR, FPR, confusion matrix. **No point adjustment** (which can make random scores look
state-of-the-art, Kim et al. 2022).

**Interval quality.** For detected labels: delay = max(0, first overlapping detection start − label
start) in timesteps; IoU and coverage with the union of overlapping detections. Recall per class.

**Threshold-free curves.** PR and ROC curves and average precision of the smoothed score. For pooled
curves, each channel's smoothed score is standardised by its own mean and sd (label-free) before
concatenation. These curves sweep a single global cut-off, which the dynamic threshold is not.

Aggregation: counts are pooled across channels (micro-average); per-channel F1 is also averaged
(macro). Results are reported overall, per spacecraft and per channel, with training/inference time.

**Uncertainty.** `bootstrap_event_f_beta` (in `backend/app/ml/evaluation/metrics.py`) resamples
channels with replacement and recomputes the pooled event F0.5 (10,000 draws, seed 0, percentile
95 % intervals). The same resampled channel sets are used for every method, so differences between
methods are paired. `scripts/results_table.py` prints these intervals under the reference table.

**Comparability.** The paper reports precision 87.5 % / recall 80.0 % overall. These are not
directly comparable to this project's numbers: the training profile, early-stopping/weight
restoration, the thresholding deviations above, and the handling of `P-2` differ.

## 6. Leakage controls

* Original train/test separation preserved; models are fitted on training data only.
* No preprocessing is fitted on test data (the benchmark's own test-based scaling is disclosed).
  Baseline scalers (Isolation Forest) are fitted on training data.
* Validation is the chronologically last part of the training split.
* Thresholds use no labels and only causal information.
* `tests/integration/test_leakage.py` (a) makes any label access during fit/score raise, and (b)
  replaces the labels and asserts that detections are identical.
* Every experiment records configuration, seed, code commit and dataset checksum.
