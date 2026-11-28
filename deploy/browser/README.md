---
title: Heliograph
colorFrom: indigo
colorTo: blue
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Viewer of LSTM anomaly detection results on NASA telemetry
---

# Heliograph: browser demo

Heliograph detects anomalies in real spacecraft telemetry from NASA's SMAP satellite and the Curiosity rover (MSL):
per-channel LSTM forecasters, nonparametric dynamic thresholding and five baselines, evaluated against the
benchmark's expert labels.

This is a **browser demo; the full app runs locally**. The page is a viewer of the stored reference results (seven
methods on all 81 channels): benchmark scores, and for each channel the model forecast, smoothed error, dynamic
threshold, detected intervals and expert labels. Nothing is trained or computed here. The telemetry is downloaded
from the public GitHub copy of the benchmark when a channel is opened.

Independent research implementation inspired by Hundman et al. (KDD 2018). Not affiliated with NASA/JPL. A detected
anomaly is an unusual pattern that warrants investigation, not proof of a hardware failure.

Source code and the full app: https://github.com/aarushagrawal267-maker/spacecraft
