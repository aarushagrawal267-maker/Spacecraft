---
title: Heliograph — Spacecraft Telemetry Anomaly Detection
emoji: 🛰️
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: LSTM anomaly detection on NASA SMAP & Curiosity telemetry
---

# Heliograph

LSTM forecasting and nonparametric dynamic thresholding on the NASA SMAP / MSL (Curiosity) telemetry
anomaly benchmark. Includes trained reference models for all 81 channels, a historical telemetry replay,
evaluation against expert labels, and a model laboratory.

Independent research implementation inspired by Hundman et al. (KDD 2018). Not affiliated with NASA/JPL.
Detected anomalies are unusual patterns that warrant investigation, not confirmed failures.

Source code: https://github.com/aarushagrawal267-maker/spacecraft · API docs: `/api/docs`

Note: this free CPU Space restarts when idle; experiments you train here are not kept after a restart.
The reference results are always available.
