// Data colours. Validated with the dataviz palette validator on a dark panel surface.
// Chart chrome (ink, grid, axis, panel) follows the graphite interface ramp in styles.css.
export const C = {
  actual: "#199e9e", // actual telemetry (teal)
  prediction: "#dcdcdf", // model forecast — neutral ink so it never competes on hue
  detect: "#9085e9", // model-detected anomaly
  label: "#c98500", // expert-labelled anomaly (always hatched as well)
  raw: "#66666d",
  smoothed: "#ededed",
  threshold: "#9085e9",
  ink: "#ededed",
  ink2: "#b9b9be",
  muted: "#8b8b92",
  dim: "#66666d",
  grid: "#29292c",
  axis: "#3a3a3e",
  panel: "#18181a",
  raised: "#222224",
  sunken: "#0d0d0e",
  accent: "#4cc9f0",
  good: "#4cc38a",
  warn: "#fab219",
  serious: "#ec835a",
  critical: "#d03b3b",
};

// Fixed method → categorical slot (dark steps of the validated reference order).
// Colour follows the method, never its rank. The random control is neutral grey.
export const METHOD_COLOR: Record<string, string> = {
  lstm_forecaster: "#3987e5",
  lstm_autoencoder: "#d95926",
  ridge_ar: "#199e70",
  persistence: "#c98500",
  rolling_zscore: "#d55181",
  isolation_forest: "#0ca30c",
  random: "#8491ab",
};

export const METHOD_ORDER = [
  "lstm_forecaster",
  "lstm_autoencoder",
  "ridge_ar",
  "persistence",
  "rolling_zscore",
  "isolation_forest",
  "random",
];

export const SPACECRAFT_INFO = {
  SMAP: {
    name: "Soil Moisture Active Passive",
    short: "SMAP",
    kind: "Earth-orbiting satellite",
    blurb:
      "NASA Earth-science mission (launched 2015) measuring global soil moisture from a near-polar, sun-synchronous orbit.",
  },
  MSL: {
    name: "Mars Science Laboratory · Curiosity",
    short: "MSL",
    kind: "Mars surface rover",
    blurb: "NASA rover operating in Gale Crater, Mars, since 2012, studying the planet's habitability.",
  },
} as const;
