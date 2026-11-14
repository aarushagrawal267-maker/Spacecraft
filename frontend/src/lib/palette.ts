// Data colours. Validated with the dataviz palette validator on the panel surface #0B1020.
export const C = {
  actual: "#199e9e", // actual telemetry (teal)
  prediction: "#d9deea", // model forecast — neutral ink so it never competes on hue
  detect: "#9085e9", // model-detected anomaly
  label: "#c98500", // expert-labelled anomaly (always hatched as well)
  raw: "#5e6a85",
  smoothed: "#e6ebf5",
  threshold: "#9085e9",
  ink: "#e6ebf5",
  ink2: "#b8c1d6",
  muted: "#8491ab",
  dim: "#5e6a85",
  grid: "#1a2238",
  axis: "#263152",
  panel: "#0b1020",
  accent: "#4cc9f0",
  good: "#0ca30c",
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
