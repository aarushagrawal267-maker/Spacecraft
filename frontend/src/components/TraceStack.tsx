import type { Data, Layout, Shape } from "plotly.js";
import { useMemo } from "react";
import type { LabelInterval } from "../api/types";
import { C } from "../lib/palette";
import { Plot, axis, baseLayout } from "./Plot";

type Arr = (number | null)[] | null | undefined;

export interface TraceDetection {
  start: number;
  end: number;
  id?: number | null;
  score?: number;
}

export interface TraceStackProps {
  actual: Arr;
  prediction?: Arr;
  raw?: Arr;
  smoothed?: Arr;
  thresholdHigh?: Arr;
  thresholdLow?: Arr;
  labels: LabelInterval[];
  detections?: TraceDetection[];
  evalStart?: number;
  xOffset?: number; // index of the first element (for context windows)
  height?: number;
  showError?: boolean;
  uirevision?: string | number;
  xRange?: [number, number];
  highlight?: [number, number] | null;
  onSelectDetection?: (d: TraceDetection) => void;
  predictionLabel?: string;
  totalLength?: number;
}

/**
 * The Trace Stack: telemetry, error/threshold and annotation lane share one index axis,
 * like a strip-chart recorder. Expert labels are amber + hatched; model detections violet.
 */
export function TraceStack({
  actual,
  prediction,
  raw,
  smoothed,
  thresholdHigh,
  thresholdLow,
  labels,
  detections = [],
  evalStart,
  xOffset = 0,
  height = 520,
  showError = true,
  uirevision,
  xRange,
  highlight,
  onSelectDetection,
  predictionLabel = "Forecast",
  totalLength,
}: TraceStackProps) {
  const fig = useMemo(() => {
    const n = actual?.length ?? 0;
    const x = Array.from({ length: n }, (_, i) => i + xOffset);
    const xmax = xOffset + n - 1;
    const useGl = n > 5000;
    const line = useGl ? "scattergl" : "scatter";
    const hasErr = showError && !!smoothed;
    const data: Data[] = [];

    data.push({
      type: line as "scatter",
      mode: "lines",
      x,
      y: actual ?? [],
      name: "Actual telemetry",
      line: { color: C.actual, width: 1.6 },
      hovertemplate: "%{y:.4f}",
    });
    if (prediction) {
      data.push({
        type: line as "scatter",
        mode: "lines",
        x,
        y: prediction,
        name: predictionLabel,
        line: { color: C.prediction, width: 1.1 },
        opacity: 0.85,
        hovertemplate: "%{y:.4f}",
      });
    }
    if (hasErr) {
      if (raw) {
        data.push({
          type: line as "scatter", mode: "lines", x, y: raw, yaxis: "y2", name: "Raw score",
          line: { color: C.raw, width: 0.8 }, opacity: 0.4, hovertemplate: "%{y:.4f}",
        });
      }
      data.push({
        type: line as "scatter", mode: "lines", x, y: smoothed!, yaxis: "y2", name: "Smoothed score",
        line: { color: C.smoothed, width: 1.5 }, hovertemplate: "%{y:.4f}",
      });
      if (thresholdHigh) {
        data.push({
          type: line as "scatter", mode: "lines", x, y: thresholdHigh, yaxis: "y2", name: "Threshold",
          line: { color: C.threshold, width: 1.4, dash: "dash" }, hovertemplate: "%{y:.4f}",
        });
      }
      if (thresholdLow && thresholdLow.some((v) => v != null)) {
        data.push({
          type: line as "scatter", mode: "lines", x, y: thresholdLow, yaxis: "y2", name: "Lower threshold",
          line: { color: C.threshold, width: 1, dash: "dot" }, opacity: 0.6, hovertemplate: "%{y:.4f}",
        });
      }
    }

    // Annotation lane: horizontal bars, labels hatched (texture = secondary encoding).
    const laneY = hasErr ? "y3" : "y2";
    // Invisible anchor so the lane axis exists even before any label/detection is shown.
    data.push({
      type: "scatter", mode: "markers", x: [xOffset, xOffset], y: ["Expert label", "Model detection"], yaxis: laneY,
      marker: { opacity: 0 }, hoverinfo: "skip", showlegend: false, name: "lane-anchor",
    } as Data);
    if (labels.length) {
      data.push({
        type: "bar",
        orientation: "h",
        y: labels.map(() => "Expert label"),
        x: labels.map((l) => l.end - l.start + 1),
        base: labels.map((l) => l.start),
        yaxis: laneY,
        width: 0.62,
        marker: {
          color: "rgba(201,133,0,0.22)",
          line: { color: C.label, width: 1 },
          pattern: { shape: "/", fgcolor: C.label, size: 6, solidity: 0.35 },
        },
        customdata: labels.map((l) => [l.start, l.end, l.cls]),
        hovertemplate: "Expert label %{customdata[0]}–%{customdata[1]} · %{customdata[2]}<extra></extra>",
        name: "Expert label",
      } as Data);
    }
    if (detections.length) {
      data.push({
        type: "bar",
        orientation: "h",
        y: detections.map(() => "Model detection"),
        x: detections.map((d) => d.end - d.start + 1),
        base: detections.map((d) => d.start),
        yaxis: laneY,
        width: 0.62,
        marker: { color: C.detect, line: { color: C.panel, width: 1 } },
        customdata: detections.map((d) => [d.start, d.end, d.score ?? null, d.id ?? null]),
        hovertemplate: "Model detection %{customdata[0]}–%{customdata[1]} · score %{customdata[2]:.2f}<extra></extra>",
        name: "Model detection",
      } as Data);
    }

    const shapes: Partial<Shape>[] = [];
    for (const l of labels) {
      shapes.push({
        type: "rect", xref: "x", yref: "y domain" as never, x0: l.start, x1: l.end + 1, y0: 0, y1: 1,
        fillcolor: C.label, opacity: 0.2, line: { width: 0 }, layer: "below",
      });
    }
    if (hasErr) {
      for (const d of detections) {
        shapes.push({
          type: "rect", xref: "x", yref: "y2 domain" as never, x0: d.start, x1: d.end + 1, y0: 0, y1: 1,
          fillcolor: C.detect, opacity: 0.22, line: { width: 0 }, layer: "below",
        });
      }
    }
    if (evalStart != null && evalStart > xOffset) {
      shapes.push({
        type: "rect", xref: "x", yref: (hasErr ? "y2 domain" : "y domain") as never, x0: xOffset, x1: evalStart, y0: 0, y1: 1,
        fillcolor: C.dim, opacity: 0.07, line: { width: 0 }, layer: "below",
      });
    }
    if (highlight) {
      shapes.push({
        type: "rect", xref: "x", yref: "paper", x0: highlight[0], x1: highlight[1] + 1, y0: 0, y1: 1,
        fillcolor: "rgba(0,0,0,0)", line: { color: C.accent, width: 1 }, layer: "above",
      });
    }

    const annotations =
      evalStart != null && evalStart > xOffset && hasErr
        ? [{
            x: (xOffset + evalStart) / 2, y: 1, xref: "x", yref: "y2 domain", yanchor: "top", showarrow: false,
            text: "not scored", font: { size: 10, color: C.dim },
          }]
        : [];

    let scoreMax: number | undefined;
    if (hasErr) {
      let m = 0;
      for (const arr of [smoothed, thresholdHigh]) for (const v of arr ?? []) if (v != null && Number.isFinite(v) && v > m) m = v;
      scoreMax = m > 0 ? m * 1.35 : undefined;
    }

    const domains = hasErr
      ? { y: [0.47, 1], y2: [0.17, 0.42], y3: [0, 0.11] }
      : { y: [0.2, 1], y2: [0, 0.13] };

    const layout: Partial<Layout> = baseLayout({
      uirevision: uirevision ?? "trace",
      hovermode: "x unified",
      bargap: 0.2,
      barmode: "overlay",
      margin: { l: 100, r: 14, t: 6, b: 34 },
      xaxis: axis({
        anchor: hasErr ? "y3" : "y2",
        range: xRange ?? [xOffset, Math.max(xmax, (totalLength ?? 0) - 1 + xOffset * 0)],
        showspikes: true, spikemode: "across", spikecolor: C.axis, spikethickness: 1, spikedash: "solid",
        title: { text: "Test-stream index (anonymised time)", font: { size: 11, color: C.muted }, standoff: 6 },
      }) as Layout["xaxis"],
      yaxis: axis({ domain: domains.y, title: { text: "Telemetry (scaled)", font: { size: 11, color: C.muted } } }) as Layout["yaxis"],
      yaxis2: hasErr
        ? (axis({ domain: domains.y2, ...(scoreMax ? { range: [0, scoreMax] } : { rangemode: "tozero" }), title: { text: "Score", font: { size: 11, color: C.muted } } }) as Layout["yaxis"])
        : (axis({ domain: domains.y2, type: "category", showgrid: false, tickfont: { size: 11, color: C.muted },
            categoryorder: "array", categoryarray: ["Expert label", "Model detection"] }) as Layout["yaxis"]),
      ...(hasErr
        ? { yaxis3: axis({ domain: domains.y3, type: "category", showgrid: false, tickfont: { size: 11, color: C.muted },
            categoryorder: "array", categoryarray: ["Expert label", "Model detection"] }) as Layout["yaxis"] }
        : {}),
      shapes,
      annotations: annotations as Layout["annotations"],
    });
    return { data, layout };
  }, [actual, prediction, raw, smoothed, thresholdHigh, thresholdLow, labels, detections, evalStart, xOffset, showError,
      uirevision, xRange, highlight, predictionLabel, totalLength]);

  return (
    <Plot
      data={fig.data}
      layout={fig.layout}
      height={height}
      onClick={(e) => {
        const p = e.points?.[0];
        const cd = p?.customdata as unknown as (number | null)[] | undefined;
        if (onSelectDetection && p && (p.data as { name?: string }).name === "Model detection" && cd) {
          onSelectDetection({ start: cd[0] as number, end: cd[1] as number, score: cd[2] ?? undefined, id: cd[3] });
        }
      }}
    />
  );
}
