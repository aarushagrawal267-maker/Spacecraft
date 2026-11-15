import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

// Plotly cannot render in jsdom: mock the library and capture the figure handed to it.
const captured: { data: unknown[]; layout: Record<string, unknown> }[] = [];
vi.mock("plotly.js-dist-min", () => ({ default: {} }));
vi.mock("react-plotly.js/factory", () => ({
  default: () => (props: { data: unknown[]; layout: Record<string, unknown> }) => {
    captured.push({ data: props.data, layout: props.layout });
    return <div data-testid="plot" />;
  },
}));

import { TraceStack } from "../components/TraceStack";
import { Legend, OutcomeTag, StatusBadge } from "../components/ui";
import { fmtDuration, fmtInt, fmtNum, fmtPct } from "../lib/format";
import { C, METHOD_COLOR, METHOD_ORDER } from "../lib/palette";

describe("format helpers", () => {
  it("formats missing values as an em dash, never as zero", () => {
    expect(fmtPct(null)).toBe("—");
    expect(fmtNum(undefined)).toBe("—");
    expect(fmtInt(null)).toBe("—");
    expect(fmtDuration(null)).toBe("—");
  });
  it("formats numbers", () => {
    expect(fmtPct(0.8754)).toBe("87.5%");
    expect(fmtNum(0.123456, 3)).toBe("0.123");
    expect(fmtInt(435826)).toBe("435,826");
    expect(fmtDuration(0.25)).toBe("250 ms");
    expect(fmtDuration(75)).toBe("1m 15s");
    expect(fmtDuration(7322)).toBe("2h 02m");
  });
});

describe("palette", () => {
  it("assigns every method a fixed colour", () => {
    for (const m of METHOD_ORDER) expect(METHOD_COLOR[m]).toMatch(/^#[0-9a-f]{6}$/i);
    expect(new Set(Object.values(METHOD_COLOR)).size).toBe(METHOD_ORDER.length);
  });
});

describe("status and reference badges", () => {
  it("pairs status colour with a text label", () => {
    render(<StatusBadge status="failed" />);
    expect(screen.getByText("Failed")).toBeInTheDocument();
  });
  it("distinguishes label overlap from no overlap in words", () => {
    render(
      <>
        <OutcomeTag outcome="matches_label" />
        <OutcomeTag outcome="no_label_overlap" />
      </>,
    );
    expect(screen.getByText("Overlaps expert label")).toBeInTheDocument();
    expect(screen.getByText("No label overlap")).toBeInTheDocument();
  });
  it("renders a legend entry per item", () => {
    render(<Legend items={[{ label: "Actual telemetry", color: C.actual }, { label: "Expert label", color: C.label, kind: "hatch" }]} />);
    expect(screen.getByText("Actual telemetry")).toBeInTheDocument();
    expect(screen.getByText("Expert label")).toBeInTheDocument();
  });
});

describe("TraceStack", () => {
  it("builds telemetry, score, threshold and annotation-lane traces from the given arrays", () => {
    captured.length = 0;
    const n = 600;
    const actual = Array.from({ length: n }, (_, i) => Math.sin(i / 10));
    const smoothed = Array.from({ length: n }, (_, i) => (i < 250 ? null : 0.1));
    render(
      <TraceStack
        actual={actual}
        prediction={actual}
        raw={smoothed}
        smoothed={smoothed}
        thresholdHigh={smoothed.map((v) => (v == null ? null : 0.3))}
        labels={[{ start: 300, end: 350, cls: "point" }]}
        detections={[{ start: 310, end: 360, score: 1.2, id: 7 }]}
        evalStart={250}
      />,
    );
    expect(screen.getByTestId("plot")).toBeInTheDocument();
    const fig = captured.at(-1)!;
    const names = (fig.data as { name?: string }[]).map((d) => d.name);
    expect(names).toEqual(expect.arrayContaining(["Actual telemetry", "Forecast", "Smoothed score", "Threshold", "Expert label", "Model detection"]));
    const label = (fig.data as { name?: string; marker?: { pattern?: { shape?: string } } }[]).find((d) => d.name === "Expert label")!;
    expect(label.marker?.pattern?.shape).toBe("/"); // texture as secondary encoding
    const shapes = fig.layout.shapes as { x0: number; x1: number; fillcolor?: string }[];
    expect(shapes.some((s) => s.x0 === 300 && s.fillcolor === C.label)).toBe(true);
    expect(shapes.some((s) => s.x0 === 310 && s.fillcolor === C.detect)).toBe(true);
    expect(shapes.some((s) => s.x0 === 0 && s.x1 === 250)).toBe(true); // unscored region
  });

  it("omits the score track when no scores are supplied (telemetry-only view)", () => {
    captured.length = 0;
    render(<TraceStack actual={[1, 2, 3]} labels={[]} showError={false} />);
    const names = (captured.at(-1)!.data as { name?: string }[]).map((d) => d.name).filter((n) => n !== "lane-anchor");
    expect(names).toEqual(["Actual telemetry"]);
  });
});
