import Plotly from "plotly.js-dist-min";
import createPlotlyComponent from "react-plotly.js/factory";
import type { Config, Data, Layout, LayoutAxis } from "plotly.js";
import { C } from "../lib/palette";

const PlotlyComponent = createPlotlyComponent(Plotly);

export const FONT = { family: "IBM Plex Sans, system-ui, sans-serif", size: 11.5, color: C.muted };

export function baseLayout(over: Partial<Layout> = {}): Partial<Layout> {
  return {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: FONT,
    margin: { l: 52, r: 16, t: 8, b: 36 },
    hoverlabel: {
      bgcolor: "#0f1629",
      bordercolor: "#263152",
      font: { family: "IBM Plex Mono, monospace", size: 11.5, color: C.ink },
      align: "left",
    },
    showlegend: false,
    xaxis: axis(),
    yaxis: axis(),
    dragmode: "zoom",
    ...over,
  };
}

export function axis(over: Record<string, unknown> = {}): Partial<LayoutAxis> {
  return {
    gridcolor: C.grid,
    gridwidth: 1,
    zeroline: false,
    linecolor: C.axis,
    showline: true,
    ticks: "outside",
    ticklen: 3,
    tickcolor: C.axis,
    tickfont: { family: "IBM Plex Mono, monospace", size: 10.5, color: C.muted },
    title: { font: { size: 11.5, color: C.muted }, standoff: 8 },
    ...over,
  } as Partial<LayoutAxis>;
}

export const plotConfig: Partial<Config> = {
  displaylogo: false,
  responsive: true,
  modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d", "toggleSpikelines"],
  toImageButtonOptions: { format: "png", scale: 2 },
};

export function Plot({
  data,
  layout,
  height,
  className = "",
  config,
  onRelayout,
  onClick,
}: {
  data: Data[];
  layout: Partial<Layout>;
  height: number;
  className?: string;
  config?: Partial<Config>;
  onRelayout?: (e: Readonly<Plotly.PlotRelayoutEvent>) => void;
  onClick?: (e: Readonly<Plotly.PlotMouseEvent>) => void;
}) {
  return (
    <div className={className} style={{ height }}>
      <PlotlyComponent
        data={data}
        layout={{ ...layout, height, autosize: true }}
        config={{ ...plotConfig, ...config }}
        useResizeHandler
        style={{ width: "100%", height: "100%" }}
        onRelayout={onRelayout}
        onClick={onClick}
      />
    </div>
  );
}
