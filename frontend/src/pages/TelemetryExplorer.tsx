import { useEffect, useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { useTelemetry } from "../api/hooks";
import type { LabelInterval } from "../api/types";
import { Plot, axis, baseLayout } from "../components/Plot";
import { ChannelPicker, useChannel } from "../components/pickers";
import { TraceStack } from "../components/TraceStack";
import { Button, ErrorState, KV, Legend, Loading, Notice, PageHeader, Panel, Segmented } from "../components/ui";
import { fmtInt, fmtNum, fmtPct } from "../lib/format";
import { C } from "../lib/palette";
import { useSelection } from "../lib/selection";

function Distribution({ values, labels }: { values: (number | null)[]; labels: LabelInterval[] }) {
  const { inside, outside } = useMemo(() => {
    const mask = new Uint8Array(values.length);
    for (const l of labels) for (let i = l.start; i <= l.end && i < values.length; i++) mask[i] = 1;
    const ins: number[] = [];
    const out: number[] = [];
    values.forEach((v, i) => {
      if (v == null) return;
      (mask[i] ? ins : out).push(v);
    });
    return { inside: ins, outside: out };
  }, [values, labels]);
  if (!inside.length) return <p className="text-sm text-muted">This channel has no labelled test points to compare.</p>;
  return (
    <>
      <Legend
        items={[
          { label: `Outside expert labels (n=${fmtInt(outside.length)})`, color: C.actual, kind: "band" },
          { label: `Inside expert labels (n=${fmtInt(inside.length)})`, color: C.label, kind: "hatch" },
        ]}
      />
      <Plot
        height={230}
        data={[
          { type: "histogram", x: outside, histnorm: "probability", name: "Outside labels", marker: { color: C.actual, line: { color: C.panel, width: 1 } }, opacity: 0.75, nbinsx: 60, hovertemplate: "%{x}: %{y:.3f}<extra>outside</extra>" },
          { type: "histogram", x: inside, histnorm: "probability", name: "Inside labels", marker: { color: "rgba(201,133,0,0.35)", line: { color: C.label, width: 1 }, pattern: { shape: "/", fgcolor: C.label, size: 6 } } as never, opacity: 0.9, nbinsx: 60, hovertemplate: "%{x}: %{y:.3f}<extra>inside</extra>" },
        ]}
        layout={baseLayout({
          barmode: "overlay",
          margin: { l: 52, r: 10, t: 8, b: 36 },
          xaxis: axis({ title: { text: "Telemetry value (scaled)" } }),
          yaxis: axis({ title: { text: "Share of points" } }),
        })}
      />
    </>
  );
}

export default function TelemetryExplorer() {
  const { chan, setChan } = useSelection();
  const [params, setParams] = useSearchParams();
  const urlChan = params.get("chan");
  useEffect(() => {
    if (urlChan && urlChan !== chan) setChan(urlChan);
  }, [urlChan]); // eslint-disable-line react-hooks/exhaustive-deps
  const [split, setSplit] = useState<"test" | "train">("test");
  const [range, setRange] = useState<[number, number] | undefined>();
  const channel = useChannel(chan);
  const tel = useTelemetry(chan, split);
  const tel2 = useTelemetry(chan, split === "test" ? "train" : "test", 600);

  const change = (c: string) => {
    setChan(c);
    setRange(undefined);
    setParams({ chan: c });
  };

  return (
    <div className="space-y-6">
      <PageHeader eyebrow="Telemetry explorer" title={`Channel ${chan}`} actions={<ChannelPicker chan={chan} onChange={change} />}>
        Inspect the raw benchmark telemetry. Column 0 of each channel file is the telemetry value; the remaining columns are one-hot
        command inputs. Indices are anonymised time steps; there are no timestamps or engineering units.
      </PageHeader>

      <div className="grid gap-6 xl:grid-cols-[1fr_300px]">
        <Panel
          eyebrow={split === "test" ? "Test split · with expert reference labels" : "Training split · used to fit models"}
          title={tel.data ? `${fmtInt(tel.data.n)} values` : " "}
          actions={
            <>
              {range && (
                <Button size="sm" variant="ghost" onClick={() => setRange(undefined)}>
                  Reset zoom
                </Button>
              )}
              <Segmented
                size="sm"
                ariaLabel="Split"
                value={split}
                onChange={(v) => {
                  setSplit(v);
                  setRange(undefined);
                }}
                options={[
                  { value: "test", label: "Test" },
                  { value: "train", label: "Train" },
                ]}
              />
            </>
          }
        >
          {tel.isError ? (
            <ErrorState error={tel.error} />
          ) : !tel.data ? (
            <Loading height={420} />
          ) : (
            <div style={{ opacity: tel.isPlaceholderData ? 0.5 : 1 }}>
              <Legend
                items={[
                  { label: "Actual telemetry", color: C.actual },
                  ...(split === "test" ? [{ label: "Expert-labelled anomaly", color: C.label, kind: "hatch" as const }] : []),
                ]}
              />
              <TraceStack
                actual={tel.data.values}
                labels={tel.data.labels}
                showError={false}
                height={420}
                uirevision={`${chan}-${split}`}
                xRange={range}
              />
              {split === "train" && (
                <Notice>
                  The training split carries no labels and is not guaranteed to be anomaly-free. Models learn "normal" behaviour from it
                  without ever seeing test labels.
                </Notice>
              )}
            </div>
          )}
        </Panel>

        <div className="space-y-6">
          <Panel eyebrow="Channel facts" title={channel ? `${channel.spacecraft} · type “${channel.type_letter}”` : " "}>
            {channel ? (
              <div>
                <KV k="Training length" v={fmtInt(channel.train_len)} />
                <KV k="Test length" v={fmtInt(channel.test_len)} />
                <KV k="Input features" v={`${channel.n_features} (1 telemetry + ${channel.n_features - 1} command)`} />
                <KV k="Command columns active (train / test)" v={`${channel.commands_active_train} / ${channel.commands_active_test}`} />
                <KV k="Value range (both splits)" v={`${fmtNum(channel.value_min, 3)} … ${fmtNum(channel.value_max, 3)}`} />
                <KV k="Training std" v={fmtNum(channel.train_std, 4)} />
                <KV k="Labelled sequences" v={channel.n_intervals} />
                <KV k="Labelled share of test" v={fmtPct(channel.label_fraction)} />
                {channel.issues.length > 0 && (
                  <div className="mt-3">
                    <Notice tone="warn">{channel.issues.join("; ")}</Notice>
                  </div>
                )}
                <p className="mt-3 text-xs leading-relaxed text-dim">
                  The type letter is the only documented part of an anonymised ID (e.g. P = power, R = radiation); other letters are not
                  interpreted here.
                </p>
              </div>
            ) : (
              <Loading height={160} />
            )}
          </Panel>
          <Panel eyebrow="Other split" title={split === "test" ? "Training split preview" : "Test split preview"}>
            {tel2.data ? (
              <Plot
                height={120}
                data={[{ type: "scatter", mode: "lines", x: tel2.data.index, y: tel2.data.values, line: { color: C.actual, width: 1 }, hovertemplate: "%{x}: %{y:.3f}<extra></extra>" }]}
                layout={baseLayout({ margin: { l: 36, r: 6, t: 4, b: 22 }, xaxis: axis({ showgrid: false }), yaxis: axis({ nticks: 3 }) })}
                config={{ displayModeBar: false }}
              />
            ) : (
              <Loading height={120} />
            )}
          </Panel>
        </div>
      </div>

      {split === "test" && tel.data && (
        <div className="grid gap-6 xl:grid-cols-[1fr_1fr]">
          <Panel eyebrow="Reference annotations" title="Expert-labelled anomaly sequences">
            {tel.data.labels.length ? (
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-xs text-muted">
                    <th className="py-1.5 font-medium">Start</th>
                    <th className="py-1.5 font-medium">End</th>
                    <th className="py-1.5 font-medium">Length</th>
                    <th className="py-1.5 font-medium">Class</th>
                    <th />
                  </tr>
                </thead>
                <tbody>
                  {tel.data.labels.map((l) => (
                    <tr key={l.start} className="border-t border-hairline">
                      <td className="num py-1.5">{l.start}</td>
                      <td className="num py-1.5">{l.end}</td>
                      <td className="num py-1.5">{l.end - l.start + 1}</td>
                      <td className="py-1.5">
                        {l.cls}
                        {l.merged_from && l.merged_from.length > 0 && (
                          <span className="ml-2 text-xs text-warn" title={`Merged from ${JSON.stringify(l.merged_from)}`}>
                            merged from {l.merged_from.length} rows
                          </span>
                        )}
                      </td>
                      <td className="py-1.5 text-right">
                        <Button size="sm" variant="ghost" onClick={() => setRange([Math.max(0, l.start - 300), Math.min(tel.data!.n - 1, l.end + 300)])}>
                          Zoom
                        </Button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <p className="text-sm text-muted">No labelled sequences.</p>
            )}
            <p className="mt-3 text-xs text-muted">
              Labels are benchmark reference intervals, not confirmed hardware failures.{" "}
              <Link to={`/detect?chan=${chan}`} className="text-accent hover:underline">
                Compare with model detections →
              </Link>
            </p>
          </Panel>
          <Panel eyebrow="Normal vs anomalous regions" title="Value distribution inside vs outside labels">
            <Distribution values={tel.data.values} labels={tel.data.labels} />
          </Panel>
        </div>
      )}
    </div>
  );
}
