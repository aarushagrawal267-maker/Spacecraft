import { Download } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { exportUrl } from "../api/client";
import { useCompare, useExperiments } from "../api/hooks";
import type { Aggregate, Experiment } from "../api/types";
import { Plot, axis, baseLayout } from "../components/Plot";
import { EmptyState, ErrorState, Legend, Loading, Notice, OriginTag, PageHeader, Panel, Segmented, Select } from "../components/ui";
import { fmtDuration, fmtInt, fmtNum, fmtPct } from "../lib/format";
import { C, METHOD_COLOR, METHOD_ORDER } from "../lib/palette";

type Scope = "overall" | "SMAP" | "MSL";
type Protocol = "event" | "point";

const agg = (e: Experiment, scope: Scope): Aggregate | null | undefined =>
  scope === "overall" ? e.metrics?.overall : e.metrics?.by_spacecraft?.[scope];

const shortLabel = (e: Experiment) => `${e.method_label}${e.profile ? ` · ${e.profile}` : ""} (#${e.id})`;

function ConfusionMatrix({ a }: { a: Aggregate }) {
  const p = a.point;
  const z = [[p.tp, p.fn], [p.fp, p.tn]];
  const total = p.tp + p.fn + p.fp + p.tn;
  const logs = z.flat().map((v) => Math.log10(v + 1));
  const lo = Math.min(...logs);
  const hi = Math.max(...logs);
  // Position of a cell on the colour scale (0 = darkest, 1 = brightest).
  const pos = (v: number) => (hi > lo ? (Math.log10(v + 1) - lo) / (hi - lo) : 0);
  return (
    <Plot
      height={260}
      data={[{
        type: "heatmap",
        z: z.map((r) => r.map((v) => Math.log10(v + 1))),
        x: ["Flagged", "Not flagged"],
        y: ["Labelled anomaly", "Not labelled"],
        colorscale: [[0, C.raised], [0.5, "#256abf"], [1, "#86b6ef"]],
        showscale: false,
        customdata: z as never,
        hovertemplate: "%{y} / %{x}: %{customdata:,}<extra></extra>",
        xgap: 3,
        ygap: 3,
      }]}
      layout={baseLayout({
        margin: { l: 120, r: 10, t: 6, b: 34 },
        xaxis: axis({ showgrid: false, showline: false, ticks: "", side: "bottom", tickfont: { size: 12, color: C.ink2, family: "IBM Plex Sans" } }),
        yaxis: axis({ showgrid: false, showline: false, ticks: "", autorange: "reversed", tickfont: { size: 12, color: C.ink2, family: "IBM Plex Sans" } }),
        annotations: [0, 1].flatMap((i) => [0, 1].map((j) => ({
          x: ["Flagged", "Not flagged"][j], y: ["Labelled anomaly", "Not labelled"][i],
          text: `<b>${fmtInt(z[i][j])}</b><br><span style="font-size:10px">${fmtPct(z[i][j] / total)}</span>`,
          showarrow: false, font: { family: "IBM Plex Mono", size: 13, color: pos(z[i][j]) > 0.6 ? C.sunken : C.ink },
        }))),
      })}
    />
  );
}

export default function Evaluation() {
  const [params, setParams] = useSearchParams();
  const exps = useExperiments(false);
  const completed = useMemo(() => (exps.data ?? []).filter((e) => ["completed", "partial"].includes(e.status)), [exps.data]);
  const defaultIds = useMemo(() => {
    const ids: number[] = [];
    for (const m of METHOD_ORDER) {
      const e = completed.find((x) => x.method === m && x.spacecraft === "ALL");
      if (e) ids.push(e.id);
    }
    return ids.length ? ids : completed.slice(0, 4).map((e) => e.id);
  }, [completed]);
  const ids = params.get("ids") ? params.get("ids")!.split(",").map(Number).filter(Boolean) : defaultIds;
  const cmp = useCompare(ids);
  const [scope, setScope] = useState<Scope>("overall");
  const [protocol, setProtocol] = useState<Protocol>("event");
  const list = useMemo(
    () => [...(cmp.data?.experiments ?? [])].sort((a, b) => METHOD_ORDER.indexOf(a.method) - METHOD_ORDER.indexOf(b.method) || a.id - b.id),
    [cmp.data],
  );
  const [focus, setFocus] = useState<number | null>(null);
  useEffect(() => {
    if (list.length && (focus == null || !list.some((e) => e.id === focus))) setFocus((list.find((e) => e.method === "lstm_forecaster") ?? list[0]).id);
  }, [list, focus]);
  const focusExp = list.find((e) => e.id === focus);
  const toggle = (id: number) => {
    const next = ids.includes(id) ? ids.filter((x) => x !== id) : [...ids, id];
    setParams(next.length ? { ids: next.join(",") } : {});
  };

  if (exps.isError) return <ErrorState error={exps.error} />;
  if (!exps.data) return <Loading height={500} />;
  if (!completed.length) {
    return (
      <div>
        <PageHeader eyebrow="Model evaluation" title="No evaluated experiments yet" />
        <EmptyState title="Nothing to evaluate" action={<Link className="text-accent hover:underline" to="/lab">Open the Model Laboratory</Link>}>
          All metrics on this page come from completed experiments. Train a model first.
        </EmptyState>
      </div>
    );
  }

  const metricNames = protocol === "event" ? ["precision", "recall", "f1", "f0_5"] : ["precision", "recall", "f1", "fpr"];
  const metricLabels: Record<string, string> = { precision: "Precision", recall: "Recall", f1: "F1", f0_5: "F0.5", fpr: "FPR" };

  return (
    <div className="space-y-6">
      <PageHeader eyebrow="Model evaluation" title="Comparing methods on the same data">
        All methods are scored on test indices ≥ 250 of the same channels, smoothed and thresholded identically, and evaluated against
        the same expert labels. Event-wise and point-wise results are kept strictly separate.
      </PageHeader>

      <Panel eyebrow="Experiments in comparison" title={`${list.length} selected`}
        actions={
          <div className="flex flex-wrap items-center gap-2">
            <Segmented ariaLabel="Protocol" value={protocol} onChange={setProtocol} options={[{ value: "event", label: "Event-wise" }, { value: "point", label: "Point-wise" }]} />
            <Segmented ariaLabel="Spacecraft" value={scope} onChange={setScope} options={[{ value: "overall", label: "Both" }, { value: "SMAP", label: "SMAP" }, { value: "MSL", label: "MSL" }]} />
          </div>
        }
      >
        <div className="flex flex-wrap gap-1.5">
          {completed.map((e) => {
            const on = ids.includes(e.id);
            return (
              <button key={e.id} onClick={() => toggle(e.id)} aria-pressed={on}
                className={`inline-flex items-center gap-1.5 rounded-md border px-2 py-1 text-xs ${on ? "border-edge bg-raised text-ink" : "border-hairline text-muted hover:text-ink"}`}>
                <span className="h-2 w-2 rounded-sm" style={{ background: on ? METHOD_COLOR[e.method] : "transparent", border: `1px solid ${METHOD_COLOR[e.method]}` }} />
                #{e.id} {e.method_label}{e.profile ? ` · ${e.profile}` : ""} · {e.spacecraft === "ALL" ? "81 ch" : `${e.n_channels} ch`}
              </button>
            );
          })}
        </div>
        <div className="mt-4">
          <Notice>
            {protocol === "event"
              ? "Event-wise (Hundman et al.): a labelled sequence counts as detected if any model detection overlaps it; a detection overlapping no label is a false positive. True negatives are undefined, so no false-positive rate is reported here."
              : "Point-wise: every evaluated timestep is a sample. No point adjustment is applied, so long labelled intervals that are only partly covered count as partly missed."}
          </Notice>
        </div>
      </Panel>

      {cmp.isError ? <ErrorState error={cmp.error} /> : !cmp.data ? <Loading height={400} /> : (
        <>
          <div className="grid gap-6 xl:grid-cols-2">
            <Panel eyebrow={`${protocol === "event" ? "Event" : "Point"}-wise · ${scope === "overall" ? "SMAP + MSL" : scope}`} title="Detection metrics by method">
              <Legend items={list.map((e) => ({ label: shortLabel(e), color: METHOD_COLOR[e.method], kind: "solid" as const }))} />
              <Plot
                height={300}
                data={list.map((e) => {
                  const a = agg(e, scope);
                  const src = (protocol === "event" ? a?.event : a?.point) as unknown as Record<string, number | null> | undefined;
                  return {
                    type: "bar", name: shortLabel(e), x: metricNames.map((m) => metricLabels[m]),
                    y: metricNames.map((m) => src?.[m] ?? null),
                    marker: { color: METHOD_COLOR[e.method], line: { color: C.panel, width: 2 } },
                    hovertemplate: `${shortLabel(e)}<br>%{x}: %{y:.3f}<extra></extra>`,
                  };
                })}
                layout={baseLayout({ barmode: "group", bargap: 0.25, bargroupgap: 0.05, margin: { l: 44, r: 8, t: 6, b: 30 }, yaxis: axis({ range: [0, 1], dtick: 0.25 }) })}
              />
            </Panel>
            <Panel eyebrow="Table view" title="Same figures, exact values">
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left text-xs text-muted">
                      <th className="py-1.5 pr-2 font-medium">Method</th>
                      {protocol === "event" ? ["TP", "FP", "FN", "P", "R", "F1", "F0.5", "FA/10k"].map((h) => <th key={h} className="py-1.5 pr-2 text-right font-medium">{h}</th>)
                        : ["TP", "FP", "FN", "TN", "P", "R", "F1", "FPR"].map((h) => <th key={h} className="py-1.5 pr-2 text-right font-medium">{h}</th>)}
                    </tr>
                  </thead>
                  <tbody>
                    {list.map((e) => {
                      const a = agg(e, scope);
                      if (!a) return null;
                      const ev = a.event;
                      const pt = a.point;
                      const cells = protocol === "event"
                        ? [fmtInt(ev.tp), fmtInt(ev.fp), fmtInt(ev.fn), fmtPct(ev.precision), fmtPct(ev.recall), fmtNum(ev.f1), fmtNum(ev.f0_5), fmtNum(ev.false_alarms_per_10k, 2)]
                        : [fmtInt(pt.tp), fmtInt(pt.fp), fmtInt(pt.fn), fmtInt(pt.tn), fmtPct(pt.precision), fmtPct(pt.recall), fmtNum(pt.f1), fmtPct(pt.fpr, 2)];
                      return (
                        <tr key={e.id} className="border-t border-hairline">
                          <td className="py-1.5 pr-2"><span className="inline-flex items-center gap-1.5"><span className="h-2 w-2 rounded-sm" style={{ background: METHOD_COLOR[e.method] }} />{shortLabel(e)}</span></td>
                          {cells.map((c, i) => <td key={i} className="num py-1.5 pr-2 text-right">{c}</td>)}
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </Panel>
          </div>

          <div className="grid gap-6 xl:grid-cols-2">
            {(["pr", "roc"] as const).map((kind) => (
              <Panel key={kind} eyebrow="Threshold-free · pooled over channels" title={kind === "pr" ? "Precision–recall curve" : "ROC curve"}>
                <Legend items={list.filter((e) => e.metrics?.curves?.[scope]).map((e) => ({
                  label: `${shortLabel(e)} · ${kind === "pr" ? "AP" : "AUC"} ${fmtNum(kind === "pr" ? e.metrics!.curves[scope]!.average_precision : e.metrics!.curves[scope]!.roc_auc, 3)}`,
                  color: METHOD_COLOR[e.method],
                }))} />
                <Plot
                  height={320}
                  data={[
                    ...list.filter((e) => e.metrics?.curves?.[scope]).map((e) => {
                      const c = e.metrics!.curves[scope]!;
                      return {
                        type: "scatter" as const, mode: "lines" as const, name: shortLabel(e),
                        x: kind === "pr" ? c.pr.recall : c.roc.fpr, y: kind === "pr" ? c.pr.precision : c.roc.tpr,
                        line: { color: METHOD_COLOR[e.method], width: 2, shape: "hv" as const },
                        hovertemplate: `${shortLabel(e)}<br>${kind === "pr" ? "recall %{x:.2f} · precision %{y:.2f}" : "FPR %{x:.2f} · TPR %{y:.2f}"}<extra></extra>`,
                      };
                    }),
                    kind === "pr"
                      ? { type: "scatter" as const, mode: "lines" as const, x: [0, 1], y: [list[0]?.metrics?.curves?.[scope]?.prevalence ?? 0, list[0]?.metrics?.curves?.[scope]?.prevalence ?? 0], line: { color: C.dim, width: 1, dash: "dot" as const }, hoverinfo: "skip" as const, name: "prevalence" }
                      : { type: "scatter" as const, mode: "lines" as const, x: [0, 1], y: [0, 1], line: { color: C.dim, width: 1, dash: "dot" as const }, hoverinfo: "skip" as const, name: "chance" },
                  ]}
                  layout={baseLayout({ margin: { l: 48, r: 10, t: 6, b: 40 },
                    xaxis: axis({ range: [0, 1], title: { text: kind === "pr" ? "Recall" : "False positive rate" } }),
                    yaxis: axis({ range: [0, 1.02], title: { text: kind === "pr" ? "Precision" : "True positive rate" } }) })}
                />
                <p className="mt-1 text-xs text-dim">
                  {kind === "pr" ? "Dotted line: share of labelled points (a random score's expected precision). " : "Dotted line: chance. "}
                  Curves sweep one global cut-off over each channel's standardised smoothed score — the dynamic threshold itself is not a single global cut-off.
                </p>
              </Panel>
            ))}
          </div>

          <div className="grid gap-6 xl:grid-cols-[1fr_1fr]">
            <Panel eyebrow="SMAP vs MSL" title={`${protocol === "event" ? "Event-wise F0.5" : "Point-wise F1"} by spacecraft`}>
              <Legend items={[{ label: "SMAP", color: "#3987e5", kind: "solid" }, { label: "MSL · Curiosity", color: "#d95926", kind: "solid" }]} />
              <Plot
                height={300}
                data={(["SMAP", "MSL"] as const).map((sc) => ({
                  type: "bar" as const, name: sc, x: list.map(shortLabel),
                  y: list.map((e) => (protocol === "event" ? agg(e, sc)?.event.f0_5 : agg(e, sc)?.point.f1) ?? null),
                  marker: { color: sc === "SMAP" ? "#3987e5" : "#d95926", line: { color: C.panel, width: 2 } },
                  hovertemplate: `${sc} · %{x}: %{y:.3f}<extra></extra>`,
                }))}
                layout={baseLayout({ barmode: "group", margin: { l: 44, r: 8, t: 6, b: 90 }, xaxis: axis({ tickangle: -25, tickfont: { size: 11, color: C.ink2, family: "IBM Plex Sans" } }), yaxis: axis({ range: [0, 1], dtick: 0.25 }) })}
              />
            </Panel>
            <Panel eyebrow="Detection quality" title="Delay, overlap and anomaly classes (event-wise)">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-xs text-muted">
                    <th className="py-1.5 pr-2 font-medium">Method</th>
                    <th className="py-1.5 pr-2 text-right font-medium">Mean delay</th>
                    <th className="py-1.5 pr-2 text-right font-medium">Median delay</th>
                    <th className="py-1.5 pr-2 text-right font-medium">Mean IoU</th>
                    <th className="py-1.5 pr-2 text-right font-medium">Point recall</th>
                    <th className="py-1.5 text-right font-medium">Contextual recall</th>
                  </tr>
                </thead>
                <tbody>
                  {list.map((e) => {
                    const a = agg(e, scope);
                    if (!a) return null;
                    return (
                      <tr key={e.id} className="border-t border-hairline">
                        <td className="py-1.5 pr-2"><span className="inline-flex items-center gap-1.5"><span className="h-2 w-2 rounded-sm" style={{ background: METHOD_COLOR[e.method] }} />{e.method_label}</span></td>
                        <td className="num py-1.5 pr-2 text-right">{a.event.mean_delay == null ? "—" : `${fmtNum(a.event.mean_delay, 0)}`}</td>
                        <td className="num py-1.5 pr-2 text-right">{a.event.median_delay == null ? "—" : `${fmtNum(a.event.median_delay, 0)}`}</td>
                        <td className="num py-1.5 pr-2 text-right">{fmtNum(a.event.mean_iou, 2)}</td>
                        <td className="num py-1.5 pr-2 text-right">{fmtPct(a.event.by_class.point?.recall)}</td>
                        <td className="num py-1.5 text-right">{fmtPct(a.event.by_class.contextual?.recall)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
              <p className="mt-3 text-xs leading-relaxed text-dim">Delay = first detected index inside a detected label minus the label start, in timesteps (0 if the detection began earlier). IoU = overlap of a detected label with the union of its overlapping detections.</p>
            </Panel>
          </div>

          {focusExp && agg(focusExp, scope) && (
            <div className="grid gap-6 xl:grid-cols-[420px_1fr]">
              <Panel eyebrow="Point-wise confusion matrix" title={<Select ariaLabel="Experiment" value={String(focus)} onChange={(v) => setFocus(Number(v))} className="h-8 text-sm">{list.map((e) => <option key={e.id} value={e.id}>{shortLabel(e)}</option>)}</Select>}>
                <ConfusionMatrix a={agg(focusExp, scope)!} />
                <p className="mt-1 text-xs text-dim">Cell shading is log-scaled; counts are exact. Evaluated points: {fmtInt(agg(focusExp, scope)!.n_evaluated)}.</p>
              </Panel>
              <Panel eyebrow="Per channel" title={`${focusExp.name}`} actions={
                <div className="flex gap-3 text-xs">
                  <a className="inline-flex items-center gap-1 text-accent hover:underline" href={exportUrl(`/export/experiments/${focusExp.id}/channels.csv`)}><Download size={12} /> CSV</a>
                  <a className="inline-flex items-center gap-1 text-accent hover:underline" href={exportUrl(`/export/experiments/${focusExp.id}.json`)}><Download size={12} /> JSON</a>
                </div>}>
                <div className="max-h-[360px] overflow-auto">
                  <table className="w-full text-sm">
                    <thead className="sticky top-0 bg-panel">
                      <tr className="text-left text-xs text-muted">
                        {["Channel", "Craft", "TP", "FP", "FN", "Event F1", "Point F1", "AP", "Train", "Infer"].map((h) => <th key={h} className="py-1.5 pr-2 font-medium">{h}</th>)}
                      </tr>
                    </thead>
                    <tbody>
                      {(focusExp.runs ?? []).filter((r) => scope === "overall" || r.spacecraft === scope).map((r) => (
                        <tr key={r.id} className="border-t border-hairline">
                          <td className="num py-1 pr-2"><Link className="text-ink hover:text-accent" to={`/detect?chan=${r.chan_id}&run=${r.id}`}>{r.chan_id}</Link></td>
                          <td className="py-1 pr-2 text-ink-2">{r.spacecraft}</td>
                          <td className="num py-1 pr-2">{r.event.tp ?? "—"}</td>
                          <td className="num py-1 pr-2">{r.event.fp ?? "—"}</td>
                          <td className="num py-1 pr-2">{r.event.fn ?? "—"}</td>
                          <td className="num py-1 pr-2">{fmtNum(r.event.f1, 2)}</td>
                          <td className="num py-1 pr-2">{fmtNum(r.point.f1, 2)}</td>
                          <td className="num py-1 pr-2">{fmtNum(r.average_precision, 2)}</td>
                          <td className="num py-1 pr-2">{fmtDuration(r.train_time_s)}</td>
                          <td className="num py-1">{fmtDuration(r.infer_time_s)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Panel>
            </div>
          )}

          <Panel eyebrow="Cost" title="Training and inference time">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-xs text-muted">
                  {["Experiment", "Origin", "Device", "Channels", "Total training", "Total inference", "Per channel (train)"].map((h) => <th key={h} className="py-1.5 pr-3 font-medium">{h}</th>)}
                </tr>
              </thead>
              <tbody>
                {list.map((e) => (
                  <tr key={e.id} className="border-t border-hairline">
                    <td className="py-1.5 pr-3"><Link to={`/experiments/${e.id}`} className="text-ink hover:text-accent">{shortLabel(e)}</Link></td>
                    <td className="py-1.5 pr-3"><OriginTag origin={e.origin} /></td>
                    <td className="py-1.5 pr-3 text-ink-2">{e.device ?? "—"}{e.environment && (e.environment as { workers?: number }).workers ? ` · ${(e.environment as { workers?: number }).workers} worker(s)` : ""}</td>
                    <td className="num py-1.5 pr-3">{e.n_channels}</td>
                    <td className="num py-1.5 pr-3">{fmtDuration(e.train_time_s)}</td>
                    <td className="num py-1.5 pr-3">{fmtDuration(e.infer_time_s)}</td>
                    <td className="num py-1.5 pr-3">{fmtDuration(e.train_time_s != null ? e.train_time_s / e.n_channels : null)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="mt-3 text-xs text-dim">
              Paper figures (Hundman et al.: overall precision 87.5%, recall 80.0%) are not shown as a comparison: preprocessing details,
              the training setup and the thresholding implementation differ, so the numbers are not directly comparable.
            </p>
          </Panel>
        </>
      )}
    </div>
  );
}
