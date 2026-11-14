import { ScanSearch } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { useDetection, useDetections, useExperiments } from "../api/hooks";
import { TraceStack } from "../components/TraceStack";
import { EmptyState, ErrorState, KV, Legend, Loading, Notice, OutcomeTag, PageHeader, Panel, Segmented, Select } from "../components/ui";
import { fmtInt, fmtNum } from "../lib/format";
import { C, METHOD_COLOR } from "../lib/palette";

function Detail({ id }: { id: number }) {
  const d = useDetection(id);
  if (d.isError) return <ErrorState error={d.error} />;
  if (!d.data) return <Loading height={520} />;
  const x = d.data;
  const w = x.window as Record<string, number | string | null | number[] | Record<string, unknown>>;
  const isNdt = w.mode !== "static";
  return (
    <div className="space-y-6">
      <Panel
        eyebrow={`${x.spacecraft} · ${x.chan_id} · detection #${x.id}`}
        title={`Model-detected interval ${x.start}–${x.end}`}
        actions={<Link to={`/detect?chan=${x.chan_id}&run=${x.run_id}`} className="text-[12.5px] text-accent hover:underline">Open full channel →</Link>}
      >
        <Legend
          items={[
            { label: "Actual telemetry", color: C.actual },
            ...(x.context.prediction ? [{ label: "Model prediction", color: C.prediction }] : []),
            { label: "Smoothed score", color: C.smoothed },
            { label: "Threshold", color: C.threshold, kind: "dash" as const },
            { label: "Model detection", color: C.detect, kind: "band" as const },
            { label: "Expert label", color: C.label, kind: "hatch" as const },
          ]}
        />
        <TraceStack
          actual={x.context.actual}
          prediction={x.context.prediction}
          raw={x.context.raw_score}
          smoothed={x.context.smoothed}
          thresholdHigh={x.context.threshold_high}
          thresholdLow={x.context.threshold_low}
          labels={x.labels_in_context.map((l) => ({ ...l, start: Math.max(l.start, x.context.start), end: Math.min(l.end, x.context.end - 1) }))}
          detections={[{ start: x.start, end: x.end, score: x.score, id: x.id }]}
          xOffset={x.context.start}
          evalStart={250}
          height={480}
          highlight={[x.start, x.end]}
          uirevision={x.id}
        />
      </Panel>
      <div className="grid gap-6 lg:grid-cols-3">
        <Panel eyebrow="Identification" title="Interval">
          <KV k="Spacecraft" v={x.spacecraft} mono={false} />
          <KV k="Channel identifier" v={x.chan_id} />
          <KV k="Start / end index" v={`${x.start} / ${x.end}`} />
          <KV k="Length" v={`${fmtInt(x.stats.length)} steps`} />
          <KV k="Model" v={<span className="inline-flex items-center gap-1.5"><span className="h-2 w-2 rounded-sm" style={{ background: METHOD_COLOR[x.method] }} />{x.method_label}</span>} mono={false} />
          <KV k="Experiment" v={<Link className="text-accent hover:underline" to={`/experiments/${x.experiment_id}`}>{x.experiment_name}</Link>} mono={false} />
        </Panel>
        <Panel eyebrow="Score & threshold" title="Why it was flagged">
          <KV k="Anomaly score (margin)" v={fmtNum(x.score, 3)} />
          <KV k="Peak smoothed error" v={fmtNum(x.peak_error, 4)} />
          <KV k="Threshold at peak" v={fmtNum(x.threshold, 4)} />
          <KV k="Direction" v={x.direction === "low" ? "unusually low error" : x.direction === "both" ? "high and low" : "unusually high error"} mono={false} />
          {isNdt ? (
            <>
              <KV k="Window z (high)" v={fmtNum(w.z_high as number, 1)} />
              <KV k="Window mean / sd" v={`${fmtNum(w.mean as number, 4)} / ${fmtNum(w.sd as number, 4)}`} />
              <KV k="Window span" v={Array.isArray(w.zone) ? `${w.start}–${w.end} (reports ${w.zone[0]}–${w.zone[1]})` : "—"} />
            </>
          ) : (
            <KV k="Static ε (training μ + kσ)" v={fmtNum(w.epsilon as number, 4)} />
          )}
          <p className="mt-3 text-[11.5px] leading-relaxed text-dim">{x.confidence_note}</p>
        </Panel>
        <Panel eyebrow="Benchmark reference" title="Expert annotation">
          <div className="mb-3"><OutcomeTag outcome={x.outcome} /></div>
          {x.matched_labels.length ? (
            x.matched_labels.map((l) => (
              <div key={l.start}>
                <KV k="Labelled interval" v={`${l.start}–${l.end}`} />
                <KV k="Anomaly class" v={l.cls} mono={false} />
                <KV k="Overlap" v={`${Math.max(0, Math.min(l.end, x.end) - Math.max(l.start, x.start) + 1)} steps`} />
              </div>
            ))
          ) : (
            <Notice>
              No expert label overlaps this interval. It counts as a false positive in the benchmark metrics, but unlabelled regions are not
              guaranteed to be normal: treat it as <em>uncertain or unexplained behaviour</em> that warrants a look, not as a confirmed fault.
            </Notice>
          )}
          <div className="mt-4">
            <div className="eyebrow mb-1">Telemetry in interval</div>
            <KV k="Min / max" v={`${fmtNum(x.stats.actual_min, 3)} / ${fmtNum(x.stats.actual_max, 3)}`} />
            <KV k="Mean" v={fmtNum(x.stats.actual_mean, 3)} />
            {x.stats.mean_abs_forecast_error != null && <KV k="Mean |forecast error|" v={fmtNum(x.stats.mean_abs_forecast_error, 4)} />}
          </div>
        </Panel>
      </div>
    </div>
  );
}

export default function Investigation() {
  const params = useParams();
  const nav = useNavigate();
  const selected = params.id ? Number(params.id) : null;
  const exps = useExperiments(false);
  const completed = (exps.data ?? []).filter((e) => ["completed", "partial"].includes(e.status));
  const [expId, setExpId] = useState<number | null>(null);
  useEffect(() => {
    if (expId == null && completed.length) {
      setExpId((completed.find((e) => e.method === "lstm_forecaster" && e.spacecraft === "ALL") ?? completed[0]).id);
    }
  }, [completed, expId]);
  const [outcome, setOutcome] = useState<"all" | "matches_label" | "no_label_overlap">("all");
  const [chanFilter, setChanFilter] = useState("");
  const dets = useDetections({ experiment_id: expId, outcome: outcome === "all" ? null : outcome });
  const rows = useMemo(
    () => (dets.data ?? []).filter((d) => !chanFilter || d.chan_id.toLowerCase().includes(chanFilter.toLowerCase())),
    [dets.data, chanFilter],
  );

  return (
    <div className="space-y-6">
      <PageHeader eyebrow="Anomaly investigation" title="Detected intervals">
        Every interval a model flagged, ranked by score. Model-detected anomalies are unusual patterns that warrant investigation; they
        are compared with — but are not the same as — the benchmark's expert-labelled anomalies.
      </PageHeader>

      <div className="grid gap-6 xl:grid-cols-[420px_1fr]">
        <Panel eyebrow="Filter" title={`${rows.length} interval${rows.length === 1 ? "" : "s"}`} pad={false}>
          <div className="space-y-3 px-5 pb-3">
            <Select ariaLabel="Experiment" value={String(expId ?? "")} onChange={(v) => setExpId(Number(v))} className="w-full">
              {completed.map((e) => (
                <option key={e.id} value={e.id}>
                  #{e.id} · {e.name}
                </option>
              ))}
            </Select>
            <div className="flex flex-wrap items-center gap-2">
              <Segmented
                size="sm"
                ariaLabel="Outcome"
                value={outcome}
                onChange={setOutcome}
                options={[
                  { value: "all", label: "All" },
                  { value: "matches_label", label: "Overlaps label" },
                  { value: "no_label_overlap", label: "No label" },
                ]}
              />
              <input
                aria-label="Filter by channel"
                placeholder="Channel…"
                value={chanFilter}
                onChange={(e) => setChanFilter(e.target.value)}
                className="num h-7 w-24 rounded-md border border-edge bg-raised px-2 text-[12.5px] text-ink outline-none"
              />
            </div>
          </div>
          <div className="max-h-[720px] overflow-y-auto border-t border-hairline">
            {dets.isError ? (
              <div className="p-4"><ErrorState error={dets.error} /></div>
            ) : !dets.data ? (
              <div className="p-4"><Loading height={300} /></div>
            ) : rows.length === 0 ? (
              <div className="p-4 text-[13px] text-muted">No detected intervals match.</div>
            ) : (
              <table className="w-full text-[12.5px]">
                <thead className="sticky top-0 bg-panel">
                  <tr className="text-left text-[11px] text-muted">
                    <th className="px-5 py-2 font-medium">Channel</th>
                    <th className="py-2 font-medium">Interval</th>
                    <th className="py-2 text-right font-medium">Score</th>
                    <th className="py-2 pr-5 pl-3 font-medium">Ref.</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((d) => (
                    <tr
                      key={d.id}
                      onClick={() => nav(`/investigate/${d.id}`)}
                      className={`cursor-pointer border-t border-hairline ${selected === d.id ? "bg-raised" : "hover:bg-raised/50"}`}
                    >
                      <td className="num px-5 py-1.5 text-ink">{d.chan_id}</td>
                      <td className="num py-1.5 text-ink-2">{d.start}–{d.end}</td>
                      <td className="num py-1.5 text-right">{fmtNum(d.score, 2)}</td>
                      <td className="py-1.5 pr-5 pl-3">
                        {d.outcome === "matches_label" ? (
                          <span className="hatch-label inline-block h-2.5 w-2.5 rounded-sm border border-label/60" title="Overlaps an expert label" />
                        ) : (
                          <span className="inline-block h-2.5 w-2.5 rounded-sm border border-dim" title="No expert label overlap" />
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </Panel>
        <div>
          {selected ? (
            <Detail id={selected} />
          ) : (
            <EmptyState icon={<ScanSearch size={28} />} title="Select an interval">
              Pick a detected interval on the left to see its telemetry, model prediction, score, threshold window and corresponding
              benchmark annotation.
            </EmptyState>
          )}
        </div>
      </div>
    </div>
  );
}
