import { useMutation } from "@tanstack/react-query";
import { Play, Radar, RotateCcw } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { api } from "../api/client";
import { useRun, useSeries } from "../api/hooks";
import type { Series } from "../api/types";
import { ChannelPicker, RunPicker, useRunsForChannel } from "../components/pickers";
import { TraceStack } from "../components/TraceStack";
import { Button, EmptyState, ErrorState, KV, Legend, Loading, Notice, OutcomeTag, PageHeader, Panel, Segmented } from "../components/ui";
import { useToast } from "../components/toast";
import { fmtInt, fmtNum, fmtPct } from "../lib/format";
import { C } from "../lib/palette";
import { useSelection } from "../lib/selection";

function MetricsBlock({ s, unscoreable }: { s: Series["metrics"] | undefined; unscoreable?: number }) {
  if (!s) return null;
  const e = s.event;
  const p = s.point;
  return (
    <div className="grid gap-6 md:grid-cols-2">
      <div>
        <div className="eyebrow mb-2">Event-wise (labelled sequences)</div>
        <KV k="Labelled sequences detected (TP)" v={`${e.tp} / ${e.n_labels}`} />
        <KV k="Detections without label overlap (FP)" v={e.fp} />
        <KV k="Missed sequences (FN)" v={e.fn} />
        <KV k="Precision" v={fmtPct(e.precision)} />
        <KV k="Recall" v={fmtPct(e.recall)} />
        <KV k="F0.5" v={fmtNum(e.f0_5)} />
        <KV k="Mean detection delay" v={e.mean_delay == null ? "—" : `${fmtNum(e.mean_delay, 0)} steps`} />
      </div>
      <div>
        <div className="eyebrow mb-2">Point-wise (every evaluated timestep)</div>
        <KV k="TP / FP" v={`${fmtInt(p.tp)} / ${fmtInt(p.fp)}`} />
        <KV k="FN / TN" v={`${fmtInt(p.fn)} / ${fmtInt(p.tn)}`} />
        <KV k="Precision" v={fmtPct(p.precision)} />
        <KV k="Recall (TPR)" v={fmtPct(p.recall)} />
        <KV k="F1" v={fmtNum(p.f1)} />
        <KV k="False positive rate" v={fmtPct(p.fpr, 2)} />
        <KV k="Evaluated points" v={fmtInt(s.n_evaluated)} />
      </div>
      {(unscoreable ?? s.unscoreable_label_points) > 0 && (
        <div className="md:col-span-2">
          <Notice>{unscoreable ?? s.unscoreable_label_points} labelled points fall before index 250 and cannot be scored.</Notice>
        </div>
      )}
    </div>
  );
}

export default function DetectionCentre() {
  const { chan, setChan, method, setMethod } = useSelection();
  const [params, setParams] = useSearchParams();
  const nav = useNavigate();
  const toast = useToast();
  const runs = useRunsForChannel(chan);
  const urlRun = params.get("run") ? Number(params.get("run")) : null;

  useEffect(() => {
    const c = params.get("chan");
    if (c && c !== chan) setChan(c);
  }, [params]); // eslint-disable-line react-hooks/exhaustive-deps

  const runId = useMemo(() => {
    if (urlRun && runs.some((r) => r.run_id === urlRun)) return urlRun;
    return (runs.find((r) => r.method === method) ?? runs[0])?.run_id ?? null;
  }, [urlRun, runs, method]);

  const run = useRun(runId);
  const stored = useSeries(runId);
  const [live, setLive] = useState<Series | null>(null);
  const [thrMode, setThrMode] = useState<"ndt" | "static">("ndt");
  const [p, setP] = useState(0.13);
  const [buffer, setBuffer] = useState(100);
  const [k, setK] = useState(3);

  useEffect(() => {
    setLive(null);
    const t = run.data?.threshold_config;
    if (t) {
      setThrMode((t.mode as "ndt" | "static") ?? "ndt");
      setP(Number(t.p ?? 0.13));
      setBuffer(Number(t.error_buffer ?? 100));
      setK(Number(t.static_k ?? 3));
    }
  }, [runId, run.data?.threshold_config]);

  const infer = useMutation({
    mutationFn: (overrides: Record<string, unknown> | null) =>
      api.post<Series>(`/runs/${runId}/infer`, { threshold: overrides }),
    onSuccess: (d) => {
      setLive(d);
      toast({
        tone: "success",
        title: "Inference complete",
        body: `Model executed on ${fmtInt(d.n)} test points in ${fmtNum((d.timing_ms?.score ?? 0) / 1000, 2)} s (${d.device}).`,
      });
    },
    onError: (e: Error) => toast({ tone: "error", title: "Inference failed", body: e.message }),
  });

  const s = live ?? stored.data;
  const isLstm = run.data?.method === "lstm_forecaster";

  const change = (c: string) => {
    setChan(c);
    setParams({ chan: c });
  };

  const labelStatus = live?.metrics?.event.per_label ?? run.data?.metrics?.event.per_label ?? [];

  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow="Anomaly detection centre"
        title={`${chan} · ${run.data ? run.data.experiment_name : "select a model"}`}
        actions={<ChannelPicker chan={chan} onChange={change} />}
      >
        Compare predicted and actual telemetry, inspect the smoothed anomaly score against the dynamic threshold, and check model
        detections against the benchmark's expert labels.
      </PageHeader>

      {!runs.length ? (
        <EmptyState
          icon={<Radar size={28} />}
          title={`No trained model for ${chan} yet`}
          action={
            <Link to={`/lab?chan=${chan}`}>
              <Button variant="primary">Train a model for {chan}</Button>
            </Link>
          }
        >
          The detection centre only shows outputs of models that were actually trained and evaluated.
        </EmptyState>
      ) : (
        <>
          <Panel>
            <div className="flex flex-wrap items-end gap-x-6 gap-y-4 pt-4">
              <div className="min-w-0 max-w-full">
                <div className="eyebrow mb-2">Trained model</div>
                <RunPicker
                  chan={chan}
                  runId={runId}
                  onChange={(id) => {
                    const r = runs.find((x) => x.run_id === id);
                    if (r) setMethod(r.method);
                    setParams({ chan, run: String(id) });
                  }}
                />
              </div>
              <div>
                <div className="eyebrow mb-2">Threshold</div>
                <Segmented
                  ariaLabel="Threshold mode"
                  value={thrMode}
                  onChange={setThrMode}
                  options={[
                    { value: "ndt", label: "Dynamic (NDT)" },
                    { value: "static", label: "Static μ+kσ" },
                  ]}
                />
              </div>
              {thrMode === "ndt" ? (
                <>
                  <label className="text-xs text-muted">
                    <div className="eyebrow mb-2">Pruning p</div>
                    <input type="number" step={0.01} min={0} max={1} value={p} onChange={(e) => setP(Number(e.target.value))}
                      className="num h-9 w-24 rounded-md border border-edge bg-raised px-2 text-ink" />
                  </label>
                  <label className="text-xs text-muted">
                    <div className="eyebrow mb-2">Error buffer</div>
                    <input type="number" step={10} min={0} max={500} value={buffer} onChange={(e) => setBuffer(Number(e.target.value))}
                      className="num h-9 w-24 rounded-md border border-edge bg-raised px-2 text-ink" />
                  </label>
                </>
              ) : (
                <label className="text-xs text-muted">
                  <div className="eyebrow mb-2">k (training σ)</div>
                  <input type="number" step={0.5} min={0.5} max={20} value={k} onChange={(e) => setK(Number(e.target.value))}
                    className="num h-9 w-24 rounded-md border border-edge bg-raised px-2 text-ink" />
                </label>
              )}
              <div className="ml-auto flex items-center gap-2">
                {live && (
                  <Button variant="ghost" onClick={() => setLive(null)}>
                    <RotateCcw size={14} /> Show stored result
                  </Button>
                )}
                <Button
                  variant="primary"
                  loading={infer.isPending}
                  disabled={!runId || (run.data && !run.data.model_available && !["persistence", "rolling_zscore", "random"].includes(run.data.method))}
                  onClick={() => infer.mutate({ mode: thrMode, p, error_buffer: buffer, static_k: k })}
                >
                  <Play size={14} /> Run inference
                </Button>
              </div>
            </div>
            {run.data && !run.data.model_available && !["persistence", "rolling_zscore", "random"].includes(run.data.method) && (
              <div className="mt-4">
                <Notice tone="warn">The model file for this run is not on this machine, so live inference is unavailable. Stored results are shown.</Notice>
              </div>
            )}
          </Panel>

          <Panel
            eyebrow={s ? (s.source === "live_inference" ? "Live inference result" : "Stored evaluation result") : "Result"}
            title={
              s ? (
                <span>
                  {s.detections.length} model-detected interval{s.detections.length === 1 ? "" : "s"} · {s.labels.length} expert label
                  {s.labels.length === 1 ? "" : "s"}
                </span>
              ) : (
                " "
              )
            }
            actions={
              live?.timing_ms && (
                <span className="num text-xs text-muted">
                  model {fmtNum(live.timing_ms.score / 1000, 2)} s · threshold {fmtNum(live.timing_ms.threshold, 0)} ms · {live.device}
                </span>
              )
            }
          >
            {stored.isError && !live ? (
              <ErrorState error={stored.error} />
            ) : !s ? (
              <Loading height={560} />
            ) : (
              <>
                <Legend
                  items={[
                    { label: "Actual telemetry", color: C.actual },
                    ...(s.prediction ? [{ label: isLstm ? "LSTM forecast" : "Model prediction", color: C.prediction }] : []),
                    { label: "Smoothed score", color: C.smoothed },
                    { label: "Raw score", color: C.raw },
                    { label: "Dynamic threshold", color: C.threshold, kind: "dash" as const },
                    { label: "Model detection", color: C.detect, kind: "band" as const },
                    { label: "Expert label", color: C.label, kind: "hatch" as const },
                  ]}
                />
                <TraceStack
                  actual={s.actual}
                  prediction={s.prediction}
                  raw={s.raw_score}
                  smoothed={s.smoothed}
                  thresholdHigh={s.threshold_high}
                  thresholdLow={s.threshold_low}
                  labels={s.labels}
                  detections={s.detections}
                  evalStart={s.eval_start}
                  height={600}
                  uirevision={`${runId}`}
                  predictionLabel={isLstm ? "LSTM forecast" : "Prediction"}
                  onSelectDetection={(d) => d.id && nav(`/investigate/${d.id}`)}
                />
                <p className="mt-2 text-xs text-muted">
                  Drag to zoom (double-click to reset). Click a violet bar in the lane to investigate it. The grey region before index{" "}
                  {s.eval_start} is not scored.
                </p>
              </>
            )}
          </Panel>

          {s && (
            <div className="grid gap-6 xl:grid-cols-[1.1fr_1fr]">
              <Panel eyebrow="Detected intervals" title="Model detections">
                {s.detections.length ? (
                  <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="text-left text-xs text-muted">
                        <th className="py-1.5 pr-4 font-medium">Start–end</th>
                        <th className="py-1.5 pr-4 text-right font-medium">Length</th>
                        <th className="py-1.5 text-right font-medium">Score</th>
                        <th className="py-1.5 pl-4 font-medium">Direction</th>
                        <th className="py-1.5 font-medium">Reference</th>
                        <th />
                      </tr>
                    </thead>
                    <tbody>
                      {s.detections.map((d) => (
                        <tr key={`${d.start}-${d.end}`} className="border-t border-hairline">
                          <td className="num whitespace-nowrap py-1.5 pr-4">{d.start}–{d.end}</td>
                          <td className="num py-1.5 pr-4 text-right">{d.end - d.start + 1}</td>
                          <td className="num py-1.5 text-right">{fmtNum(d.score, 2)}</td>
                          <td className="py-1.5 pl-4 text-ink-2">{d.direction === "low" ? "unusually low error" : d.direction === "both" ? "high + low" : "high error"}</td>
                          <td className="py-1.5"><OutcomeTag outcome={d.outcome} /></td>
                          <td className="py-1.5 text-right">
                            {d.id ? (
                              <Link to={`/investigate/${d.id}`} className="text-sm text-accent hover:underline">Investigate</Link>
                            ) : (
                              <span className="text-xs text-dim" title="Live inference results are not stored">not stored</span>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  </div>
                ) : (
                  <p className="text-sm text-muted">The model flagged no interval on this channel.</p>
                )}
                <p className="mt-3 text-xs leading-relaxed text-dim">
                  Score = margin of the smoothed error above the window threshold ÷ (window mean + sd). It indicates severity; it is not a
                  calibrated probability.
                </p>
              </Panel>
              <Panel eyebrow="Against the reference" title="Expert labels and metrics">
                {labelStatus.length > 0 && (
                  <div className="mb-4 flex flex-wrap gap-2">
                    {labelStatus.map((l) => (
                      <span key={l.start} className={`num inline-flex items-center gap-2 rounded border px-2 py-1 text-xs ${l.detected ? "border-detect/60 text-ink" : "border-hairline text-ink-2"}`}>
                        <span className="hatch-label inline-block h-2.5 w-2.5 rounded-sm border border-label/60" aria-hidden />
                        {l.start}–{l.end} · {l.cls} · {l.detected ? `detected (delay ${l.delay})` : "missed"}
                      </span>
                    ))}
                  </div>
                )}
                <MetricsBlock s={live?.metrics ?? (run.data?.metrics ? { event: run.data.metrics.event, point: run.data.metrics.point, unscoreable_label_points: run.data.metrics.unscoreable_label_points, n_evaluated: run.data.metrics.n_evaluated } : undefined)} />
              </Panel>
            </div>
          )}
        </>
      )}
    </div>
  );
}
