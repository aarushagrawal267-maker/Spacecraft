import { Download, Square } from "lucide-react";
import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { exportUrl } from "../api/client";
import { useCancelExperiment, useExperiment, useInvalidateOnFinish, useRun } from "../api/hooks";
import { Button, ErrorState, KV, Loading, Notice, OriginTag, PageHeader, Panel, StatusBadge } from "../components/ui";
import { fmtDate, fmtDuration, fmtNum, fmtPct } from "../lib/format";
import { METHOD_COLOR } from "../lib/palette";
import { LossChart } from "./ModelLab";

export default function ExperimentDetail() {
  const id = Number(useParams().id);
  const exp = useExperiment(id, true);
  useInvalidateOnFinish(exp.data?.status);
  const cancel = useCancelExperiment();
  const [runId, setRunId] = useState<number | null>(null);
  const run = useRun(runId);

  if (exp.isError) return <ErrorState error={exp.error} />;
  if (!exp.data) return <Loading height={400} />;
  const e = exp.data;
  const o = e.metrics?.overall;
  const env = (e.environment ?? {}) as { workers?: number; torch_threads?: number; system?: Record<string, unknown> };
  const trainable = ["lstm_forecaster", "lstm_autoencoder"].includes(e.method);

  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow={`Experiment #${e.id}`}
        title={e.name}
        actions={
          <div className="flex items-center gap-3">
            <OriginTag origin={e.origin} />
            <StatusBadge status={e.status} />
            {["queued", "running"].includes(e.status) && <Button variant="danger" size="sm" onClick={() => cancel.mutate(e.id)}><Square size={12} /> Cancel</Button>}
            <a href={exportUrl(`/export/experiments/${e.id}.json`)}><Button size="sm"><Download size={13} /> JSON</Button></a>
            <a href={exportUrl(`/export/experiments/${e.id}/detections.csv`)}><Button size="sm"><Download size={13} /> Detections CSV</Button></a>
          </div>
        }
      >
        <span className="inline-flex items-center gap-2"><span className="h-2.5 w-2.5 rounded-sm" style={{ background: METHOD_COLOR[e.method] }} />{e.method_label}{e.profile ? ` · ${e.profile} profile` : ""} · {e.spacecraft} · {e.n_channels} channels</span>
      </PageHeader>

      {e.origin === "reference" && (
        <Notice>
          Reference result: computed with this application's code (commit <span className="num">{e.git_commit ?? "—"}</span>) on the full benchmark
          and imported on first start. {e.notes}
        </Notice>
      )}
      {e.message && <Notice tone="warn">{e.message}</Notice>}

      <div className="grid gap-6 lg:grid-cols-3">
        <Panel eyebrow="Evaluation" title="Headline metrics">
          <KV k="Event precision" v={fmtPct(o?.event.precision)} />
          <KV k="Event recall" v={fmtPct(o?.event.recall)} />
          <KV k="Event F1 / F0.5" v={`${fmtNum(o?.event.f1)} / ${fmtNum(o?.event.f0_5)}`} />
          <KV k="Event TP / FP / FN" v={o ? `${o.event.tp} / ${o.event.fp} / ${o.event.fn}` : "—"} />
          <KV k="Point precision / recall" v={`${fmtPct(o?.point.precision)} / ${fmtPct(o?.point.recall)}`} />
          <KV k="Point F1" v={fmtNum(o?.point.f1)} />
          <KV k="Point FPR" v={fmtPct(o?.point.fpr, 2)} />
          <KV k="Mean AP (per channel)" v={fmtNum(o?.mean_average_precision)} />
          <KV k="Channels completed / failed" v={`${e.metrics?.n_completed ?? 0} / ${e.metrics?.n_failed ?? 0}`} />
        </Panel>
        <Panel eyebrow="Configuration" title="Model & threshold">
          {Object.entries(e.config).map(([k, v]) => <KV key={k} k={k} v={String(v)} />)}
          <div className="mt-3 eyebrow">Threshold</div>
          {["mode", "batch_size", "window_size", "smoothing_perc", "error_buffer", "p", "inverse", "static_k"].map((k) => (
            <KV key={k} k={k} v={String(e.threshold_config[k])} />
          ))}
        </Panel>
        <Panel eyebrow="Reproducibility" title="Run record">
          <KV k="Created" v={fmtDate(e.created_at)} mono={false} />
          <KV k="Started / finished" v={`${fmtDate(e.started_at)} → ${fmtDate(e.finished_at)}`} mono={false} />
          <KV k="Seed" v={e.seed} />
          <KV k="Code commit" v={e.git_commit ?? "—"} />
          <KV k="Dataset checksum" v={e.dataset_checksum ? e.dataset_checksum.slice(0, 16) + "…" : "—"} />
          <KV k="Device" v={e.device ?? "—"} />
          <KV k="Workers × threads" v={`${env.workers ?? 1} × ${env.torch_threads ?? "default"}`} />
          <KV k="Platform" v={String(env.system?.platform ?? "—")} mono={false} />
          <KV k="PyTorch" v={String(env.system?.torch ?? "—")} />
          <KV k="Total training time" v={fmtDuration(e.train_time_s)} />
          <KV k="Total inference time" v={fmtDuration(e.infer_time_s)} />
        </Panel>
      </div>

      <div className="grid gap-6 xl:grid-cols-[1fr_440px]">
        <Panel eyebrow="Channel runs" title={`${e.runs?.length ?? 0} channels`} pad={false}>
          <div className="max-h-[520px] overflow-auto">
            <table className="w-full text-[12.5px]">
              <thead className="sticky top-0 bg-panel">
                <tr className="text-left text-[11px] text-muted">
                  {["Channel", "Status", "Epochs", "Best val", "TP/FP/FN", "Event F1", "Point F1", "AP", "Train", ""].map((h) => <th key={h} className="px-3 py-2 font-medium first:pl-5">{h}</th>)}
                </tr>
              </thead>
              <tbody>
                {(e.runs ?? []).map((r) => (
                  <tr key={r.id} onClick={() => setRunId(r.id)} className={`cursor-pointer border-t border-hairline ${runId === r.id ? "bg-raised" : "hover:bg-raised/40"}`}>
                    <td className="num py-1.5 pr-3 pl-5">{r.chan_id}</td>
                    <td className="px-3 py-1.5"><StatusBadge status={r.status} /></td>
                    <td className="num px-3 py-1.5">{r.epochs_run ?? "—"}{r.best_epoch ? ` (best ${r.best_epoch})` : ""}</td>
                    <td className="num px-3 py-1.5">{fmtNum(r.best_val_loss, 4)}</td>
                    <td className="num px-3 py-1.5">{r.event.tp ?? "—"}/{r.event.fp ?? "—"}/{r.event.fn ?? "—"}</td>
                    <td className="num px-3 py-1.5">{fmtNum(r.event.f1, 2)}</td>
                    <td className="num px-3 py-1.5">{fmtNum(r.point.f1, 2)}</td>
                    <td className="num px-3 py-1.5">{fmtNum(r.average_precision, 2)}</td>
                    <td className="num px-3 py-1.5">{fmtDuration(r.train_time_s)}</td>
                    <td className="px-3 py-1.5 text-right">{r.status === "completed" && <Link onClick={(ev) => ev.stopPropagation()} to={`/detect?chan=${r.chan_id}&run=${r.id}`} className="text-accent hover:underline">Inspect</Link>}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
        <Panel eyebrow={run.data ? `Run ${run.data.chan_id}` : "Run"} title={run.data ? (trainable ? "Training history" : "Run details") : "Select a channel run"}>
          {!runId ? (
            <p className="text-[13px] text-muted">Click a row to see its loss curves and detections.</p>
          ) : !run.data ? (
            <Loading height={260} />
          ) : (
            <>
              {trainable && <LossChart epochs={run.data.epochs} height={240} />}
              <div className="mt-3">
                <KV k="Model file" v={run.data.model_file ?? "none (stateless method)"} />
                <KV k="Model available on this machine" v={run.data.model_available ? "yes" : "no"} mono={false} />
                <KV k="Detections" v={run.data.detections.length} />
                {run.data.error && <KV k="Error" v={run.data.error} mono={false} />}
              </div>
            </>
          )}
        </Panel>
      </div>
    </div>
  );
}
