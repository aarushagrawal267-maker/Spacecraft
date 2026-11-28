import { FlaskConical, Play, Square } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { useCancelExperiment, useChannels, useCreateExperiment, useExperiment, useExperiments, useInvalidateOnFinish, useMethods, useSystem } from "../api/hooks";
import type { EpochLog } from "../api/types";
import { Plot, axis, baseLayout } from "../components/Plot";
import { Button, EmptyState, ErrorState, KV, Legend, Loading, Notice, PageHeader, Panel, Segmented, Select, StatusBadge } from "../components/ui";
import { useToast } from "../components/toast";
import { fmtDuration, fmtInt, fmtNum } from "../lib/format";
import { C, METHOD_COLOR } from "../lib/palette";
import { useSelection } from "../lib/selection";

const TRAIN_COLOR = "#3987e5";
const VAL_COLOR = "#d95926";

const PARAM_HELP: Record<string, string> = {
  window: "History length (timesteps) fed to the model",
  horizon: "Future values predicted per window (first is scored)",
  hidden: "LSTM units per layer",
  layers: "Stacked LSTM layers",
  latent: "Autoencoder bottleneck size",
  dropout: "Dropout rate",
  epochs: "Maximum training epochs",
  batch_size: "Mini-batch size",
  lr: "Adam learning rate",
  patience: "Early-stopping patience (epochs)",
  min_delta: "Minimum validation improvement",
  val_fraction: "Chronological validation share of the training split",
  train_stride: "Stride between training windows",
  lags: "Lagged values used as features",
  alpha: "Ridge regularisation",
  n_estimators: "Number of trees",
  max_samples: "Samples per tree",
  sd_floor: "Minimum standard deviation",
};

export function useEpochStream(expId: number | null, active: boolean) {
  const [epochs, setEpochs] = useState<Record<string, EpochLog[]>>({});
  useEffect(() => {
    setEpochs({});
    if (!expId || !active) return;
    const es = new EventSource(`/api/experiments/${expId}/events`);
    es.addEventListener("epoch", (ev) => {
      const d = JSON.parse((ev as MessageEvent).data);
      setEpochs((prev) => ({ ...prev, [d.chan_id]: [...(prev[d.chan_id] ?? []).filter((e) => e.epoch !== d.epoch), d] }));
    });
    es.addEventListener("done", () => es.close());
    es.onerror = () => es.close();
    return () => es.close();
  }, [expId, active]);
  return epochs;
}

export function LossChart({ epochs, height = 260 }: { epochs: EpochLog[]; height?: number }) {
  if (!epochs.length) return <div className="flex items-center justify-center text-sm text-muted" style={{ height }}>Waiting for the first epoch…</div>;
  const x = epochs.map((e) => e.epoch);
  const hasVal = epochs.some((e) => e.val_loss != null);
  const last = epochs[epochs.length - 1];
  return (
    <>
      <Legend items={[{ label: "Training loss (MSE)", color: TRAIN_COLOR }, ...(hasVal ? [{ label: "Validation loss", color: VAL_COLOR }] : [])]} />
      <Plot
        height={height}
        data={[
          { type: "scatter", mode: "lines+markers", x, y: epochs.map((e) => e.train_loss), name: "train", line: { color: TRAIN_COLOR, width: 2 }, marker: { size: 6, color: TRAIN_COLOR, line: { color: C.panel, width: 2 } }, hovertemplate: "epoch %{x}: %{y:.5f}<extra>train</extra>" },
          ...(hasVal ? [{ type: "scatter" as const, mode: "lines+markers" as const, x, y: epochs.map((e) => e.val_loss), name: "validation", line: { color: VAL_COLOR, width: 2 }, marker: { size: 6, color: VAL_COLOR, line: { color: C.panel, width: 2 } }, hovertemplate: "epoch %{x}: %{y:.5f}<extra>validation</extra>" }] : []),
        ]}
        layout={baseLayout({
          hovermode: "x unified",
          margin: { l: 60, r: 70, t: 8, b: 36 },
          xaxis: axis({ title: { text: "Epoch" }, dtick: epochs.length > 20 ? 5 : 1 }),
          yaxis: axis({ title: { text: "MSE (scaled units²)" }, rangemode: "tozero" }),
          annotations: [
            { x: last.epoch, y: last.train_loss, text: fmtNum(last.train_loss, 4), xanchor: "left", xshift: 8, showarrow: false, font: { size: 11, family: "IBM Plex Mono", color: C.ink2 } },
            ...(last.val_loss != null ? [{ x: last.epoch, y: last.val_loss, text: fmtNum(last.val_loss, 4), xanchor: "left" as const, xshift: 8, showarrow: false, font: { size: 11, family: "IBM Plex Mono", color: C.ink2 } }] : []),
          ],
        })}
      />
    </>
  );
}

export default function ModelLab() {
  const methods = useMethods();
  const channels = useChannels();
  const sys = useSystem();
  const exps = useExperiments();
  const create = useCreateExperiment();
  const cancel = useCancelExperiment();
  const toast = useToast();
  const { chan } = useSelection();
  const [params] = useSearchParams();

  const [method, setMethod] = useState("lstm_forecaster");
  const [profile, setProfile] = useState("quick");
  const [cfg, setCfg] = useState<Record<string, number>>({});
  const [scope, setScope] = useState<"channel" | "list" | "SMAP" | "MSL" | "ALL">("channel");
  const [selChan, setSelChan] = useState(params.get("chan") ?? chan);
  const [list, setList] = useState<string[]>([]);
  const [thrMode, setThrMode] = useState<"ndt" | "static">("ndt");
  const [workers, setWorkers] = useState(1);
  const [seed, setSeed] = useState(0);
  const [name, setName] = useState("");
  const [watchId, setWatchId] = useState<number | null>(null);

  const spec = methods.data?.methods.find((m) => m.key === method);
  useEffect(() => {
    if (!spec) return;
    const profiles = Object.keys(spec.profiles);
    const p = profiles.includes(profile) ? profile : profiles[0] ?? "";
    if (p !== profile) setProfile(p);
    setCfg({ ...(p ? spec.profiles[p] : spec.defaults) });
  }, [spec, profile]);

  useEffect(() => {
    if (watchId == null && exps.data) {
      const running = exps.data.find((e) => e.status === "running" && e.origin === "local") ?? exps.data.find((e) => e.origin === "local");
      if (running) setWatchId(running.id);
    }
  }, [exps.data, watchId]);

  const watched = useExperiment(watchId, true);
  useInvalidateOnFinish(watched.data?.status);
  const active = !!watched.data && ["queued", "running"].includes(watched.data.status);
  const live = useEpochStream(watchId, active);
  const [viewChan, setViewChan] = useState<string | null>(null);
  const runs = watched.data?.runs ?? [];
  const shownChan = viewChan && runs.some((r) => r.chan_id === viewChan) ? viewChan : watched.data?.current_channel ?? runs[0]?.chan_id ?? null;
  const shownRun = runs.find((r) => r.chan_id === shownChan);
  const [storedEpochs, setStoredEpochs] = useState<EpochLog[]>([]);
  useEffect(() => {
    if (!shownRun || active) return;
    fetch(`/api/runs/${shownRun.id}`).then((r) => r.json()).then((d) => setStoredEpochs(d.epochs ?? [])).catch(() => setStoredEpochs([]));
  }, [shownRun?.id, active]); // eslint-disable-line react-hooks/exhaustive-deps
  const epochs = active ? live[shownChan ?? ""] ?? [] : storedEpochs;

  const selectedChannels = useMemo(() => {
    if (scope === "channel") return [selChan];
    if (scope === "list") return list;
    return (channels.data ?? []).filter((c) => scope === "ALL" || c.spacecraft === scope).map((c) => c.chan_id);
  }, [scope, selChan, list, channels.data]);

  const submit = () => {
    const body: Record<string, unknown> = {
      method,
      profile: profile || null,
      config: cfg,
      threshold: { mode: thrMode },
      channels: selectedChannels,
      workers,
      seed,
      name: name || undefined,
    };
    create.mutate(body, {
      onSuccess: (e) => {
        setWatchId(e.id);
        setViewChan(null);
        toast({ tone: "success", title: `Experiment #${e.id} queued`, body: `${e.method_label} on ${e.n_channels} channel(s). Training runs in the background.` });
      },
      onError: (e: Error) => toast({ tone: "error", title: "Could not create experiment", body: e.message }),
    });
  };

  if (methods.isError) return <ErrorState error={methods.error} />;
  if (!methods.data || !channels.data) return <Loading height={500} />;
  const heavy = method.startsWith("lstm") && selectedChannels.length > 5;

  return (
    <div className="space-y-6">
      <PageHeader eyebrow="Model laboratory" title="Train and evaluate on real telemetry">
        Configure a method, pick channels and start a job. Training uses the channel's training split only; thresholds are chosen
        without labels; labels are used afterwards for evaluation. Start small — one channel with the Quick profile — and scale up.
      </PageHeader>

      <div className="grid gap-6 xl:grid-cols-[440px_1fr]">
        <div className="space-y-6">
          <Panel eyebrow="1 · Method" title={spec?.label ?? method}>
            <Select ariaLabel="Method" value={method} onChange={setMethod} className="w-full">
              {methods.data.methods.map((m) => (
                <option key={m.key} value={m.key}>{m.label}</option>
              ))}
            </Select>
            <p className="mt-2 text-sm leading-relaxed text-muted">{spec?.purpose}</p>
            {spec && Object.keys(spec.profiles).length > 0 && (
              <div className="mt-4">
                <div className="eyebrow mb-2">Profile</div>
                <Segmented ariaLabel="Profile" value={profile} onChange={setProfile} options={Object.keys(spec.profiles).map((p) => ({ value: p, label: p[0].toUpperCase() + p.slice(1) }))} />
              </div>
            )}
          </Panel>

          <Panel eyebrow="2 · Parameters" title="Hyper-parameters">
            {Object.keys(cfg).length === 0 ? (
              <p className="text-sm text-muted">This method has no trainable parameters.</p>
            ) : (
              <div className="grid grid-cols-2 gap-x-4 gap-y-3">
                {Object.entries(cfg).map(([k, v]) => (
                  <label key={k} className="block" title={PARAM_HELP[k]}>
                    <div className="mb-1 text-xs text-muted">{k}</div>
                    <input
                      type="number"
                      step={k === "lr" || k === "min_delta" ? "any" : k === "dropout" || k === "val_fraction" ? 0.05 : 1}
                      value={v}
                      onChange={(e) => setCfg((c) => ({ ...c, [k]: Number(e.target.value) }))}
                      className="num h-8 w-full rounded-md border border-edge bg-raised px-2 text-sm text-ink"
                    />
                  </label>
                ))}
              </div>
            )}
            <div className="mt-4">
              <div className="eyebrow mb-2">Thresholding</div>
              <Segmented ariaLabel="Threshold" value={thrMode} onChange={setThrMode} options={[{ value: "ndt", label: "Dynamic (NDT)" }, { value: "static", label: "Static μ+3σ (train)" }]} />
            </div>
          </Panel>

          <Panel eyebrow="3 · Channels" title={`${selectedChannels.length} selected`}>
            <Segmented
              ariaLabel="Channel scope"
              size="sm"
              value={scope}
              onChange={setScope}
              options={[{ value: "channel", label: "One" }, { value: "list", label: "Pick" }, { value: "SMAP", label: "All SMAP" }, { value: "MSL", label: "All MSL" }, { value: "ALL", label: "All 81" }]}
            />
            <div className="mt-3">
              {scope === "channel" && (
                <Select ariaLabel="Channel" value={selChan} onChange={setSelChan} className="num w-full">
                  {channels.data.map((c) => <option key={c.chan_id} value={c.chan_id}>{c.spacecraft} · {c.chan_id} (train {c.train_len}, test {c.test_len})</option>)}
                </Select>
              )}
              {scope === "list" && (
                <div className="flex max-h-[180px] flex-wrap gap-1 overflow-y-auto">
                  {channels.data.map((c) => {
                    const on = list.includes(c.chan_id);
                    return (
                      <button key={c.chan_id} onClick={() => setList((l) => (on ? l.filter((x) => x !== c.chan_id) : [...l, c.chan_id]))}
                        className={`num rounded border px-1.5 py-0.5 text-xs ${on ? "border-accent/70 bg-accent/10 text-ink" : "border-hairline text-muted hover:text-ink"}`}>
                        {c.chan_id}
                      </button>
                    );
                  })}
                </div>
              )}
            </div>
            <div className="mt-4 grid grid-cols-3 gap-3">
              <label><div className="mb-1 text-xs text-muted">Worker processes</div>
                <input type="number" min={1} max={sys.data?.cpu_count ?? 8} value={workers} onChange={(e) => setWorkers(Number(e.target.value))} className="num h-8 w-full rounded-md border border-edge bg-raised px-2 text-sm" /></label>
              <label><div className="mb-1 text-xs text-muted">Seed</div>
                <input type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))} className="num h-8 w-full rounded-md border border-edge bg-raised px-2 text-sm" /></label>
              <label><div className="mb-1 text-xs text-muted">Name (optional)</div>
                <input value={name} onChange={(e) => setName(e.target.value)} className="h-8 w-full rounded-md border border-edge bg-raised px-2 text-sm" /></label>
            </div>
            {heavy && (
              <div className="mt-4"><Notice tone="warn">Training an LSTM on {selectedChannels.length} channels can take a long time on a laptop CPU (minutes per channel). Consider the Quick profile or fewer channels first.</Notice></div>
            )}
            <Button variant="primary" className="mt-4 w-full" onClick={submit} loading={create.isPending} disabled={!selectedChannels.length}>
              <Play size={14} /> Start experiment
            </Button>
          </Panel>

          {sys.data && (
            <Panel eyebrow="Compute" title={sys.data.cuda_available ? `GPU · ${sys.data.cuda_device}` : "CPU only"}>
              <KV k="CPU cores" v={sys.data.cpu_count} />
              <KV k="RAM available / total" v={sys.data.ram_total_gb ? `${sys.data.ram_available_gb} / ${sys.data.ram_total_gb} GB` : "—"} />
              <KV k="PyTorch" v={sys.data.torch} />
              <KV k="Device preference" v={sys.data.device_preference} />
              <p className="mt-2 text-xs text-muted">{sys.data.recommendation}</p>
            </Panel>
          )}
        </div>

        <div className="space-y-6">
          {!watched.data ? (
            <EmptyState icon={<FlaskConical size={28} />} title="No local experiment yet">
              Configure a method on the left and start an experiment. Progress, epoch-by-epoch training and validation loss, and results
              appear here as they are produced.
            </EmptyState>
          ) : (
            <>
              <Panel
                eyebrow={`Experiment #${watched.data.id}`}
                title={watched.data.name}
                actions={
                  <div className="flex items-center gap-3">
                    <StatusBadge status={watched.data.status} />
                    {active && <Button size="sm" variant="danger" onClick={() => cancel.mutate(watched.data!.id)} loading={cancel.isPending}><Square size={12} /> Cancel</Button>}
                    <Link to={`/experiments/${watched.data.id}`} className="text-sm text-accent hover:underline">Details</Link>
                  </div>
                }
              >
                <div className="h-1.5 overflow-hidden rounded bg-sunken"><div className="h-full rounded bg-accent/70 transition-[width]" style={{ width: `${watched.data.progress * 100}%` }} /></div>
                <div className="num mt-1.5 flex justify-between text-xs text-muted">
                  <span>{runs.filter((r) => ["completed", "failed", "cancelled"].includes(r.status)).length} / {watched.data.n_channels} channels finished</span>
                  <span>{watched.data.device ?? "—"} · {watched.data.current_channel ? `training ${watched.data.current_channel}` : watched.data.status}</span>
                </div>
                {watched.data.message && <div className="mt-3"><Notice tone="warn">{watched.data.message}</Notice></div>}
                <div className="mt-5 grid gap-6 lg:grid-cols-[1fr_220px]">
                  <div>
                    <div className="eyebrow mb-2">Loss · {shownChan ?? "—"}</div>
                    {["lstm_forecaster", "lstm_autoencoder"].includes(watched.data.method) ? (
                      <LossChart epochs={epochs} />
                    ) : (
                      <p className="text-sm text-muted">This method has no iterative training, so there is no loss curve.</p>
                    )}
                  </div>
                  <div className="max-h-[320px] overflow-y-auto">
                    <div className="eyebrow mb-2">Channels</div>
                    {runs.map((r) => (
                      <button key={r.id} onClick={() => setViewChan(r.chan_id)}
                        className={`flex w-full items-center justify-between rounded px-2 py-1 text-left text-sm ${r.chan_id === shownChan ? "bg-raised" : "hover:bg-raised/50"}`}>
                        <span className="num">{r.chan_id}</span>
                        <span className="flex items-center gap-2">
                          {r.event.f0_5 != null && <span className="num text-xs text-muted">F0.5 {fmtNum(r.event.f0_5, 2)}</span>}
                          <StatusBadge status={r.status} />
                        </span>
                      </button>
                    ))}
                  </div>
                </div>
              </Panel>
              {!active && watched.data.metrics?.overall && (
                <Panel eyebrow="Result" title="Evaluation of this experiment">
                  <div className="grid grid-cols-2 gap-5 md:grid-cols-5">
                    <KV k="Event precision" v={fmtNum(watched.data.headline.event_precision, 3)} />
                    <KV k="Event recall" v={fmtNum(watched.data.headline.event_recall, 3)} />
                    <KV k="Event F0.5" v={fmtNum(watched.data.headline.event_f0_5, 3)} />
                    <KV k="Point F1" v={fmtNum(watched.data.headline.point_f1, 3)} />
                    <KV k="Training time" v={fmtDuration(watched.data.train_time_s)} />
                  </div>
                  <div className="mt-4 flex gap-3">
                    {shownRun && <Link to={`/detect?chan=${shownRun.chan_id}&run=${shownRun.id}`}><Button>Inspect {shownRun.chan_id} in Detection Centre</Button></Link>}
                    <Link to={`/evaluation?ids=${watched.data.id}`}><Button variant="ghost">Open in evaluation</Button></Link>
                  </div>
                </Panel>
              )}
            </>
          )}

          <Panel eyebrow="Queue" title="Recent local experiments">
            {(exps.data ?? []).filter((e) => e.origin === "local").length === 0 ? (
              <p className="text-sm text-muted">None yet.</p>
            ) : (
              <div className="divide-y divide-hairline">
                {(exps.data ?? []).filter((e) => e.origin === "local").slice(0, 8).map((e) => (
                  <button key={e.id} onClick={() => { setWatchId(e.id); setViewChan(null); }} className={`flex w-full items-center justify-between gap-3 py-2 text-left ${e.id === watchId ? "text-ink" : "text-ink-2"}`}>
                    <span className="flex min-w-0 items-center gap-2">
                      <span className="h-2 w-2 shrink-0 rounded-sm" style={{ background: METHOD_COLOR[e.method] }} />
                      <span className="truncate text-sm">#{e.id} {e.name}</span>
                    </span>
                    <span className="flex shrink-0 items-center gap-3 text-xs text-muted">
                      <span className="num">{fmtInt(e.n_channels)} ch</span>
                      <StatusBadge status={e.status} />
                    </span>
                  </button>
                ))}
              </div>
            )}
          </Panel>
        </div>
      </div>
    </div>
  );
}
