import { ArrowRight, Database, FlaskConical, Satellite } from "lucide-react";
import { Link, useNavigate } from "react-router-dom";
import { useChannels, useExperiment, useOverview } from "../api/hooks";
import type { Channel, Experiment, RunSummary } from "../api/types";
import { Plot, axis, baseLayout } from "../components/Plot";
import { Button, EmptyState, ErrorState, Loading, OriginTag, PageHeader, Panel, Stat, StatusBadge } from "../components/ui";
import { fmtDate, fmtDuration, fmtInt, fmtNum, fmtPct } from "../lib/format";
import { C, METHOD_COLOR, METHOD_ORDER, SPACECRAFT_INFO } from "../lib/palette";
import { useSelection } from "../lib/selection";

function MissionCard({ sc, d }: { sc: "SMAP" | "MSL"; d?: { channels: number; test_values: number; train_values: number; label_sequences: number } }) {
  const info = SPACECRAFT_INFO[sc];
  return (
    <Panel className="relative overflow-hidden">
      <div className="pointer-events-none absolute -top-16 -right-16 h-48 w-48 rounded-full border border-hairline" aria-hidden />
      <div className="pointer-events-none absolute -top-6 -right-6 h-28 w-28 rounded-full border border-hairline/70" aria-hidden />
      <div className="relative pt-4">
        <div className="eyebrow flex items-center gap-2">
          <Satellite size={12} /> {info.kind}
        </div>
        <div className="mt-2 text-[20px] font-semibold tracking-tight text-ink">{info.name}</div>
        <p className="mt-1.5 max-w-md text-[13px] leading-relaxed text-muted">{info.blurb}</p>
        <div className="mt-5 grid grid-cols-2 gap-4 sm:grid-cols-4">
          <Stat label="Channels" value={fmtInt(d?.channels)} />
          <Stat label="Label seq." value={fmtInt(d?.label_sequences)} />
          <Stat label="Train values" value={fmtInt(d?.train_values)} />
          <Stat label="Test values" value={fmtInt(d?.test_values)} />
        </div>
      </div>
    </Panel>
  );
}

function Leaderboard({ rows }: { rows: Experiment[] }) {
  const sorted = [...rows].sort((a, b) => METHOD_ORDER.indexOf(a.method) - METHOD_ORDER.indexOf(b.method));
  const chartRows = sorted.filter((r) => r.headline.event_f0_5 != null);
  return (
    <div className="grid gap-6 lg:grid-cols-[1fr_380px]">
      <div className="overflow-x-auto">
        <table className="w-full text-[13px]">
          <thead>
            <tr className="text-left text-[11.5px] text-muted">
              <th className="py-2 pr-3 font-medium">Method</th>
              <th className="py-2 pr-3 text-right font-medium">Event P</th>
              <th className="py-2 pr-3 text-right font-medium">Event R</th>
              <th className="py-2 pr-3 text-right font-medium">F0.5</th>
              <th className="py-2 pr-3 text-right font-medium">Point F1</th>
              <th className="py-2 pr-3 text-right font-medium">mAP</th>
              <th className="py-2 text-right font-medium">Detections</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((e) => (
              <tr key={e.id} className="border-t border-hairline">
                <td className="py-2 pr-3">
                  <Link to={`/experiments/${e.id}`} className="inline-flex items-center gap-2 text-ink hover:text-accent">
                    <span className="h-2.5 w-2.5 rounded-sm" style={{ background: METHOD_COLOR[e.method] }} />
                    {e.method_label}
                    {e.profile && <span className="text-muted">· {e.profile}</span>}
                  </Link>
                  <span className="ml-2">
                    <OriginTag origin={e.origin} />
                  </span>
                </td>
                <td className="num py-2 pr-3 text-right">{fmtPct(e.headline.event_precision)}</td>
                <td className="num py-2 pr-3 text-right">{fmtPct(e.headline.event_recall)}</td>
                <td className="num py-2 pr-3 text-right text-ink">{fmtNum(e.headline.event_f0_5, 3)}</td>
                <td className="num py-2 pr-3 text-right">{fmtNum(e.headline.point_f1, 3)}</td>
                <td className="num py-2 pr-3 text-right">{fmtNum(e.headline.map, 3)}</td>
                <td className="num py-2 text-right">{fmtInt(e.headline.n_detections)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="mt-3 text-[12px] leading-relaxed text-muted">
          Event-wise metrics follow the protocol of Hundman et al.; point-wise F1 counts every evaluated timestep without point
          adjustment. Values are computed by this application on test indices ≥ 250 of all 81 channels and are{" "}
          <em>not</em> directly comparable with the paper (different preprocessing details and implementation).
          With only 104 labelled sequences, event F0.5 carries a 95 % channel-bootstrap interval of roughly ±0.07–0.10 for the
          stronger methods, and each reference run uses a single seed, so small gaps between methods are not a reliable ranking.
        </p>
      </div>
      <div>
        <div className="eyebrow mb-2">Event-wise F0.5 by method</div>
        <Plot
          height={Math.max(160, chartRows.length * 30 + 40)}
          data={[
            {
              type: "bar",
              orientation: "h",
              y: chartRows.map((r) => r.method_label),
              x: chartRows.map((r) => r.headline.event_f0_5 ?? 0),
              marker: { color: chartRows.map((r) => METHOD_COLOR[r.method]) },
              text: chartRows.map((r) => fmtNum(r.headline.event_f0_5, 2)),
              textposition: "outside",
              textfont: { family: "IBM Plex Mono", size: 11, color: C.ink2 },
              hovertemplate: "%{y}: %{x:.3f}<extra></extra>",
              cliponaxis: false,
            },
          ]}
          layout={baseLayout({
            margin: { l: 160, r: 40, t: 4, b: 28 },
            xaxis: axis({ range: [0, 1], dtick: 0.25 }),
            yaxis: axis({ autorange: "reversed", showline: false, ticks: "", ticksuffix: "   ", tickfont: { size: 11.5, color: C.ink2, family: "IBM Plex Sans" } }),
          })}
        />
      </div>
    </div>
  );
}

function ChannelMap({ channels, runs, exp }: { channels: Channel[]; runs: RunSummary[]; exp?: Experiment }) {
  const nav = useNavigate();
  const { setChan } = useSelection();
  const byChan = new Map(runs.map((r) => [r.chan_id, r]));
  return (
    <div className="grid gap-6 lg:grid-cols-[1fr_auto]">
      <div className="space-y-5">
        {(["SMAP", "MSL"] as const).map((sc) => (
          <div key={sc}>
            <div className="eyebrow mb-2">{sc}</div>
            <div className="grid grid-cols-[repeat(auto-fill,minmax(76px,1fr))] gap-1.5">
              {channels
                .filter((c) => c.spacecraft === sc)
                .map((c) => {
                  const r = byChan.get(c.chan_id);
                  const tp = r?.event.tp ?? 0;
                  const fp = r?.event.fp ?? 0;
                  return (
                    <button
                      key={c.chan_id}
                      onClick={() => {
                        setChan(c.chan_id);
                        nav(`/detect?chan=${c.chan_id}${r ? `&run=${r.id}` : ""}`);
                      }}
                      className="group rounded-md border border-hairline bg-sunken px-2 py-1.5 text-left transition-colors hover:border-edge hover:bg-raised"
                      title={
                        r
                          ? `${c.chan_id}: ${tp}/${c.n_intervals} labelled sequences overlapped by a detection, ${fp} detection(s) with no label overlap`
                          : `${c.chan_id}: no result for this method yet`
                      }
                    >
                      <div className="num text-[12px] text-ink-2 group-hover:text-ink">{c.chan_id}</div>
                      <div className="mt-1 flex flex-wrap gap-[3px]">
                        {Array.from({ length: c.n_intervals }).map((_, i) => (
                          <span
                            key={i}
                            className={`h-2 w-2 rounded-[2px] ${i < tp ? "bg-detect" : "hatch-label border border-label/60"}`}
                          />
                        ))}
                        {Array.from({ length: Math.min(fp, 6) }).map((_, i) => (
                          <span key={`fp${i}`} className="h-2 w-2 rounded-[2px] border border-detect/70" />
                        ))}
                      </div>
                    </button>
                  );
                })}
            </div>
          </div>
        ))}
      </div>
      <div className="w-[220px] space-y-3 text-[12px] text-ink-2">
        <div className="eyebrow">Reading the map</div>
        <div className="flex items-center gap-2">
          <span className="h-2.5 w-2.5 rounded-[2px] bg-detect" /> Labelled sequence overlapped by a model detection
        </div>
        <div className="flex items-center gap-2">
          <span className="hatch-label h-2.5 w-2.5 rounded-[2px] border border-label/60" /> Labelled sequence not detected
        </div>
        <div className="flex items-center gap-2">
          <span className="h-2.5 w-2.5 rounded-[2px] border border-detect/70" /> Detection with no label overlap (max 6 shown)
        </div>
        <p className="pt-2 text-muted">
          Source: {exp ? <Link className="text-accent hover:underline" to={`/experiments/${exp.id}`}>{exp.name}</Link> : "—"}. Click a channel to open it in
          the Detection Centre.
        </p>
      </div>
    </div>
  );
}

export default function Overview() {
  const ov = useOverview();
  const ch = useChannels();
  const lstm = ov.data?.leaderboard.find((e) => e.method === "lstm_forecaster");
  const lstmExp = useExperiment(lstm?.id ?? null);

  if (ov.isError) return <ErrorState error={ov.error} title="Could not load the mission overview" />;
  if (!ov.data) return <Loading height={500} />;
  const o = ov.data;
  const status = o.experiments.by_status;

  return (
    <div className="space-y-6">
      <PageHeader eyebrow="Mission overview" title="Spacecraft telemetry intelligence">
        LSTM forecasting and nonparametric dynamic thresholding on expert-labelled telemetry from NASA's SMAP satellite and the
        Curiosity rover. Every figure on this page is read from the installed benchmark and stored experiment results.
      </PageHeader>

      {!o.dataset.available && (
        <EmptyState
          icon={<Database size={28} />}
          title="The SMAP / MSL benchmark is not installed"
          action={<Link to="/dataset"><Button variant="primary">Open Dataset Explorer</Button></Link>}
        >
          Run <code className="num">setup.ps1 -DownloadData</code> (Windows) or <code className="num">./setup.sh</code> (Linux/macOS), or start the acquisition from the Dataset Explorer. Nothing is shown
          until real data is present.
        </EmptyState>
      )}

      <div className="grid gap-6 lg:grid-cols-2">
        <MissionCard sc="SMAP" d={o.spacecraft.SMAP} />
        <MissionCard sc="MSL" d={o.spacecraft.MSL} />
      </div>

      <Panel>
        <div className="grid grid-cols-2 gap-6 pt-5 md:grid-cols-3 xl:grid-cols-6">
          <Stat
            label="Dataset"
            mono={false}
            value={<span className="text-[17px]">{o.dataset.available ? (o.dataset.validation_ok ? "Validated" : "Issues") : "Missing"}</span>}
            sub={o.dataset.source ? `via ${o.dataset.source.replace("_", " ")}` : undefined}
          />
          <Stat label="Channels analysed" value={`${o.channels_analysed}/${(o.spacecraft.SMAP?.channels ?? 0) + (o.spacecraft.MSL?.channels ?? 0)}`} />
          <Stat label="Channel runs" value={fmtInt(o.completed_channel_runs)} sub="completed (all methods)" />
          <Stat label="Detected intervals" value={fmtInt(o.detections_total)} sub="model-detected, all experiments" />
          <Stat label="Experiments" value={fmtInt(o.experiments.total)} sub={`${status.completed ?? 0} completed · ${status.running ?? 0} running`} />
          <Stat label="Label sequences" value={fmtInt((o.spacecraft.SMAP?.label_sequences ?? 0) + (o.spacecraft.MSL?.label_sequences ?? 0))} sub="expert reference (P-2 merged)" />
        </div>
      </Panel>

      <Panel
        eyebrow="Model evaluation summary"
        title="Latest all-channel result per method"
        actions={<Link to="/evaluation" className="inline-flex items-center gap-1 text-[12.5px] text-accent hover:underline">Full evaluation <ArrowRight size={13} /></Link>}
      >
        {o.leaderboard.length ? (
          <Leaderboard rows={o.leaderboard} />
        ) : (
          <EmptyState icon={<FlaskConical size={26} />} title="No models evaluated yet" action={<Link to="/lab"><Button variant="primary">Open Model Laboratory</Button></Link>}>
            Train a model in the laboratory. Results appear here only after a real evaluation has completed.
          </EmptyState>
        )}
      </Panel>

      {lstm && ch.data && (
        <Panel eyebrow="Channel map" title="LSTM forecaster vs expert labels, per channel">
          {lstmExp.data?.runs ? <ChannelMap channels={ch.data} runs={lstmExp.data.runs} exp={lstm} /> : <Loading height={200} />}
        </Panel>
      )}

      <Panel eyebrow="Recent experiments" title="Latest activity" actions={<Link to="/experiments" className="text-[12.5px] text-accent hover:underline">All experiments</Link>}>
        {o.recent_experiments.length ? (
          <div className="divide-y divide-hairline">
            {o.recent_experiments.map((e) => (
              <Link key={e.id} to={`/experiments/${e.id}`} className="flex items-center justify-between gap-4 py-2.5 hover:bg-raised/40">
                <div className="flex min-w-0 items-center gap-3">
                  <span className="h-2.5 w-2.5 shrink-0 rounded-sm" style={{ background: METHOD_COLOR[e.method] }} />
                  <span className="truncate text-[13px] text-ink">{e.name}</span>
                  <OriginTag origin={e.origin} />
                </div>
                <div className="flex shrink-0 items-center gap-5 text-[12px] text-muted">
                  <span className="num">F0.5 {fmtNum(e.headline.event_f0_5, 3)}</span>
                  <span className="num hidden md:inline">{fmtDuration(e.train_time_s)}</span>
                  <span className="hidden md:inline">{fmtDate(e.created_at)}</span>
                  <StatusBadge status={e.status} />
                </div>
              </Link>
            ))}
          </div>
        ) : (
          <EmptyState title="No experiments yet" />
        )}
      </Panel>
    </div>
  );
}
