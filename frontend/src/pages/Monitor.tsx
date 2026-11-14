import { Pause, Play, RadioTower, RotateCcw } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import type { ReplaySession, ReplayStep } from "../api/types";
import { ChannelPicker, RunPicker, isStateless, useRunsForChannel } from "../components/pickers";
import { TraceStack } from "../components/TraceStack";
import { Button, EmptyState, ErrorState, KV, Legend, Notice, PageHeader, Panel, Segmented } from "../components/ui";
import { fmtInt, fmtNum } from "../lib/format";
import { C } from "../lib/palette";
import { useSelection } from "../lib/selection";

const SPEEDS = [
  { value: "20", label: "20 pts/s" },
  { value: "80", label: "80 pts/s" },
  { value: "300", label: "300 pts/s" },
  { value: "1000", label: "1000 pts/s" },
];
const TICK_MS = 200;

type Buf = (number | null)[];
interface LogEntry { at: number; text: string; kind: "flag" | "info" }

export default function Monitor() {
  const { chan, setChan } = useSelection();
  const runs = useRunsForChannel(chan).filter((r) => r.has_model || isStateless(r.method));
  const [runId, setRunId] = useState<number | null>(null);
  useEffect(() => {
    setRunId((cur) => (cur && runs.some((r) => r.run_id === cur) ? cur : runs.find((r) => r.method === "lstm_forecaster")?.run_id ?? runs[0]?.run_id ?? null));
  }, [runs]);

  const [session, setSession] = useState<ReplaySession | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [playing, setPlaying] = useState(false);
  const [everPlayed, setEverPlayed] = useState(false);
  useEffect(() => { if (playing) setEverPlayed(true); }, [playing]);
  const [speed, setSpeed] = useState("80");
  const [cursor, setCursor] = useState(0);
  const [done, setDone] = useState(false);
  const [bufs, setBufs] = useState<{ actual: Buf; prediction: Buf; raw: Buf; smoothed: Buf; thrHi: Buf; thrLo: Buf } | null>(null);
  const [intervals, setIntervals] = useState<NonNullable<ReplayStep["intervals"]>>([]);
  const [verdictTo, setVerdictTo] = useState(0);
  const [log, setLog] = useState<LogEntry[]>([]);
  const [computeMs, setComputeMs] = useState<number | null>(null);
  const busy = useRef(false);
  const seen = useRef(new Set<string>());

  const start = useCallback(async () => {
    if (!runId) return;
    setError(null);
    setPlaying(false);
    try {
      if (session) api.del(`/replay/${session.session_id}`).catch(() => {});
      const s = await api.post<ReplaySession>("/replay", { run_id: runId });
      const empty = () => Array<number | null>(s.n).fill(null);
      setSession(s);
      setBufs({ actual: empty(), prediction: empty(), raw: empty(), smoothed: empty(), thrHi: empty(), thrLo: empty() });
      setCursor(0);
      setEverPlayed(false);
      setDone(false);
      setIntervals([]);
      setVerdictTo(0);
      setLog([{ at: 0, kind: "info", text: `Replay prepared: ${s.method_label} on ${s.chan_id} (${fmtInt(s.n)} historical test points).` }]);
      seen.current = new Set();
    } catch (e) {
      setError(e);
    }
  }, [runId, session]);

  useEffect(() => {
    setSession(null);
    setBufs(null);
    setPlaying(false);
  }, [runId, chan]);

  const step = useCallback(async () => {
    if (!session || busy.current || done) return;
    busy.current = true;
    try {
      const n = Math.max(1, Math.round((Number(speed) * TICK_MS) / 1000));
      const r = await api.post<ReplayStep>(`/replay/${session.session_id}/advance`, { n });
      setComputeMs(r.compute_ms ?? null);
      setBufs((b) => {
        if (!b || r.actual == null) return b;
        const nb = { actual: b.actual.slice(), prediction: b.prediction.slice(), raw: b.raw.slice(), smoothed: b.smoothed.slice(), thrHi: b.thrHi.slice(), thrLo: b.thrLo.slice() };
        for (let i = 0; i < r.actual.length; i++) {
          const t = r.start + i;
          nb.actual[t] = r.actual[i];
          nb.prediction[t] = r.prediction ? r.prediction[i] : null;
          nb.raw[t] = r.raw_score ? r.raw_score[i] : null;
          nb.smoothed[t] = r.smoothed ? r.smoothed[i] : null;
        }
        if (r.verdict) {
          for (let i = 0; i < r.verdict.threshold_high.length; i++) {
            nb.thrHi[r.verdict.from + i] = r.verdict.threshold_high[i];
            nb.thrLo[r.verdict.from + i] = r.verdict.threshold_low[i];
          }
        }
        return nb;
      });
      if (r.verdict) setVerdictTo(r.verdict.to);
      if (r.intervals) {
        setIntervals(r.intervals);
        const fresh = r.intervals.filter((iv) => !seen.current.has(`${iv.start}`));
        if (fresh.length) {
          setLog((l) => [
            ...fresh.map((iv) => {
              seen.current.add(`${iv.start}`);
              return { at: r.cursor, kind: "flag" as const, text: `Interval ${iv.start}–${iv.end} flagged (score ${fmtNum(iv.score, 2)}, ${iv.direction} error) — verdict issued at index ${r.cursor}.` };
            }),
            ...l,
          ]);
        }
      }
      setCursor(r.cursor);
      if (r.done) {
        setDone(true);
        setPlaying(false);
        setLog((l) => [{ at: r.cursor, kind: "info", text: "End of the historical test stream reached." }, ...l]);
      }
    } catch (e) {
      setError(e);
      setPlaying(false);
    } finally {
      busy.current = false;
    }
  }, [session, speed, done]);

  useEffect(() => {
    if (!playing) return;
    const id = setInterval(step, TICK_MS);
    return () => clearInterval(id);
  }, [playing, step]);

  useEffect(() => () => { if (session) api.del(`/replay/${session.session_id}`).catch(() => {}); }, [session]);

  const revealedLabels = useMemo(
    () => (session?.labels ?? []).filter((l) => l.start < cursor).map((l) => ({ ...l, end: Math.min(l.end, cursor - 1) })),
    [session, cursor],
  );
  const last = cursor > 0 && bufs ? cursor - 1 : null;
  const latestThr = useMemo(() => {
    if (!bufs) return null;
    for (let i = Math.min(verdictTo, bufs.thrHi.length) - 1; i >= 0; i--) if (bufs.thrHi[i] != null) return bufs.thrHi[i];
    return null;
  }, [bufs, verdictTo]);
  const flaggedNow = intervals.some((iv) => last != null && iv.end >= verdictTo - 1 - (session?.batch_size ?? 70));

  return (
    <div className="space-y-6">
      <PageHeader eyebrow="Live-style telemetry monitor" title="Historical telemetry replay" actions={<ChannelPicker chan={chan} onChange={setChan} />}>
        Successive points of an existing benchmark test sequence are revealed one batch at a time. The trained model is executed on each
        newly revealed point, the score is smoothed incrementally, and each threshold window is evaluated as soon as it is complete.
      </PageHeader>

      <div className="flex items-center gap-3 rounded-lg border border-[#5c4a1a] bg-[#171308] px-4 py-2.5 text-[13px] text-[#e8cf8f]" role="note">
        <RadioTower size={15} />
        <span>
          <strong>Historical replay.</strong> This is recorded benchmark telemetry played back for analysis — not a live connection to an
          operating spacecraft.
        </span>
      </div>

      {!runs.length ? (
        <EmptyState title={`No trained model with a saved file for ${chan}`} action={<Link to="/lab"><Button variant="primary">Open Model Laboratory</Button></Link>}>
          The replay executes a real model, so it needs a trained run on this machine.
        </EmptyState>
      ) : (
        <>
          <Panel>
            <div className="flex flex-wrap items-end gap-x-6 gap-y-4 pt-4">
              <div>
                <div className="eyebrow mb-1.5">Model</div>
                <RunPicker chan={chan} runId={runId} onChange={setRunId} requireModel />
              </div>
              <div>
                <div className="eyebrow mb-1.5">Playback speed</div>
                <Segmented ariaLabel="Playback speed" value={speed} onChange={setSpeed} options={SPEEDS} />
              </div>
              <div className="ml-auto flex items-center gap-2">
                {!session ? (
                  <Button variant="primary" onClick={async () => { await start(); setPlaying(true); }}>
                    <Play size={14} /> Start monitoring
                  </Button>
                ) : (
                  <>
                    <Button variant="secondary" onClick={start}>
                      <RotateCcw size={14} /> Reset
                    </Button>
                    {playing ? (
                      <Button variant="primary" onClick={() => setPlaying(false)}>
                        <Pause size={14} /> Pause
                      </Button>
                    ) : (
                      <Button variant="primary" disabled={done} onClick={() => setPlaying(true)}>
                        <Play size={14} /> {cursor > 0 || everPlayed ? "Resume" : "Start"}
                      </Button>
                    )}
                  </>
                )}
              </div>
            </div>
            {session && (
              <div className="mt-4">
                <div className="h-1.5 overflow-hidden rounded bg-sunken" role="progressbar" aria-valuenow={cursor} aria-valuemax={session.n}>
                  <div className="h-full bg-accent/70 transition-[width]" style={{ width: `${(100 * cursor) / session.n}%` }} />
                </div>
                <div className="num mt-1.5 flex justify-between text-[11.5px] text-muted">
                  <span>revealed {fmtInt(cursor)} / {fmtInt(session.n)}</span>
                  <span>verdicts issued up to index {fmtInt(verdictTo)}{computeMs != null ? ` · last step ${fmtNum(computeMs, 0)} ms server compute` : ""}</span>
                </div>
              </div>
            )}
          </Panel>
          {error != null && <ErrorState error={error} title="Replay error" />}

          <div className="grid gap-6 xl:grid-cols-[1fr_320px]">
            <Panel eyebrow={session ? `${session.method_label} · ${session.chan_id}` : "Replay"} title={session ? "Telemetry stream" : "Press “Start monitoring”"}>
              {bufs && session ? (
                <>
                  <Legend
                    items={[
                      { label: "Actual telemetry", color: C.actual },
                      ...(session.produces_prediction ? [{ label: "Model prediction", color: C.prediction }] : []),
                      { label: "Smoothed score", color: C.smoothed },
                      { label: "Threshold (once window evaluated)", color: C.threshold, kind: "dash" as const },
                      { label: "Model detection", color: C.detect, kind: "band" as const },
                      { label: "Expert label (revealed)", color: C.label, kind: "hatch" as const },
                    ]}
                  />
                  <TraceStack
                    actual={bufs.actual}
                    prediction={session.produces_prediction ? bufs.prediction : null}
                    raw={bufs.raw}
                    smoothed={bufs.smoothed}
                    thresholdHigh={bufs.thrHi}
                    thresholdLow={bufs.thrLo}
                    labels={revealedLabels}
                    detections={intervals}
                    evalStart={session.eval_start}
                    xRange={[0, session.n - 1]}
                    height={540}
                    uirevision={session.session_id}
                  />
                </>
              ) : (
                <div className="flex h-[300px] items-center justify-center text-[13px] text-muted">The stream appears here once monitoring starts.</div>
              )}
            </Panel>
            <div className="space-y-6">
              <Panel eyebrow="Current reading" title={last != null ? `Index ${fmtInt(last)}` : "—"}>
                <div
                  className={`mb-3 flex items-center gap-2 rounded-md border px-3 py-2 text-[13px] ${
                    flaggedNow ? "border-detect/60 bg-[#151233] text-ink" : "border-hairline bg-sunken text-ink-2"
                  }`}
                  role="status"
                >
                  <span className={`h-2 w-2 rounded-full ${flaggedNow ? "bg-detect pulse-dot" : "bg-dim"}`} />
                  {flaggedNow ? "Model flagged an unusual interval in the latest evaluated window" : "No flag in the latest evaluated window"}
                </div>
                <KV k="Telemetry value" v={fmtNum(last != null ? bufs?.actual[last] : null, 4)} />
                <KV k="Prediction" v={fmtNum(last != null ? bufs?.prediction[last] : null, 4)} />
                <KV k="Smoothed score" v={fmtNum(last != null ? bufs?.smoothed[last] : null, 4)} />
                <KV k="Latest threshold" v={fmtNum(latestThr, 4)} />
                <KV k="Intervals flagged" v={intervals.length} />
                <KV k="Threshold window" v={session ? `${session.window_batches ?? "—"} × ${session.batch_size ?? "—"} pts` : "—"} />
                <p className="mt-3 text-[11.5px] leading-relaxed text-dim">
                  {session && session.window_batches && session.batch_size
                    ? `The first verdict needs one full threshold window (${session.window_batches} × ${session.batch_size} = ${fmtInt(session.window_batches * session.batch_size)} scored points, i.e. up to index ${fmtInt(session.eval_start + session.window_batches * session.batch_size)}). After that a window is evaluated every ${session.batch_size} new points, so verdicts lag the newest reading by up to one batch.`
                    : "Verdicts are issued once each threshold window is complete."}{" "}
                  Points before index {session?.eval_start ?? 250} are not scored.
                </p>
              </Panel>
              <Panel eyebrow="Event log" title="Model flags">
                <div className="max-h-[260px] space-y-2 overflow-y-auto pr-1 text-[12.5px]">
                  {log.length ? (
                    log.map((e, i) => (
                      <div key={i} className={`flex gap-2 ${e.kind === "flag" ? "text-ink" : "text-muted"}`}>
                        <span className={`mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full ${e.kind === "flag" ? "bg-detect" : "bg-dim"}`} />
                        <span>{e.text}</span>
                      </div>
                    ))
                  ) : (
                    <span className="text-muted">Nothing yet.</span>
                  )}
                </div>
              </Panel>
              {done && (
                <Notice>
                  Replay complete. The flagged intervals equal the offline detection for this run (verified by the automated tests).
                </Notice>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
