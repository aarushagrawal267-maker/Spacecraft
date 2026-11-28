import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { CheckCircle2, Download, ShieldCheck } from "lucide-react";
import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import { useChannels, useDatasetStatus } from "../api/hooks";
import { Plot, axis, baseLayout } from "../components/Plot";
import { Button, ErrorState, KV, Legend, Loading, Notice, PageHeader, Panel, Segmented, Select, Stat } from "../components/ui";
import { useToast } from "../components/toast";
import { fmtBytes, fmtDate, fmtInt, fmtNum, fmtPct } from "../lib/format";
import { C } from "../lib/palette";

const SC_COLOR = { SMAP: "#3987e5", MSL: "#d95926" } as const;

interface ReportChannel { chan_id: string; nonfinite_train: number; nonfinite_test: number; commands_binary: boolean; test_len_matches_labels: boolean }

export default function DatasetExplorer() {
  const toast = useToast();
  const qc = useQueryClient();
  const [polling, setPolling] = useState(false);
  const ds = useDatasetStatus(polling);
  const ch = useChannels();
  const report = useQuery({ queryKey: ["dataset-report"], queryFn: () => api.get<{ channels: ReportChannel[] }>("/dataset/report"), enabled: !!ds.data?.snapshot });
  const [source, setSource] = useState("auto");
  const [scFilter, setScFilter] = useState<"ALL" | "SMAP" | "MSL">("ALL");
  const [search, setSearch] = useState("");
  const [sortKey, setSortKey] = useState<"chan_id" | "test_len" | "train_len" | "label_fraction" | "n_intervals">("chan_id");

  const acquire = useMutation({
    mutationFn: () => api.post("/dataset/acquire", { source }),
    onSuccess: () => { setPolling(true); toast({ tone: "success", title: "Acquisition started" }); },
    onError: (e: Error) => toast({ tone: "error", title: "Could not start acquisition", body: e.message }),
  });
  const verify = useMutation({
    mutationFn: () => api.post<{ ok: boolean; checksum_mismatches: string[]; errors: string[] }>("/dataset/validate"),
    onSuccess: (r) => toast({ tone: r.ok ? "success" : "error", title: r.ok ? "Files verified" : "Verification found problems",
      body: r.ok ? "All checksums match the manifest and every structural check passed." : [...r.checksum_mismatches, ...r.errors].slice(0, 3).join("; ") }),
  });

  if (polling && ds.data && !ds.data.acquisition.running) {
    setPolling(false);
    qc.invalidateQueries();
  }

  const nonfinite = useMemo(() => new Map((report.data?.channels ?? []).map((c) => [c.chan_id, c.nonfinite_train + c.nonfinite_test])), [report.data]);
  const rows = useMemo(() => {
    const list = (ch.data ?? []).filter((c) => (scFilter === "ALL" || c.spacecraft === scFilter) && c.chan_id.toLowerCase().includes(search.toLowerCase()));
    return [...list].sort((a, b) => (sortKey === "chan_id" ? a.chan_id.localeCompare(b.chan_id, undefined, { numeric: true }) : (b[sortKey] as number) - (a[sortKey] as number)));
  }, [ch.data, scFilter, search, sortKey]);

  if (ds.isError) return <ErrorState error={ds.error} />;
  if (!ds.data) return <Loading height={500} />;
  const snap = ds.data.snapshot;
  const sum = snap?.summary;
  const classes = (ch.data ?? []).flatMap((c) => c.intervals.map((i) => ({ sc: c.spacecraft, cls: i.cls })));
  const classCount = (sc: string, cls: string) => classes.filter((x) => x.sc === sc && x.cls === cls).length;
  const sorted = [...(ch.data ?? [])].sort((a, b) => b.test_len - a.test_len);

  return (
    <div className="space-y-6">
      <PageHeader eyebrow="Dataset explorer" title="NASA SMAP / MSL telemetry benchmark">
        The expert-labelled anomaly benchmark released with Hundman et al. (KDD 2018). Each channel has a training sequence, a test
        sequence and reference anomaly intervals for the test sequence.
      </PageHeader>

      <Panel eyebrow="Acquisition" title={ds.data.available ? "Installed and validated" : "Not installed"}
        actions={
          <div className="flex items-center gap-2">
            <Select ariaLabel="Source" value={source} onChange={setSource}>
              <option value="auto">Auto (Kaggle → GitHub mirror → manual)</option>
              <option value="kaggle">Kaggle</option>
              <option value="github_mirror">GitHub mirror (pinned commit)</option>
              <option value="manual">Manual data.zip in data/incoming/</option>
            </Select>
            <Button onClick={() => acquire.mutate()} loading={acquire.isPending || ds.data.acquisition.running}>
              <Download size={14} /> {ds.data.available ? "Re-download" : "Download"}
            </Button>
            {ds.data.available && (
              <Button onClick={() => verify.mutate()} loading={verify.isPending}><ShieldCheck size={14} /> Verify files</Button>
            )}
          </div>
        }
      >
        {ds.data.acquisition.running && (
          <div className="mb-4">
            <div className="mb-1 text-sm text-ink-2">{ds.data.acquisition.message}</div>
            <div className="h-1.5 overflow-hidden rounded bg-sunken"><div className="h-full bg-accent/70" style={{ width: `${ds.data.acquisition.progress * 100}%` }} /></div>
          </div>
        )}
        {ds.data.acquisition.result && !ds.data.acquisition.running && (
          <div className="mb-4 space-y-1 text-sm">
            {ds.data.acquisition.result.attempts.map((a, i) => (
              <div key={i} className={a.ok ? "text-good" : "text-muted"}>{a.ok ? "✓" : "✗"} {a.source}: {a.message}</div>
            ))}
          </div>
        )}
        {snap ? (
          <div className="grid gap-8 lg:grid-cols-2">
            <div>
              <KV k="Telemetry source" v={snap.provenance.source_title ?? snap.source} mono={false} />
              <KV k="Source location" v={<a href={snap.source_url} target="_blank" rel="noreferrer" title={snap.source_url} className="break-all text-accent hover:underline">{snap.source_url.replace("https://", "")}</a>} mono={false} />
              <KV k="Retrieved" v={fmtDate(snap.retrieved_at)} mono={false} />
              <KV k="Labels" v={<a href={snap.provenance.labels_url ?? "#"} className="text-accent hover:underline" target="_blank" rel="noreferrer">official Telemanom labeled_anomalies.csv</a>} mono={false} />
              <KV k="Labels SHA-256" v={snap.labels_sha256.slice(0, 24) + "…"} />
              <KV k="Dataset checksum" v={snap.dataset_checksum.slice(0, 24) + "…"} />
              <KV k="Files / size" v={`${snap.n_files} · ${fmtBytes(snap.total_bytes)}`} />
              <KV k="Validation" v={<span className="inline-flex items-center gap-1.5 text-good"><CheckCircle2 size={13} /> {snap.errors.length} errors · {snap.warnings.length} warnings</span>} mono={false} />
            </div>
            <div className="space-y-2">
              <div className="eyebrow">Source notes & licensing</div>
              <p className="text-sm leading-relaxed text-ink-2">{snap.provenance.source_note}</p>
              <p className="text-sm leading-relaxed text-muted">{ds.data.license_note}</p>
            </div>
          </div>
        ) : (
          <Notice tone="warn">No dataset registered yet. Use the download button; the files are validated against the official label file before they are used.</Notice>
        )}
      </Panel>

      <Panel eyebrow="Integrity disclosures" title="What this benchmark is — and is not">
        <ul className="grid gap-2 md:grid-cols-2">
          {ds.data.disclosures.map((d) => (
            <li key={d} className="flex gap-2 text-sm leading-relaxed text-ink-2"><span className="mt-2 h-1 w-1 shrink-0 rounded-full bg-accent" />{d}</li>
          ))}
        </ul>
      </Panel>

      {sum && (
        <div className="grid gap-6 lg:grid-cols-2">
          {(["SMAP", "MSL"] as const).map((sc) => {
            const s = sum.by_spacecraft[sc];
            return (
              <Panel key={sc} eyebrow={sc === "SMAP" ? "Soil Moisture Active Passive" : "Mars Science Laboratory · Curiosity"} title={sc}>
                <div className="grid grid-cols-3 gap-5">
                  <Stat label="Channels" value={s.channels} />
                  <Stat label="Features / step" value={s.features.join(", ")} sub="1 telemetry + commands" />
                  <Stat label="Label sequences" value={s.sequences} sub={`${classCount(sc, "point")} point · ${classCount(sc, "contextual")} contextual`} />
                  <Stat label="Train values" value={fmtInt(s.train_values)} />
                  <Stat label="Test values" value={fmtInt(s.test_values)} />
                  <Stat label="Labelled test" value={fmtPct(s.labelled_points / s.test_values)} sub={`${fmtInt(s.labelled_points)} points`} />
                </div>
              </Panel>
            );
          })}
        </div>
      )}
      {sum && (
        <Notice>
          The label file has {sum.label_rows} rows and {sum.label_sequences_raw} sequences. Channel <span className="num">P-2</span> appears twice with overlapping
          intervals; this project merges them into one stream with {sum.label_sequences_after_merge} sequences in total ({sum.unique_channels} unique channels).
        </Notice>
      )}

      {ch.data && (
        <div className="grid gap-6 xl:grid-cols-2">
          <Panel eyebrow="Training / testing split" title="Sequence lengths per channel">
            <Legend items={[{ label: "SMAP (test ↑, train ↓ lighter)", color: SC_COLOR.SMAP, kind: "solid" }, { label: "MSL", color: SC_COLOR.MSL, kind: "solid" }]} />
            <Plot
              height={300}
              data={(["SMAP", "MSL"] as const).flatMap((sc) => {
                const cs = sorted.filter((c) => c.spacecraft === sc);
                return [
                  { type: "bar", name: `${sc} test`, x: cs.map((c) => c.chan_id), y: cs.map((c) => c.test_len), marker: { color: SC_COLOR[sc], line: { color: C.panel, width: 1 } }, hovertemplate: "%{x} test: %{y}<extra></extra>" },
                  { type: "bar", name: `${sc} train`, x: cs.map((c) => c.chan_id), y: cs.map((c) => -c.train_len), marker: { color: SC_COLOR[sc], opacity: 0.45, line: { color: C.panel, width: 1 } }, customdata: cs.map((c) => c.train_len), hovertemplate: "%{x} train: %{customdata}<extra></extra>" },
                ];
              }) as never}
              layout={baseLayout({ barmode: "relative", bargap: 0.15, margin: { l: 56, r: 8, t: 6, b: 48 },
                xaxis: axis({ tickangle: -90, tickfont: { size: 8.5, family: "IBM Plex Mono", color: C.muted }, categoryorder: "array", categoryarray: sorted.map((c) => c.chan_id) }),
                yaxis: axis({ title: { text: "↑ test   values   train ↓" }, tickformat: "~s" }) })}
            />
          </Panel>
          <Panel eyebrow="Anomaly annotations" title="Share of each test sequence that is labelled">
            <Plot
              height={320}
              data={[{
                type: "bar", x: [...(ch.data ?? [])].sort((a, b) => b.label_fraction - a.label_fraction).map((c) => c.chan_id),
                y: [...(ch.data ?? [])].sort((a, b) => b.label_fraction - a.label_fraction).map((c) => c.label_fraction),
                marker: { color: "rgba(201,133,0,0.35)", line: { color: C.label, width: 1 }, pattern: { shape: "/", fgcolor: C.label, size: 5 } } as never,
                customdata: [...(ch.data ?? [])].sort((a, b) => b.label_fraction - a.label_fraction).map((c) => [c.spacecraft, c.n_intervals]),
                hovertemplate: "%{x} (%{customdata[0]}): %{y:.1%} · %{customdata[1]} sequence(s)<extra></extra>",
              }]}
              layout={baseLayout({ margin: { l: 52, r: 8, t: 6, b: 48 }, xaxis: axis({ tickangle: -90, tickfont: { size: 8.5, family: "IBM Plex Mono", color: C.muted } }), yaxis: axis({ tickformat: ".0%", title: { text: "Labelled share" } }) })}
            />
          </Panel>
        </div>
      )}

      {snap && snap.warnings.length > 0 && (
        <Panel eyebrow="Data quality" title="Validation warnings">
          <ul className="grid gap-1.5 md:grid-cols-2">
            {snap.warnings.map((w) => <li key={w} className="num text-xs text-ink-2">• {w}</li>)}
          </ul>
          <p className="mt-3 text-xs text-muted">No missing (non-finite) values were found in any file. Constant training channels are kept: the models then learn a constant, and any test deviation produces forecast error.</p>
        </Panel>
      )}

      <Panel eyebrow="Channel inventory" title={`${rows.length} channels`}
        actions={
          <div className="flex items-center gap-2">
            <Segmented size="sm" ariaLabel="Spacecraft filter" value={scFilter} onChange={setScFilter} options={[{ value: "ALL", label: "All" }, { value: "SMAP", label: "SMAP" }, { value: "MSL", label: "MSL" }]} />
            <input aria-label="Search channels" placeholder="Search…" value={search} onChange={(e) => setSearch(e.target.value)} className="num h-7 w-28 rounded-md border border-edge bg-raised px-2 text-sm text-ink outline-none" />
            <Select ariaLabel="Sort" value={sortKey} onChange={(v) => setSortKey(v as typeof sortKey)} className="h-7 text-xs">
              <option value="chan_id">Sort: channel</option>
              <option value="test_len">Sort: test length</option>
              <option value="train_len">Sort: train length</option>
              <option value="label_fraction">Sort: labelled share</option>
              <option value="n_intervals">Sort: label count</option>
            </Select>
          </div>
        }
      >
        <div className="max-h-[560px] overflow-auto">
          <table className="w-full text-sm">
            <thead className="sticky top-0 bg-panel">
              <tr className="text-left text-xs text-muted">
                {["Channel", "Craft", "Train", "Test", "Features", "Labels", "Labelled", "Range", "Train σ", "Missing", "Notes"].map((h) => (
                  <th key={h} className="py-2 pr-3 font-medium">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((c) => (
                <tr key={c.chan_id} className="border-t border-hairline">
                  <td className="num py-1.5 pr-3"><Link to={`/telemetry?chan=${c.chan_id}`} className="text-ink hover:text-accent">{c.chan_id}</Link></td>
                  <td className="py-1.5 pr-3 text-ink-2">{c.spacecraft}</td>
                  <td className="num py-1.5 pr-3">{fmtInt(c.train_len)}</td>
                  <td className="num py-1.5 pr-3">{fmtInt(c.test_len)}</td>
                  <td className="num py-1.5 pr-3">{c.n_features}</td>
                  <td className="num py-1.5 pr-3">{c.n_intervals}{c.label_rows > 1 && <span className="text-warn" title="merged duplicate label rows">*</span>}</td>
                  <td className="num py-1.5 pr-3">{fmtPct(c.label_fraction)}</td>
                  <td className="num py-1.5 pr-3 text-ink-2">{fmtNum(c.value_min, 2)} … {fmtNum(c.value_max, 2)}</td>
                  <td className="num py-1.5 pr-3">{fmtNum(c.train_std, 3)}</td>
                  <td className="num py-1.5 pr-3">{nonfinite.has(c.chan_id) ? nonfinite.get(c.chan_id) : "—"}</td>
                  <td className="py-1.5 text-xs text-warn">{c.issues.join("; ")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
    </div>
  );
}
