import { GitCompare, Trash2 } from "lucide-react";
import { useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useDeleteExperiment, useExperiments } from "../api/hooks";
import { Button, EmptyState, ErrorState, Loading, OriginTag, PageHeader, Panel, Segmented, StatusBadge } from "../components/ui";
import { useToast } from "../components/toast";
import { fmtDate, fmtDuration, fmtNum } from "../lib/format";
import { METHOD_COLOR } from "../lib/palette";

export default function Experiments() {
  const exps = useExperiments();
  const del = useDeleteExperiment();
  const toast = useToast();
  const nav = useNavigate();
  const [sel, setSel] = useState<number[]>([]);
  const [origin, setOrigin] = useState<"all" | "local" | "reference">("all");
  const rows = useMemo(() => (exps.data ?? []).filter((e) => origin === "all" || e.origin === origin), [exps.data, origin]);

  if (exps.isError) return <ErrorState error={exps.error} />;
  if (!exps.data) return <Loading height={400} />;

  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow="Experiment history"
        title="Every run, with its full configuration"
        actions={
          <Button variant="primary" disabled={sel.length < 1} onClick={() => nav(`/evaluation?ids=${sel.join(",")}`)}>
            <GitCompare size={14} /> Compare {sel.length || ""}
          </Button>
        }
      >
        Experiments record the method, channels, hyper-parameters, threshold settings, seed, code commit, dataset checksum, device,
        timings and evaluation metrics. Reference experiments were computed with this code on the full benchmark and imported.
      </PageHeader>

      <Panel
        title={`${rows.length} experiment${rows.length === 1 ? "" : "s"}`}
        actions={<Segmented size="sm" ariaLabel="Origin" value={origin} onChange={setOrigin} options={[{ value: "all", label: "All" }, { value: "local", label: "Local" }, { value: "reference", label: "Reference" }]} />}
        pad={false}
      >
        {rows.length === 0 ? (
          <div className="p-5"><EmptyState title="No experiments" action={<Link to="/lab"><Button variant="primary">Open Model Laboratory</Button></Link>} /></div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-[12.5px]">
              <thead>
                <tr className="border-b border-hairline text-left text-[11px] text-muted">
                  <th className="w-10 px-5 py-2" />
                  {["#", "Name", "Method", "Scope", "Status", "Event P", "Event R", "F0.5", "Point F1", "Train time", "Created", ""].map((h) => (
                    <th key={h} className="py-2 pr-3 font-medium">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map((e) => (
                  <tr key={e.id} className="border-b border-hairline hover:bg-raised/40">
                    <td className="px-5 py-2">
                      <input
                        type="checkbox"
                        aria-label={`Select experiment ${e.id}`}
                        checked={sel.includes(e.id)}
                        disabled={!["completed", "partial"].includes(e.status)}
                        onChange={() => setSel((s) => (s.includes(e.id) ? s.filter((x) => x !== e.id) : [...s, e.id]))}
                        className="accent-[#4d7cfe]"
                      />
                    </td>
                    <td className="num py-2 pr-3 text-muted">{e.id}</td>
                    <td className="max-w-[280px] py-2 pr-3">
                      <Link to={`/experiments/${e.id}`} className="block truncate text-ink hover:text-accent">{e.name}</Link>
                      <OriginTag origin={e.origin} />
                    </td>
                    <td className="py-2 pr-3"><span className="inline-flex items-center gap-1.5"><span className="h-2 w-2 rounded-sm" style={{ background: METHOD_COLOR[e.method] }} />{e.method_label}{e.profile ? ` · ${e.profile}` : ""}</span></td>
                    <td className="py-2 pr-3 text-ink-2">{e.spacecraft} · {e.n_channels} ch</td>
                    <td className="py-2 pr-3"><StatusBadge status={e.status} />{e.status === "running" && <span className="num ml-2 text-muted">{Math.round(e.progress * 100)}%</span>}</td>
                    <td className="num py-2 pr-3">{fmtNum(e.headline.event_precision, 3)}</td>
                    <td className="num py-2 pr-3">{fmtNum(e.headline.event_recall, 3)}</td>
                    <td className="num py-2 pr-3">{fmtNum(e.headline.event_f0_5, 3)}</td>
                    <td className="num py-2 pr-3">{fmtNum(e.headline.point_f1, 3)}</td>
                    <td className="num py-2 pr-3">{fmtDuration(e.train_time_s)}</td>
                    <td className="py-2 pr-3 text-muted">{fmtDate(e.created_at)}</td>
                    <td className="py-2 pr-5 text-right">
                      <button
                        aria-label={`Delete experiment ${e.id}`}
                        className="text-dim hover:text-[#f07070] disabled:opacity-30"
                        disabled={["running", "queued"].includes(e.status)}
                        onClick={() => {
                          if (!window.confirm(`Delete experiment #${e.id} (${e.name}) and its local model files?`)) return;
                          del.mutate(e.id, {
                            onSuccess: () => toast({ tone: "success", title: `Experiment #${e.id} deleted` }),
                            onError: (err: Error) => toast({ tone: "error", title: "Delete failed", body: err.message }),
                          });
                        }}
                      >
                        <Trash2 size={14} />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>
    </div>
  );
}
