import {
  Activity,
  BookOpen,
  Cpu,
  Database,
  FlaskConical,
  Gauge,
  History,
  Orbit,
  Radar,
  RadioTower,
  ScanSearch,
} from "lucide-react";
import type { ReactNode } from "react";
import { NavLink } from "react-router-dom";
import { useDatasetStatus, useExperiments, useSystem } from "../api/hooks";
import { Wordmark } from "./Logo";

const NAV = [
  { group: "Mission", items: [
    { to: "/", label: "Mission Overview", icon: Orbit, end: true },
    { to: "/telemetry", label: "Telemetry Explorer", icon: Activity },
    { to: "/detect", label: "Detection Centre", icon: Radar },
    { to: "/monitor", label: "Telemetry Replay", icon: RadioTower },
    { to: "/investigate", label: "Investigation", icon: ScanSearch },
  ] },
  { group: "Research", items: [
    { to: "/lab", label: "Model Laboratory", icon: FlaskConical },
    { to: "/evaluation", label: "Model Evaluation", icon: Gauge },
    { to: "/experiments", label: "Experiment History", icon: History },
    { to: "/dataset", label: "Dataset Explorer", icon: Database },
    { to: "/docs", label: "Research & Docs", icon: BookOpen },
  ] },
];

function ContextBar() {
  const ds = useDatasetStatus();
  const sys = useSystem();
  const exps = useExperiments();
  const running = (exps.data ?? []).filter((e) => e.status === "running").length;
  const queued = (exps.data ?? []).filter((e) => e.status === "queued").length;
  const backendDown = ds.isError;
  return (
    <div className="flex h-12 items-center justify-between gap-4 border-b border-hairline bg-void/80 px-4 backdrop-blur md:px-6">
      <div className="flex min-w-0 items-center gap-5 text-[12px]">
        <span className="inline-flex items-center gap-2 whitespace-nowrap text-ink-2">
          <span
            className={`h-1.5 w-1.5 rounded-full ${backendDown ? "bg-critical" : ds.data?.available ? "bg-good" : "bg-warn"}`}
            aria-hidden
          />
          {backendDown ? "Backend unreachable" : ds.data?.available ? "Benchmark loaded · SMAP + MSL" : "Dataset not installed"}
        </span>
        {ds.data?.snapshot && (
          <span className="hidden text-muted md:inline">
            source <span className="text-ink-2">{ds.data.snapshot.source.replace("_", " ")}</span> · checksum{" "}
            <span className="num text-ink-2">{ds.data.snapshot.dataset_checksum.slice(0, 10)}</span>
          </span>
        )}
      </div>
      <div className="flex items-center gap-5 text-[12px] text-muted">
        {(running > 0 || queued > 0) && (
          <NavLink to="/experiments" className="inline-flex items-center gap-1.5 text-accent hover:underline">
            <span className="pulse-dot h-1.5 w-1.5 rounded-full bg-accent" />
            {running} running{queued ? ` · ${queued} queued` : ""}
          </NavLink>
        )}
        {sys.data && (
          <span className="inline-flex items-center gap-1.5 whitespace-nowrap" title={sys.data.platform}>
            <Cpu size={13} />
            {sys.data.cuda_available ? `CUDA · ${sys.data.cuda_device}` : `CPU · ${sys.data.cpu_count} cores`}
          </span>
        )}
        <span className="hidden rounded border border-hairline px-2 py-0.5 text-[11px] lg:inline" title="Independent research implementation — not affiliated with NASA/JPL">
          Independent research · benchmark data
        </span>
      </div>
    </div>
  );
}

export function Shell({ children }: { children: ReactNode }) {
  return (
    <div className="flex h-full flex-col md:flex-row">
      {/* Below the md breakpoint the sidebar becomes a top bar with a horizontally scrolling menu. */}
      <aside className="flex shrink-0 flex-col border-b border-hairline bg-panel md:w-[232px] md:border-r md:border-b-0">
        <div className="px-4 pt-3 pb-2 md:px-5 md:pt-5 md:pb-6">
          <Wordmark />
        </div>
        <nav className="flex gap-1 overflow-x-auto px-3 pb-2 md:block md:flex-1 md:overflow-y-auto md:pb-0" aria-label="Main">
          {NAV.map((g) => (
            <div key={g.group} className="flex shrink-0 gap-1 md:mb-5 md:block">
              <div className="eyebrow mb-1.5 hidden px-2.5 text-dim md:block">{g.group}</div>
              {g.items.map((it) => (
                <NavLink
                  key={it.to}
                  to={it.to}
                  end={it.end}
                  className={({ isActive }) =>
                    `group relative flex shrink-0 items-center gap-2.5 whitespace-nowrap rounded-md px-2.5 py-[7px] text-[13px] transition-colors ${
                      isActive ? "bg-raised text-ink" : "text-ink-2 hover:bg-raised/60 hover:text-ink"
                    }`
                  }
                >
                  {({ isActive }) => (
                    <>
                      {isActive && <span className="absolute top-1.5 bottom-1.5 left-0 w-[2px] rounded bg-accent" />}
                      <it.icon size={15} className={isActive ? "text-accent" : "text-muted group-hover:text-ink-2"} />
                      {it.label}
                    </>
                  )}
                </NavLink>
              ))}
            </div>
          ))}
        </nav>
        <div className="hidden border-t border-hairline px-5 py-4 text-[11px] leading-relaxed text-dim md:block">
          Inspired by Hundman et al., KDD 2018. Not affiliated with NASA/JPL. Anomalies are patterns that warrant investigation, not
          confirmed failures.
        </div>
      </aside>
      <div className="flex min-w-0 flex-1 flex-col">
        <ContextBar />
        <main className="graticule flex-1 overflow-y-auto">
          <div className="mx-auto max-w-[1440px] px-4 py-5 md:px-8 md:py-7">
            {children}
            <p className="mt-8 text-[11px] leading-relaxed text-dim md:hidden">
              Inspired by Hundman et al., KDD 2018. Not affiliated with NASA/JPL. Anomalies are patterns that warrant investigation, not
              confirmed failures.
            </p>
          </div>
        </main>
      </div>
    </div>
  );
}
