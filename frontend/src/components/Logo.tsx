export function LogoMark({ size = 28 }: { size?: number }) {
  // Aperture ring + signal trace: a heliograph flashes signals; the trace is telemetry.
  // Drawn from the interface tokens: graphite tile, cyan aperture arc, ink trace.
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden>
      <rect width="32" height="32" rx="8" className="fill-raised" />
      <circle cx="16" cy="16" r="10" fill="none" className="stroke-edge" strokeWidth="1.5" />
      <path d="M8.2 10.2 A10 10 0 0 1 23.8 10.2" fill="none" className="stroke-accent" strokeWidth="1.5" strokeLinecap="round" />
      <path d="M4 17h6l2.5-6 4 12 3-9 2 3H28" fill="none" className="stroke-ink" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
      <circle cx="22.5" cy="17" r="2" className="fill-accent" />
    </svg>
  );
}

export function Wordmark() {
  return (
    <div className="flex items-center gap-2.5">
      <LogoMark />
      <div className="hidden leading-none md:block">
        <div className="text-lg leading-5 font-medium text-ink">Heliograph</div>
        <div className="text-xs text-muted">SMAP · MSL telemetry</div>
      </div>
    </div>
  );
}
