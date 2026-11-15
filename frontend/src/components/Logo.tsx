export function LogoMark({ size = 28 }: { size?: number }) {
  // Aperture ring + signal trace: a heliograph flashes signals; the trace is telemetry.
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden>
      <rect width="32" height="32" rx="8" fill="#0F1629" />
      <circle cx="16" cy="16" r="10" fill="none" stroke="#263152" strokeWidth="1.5" />
      <path d="M8.2 10.2 A10 10 0 0 1 23.8 10.2" fill="none" stroke="#4CC9F0" strokeWidth="1.5" strokeLinecap="round" opacity="0.7" />
      <path d="M4 17h6l2.5-6 4 12 3-9 2 3H28" fill="none" stroke="#E6EBF5" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
      <circle cx="22.5" cy="17" r="2" fill="#9085E9" />
    </svg>
  );
}

export function Wordmark() {
  return (
    <div className="flex items-center gap-2.5">
      <LogoMark />
      <div className="leading-none">
        <div className="text-[15px] font-semibold tracking-tight text-ink">Heliograph</div>
        <div className="mt-1 text-[10px] tracking-[0.14em] text-muted uppercase">Telemetry intelligence</div>
      </div>
    </div>
  );
}
