import { AlertTriangle, CheckCircle2, CircleDashed, Info, Loader2, Square, XCircle } from "lucide-react";
import type { ReactNode } from "react";
import { ApiError } from "../api/client";

export function Panel({
  title,
  eyebrow,
  actions,
  children,
  className = "",
  pad = true,
}: {
  title?: ReactNode;
  eyebrow?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  pad?: boolean;
}) {
  return (
    <section className={`panel ${className}`}>
      {(title || actions || eyebrow) && (
        <header className="flex items-start justify-between gap-4 px-5 pt-4 pb-3">
          <div className="min-w-0">
            {eyebrow && <div className="eyebrow mb-1">{eyebrow}</div>}
            {title && <h2 className="text-lg font-medium text-ink">{title}</h2>}
          </div>
          {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
        </header>
      )}
      <div className={pad ? "px-5 pb-5" : ""}>{children}</div>
    </section>
  );
}

export function PageHeader({ eyebrow, title, children, actions }: { eyebrow: string; title: string; children?: ReactNode; actions?: ReactNode }) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
      <div className="max-w-3xl">
        <div className="eyebrow mb-2">{eyebrow}</div>
        <h1 className="text-3xl font-medium text-ink">{title}</h1>
        {children && <p className="mt-2 text-base leading-relaxed text-ink-2">{children}</p>}
      </div>
      {actions && <div className="flex items-center gap-2">{actions}</div>}
    </div>
  );
}

export function Stat({ label, value, sub, mono = true }: { label: string; value: ReactNode; sub?: ReactNode; mono?: boolean }) {
  return (
    <div className="min-w-0">
      <div className="eyebrow">{label}</div>
      <div className={`mt-1 text-2xl font-medium text-ink ${mono ? "font-mono" : ""}`}>{value}</div>
      {sub && <div className="mt-0.5 text-xs text-muted">{sub}</div>}
    </div>
  );
}

export function KV({ k, v, mono = true }: { k: ReactNode; v: ReactNode; mono?: boolean }) {
  return (
    <div className="flex items-baseline justify-between gap-3 py-1.5 hairline-b last:border-b-0">
      <span className="text-sm text-muted">{k}</span>
      <span className={`text-right text-sm text-ink ${mono ? "num" : ""}`}>{v}</span>
    </div>
  );
}

type BtnVariant = "primary" | "secondary" | "ghost" | "danger";
export function Button({
  children,
  variant = "secondary",
  size = "md",
  className = "",
  loading = false,
  ...rest
}: React.ButtonHTMLAttributes<HTMLButtonElement> & { variant?: BtnVariant; size?: "sm" | "md"; loading?: boolean }) {
  const base =
    "inline-flex items-center justify-center gap-2 rounded-md font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-45";
  const sizes = size === "sm" ? "h-7 px-2.5 text-sm" : "h-9 px-3.5 text-sm";
  const variants: Record<BtnVariant, string> = {
    primary: "bg-accent text-accent-ink hover:bg-accent/85",
    secondary: "border border-edge bg-raised text-ink hover:border-muted/50 hover:bg-edge/40",
    ghost: "text-ink-2 hover:bg-raised hover:text-ink",
    danger: "border border-critical/40 bg-critical/10 text-critical-ink hover:bg-critical/20",
  };
  return (
    <button className={`${base} ${sizes} ${variants[variant]} ${className}`} disabled={loading || rest.disabled} {...rest}>
      {loading && <Loader2 size={14} className="animate-spin" />}
      {children}
    </button>
  );
}

export function Segmented<T extends string>({
  value,
  options,
  onChange,
  size = "md",
  ariaLabel,
}: {
  value: T;
  options: { value: T; label: ReactNode }[];
  onChange: (v: T) => void;
  size?: "sm" | "md";
  ariaLabel?: string;
}) {
  return (
    <div role="radiogroup" aria-label={ariaLabel} className="inline-flex rounded-md border border-hairline bg-sunken p-0.5">
      {options.map((o) => (
        <button
          key={o.value}
          role="radio"
          aria-checked={value === o.value}
          onClick={() => onChange(o.value)}
          className={`rounded-sm font-medium transition-colors ${size === "sm" ? "px-2 py-0.5 text-xs" : "px-3 py-1 text-sm"} ${
            value === o.value ? "bg-raised text-ink shadow-[inset_0_0_0_1px_var(--color-edge)]" : "text-muted hover:text-ink"
          }`}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Select({
  value,
  onChange,
  children,
  className = "",
  ariaLabel,
}: {
  value: string;
  onChange: (v: string) => void;
  children: ReactNode;
  className?: string;
  ariaLabel?: string;
}) {
  return (
    <select
      aria-label={ariaLabel}
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className={`h-9 rounded-md border border-edge bg-raised px-2.5 text-sm text-ink outline-none hover:border-muted/50 ${className}`}
    >
      {children}
    </select>
  );
}

const STATUS: Record<string, { label: string; color: string; Icon: typeof CheckCircle2; spin?: boolean }> = {
  completed: { label: "Completed", color: "text-good", Icon: CheckCircle2 },
  partial: { label: "Partial", color: "text-warn", Icon: AlertTriangle },
  running: { label: "Running", color: "text-accent", Icon: Loader2, spin: true },
  queued: { label: "Queued", color: "text-muted", Icon: CircleDashed },
  failed: { label: "Failed", color: "text-critical-ink", Icon: XCircle },
  cancelled: { label: "Cancelled", color: "text-muted", Icon: Square },
  pending: { label: "Pending", color: "text-muted", Icon: CircleDashed },
};

export function StatusBadge({ status }: { status: string }) {
  const s = STATUS[status] ?? { label: status, color: "text-muted", Icon: Info };
  return (
    <span className={`inline-flex items-center gap-1.5 text-sm font-medium ${s.color}`}>
      <s.Icon size={13} className={s.spin ? "animate-spin" : ""} aria-hidden />
      {s.label}
    </span>
  );
}

export function OriginTag({ origin }: { origin: string }) {
  return origin === "reference" ? (
    <span
      title="Imported reference result: computed with this code on the full benchmark (see Research → Reproducibility)"
      className="rounded border border-edge bg-raised px-1.5 text-xs text-ink-2"
    >
      Reference
    </span>
  ) : (
    <span className="rounded border border-hairline px-1.5 text-xs text-muted">Local</span>
  );
}

export function Skeleton({ className = "" }: { className?: string }) {
  return <div className={`skeleton ${className}`} aria-hidden />;
}

export function Loading({ label = "Loading", height = 240 }: { label?: string; height?: number }) {
  return (
    <div className="flex flex-col gap-3" style={{ minHeight: height }} role="status" aria-live="polite">
      <Skeleton className="h-4 w-40" />
      <Skeleton className="w-full flex-1" />
      <span className="sr-only">{label}</span>
    </div>
  );
}

export function EmptyState({ icon, title, children, action }: { icon?: ReactNode; title: string; children?: ReactNode; action?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center rounded-lg border border-dashed border-edge px-6 py-12 text-center">
      {icon && <div className="mb-3 text-dim">{icon}</div>}
      <div className="text-base font-medium text-ink">{title}</div>
      {children && <div className="mt-1.5 max-w-md text-sm leading-relaxed text-muted">{children}</div>}
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}

export function ErrorState({ error, title = "Something went wrong" }: { error: unknown; title?: string }) {
  const msg = error instanceof ApiError || error instanceof Error ? error.message : String(error);
  return (
    <div role="alert" className="flex items-start gap-3 rounded-lg border border-critical/40 bg-critical/10 px-4 py-3">
      <XCircle size={16} className="mt-0.5 shrink-0 text-critical-ink" />
      <div>
        <div className="text-sm font-medium text-ink">{title}</div>
        <div className="mt-0.5 text-sm text-ink-2">{msg}</div>
      </div>
    </div>
  );
}

export function Notice({ children, tone = "info" }: { children: ReactNode; tone?: "info" | "warn" }) {
  return (
    <div
      className={`flex items-start gap-2.5 rounded-lg border px-4 py-3 text-sm leading-relaxed ${
        tone === "warn" ? "border-warn/30 bg-warn/5 text-ink-2" : "border-hairline bg-raised/60 text-ink-2"
      }`}
    >
      {tone === "warn" ? <AlertTriangle size={14} className="mt-0.5 shrink-0 text-warn" /> : <Info size={14} className="mt-0.5 shrink-0 text-muted" />}
      <div>{children}</div>
    </div>
  );
}

export function Legend({ items }: { items: { label: string; color: string; kind?: "line" | "band" | "solid" | "hatch" | "dash" }[] }) {
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 text-xs text-ink-2">
      {items.map((it) => (
        <span key={it.label} className="inline-flex items-center gap-1.5">
          {it.kind === "solid" ? (
            <span className="inline-block h-3 w-3 rounded-xs" style={{ background: it.color }} />
          ) : it.kind === "band" ? (
            <span className="inline-block h-3 w-4 rounded-xs" style={{ background: it.color, opacity: 0.55 }} />
          ) : it.kind === "hatch" ? (
            <span className="hatch-label inline-block h-3 w-4 rounded-xs border border-label/60" />
          ) : it.kind === "dash" ? (
            <span className="inline-block w-4 border-t-2 border-dashed" style={{ borderColor: it.color }} />
          ) : (
            <span className="inline-block h-[2px] w-4 rounded" style={{ background: it.color }} />
          )}
          {it.label}
        </span>
      ))}
    </div>
  );
}

export function OutcomeTag({ outcome }: { outcome: string }) {
  return outcome === "matches_label" ? (
    <span className="inline-flex items-center gap-1 text-xs text-ink-2">
      <span className="hatch-label inline-block h-2.5 w-2.5 rounded-sm border border-label/60" />
      Overlaps expert label
    </span>
  ) : (
    <span className="inline-flex items-center gap-1 text-xs text-muted">
      <span className="inline-block h-2.5 w-2.5 rounded-sm border border-dim" />
      No label overlap
    </span>
  );
}
