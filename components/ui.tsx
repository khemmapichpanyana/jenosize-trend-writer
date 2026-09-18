import type { ReactNode } from "react";

export function Card({ title, action, children, className = "" }: { title?: ReactNode; action?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={`rounded-xl border border-line bg-surface-1 ${className}`}>
      {(title || action) && (
        <header className="flex items-center justify-between gap-3 border-b border-line px-4 py-3">
          <h2 className="text-sm font-semibold text-ink">{title}</h2>
          {action}
        </header>
      )}
      <div className="p-4">{children}</div>
    </section>
  );
}

export function PageHeader({ title, subtitle, action }: { title: string; subtitle?: string; action?: ReactNode }) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-ink">{title}</h1>
        {subtitle && <p className="mt-1 text-sm text-ink-2">{subtitle}</p>}
      </div>
      {action}
    </div>
  );
}

type ButtonProps = React.ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "primary" | "secondary" | "danger"; busy?: boolean };

export function Button({ variant = "secondary", busy, children, className = "", disabled, ...rest }: ButtonProps) {
  const styles = {
    primary: "bg-accent text-white hover:opacity-90",
    secondary: "border border-line bg-surface-1 text-ink hover:bg-surface-2",
    danger: "border border-line bg-surface-1 text-critical hover:bg-surface-2",
  }[variant];
  return (
    <button
      className={`inline-flex items-center justify-center gap-2 rounded-lg px-3 py-2 text-sm font-medium transition disabled:cursor-not-allowed disabled:opacity-50 ${styles} ${className}`}
      disabled={disabled || busy}
      {...rest}
    >
      {busy && <span className="size-3 animate-spin rounded-full border-2 border-current border-t-transparent" aria-hidden />}
      {children}
    </button>
  );
}

/** Stat tile: label (sentence case), value, optional hint. */
export function StatTile({ label, value, hint }: { label: string; value: ReactNode; hint?: ReactNode }) {
  return (
    <div className="rounded-xl border border-line bg-surface-1 p-4">
      <div className="text-xs font-medium text-muted">{label}</div>
      <div className="mt-1 text-2xl font-semibold tabular-nums text-ink">{value}</div>
      {hint && <div className="mt-1 text-xs text-ink-2">{hint}</div>}
    </div>
  );
}

/** Meter: fill carries severity; the track is a lighter step of the same ramp. */
export function Meter({ label, value, max, unit = "", warnAt, dangerAt }: { label: string; value: number | null | undefined; max: number; unit?: string; warnAt?: number; dangerAt?: number }) {
  const v = value ?? null;
  const ratio = v === null || !max ? 0 : Math.min(1, Math.max(0, v / max));
  const severity = dangerAt !== undefined && ratio >= dangerAt ? "critical" : warnAt !== undefined && ratio >= warnAt ? "warning" : "accent";
  const fill = { accent: "bg-accent", warning: "bg-warning", critical: "bg-critical" }[severity];
  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between text-xs">
        <span className="font-medium text-ink-2">{label}</span>
        <span className="tabular-nums text-ink">
          {v === null ? "—" : `${v.toFixed(v < 10 ? 1 : 0)}${unit}`} <span className="text-muted">/ {max}{unit}</span>
        </span>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-accent-wash" role="meter" aria-label={label} aria-valuenow={v ?? undefined} aria-valuemin={0} aria-valuemax={max}>
        <div className={`h-full rounded-full ${fill} transition-[width] duration-500`} style={{ width: `${ratio * 100}%` }} />
      </div>
    </div>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <p className="py-6 text-center text-sm text-muted">{children}</p>;
}

export function ErrorNote({ message }: { message: string | null | undefined }) {
  if (!message) return null;
  return (
    <p role="alert" className="rounded-lg border border-line bg-surface-2 px-3 py-2 text-sm text-ink">
      <span className="mr-1 font-semibold text-critical" aria-hidden>✕</span>
      {message}
    </p>
  );
}

export function Field({ label, children, hint }: { label: string; children: ReactNode; hint?: string }) {
  return (
    <label className="block text-sm">
      <span className="mb-1 block font-medium text-ink-2">{label}</span>
      {children}
      {hint && <span className="mt-1 block text-xs text-muted">{hint}</span>}
    </label>
  );
}

export const inputClass =
  "w-full rounded-lg border border-line bg-surface-1 px-3 py-2 text-sm text-ink placeholder:text-muted focus:border-accent focus:outline-none";
