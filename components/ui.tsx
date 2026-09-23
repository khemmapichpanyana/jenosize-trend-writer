import type { ReactNode } from "react";
import { Button as ShadcnButton } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export function Card({ title, action, children, className = "" }: { title?: ReactNode; action?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={`rounded-xl border border-line bg-surface-1 shadow-[0_6px_20px_rgba(7,19,38,0.03)] ${className}`}>
      {(title || action) && (
        <header className="flex h-10 items-center justify-between gap-3 border-b border-line px-3.5 py-2.5">
          <h2 className="min-w-0 text-[13px] font-semibold tracking-[-0.01em] text-ink">{title}</h2>
          {action}
        </header>
      )}
      {/* overflow-x-auto: wide tables scroll inside the card on phones instead of widening the page */}
      <div className="overflow-x-auto p-3.5">{children}</div>
    </section>
  );
}

export function PageHeader({ title, subtitle, action }: { title: string; subtitle?: string; action?: ReactNode }) {
  return (
    <div className="t-stagger is-shown mb-5 flex flex-wrap items-center justify-between gap-3">
      <div className="min-w-0">
        <h1 className="t-stagger-line t-stagger-line--1 text-[18px] font-semibold tracking-[-0.02em] text-ink">{title}</h1>
        {subtitle && <p className="t-stagger-line t-stagger-line--2 mt-0.5 max-w-xl truncate text-xs text-muted">{subtitle}</p>}
      </div>
      <div className="t-stagger-line t-stagger-line--2">{action}</div>
    </div>
  );
}

type ButtonProps = React.ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "primary" | "secondary" | "danger"; busy?: boolean };

export function Button({ variant = "secondary", busy, children, className = "", disabled, ...rest }: ButtonProps) {
  const styles = {
    primary: "bg-accent text-white shadow-[0_4px_14px_rgba(36,87,214,0.16)] hover:bg-accent/90",
    secondary: "border-line bg-surface-1 text-ink hover:border-accent/35 hover:bg-accent-wash",
    danger: "border-line bg-surface-1 text-critical hover:bg-critical/10",
  }[variant];
  return (
    <ShadcnButton
      variant={variant === "primary" ? "default" : variant === "danger" ? "destructive" : "outline"}
      className={cn("h-8 rounded-full px-3.5 text-xs", styles, className)}
      disabled={disabled || busy}
      {...rest}
    >
      {busy && <span className="size-3 animate-spin rounded-full border-2 border-current border-t-transparent" aria-hidden />}
      {children}
    </ShadcnButton>
  );
}

/** Stat tile: label (sentence case), value, optional hint. */
export function StatTile({ label, value, hint }: { label: string; value: ReactNode; hint?: ReactNode }) {
  return (
    <div className="rounded-xl border border-line bg-surface-1 p-3 shadow-[0_6px_20px_rgba(7,19,38,0.02)]">
      <div className="text-[11px] font-medium text-ink-2">{label}</div>
      <div className="mt-0.5 text-lg font-semibold tabular-nums tracking-tight text-ink">{value}</div>
      {hint && <div className="mt-0.5 text-[11px] text-ink-2">{hint}</div>}
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
  return <p className="py-4 text-center text-xs text-muted">{children}</p>;
}

export function ErrorNote({ message }: { message: string | null | undefined }) {
  if (!message) return null;
  return (
    <p role="alert" className="rounded-lg border border-critical/30 bg-critical/5 px-3 py-2 text-[13px] leading-5 text-ink">
      <span className="mr-1 font-semibold text-critical" aria-hidden>×</span>
      {message}
    </p>
  );
}

export function Field({ label, children, hint }: { label: string; children: ReactNode; hint?: string }) {
  return (
    <label className="block text-[13px]">
      <span className="mb-1 block font-medium text-ink-2">{label}</span>
      {children}
      {hint && <span className="mt-1 block text-[11px] text-muted">{hint}</span>}
    </label>
  );
}

export const inputClass =
  "w-full rounded-lg border border-line bg-surface-1 px-2.5 py-2 text-[13px] text-ink placeholder:text-muted transition-[border-color,box-shadow] focus:border-accent focus:outline-none focus:ring-4 focus:ring-accent/10";
