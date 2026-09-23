import type { JobStatus } from "@/lib/types";

// Status colours are reserved for state and always ship with an icon + label.
const STATUS: Record<JobStatus, { icon: string; label: string; className: string }> = {
  queued: { icon: "○", label: "Queued", className: "text-muted" },
  running: { icon: "●", label: "Running", className: "text-accent-ink" },
  succeeded: { icon: "✓", label: "Succeeded", className: "text-good" },
  failed: { icon: "✕", label: "Failed", className: "text-critical" },
  cancelled: { icon: "–", label: "Cancelled", className: "text-muted" },
};

export function StatusBadge({ status }: { status: JobStatus }) {
  const s = STATUS[status] ?? STATUS.queued;
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full border border-line bg-surface-1 px-2 py-0.5 text-[11px] font-medium text-ink">
      <span className={`${s.className} ${status === "running" ? "animate-pulse" : ""}`} aria-hidden>
        {s.icon}
      </span>
      {s.label}
    </span>
  );
}

export function CheckBadge({ ok, label }: { ok: boolean; label: string }) {
  return (
    <span className="inline-flex items-center gap-1.5 text-[13px] text-ink">
      <span className={ok ? "text-good" : "text-critical"} aria-hidden>{ok ? "✓" : "✕"}</span>
      {label}
    </span>
  );
}
