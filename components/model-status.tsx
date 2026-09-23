"use client";

import { Cpu, LoaderCircle, Power } from "lucide-react";
import { useModelStatus, wakeModel } from "@/lib/model-status";

/** Icon-only status for the collapsed sidebar rail: model glyph + status dot; clicking wakes it when offline. */
export function ModelStatusIcon() {
  const status = useModelStatus();
  const look = {
    idle: { dot: "bg-good", label: "Model ready" },
    ready: { dot: "bg-good", label: "Model ready" },
    warming: { dot: "bg-warning", label: "Waking the model (1–3 minutes after idle)" },
    offline: { dot: "bg-critical", label: "Model offline — click to wake it" },
  }[status];
  const offline = status === "offline";
  return (
    <button
      type="button"
      onClick={offline ? wakeModel : undefined}
      disabled={!offline}
      title={look.label}
      aria-label={look.label}
      className={`relative flex size-8 items-center justify-center rounded-lg text-ink-2 transition-colors disabled:cursor-default ${offline ? "hover:bg-critical/10 hover:text-critical" : ""}`}
    >
      {status === "warming" ? <LoaderCircle size={16} className="animate-spin text-[#9a6a00]" /> : offline ? <Power size={16} /> : <Cpu size={16} />}
      <span className={`absolute right-1 top-1 size-2 rounded-full ring-2 ring-surface-1 ${look.dot} ${status === "warming" ? "animate-pulse" : ""}`} aria-hidden />
    </button>
  );
}

/**
 * The model's state, with a way out when it's down: "offline" shows a Wake
 * button (the GPU scales to zero, and a warmup can time out mid-cold-start).
 */
export function ModelStatusControl({ compact = false }: { compact?: boolean }) {
  const status = useModelStatus();
  const pill = "inline-flex shrink-0 items-center gap-1 rounded-md px-1.5 py-0.5 text-[9px] font-semibold uppercase tracking-[0.1em]";

  if (status === "offline") {
    return (
      <span className="inline-flex shrink-0 items-center gap-1">
        <span className={`${pill} bg-critical/5 text-critical`}>
          <span className="size-1.5 rounded-full bg-critical" /> {compact ? "Offline" : "Model offline"}
        </span>
        <button
          type="button"
          onClick={wakeModel}
          title="Start the model's GPU (1–3 minutes after idle)"
          className="inline-flex items-center gap-1 rounded-md border border-line bg-surface-1 px-1.5 py-0.5 text-[10.5px] font-semibold text-ink transition-colors hover:border-accent/40 hover:bg-accent-wash hover:text-accent-ink"
        >
          <Power size={10} aria-hidden /> Wake
        </button>
      </span>
    );
  }
  if (status === "warming") {
    return (
      <span className={`${pill} bg-warning/10 text-[#9a6a00]`} title="The GPU scales to zero when idle; it is starting up (1–3 minutes)" role="status">
        <LoaderCircle size={9} className="animate-spin" aria-hidden /> {compact ? "Waking" : "Waking model"}
      </span>
    );
  }
  return (
    <span className={`${pill} bg-good/5 text-good`}>
      <span className="size-1.5 rounded-full bg-good" /> {compact ? "Ready" : "Model ready"}
    </span>
  );
}
