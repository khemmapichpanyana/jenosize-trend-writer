"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { ApiError, post } from "@/lib/api";
import type { JobRun } from "@/lib/types";
import { Button } from "./ui";

/** Starts a job, then opens its live run page. A 409 opens the run already active. */
export function JobButton({ path, body, label, variant = "primary", disabled }: { path: string; body?: unknown; label: string; variant?: "primary" | "secondary"; disabled?: boolean }) {
  const router = useRouter();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  async function start() {
    setBusy(true);
    setError(null);
    try {
      const run = await post<JobRun>(path, body ?? {});
      router.push(`/runs/${run.id}`);
    } catch (e) {
      if (e instanceof ApiError && e.status === 409) {
        const runId = (e.body as { error?: { run_id?: string } })?.error?.run_id;
        if (runId) return router.push(`/runs/${runId}`);
      }
      setError(e instanceof Error ? e.message : String(e));
      setBusy(false);
    }
  }
  return (
    <div className="flex flex-col items-start gap-1">
      <Button variant={variant} busy={busy} onClick={start} disabled={disabled}>{label}</Button>
      {error && <span role="alert" className="text-xs text-critical">{error}</span>}
    </div>
  );
}
