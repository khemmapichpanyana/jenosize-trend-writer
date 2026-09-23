"use client";

import Link from "next/link";
import { CheckBadge, StatusBadge } from "@/components/status";
import { LoadingState } from "@/components/loading-state";
import { Card, Empty, ErrorNote, Meter, PageHeader, StatTile } from "@/components/ui";
import { compact, fixed } from "@/lib/format";
import { usePoll } from "@/lib/hooks";
import type { Adapter, CorpusStatus, DoctorCheck, Resources } from "@/lib/types";

export default function SystemPage() {
  // Each poll is a round trip to Modal; these values change slowly, so poll lazily.
  const doctor = usePoll<DoctorCheck[]>("/doctor", 120_000);
  const corpus = usePoll<CorpusStatus>("/corpus", 60_000);
  const resources = usePoll<Resources>("/resources", 20_000);
  const adapters = usePoll<Adapter[]>("/adapters", 120_000);
  const counts = corpus.data?.counts;
  const active = adapters.data?.find((a) => a.active);

  return (
    <>
      <PageHeader title="System" subtitle="Corpus, adapters, live compute and service health." />
      <ErrorNote message={doctor.error ?? corpus.error} />

      <div className="grid grid-cols-2 gap-2.5 lg:grid-cols-5">
        <StatTile label="Articles discovered" value={compact(counts?.discovered)} />
        <StatTile label="Cleaned" value={compact(counts?.cleaned)} hint={counts ? `${compact(counts.errors)} rejected` : undefined} />
        <StatTile label="Labelled" value={compact(counts?.labelled)} />
        <StatTile label="Adapters here" value={compact(adapters.data?.length ?? null)} hint={active ? `active: ${active.served_as}` : "local registry only"} />
        <StatTile label="Active jobs" value={compact(resources.data?.active_jobs.length ?? null)} />
      </div>

      <div className="mt-4 grid gap-4 xl:grid-cols-3">
        <Card title="Live compute" className="xl:col-span-2" action={<span className="text-xs text-muted">every 20s</span>}>
          {resources.data?.active_jobs.length ? (
            <ul className="space-y-4">
              {resources.data.active_jobs.map((job) => (
                <li key={job.id} className="rounded-lg border border-line p-3">
                  <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                    <Link href={`/runs/${job.id}`} className="text-[13px] font-medium text-ink hover:text-accent-ink">{job.kind} {String(job.params.version ?? "")}</Link>
                    <StatusBadge status={job.status} />
                  </div>
                  {job.latest ? (
                    <div className="grid gap-3 sm:grid-cols-3">
                      <Meter label="GPU utilisation" value={job.latest.gpu_util} max={100} unit="%" />
                      <Meter label="GPU memory" value={job.latest.gpu_mem_used_gb} max={Math.round(job.latest.gpu_mem_total_gb ?? 24)} unit=" GB" warnAt={0.9} dangerAt={0.97} />
                      <div className="text-xs text-ink-2"><div>step {job.latest.step ?? "—"} / {job.latest.total_steps ?? "—"}</div><div className="tabular-nums">loss {fixed(job.latest.loss)}</div><div className="text-muted">{job.latest.phase}</div></div>
                    </div>
                  ) : <p className="text-xs text-muted">No telemetry yet (CPU job, or the GPU container is starting).</p>}
                </li>
              ))}
            </ul>
          ) : resources.loading ? <LoadingState label="Loading compute" /> : <Empty>No jobs running. Start one from Data or Training.</Empty>}
          {resources.data && (
            <table className="mt-4 w-full text-left text-xs">
              <thead className="text-muted"><tr><th className="pb-1.5 font-medium">Modal function</th><th className="pb-1.5 font-medium">Containers</th><th className="pb-1.5 font-medium">Running</th><th className="pb-1.5 font-medium">Queued</th></tr></thead>
              <tbody className="tabular-nums text-ink">{Object.entries(resources.data.functions).map(([name, s]) => <tr key={name} className="border-t border-line"><td className="py-1.5 font-mono">{name}</td><td>{s.num_total_runners ?? "—"}</td><td>{s.num_running_inputs ?? "—"}</td><td>{s.backlog ?? "—"}</td></tr>)}</tbody>
            </table>
          )}
        </Card>

        <Card title="System health">
          {doctor.data ? <ul className="space-y-3">{doctor.data.map((c) => <li key={c.check}><CheckBadge ok={c.ok} label={c.check} /><p className="ml-5 text-xs text-ink-2">{c.detail}</p></li>)}</ul> : doctor.loading ? <LoadingState label="Checking system health" /> : <Empty>Unavailable</Empty>}
        </Card>
      </div>
    </>
  );
}
