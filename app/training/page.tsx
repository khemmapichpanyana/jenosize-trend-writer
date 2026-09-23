"use client";

import Link from "next/link";
import { useState } from "react";
import { JobButton } from "@/components/job-button";
import { StatusBadge } from "@/components/status";
import { Card, Empty, Field, PageHeader, inputClass } from "@/components/ui";
import { LoadingState } from "@/components/loading-state";
import { ago, duration, fixed } from "@/lib/format";
import { usePoll } from "@/lib/hooks";
import type { DatasetVersion, JobRun } from "@/lib/types";

export default function TrainingPage() {
  const datasets = usePoll<DatasetVersion[]>("/datasets", 30_000);
  const runs = usePoll<JobRun[]>("/runs?kind=train&limit=20", 5_000);
  const [version, setVersion] = useState("");
  const [epochs, setEpochs] = useState(3);
  const [rank, setRank] = useState(16);
  const [lr, setLr] = useState("0.0002");
  const chosen = version || datasets.data?.[0]?.version || "";

  return (
    <>
      <PageHeader title="Training" subtitle="Fine-tuning runs and live progress." />
      <div className="grid gap-4 lg:grid-cols-3">
        <Card title="New training run">
          <div className="space-y-3">
            <Field label="Dataset version">
              <select className={inputClass} value={chosen} onChange={(e) => setVersion(e.target.value)}>
                {(datasets.data ?? []).map((d) => (
                  <option key={d.version} value={d.version}>{d.version} — {d.train_count} examples</option>
                ))}
              </select>
            </Field>
            <div className="grid grid-cols-3 gap-2">
              <Field label="Epochs"><input type="number" min={1} max={10} className={inputClass} value={epochs} onChange={(e) => setEpochs(Number(e.target.value))} /></Field>
              <Field label="LoRA r">
                <select className={inputClass} value={rank} onChange={(e) => setRank(Number(e.target.value))}>
                  {[8, 16, 32, 64].map((r) => <option key={r}>{r}</option>)}
                </select>
              </Field>
              <Field label="LR"><input className={inputClass} value={lr} onChange={(e) => setLr(e.target.value)} /></Field>
            </div>
            <p className="text-xs text-muted">Tip: run 1 epoch first (~$0.50) to prove the GPU path, then the full run. The adapter is saved as jeno-lora-{chosen || "vN"}.</p>
            <JobButton
              path="/train"
              label="Start training"
              disabled={!chosen}
              body={{ version: chosen, epochs, lora_r: rank, learning_rate: Number(lr) }}
            />
            {datasets.loading ? <LoadingState label="Loading datasets" /> : !datasets.data?.length && <p className="text-xs text-critical">Publish a dataset on the Data page first.</p>}
          </div>
        </Card>

        <Card title="Training runs" className="lg:col-span-2">
          {runs.data?.length ? (
            <table className="w-full text-left text-xs">
              <thead className="text-xs text-muted">
                <tr>
                  <th className="pb-1.5 font-medium">Dataset</th>
                  <th className="pb-1.5 font-medium">Status</th>
                  <th className="pb-1.5 font-medium">Final loss</th>
                  <th className="pb-1.5 font-medium">Duration</th>
                  <th className="pb-1.5 font-medium">Started</th>
                </tr>
              </thead>
              <tbody>
                {runs.data.map((run) => (
                  <tr key={run.id} className="border-t border-line">
                    <td className="py-1.5">
                      <Link href={`/runs/${run.id}`} className="font-medium hover:text-accent-ink">
                        {String(run.params.version)} · r{String(run.params.lora_r)} · {String(run.params.epochs)}ep
                      </Link>
                    </td>
                    <td><StatusBadge status={run.status} /></td>
                    <td className="tabular-nums">{fixed(run.result?.train_loss as number | undefined)}</td>
                    <td className="tabular-nums text-ink-2">{duration(run.started_at, run.finished_at)}</td>
                    <td className="text-ink-2">{ago(run.created_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <Empty>No training runs yet.</Empty>
          )}
        </Card>
      </div>
    </>
  );
}
