"use client";

import Link from "next/link";
import { useState } from "react";
import { JobButton } from "@/components/job-button";
import { StatusBadge } from "@/components/status";
import { Button, Card, Empty, Field, PageHeader, inputClass } from "@/components/ui";
import { post } from "@/lib/api";
import { ago, fixed } from "@/lib/format";
import { useLocalSetting, usePoll } from "@/lib/hooks";
import type { Adapter, JobRun } from "@/lib/types";

const ENDPOINT_KEY = "jenosize.vllmEndpoint";

export default function ModelsPage() {
  const adapters = usePoll<Adapter[]>("/adapters", 15_000);
  const evals = usePoll<JobRun[]>("/runs?kind=eval&limit=10", 10_000);
  const [endpoint, saveEndpoint] = useLocalSetting(ENDPOINT_KEY);
  const [judge, setJudge] = useState(false);
  const [repo, setRepo] = useState("");
  const [activating, setActivating] = useState<string | null>(null);

  async function activate(version: string) {
    setActivating(version);
    try {
      await post(`/adapters/${version}/activate`);
      await adapters.refresh();
    } finally {
      setActivating(null);
    }
  }

  return (
    <>
      <PageHeader title="Models" subtitle="Trained LoRA adapters and which one is live." />
      <div className="grid gap-4 xl:grid-cols-3">
        <Card title="Adapters" className="xl:col-span-2">
          {adapters.data?.length ? (
            <table className="w-full text-left text-xs">
              <thead className="text-xs text-muted">
                <tr>
                  <th className="pb-1.5 font-medium">Version</th>
                  <th className="pb-1.5 font-medium">Served as</th>
                  <th className="pb-1.5 font-medium">Train loss</th>
                  <th className="pb-1.5 font-medium">Steps</th>
                  <th className="pb-1.5" />
                </tr>
              </thead>
              <tbody>
                {adapters.data.map((a) => (
                  <tr key={a.version} className="border-t border-line">
                    <td className="py-1.5 font-semibold">
                      {a.version}
                      {a.active && <span className="ml-2 rounded bg-accent-wash px-1.5 py-0.5 text-xs font-medium text-accent-ink">active</span>}
                    </td>
                    <td className="font-mono text-xs">{a.served_as}</td>
                    <td className="tabular-nums">{fixed(a.metrics.train_loss)}</td>
                    <td className="tabular-nums">{a.metrics.steps ?? "—"}</td>
                    <td className="py-1.5 text-right">
                      <div className="flex justify-end gap-2">
                        {!a.active && <Button onClick={() => activate(a.version)} busy={activating === a.version}>Make active</Button>}
                        <JobButton
                          path="/eval"
                          label="Evaluate"
                          variant="secondary"
                          disabled={!endpoint}
                          body={{ version: a.version, endpoint, judge }}
                        />
                        <JobButton
                          path={`/adapters/${a.version}/publish`}
                          label="Publish to HF"
                          variant="secondary"
                          disabled={!repo.includes("/")}
                          body={{ version: a.version, repo_id: repo }}
                        />
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : adapters.error ? (
            <Empty>Adapter registry is unavailable: {adapters.error}</Empty>
          ) : adapters.loading ? (
            <Empty>Loading adapter registry…</Empty>
          ) : (
            <Empty>No adapters are visible in this backend&apos;s model volume. A local API cannot read the deployed Modal volume; use the deployed Jobs API to inspect or activate the served LoRA.</Empty>
          )}
          <p className="mt-3 text-xs text-muted">Activation takes effect the next time the vLLM server starts (it scales to zero after 5 idle minutes).</p>
        </Card>

        <Card title="Evaluation & publishing settings">
          <div className="space-y-3">
            <Field label="vLLM endpoint" hint="The server URL printed by `make deploy-modal`, ending in /v1. Saved in this browser.">
              <input className={inputClass} value={endpoint} onChange={(e) => saveEndpoint(e.target.value)} placeholder="https://…-vllmserver.modal.run/v1" />
            </Field>
            <label className="flex items-center gap-2 text-[13px] text-ink-2">
              <input type="checkbox" checked={judge} onChange={(e) => setJudge(e.target.checked)} />
              Blind judge (the labelling LLM compares base vs fine-tuned)
            </label>
            <Field label="Hugging Face repo" hint="e.g. your-user/jeno-trend-writer-lora">
              <input className={inputClass} value={repo} onChange={(e) => setRepo(e.target.value)} />
            </Field>
          </div>
        </Card>
      </div>

      <Card title="Evaluations" className="mt-4">
        {evals.data?.length ? (
          <ul className="divide-y divide-line">
            {evals.data.map((run) => {
              const summary = (run.result?.summary ?? {}) as { deterministic?: Record<string, Record<string, number>>; judge?: Record<string, number> | null };
              return (
                <li key={run.id} className="flex flex-wrap items-center justify-between gap-3 py-2 text-[13px]">
                  <Link href={`/runs/${run.id}`} className="font-medium hover:text-accent-ink">{String(run.params.version)} · {ago(run.created_at)}</Link>
                  <StatusBadge status={run.status} />
                  {summary.deterministic && (
                    <span className="text-xs tabular-nums text-ink-2">
                      pass rate {fixed(summary.deterministic.base?.pass_rate, 2)} → {fixed(summary.deterministic.finetuned?.pass_rate, 2)}
                      {summary.judge && ` · judge: fine-tuned ${summary.judge.finetuned_wins} / base ${summary.judge.base_wins} / ties ${summary.judge.ties}`}
                    </span>
                  )}
                </li>
              );
            })}
          </ul>
        ) : (
          <Empty>No evaluations yet.</Empty>
        )}
      </Card>
    </>
  );
}
