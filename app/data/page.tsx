"use client";

import { useState } from "react";
import { JobButton } from "@/components/job-button";
import { Button, Card, Empty, ErrorNote, Field, PageHeader, StatTile, inputClass } from "@/components/ui";
import { ApiError, post } from "@/lib/api";
import { ago, compact } from "@/lib/format";
import { usePoll } from "@/lib/hooks";
import type { CorpusStatus, DatasetVersion } from "@/lib/types";

type Stats = {
  usable: number;
  words_median: number;
  words_min: number;
  words_max: number;
  by_category: Record<string, number>;
  errors_by_reason: Record<string, number>;
};
type ArticlePage = { total: number; items: { url: string; title: string | null; word_count: number; error: string | null; category_slug: string | null; labelled: boolean }[] };
const STATES = ["cleaned", "rejected", "pending", "labelled", "duplicates", "all"] as const;

export default function DataPage() {
  const corpus = usePoll<CorpusStatus>("/corpus", 10_000);
  const stats = usePoll<Stats>("/corpus/stats", 30_000);
  const datasets = usePoll<DatasetVersion[]>("/datasets", 30_000);
  const [state, setState] = useState<(typeof STATES)[number]>("cleaned");
  const articles = usePoll<ArticlePage>(`/corpus/articles?state=${state}&limit=50`, 0);
  const [version, setVersion] = useState("v1");
  const [publishing, setPublishing] = useState(false);
  const [datasetMsg, setDatasetMsg] = useState<string | null>(null);
  const [migrating, setMigrating] = useState(false);
  const [migrateMsg, setMigrateMsg] = useState<string | null>(null);
  const counts = corpus.data?.counts;

  async function publishDataset() {
    setPublishing(true);
    setDatasetMsg(null);
    try {
      const r = await post<{ status: string; train: number; eval: number }>("/datasets", { version });
      setDatasetMsg(`${version}: ${r.status} (${r.train} train / ${r.eval} eval)`);
      await datasets.refresh();
    } catch (e) {
      setDatasetMsg(e instanceof ApiError ? e.message : String(e));
    } finally {
      setPublishing(false);
    }
  }

  async function migrate() {
    setMigrating(true);
    try {
      const r = await post<{ status: string; applied: string[] }>("/migrations/apply");
      setMigrateMsg(r.applied.length ? `Applied ${r.applied.join(", ")}` : "Database is up to date");
    } catch (e) {
      setMigrateMsg(e instanceof Error ? e.message : String(e));
    } finally {
      setMigrating(false);
    }
  }

  return (
    <>
      <PageHeader
        title="Data"
        subtitle="Pull Jenosize Ideas into Postgres + R2, review it, label it, publish a dataset. Re-runs only touch what changed."
        action={
          <div className="flex flex-wrap gap-2">
            <JobButton path="/scrape" label="Pull new articles" />
            <JobButton path="/scrape" body={{ recheck_days: 30 }} label="Re-check for edits" variant="secondary" />
            <JobButton path="/label" label="Label new articles" variant="secondary" />
          </div>
        }
      />
      <ErrorNote message={corpus.error} />

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-6">
        {["discovered", "fetched", "cleaned", "labelled", "duplicates", "errors"].map((key) => (
          <StatTile key={key} label={key[0].toUpperCase() + key.slice(1)} value={compact(counts?.[key])} />
        ))}
      </div>

      <div className="mt-6 grid gap-6 xl:grid-cols-3">
        <Card title="Corpus health">
          {stats.data ? (
            <dl className="space-y-3 text-sm">
              <div className="flex justify-between"><dt className="text-ink-2">Usable articles</dt><dd className="tabular-nums">{stats.data.usable}</dd></div>
              <div className="flex justify-between"><dt className="text-ink-2">Words (min · median · max)</dt><dd className="tabular-nums">{stats.data.words_min} · {stats.data.words_median} · {stats.data.words_max}</dd></div>
              <div>
                <dt className="mb-1 text-ink-2">By category</dt>
                {Object.entries(stats.data.by_category).map(([k, v]) => (
                  <dd key={k} className="flex justify-between text-xs"><span className="font-mono">{k}</span><span className="tabular-nums">{v}</span></dd>
                ))}
              </div>
              {Object.keys(stats.data.errors_by_reason).length > 0 && (
                <div>
                  <dt className="mb-1 text-ink-2">Rejected, by reason</dt>
                  {Object.entries(stats.data.errors_by_reason).map(([k, v]) => (
                    <dd key={k} className="flex justify-between gap-3 text-xs"><span>{k}</span><span className="tabular-nums">{v}</span></dd>
                  ))}
                </div>
              )}
            </dl>
          ) : (
            <Empty>{stats.loading ? "Loading…" : "No data yet — pull articles first."}</Empty>
          )}
        </Card>

        <Card title="Articles" className="xl:col-span-2" action={
          <select className={`${inputClass} w-auto py-1`} value={state} onChange={(e) => setState(e.target.value as typeof state)} aria-label="Filter by state">
            {STATES.map((s) => <option key={s}>{s}</option>)}
          </select>
        }>
          {articles.data?.items.length ? (
            <>
              <p className="mb-2 text-xs text-muted">{articles.data.total} {state}</p>
              <ul className="max-h-96 divide-y divide-line overflow-auto">
                {articles.data.items.map((a) => (
                  <li key={a.url} className="py-2 text-sm">
                    <a href={a.url} target="_blank" rel="noreferrer" className="font-medium text-ink hover:text-accent-ink">{a.title ?? a.url.split("/").pop()}</a>
                    <div className="text-xs text-ink-2">
                      <span className="font-mono">{a.category_slug}</span> · {a.word_count} words{a.labelled ? " · labelled" : ""}
                      {a.error && <span className="text-critical"> · {a.error}</span>}
                    </div>
                  </li>
                ))}
              </ul>
            </>
          ) : (
            <Empty>{articles.loading ? "Loading…" : `No ${state} articles.`}</Empty>
          )}
        </Card>
      </div>

      <div className="mt-6 grid gap-6 lg:grid-cols-2">
        <Card title="Datasets" action={<span className="text-xs text-muted">versions are immutable</span>}>
          <div className="mb-4 flex items-end gap-2">
            <Field label="Version"><input className={`${inputClass} w-28`} value={version} onChange={(e) => setVersion(e.target.value)} pattern="v[0-9]+" /></Field>
            <Button variant="primary" onClick={publishDataset} busy={publishing}>Publish dataset</Button>
          </div>
          {datasetMsg && <p className="mb-3 text-sm text-ink-2">{datasetMsg}</p>}
          {datasets.data?.length ? (
            <ul className="divide-y divide-line text-sm">
              {datasets.data.map((d) => (
                <li key={d.version} className="flex justify-between py-2">
                  <span className="font-semibold">{d.version}</span>
                  <span className="tabular-nums text-ink-2">{d.train_count} train · {d.eval_count} eval · {ago(d.created_at)}</span>
                </li>
              ))}
            </ul>
          ) : (
            <Empty>No datasets published yet.</Empty>
          )}
        </Card>
        <Card title="Database">
          <p className="mb-3 text-sm text-ink-2">Apply pending schema migrations. Safe to repeat.</p>
          <Button onClick={migrate} busy={migrating}>Apply migrations</Button>
          {migrateMsg && <p className="mt-3 text-sm text-ink-2">{migrateMsg}</p>}
        </Card>
      </div>
    </>
  );
}
