"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { ChevronLeft, ChevronRight, Search } from "lucide-react";
import { Button, Card, Empty, ErrorNote, PageHeader, inputClass } from "@/components/ui";
import { LoadingState } from "@/components/loading-state";
import { post } from "@/lib/api";
import { ago } from "@/lib/format";
import { invalidate, usePoll } from "@/lib/hooks";
import type { GeneratedArticlePage } from "@/lib/types";

export default function ContentPage() {
  const [query, setQuery] = useState("");
  const [search, setSearch] = useState("");
  const [status, setStatus] = useState("all");
  const [page, setPage] = useState(0);
  const pageSize = 12;
  useEffect(() => {
    const timer = window.setTimeout(() => setSearch(query.trim()), 250);
    return () => window.clearTimeout(timer);
  }, [query]);
  const path = `/studio/generated?limit=${pageSize}&offset=${page * pageSize}${search ? `&q=${encodeURIComponent(search)}` : ""}${status !== "all" ? `&status=${status}` : ""}`;
  const results = usePoll<GeneratedArticlePage>(path, 20_000);
  const [busy, setBusy] = useState<string | null>(null);
  const [copied, setCopied] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const pages = results.data?.items ?? [];
  const total = results.data?.total ?? 0;

  async function unpublish(slug: string) {
    setBusy(slug);
    try {
      await post(`/studio/content/${slug}/unpublish`);
      invalidate("/studio/generated");
      await results.refresh();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setBusy(null);
    }
  }

  async function copy(slug: string) {
    try {
      await navigator.clipboard.writeText(`${location.origin}/p/${slug}`);
      setCopied(slug);
      setTimeout(() => setCopied(null), 1500);
    } catch {
      setError("Could not copy the link. Open the page and copy its address instead.");
    }
  }

  return (
    <>
      <PageHeader title="Articles" subtitle="Drafts and published pages from every conversation." />
      <Card title="Article library" action={<span className="text-xs text-ink-2">{total} {total === 1 ? "article" : "articles"}</span>}>
        <div className="mb-3 grid gap-3 border-b border-line pb-5 sm:grid-cols-[minmax(0,1fr)_11rem]">
          <label className="relative block w-full">
            <Search size={14} className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-muted" aria-hidden />
            <input className={`${inputClass} pl-8`} value={query} onChange={(e) => { setQuery(e.target.value); setPage(0); }} placeholder="Search generated articles" aria-label="Search generated articles" />
          </label>
          <select className={inputClass} aria-label="Filter articles by status" value={status} onChange={(e) => { setStatus(e.target.value); setPage(0); }}>
            <option value="all">All articles</option>
            <option value="draft">Drafts</option>
            <option value="published">Published</option>
          </select>
        </div>
        <ErrorNote message={error ?? results.error} />
        {pages.length ? (
          <ul className="divide-y divide-line">
            {pages.map((p) => (
              <li key={p.id} className="flex flex-wrap items-center justify-between gap-3 py-4">
                <div className="min-w-0">
                  <Link href={`/studio/${p.thread_id}?artifact=${p.id}`} className="font-semibold text-ink hover:text-accent-ink">{p.title}</Link>
                  <div className="text-xs text-ink-2">
                    v{p.current_version} · {ago(p.updated_at)} ·{" "}
                    <span className={p.status === "published" ? "text-good" : "text-ink-2"}>{p.status === "published" ? "Published" : "Draft"}</span>
                  </div>
                </div>
                <div className="flex gap-2">
                  {p.status === "published" && p.slug && (
                    <>
                      <a className="inline-flex h-8 items-center rounded-full border border-line bg-surface-1 px-3.5 text-xs font-medium text-ink transition-colors hover:border-accent/35 hover:bg-accent-wash" href={`/p/${p.slug}`} target="_blank" rel="noreferrer">Open</a>
                      <Button onClick={() => copy(p.slug!)}>{copied === p.slug ? "Copied" : "Copy link"}</Button>
                      <Button variant="danger" onClick={() => unpublish(p.slug!)} busy={busy === p.slug}>Unpublish</Button>
                    </>
                  )}
                  <Link className="inline-flex h-8 items-center rounded-full border border-line bg-surface-1 px-3.5 text-xs font-medium text-ink transition-colors hover:border-accent/35 hover:bg-accent-wash" href={`/studio/${p.thread_id}?artifact=${p.id}`}>{p.status === "draft" ? "Review draft" : "Edit in Studio"}</Link>
                </div>
              </li>
            ))}
          </ul>
        ) : (
          results.loading ? <LoadingState label="Loading generated articles" /> : <Empty>{search || status !== "all" ? "No articles match those filters." : "No articles yet. Start a conversation in the Studio."}</Empty>
        )}
        {total > pageSize && (
          <div className="mt-4 flex items-center justify-between border-t border-line pt-3">
            <span className="text-xs text-muted">Page {page + 1} of {Math.ceil(total / pageSize)}</span>
            <div className="flex gap-2">
              <Button variant="secondary" className="h-8 px-3" disabled={page === 0} onClick={() => setPage((p) => Math.max(0, p - 1))}><ChevronLeft size={14} /> Previous</Button>
              <Button variant="secondary" className="h-8 px-3" disabled={(page + 1) * pageSize >= total} onClick={() => setPage((p) => p + 1)}>Next <ChevronRight size={14} /></Button>
            </div>
          </div>
        )}
      </Card>
    </>
  );
}
