"use client";

import Link from "next/link";
import { useState } from "react";
import { Button, Card, Empty, PageHeader } from "@/components/ui";
import { post } from "@/lib/api";
import { ago } from "@/lib/format";
import { usePoll } from "@/lib/hooks";
import type { PublishedPage } from "@/lib/types";

export default function ContentPage() {
  const pages = usePoll<PublishedPage[]>("/studio/content", 20_000);
  const [busy, setBusy] = useState<string | null>(null);
  const [copied, setCopied] = useState<string | null>(null);

  async function unpublish(slug: string) {
    setBusy(slug);
    try {
      await post(`/studio/content/${slug}/unpublish`);
      await pages.refresh();
    } finally {
      setBusy(null);
    }
  }

  async function copy(url: string) {
    try {
      await navigator.clipboard.writeText(url);
      setCopied(url);
      setTimeout(() => setCopied(null), 1500);
    } catch {
      /* clipboard blocked */
    }
  }

  return (
    <>
      <PageHeader title="Published" subtitle="Pages published from the Studio. Share links are public; unpublishing takes them (and their images) down." />
      <Card>
        {pages.data?.length ? (
          <ul className="divide-y divide-line">
            {pages.data.map((p) => (
              <li key={p.slug} className="flex flex-wrap items-center justify-between gap-3 py-3">
                <div className="min-w-0">
                  <div className="truncate font-medium text-ink">{p.title}</div>
                  <div className="text-xs text-ink-2">
                    /p/{p.slug} · v{p.version} · {ago(p.updated_at)} ·{" "}
                    <span className={p.status === "published" ? "text-good" : "text-muted"}>{p.status === "published" ? "✓ published" : "– unpublished"}</span>
                  </div>
                </div>
                <div className="flex gap-2">
                  {p.status === "published" && (
                    <>
                      <a className="rounded-lg border border-line px-3 py-2 text-sm hover:bg-surface-2" href={`/p/${p.slug}`} target="_blank" rel="noreferrer">Open</a>
                      <Button onClick={() => copy(`${location.origin}/p/${p.slug}`)}>{copied === `${location.origin}/p/${p.slug}` ? "Copied" : "Copy link"}</Button>
                      <Button variant="danger" onClick={() => unpublish(p.slug)} busy={busy === p.slug}>Unpublish</Button>
                    </>
                  )}
                  <Link className="rounded-lg border border-line px-3 py-2 text-sm hover:bg-surface-2" href={`/studio?artifact=${p.artifact_id}`}>Edit in Studio</Link>
                </div>
              </li>
            ))}
          </ul>
        ) : (
          <Empty>{pages.loading ? "Loading…" : "Nothing published yet. Write an article in the Studio and press Publish."}</Empty>
        )}
      </Card>
    </>
  );
}
