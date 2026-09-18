"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";
import { Button, Card, Empty, ErrorNote, PageHeader } from "@/components/ui";
import { get, post } from "@/lib/api";
import { ago } from "@/lib/format";
import { usePoll } from "@/lib/hooks";
import type { Thread } from "@/lib/types";

function StudioIndex() {
  const router = useRouter();
  const search = useSearchParams();
  const threads = usePoll<Thread[]>("/studio/threads", 15_000);
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // /studio?artifact=<id> (from the Published page) opens that artifact's chat.
  useEffect(() => {
    const artifact = search.get("artifact");
    if (!artifact) return;
    get<{ thread_id: string }>(`/studio/artifacts/${artifact}`)
      .then((a) => router.replace(`/studio/${a.thread_id}?artifact=${artifact}`))
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, [search, router]);

  async function newChat() {
    setCreating(true);
    try {
      const thread = await post<Thread>("/studio/threads");
      router.push(`/studio/${thread.id}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setCreating(false);
    }
  }

  return (
    <>
      <PageHeader
        title="Studio"
        subtitle="Chat with the content agent. It writes with the fine-tuned Jenosize model, designs branded pages, and you publish them."
        action={<Button variant="primary" onClick={newChat} busy={creating}>New chat</Button>}
      />
      <ErrorNote message={error ?? threads.error} />
      <Card>
        {threads.data?.length ? (
          <ul className="divide-y divide-line">
            {threads.data.map((t) => (
              <li key={t.id}>
                <Link href={`/studio/${t.id}`} className="flex items-center justify-between gap-3 py-3 hover:text-accent-ink">
                  <span className="truncate font-medium">{t.title}</span>
                  <span className="shrink-0 text-xs text-ink-2">
                    {t.artifact_count ? `${t.artifact_count} artifact${t.artifact_count > 1 ? "s" : ""} · ` : ""}
                    {ago(t.updated_at)}
                  </span>
                </Link>
              </li>
            ))}
          </ul>
        ) : (
          <Empty>{threads.loading ? "Loading…" : "No chats yet — start one."}</Empty>
        )}
      </Card>
    </>
  );
}

export default function StudioPage() {
  // useSearchParams needs a Suspense boundary for static rendering.
  return (
    <Suspense>
      <StudioIndex />
    </Suspense>
  );
}
