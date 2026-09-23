"use client";

import Image from "next/image";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useRef, useState, type KeyboardEvent } from "react";
import { ArrowUp, Building2, FileText, HeartPulse, LoaderCircle, Plus, Recycle, ShoppingBag } from "lucide-react";
import { LoadingState } from "@/components/loading-state";
import { ErrorNote } from "@/components/ui";
import { ago } from "@/lib/format";
import { invalidate, usePoll } from "@/lib/hooks";
import { setPendingBrief } from "@/lib/pending-brief";
import { createThread } from "@/lib/thread";
import type { GeneratedArticlePage } from "@/lib/types";

const IDEAS = [
  { icon: ShoppingBag, text: "Write a medium article on agentic AI in Southeast Asian retail for C-suite readers." },
  { icon: Building2, text: "Draft a short piece on tokenised real-world assets for Thai banking leaders." },
  { icon: Recycle, text: "Write about circular supply chains after the tariff shock, then design it as a page." },
  { icon: HeartPulse, text: "Research recent data on AI in Thai healthcare, then write a short article for hospital executives." },
];

export default function Home() {
  const router = useRouter();
  const [text, setText] = useState("");
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const submitting = useRef(false);
  const textarea = useRef<HTMLTextAreaElement>(null);
  const recent = usePoll<GeneratedArticlePage>("/studio/generated?limit=3", 60_000);
  const articles = recent.data?.items ?? [];

  async function start() {
    const brief = text.trim();
    // Ref, not state: Enter and a click in the same tick must not create two threads.
    if (!brief || submitting.current) return;
    submitting.current = true;
    setCreating(true);
    setError(null);
    try {
      const thread = await createThread();
      setPendingBrief(thread.id, brief);
      invalidate("/studio/threads");
      router.push(`/studio/${thread.id}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      submitting.current = false;
      setCreating(false);
    }
  }

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      void start();
    }
  }

  function pick(idea: string) {
    setText(idea);
    textarea.current?.focus();
  }

  return (
    <div className="flex min-h-[calc(100vh-3rem)] flex-col items-center justify-center py-10">
      <div className="t-stagger is-shown w-full max-w-2xl">
        {/* .t-stagger-line forces display:block, so the flex row lives on an inner element. */}
        <h1 className="t-stagger-line t-stagger-line--1 text-[26px] font-semibold tracking-[-0.03em] text-ink md:text-[30px]">
          <span className="flex items-center justify-center gap-3">
            <Image src="/jenosize-icon.png" alt="" width={180} height={180} priority className="size-8 md:size-9" />
            What should we write today?
          </span>
        </h1>
        <p className="t-stagger-line t-stagger-line--2 mt-1.5 text-center text-[13px] text-ink-2">
          Brief the agent — it researches, writes with the fine-tuned model, and lays out a branded page.
        </p>

        <div className="t-stagger-line t-stagger-line--2 mt-6">
          <ErrorNote message={error} />
          <div className="mt-2 rounded-2xl border border-line bg-surface-1 shadow-[0_10px_28px_rgba(7,19,38,0.05)] transition-colors focus-within:border-accent/45 focus-within:shadow-[0_0_0_3px_rgba(36,87,214,0.08)]">
            <textarea
              ref={textarea}
              value={text}
              onChange={(e) => setText(e.target.value)}
              onKeyDown={onKeyDown}
              rows={2}
              placeholder="Describe an article or page to create…"
              aria-label="Article brief"
              className="block max-h-40 min-h-[64px] w-full resize-none bg-transparent px-4 pb-1 pt-3.5 text-[14px] leading-6 text-ink outline-none placeholder:text-muted"
              disabled={creating}
            />
            <div className="flex items-center justify-between px-2 pb-2 pt-1">
              <Link href="/studio" title="Open Studio" className="flex size-8 items-center justify-center rounded-full text-ink-2 transition-colors hover:bg-surface-2 hover:text-accent-ink">
                <Plus size={18} aria-hidden />
                <span className="sr-only">Open Studio</span>
              </Link>
              <button
                type="button"
                onClick={() => void start()}
                disabled={!text.trim() || creating}
                title="Start"
                className="flex size-8 items-center justify-center rounded-full bg-accent text-white transition-opacity hover:opacity-90 disabled:opacity-30"
              >
                {creating ? <LoaderCircle size={15} className="animate-spin" /> : <ArrowUp size={16} strokeWidth={2.4} />}
                <span className="sr-only">Start</span>
              </button>
            </div>
          </div>
        </div>

        <div className="t-stagger-line t-stagger-line--2 mt-7">
          <p className="mb-1.5 px-2 text-[11px] font-medium text-muted">Ideas for you</p>
          <ul>
            {IDEAS.map(({ icon: Icon, text: idea }) => (
              <li key={idea}>
                <button
                  type="button"
                  onClick={() => pick(idea)}
                  className="group flex w-full items-center gap-3 rounded-lg px-2 py-2 text-left transition-colors hover:bg-surface-2"
                >
                  <span className="flex size-7 shrink-0 items-center justify-center rounded-md border border-line bg-surface-1 text-ink-2 group-hover:text-accent-ink">
                    <Icon size={14} aria-hidden />
                  </span>
                  <span className="min-w-0 truncate text-[13px] text-ink-2 group-hover:text-ink">{idea}</span>
                </button>
              </li>
            ))}
          </ul>
        </div>

        {(recent.loading || articles.length > 0) && (
          <div className="mt-6">
            <div className="mb-1.5 flex items-center justify-between px-2">
              <p className="text-[11px] font-medium text-muted">Recent articles</p>
              <Link href="/content" className="text-[11px] font-medium text-accent-ink hover:underline">View all</Link>
            </div>
            {articles.length ? (
              <ul>
                {articles.map((a) => (
                  <li key={a.id}>
                    <Link href={`/studio/${a.thread_id}?artifact=${a.id}`} className="group flex items-center gap-3 rounded-lg px-2 py-2 transition-colors hover:bg-surface-2">
                      <FileText size={14} className="shrink-0 text-muted group-hover:text-accent-ink" aria-hidden />
                      <span className="min-w-0 flex-1 truncate text-[13px] text-ink">{a.title}</span>
                      <span className="shrink-0 text-[11px] text-muted">{a.status === "published" ? "Published" : "Draft"} · {ago(a.updated_at)}</span>
                    </Link>
                  </li>
                ))}
              </ul>
            ) : (
              <div className="px-2"><LoadingState label="Loading recent articles" rows={2} /></div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
