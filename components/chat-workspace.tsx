"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Button, ErrorNote } from "@/components/ui";
import { API, ApiError, get, getText, post, upload } from "@/lib/api";
import { streamSse } from "@/lib/sse";
import type { AgentRun, Artifact, Asset, ChatMessage, PublishedPage, ThreadDetail, ToolCallRecord } from "@/lib/types";

const ASSET_BASE = `${API}/studio/assets`;
const TOOL_LABEL: Record<string, string> = {
  write_article: "Writing with the fine-tuned model",
  design_page: "Designing the branded page",
  list_images: "Checking your images",
  get_artifact: "Reading the artifact",
};

interface LiveTool {
  id: string;
  name: string;
  state: "running" | "done" | "error";
  note?: string;
}

interface LiveTurn {
  runId: string | null;
  user: string;
  assetIds: string[];
  reply: string;
  tools: LiveTool[];
}

export function ChatWorkspace({ threadId }: { threadId: string }) {
  const search = useSearchParams();
  const [thread, setThread] = useState<ThreadDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [live, setLive] = useState<LiveTurn | null>(null);
  const [input, setInput] = useState("");
  const [pending, setPending] = useState<Asset[]>([]);
  const [uploading, setUploading] = useState(0);
  const [selected, setSelected] = useState<{ artifactId: string; version: number } | null>(null);
  const [draft, setDraft] = useState<{ artifactId: string; text: string } | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  const load = useCallback(async () => {
    try {
      const detail = await get<ThreadDetail>(`/studio/threads/${threadId}`);
      setThread(detail);
      return detail;
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      return null;
    }
  }, [threadId]);

  const handleEvent = useCallback((event: string, data: Record<string, unknown>) => {
    switch (event) {
      case "token":
        setLive((t) => (t ? { ...t, reply: t.reply + String(data.text ?? "") } : t));
        break;
      case "tool_start":
        setLive((t) => (t ? { ...t, tools: [...t.tools, { id: String(data.id), name: String(data.name), state: "running" }] } : t));
        break;
      case "tool_progress":
        setLive((t) =>
          t ? { ...t, tools: t.tools.map((x) => (x.name === data.tool && x.state === "running" ? { ...x, note: String(data.message ?? "") } : x)) } : t,
        );
        break;
      case "tool_end": {
        const result = data.result as Record<string, unknown> | string;
        const failed = typeof result === "object" && result && "error" in result;
        setLive((t) =>
          t
            ? {
                ...t,
                tools: t.tools.map((x) =>
                  x.id === String(data.id) ? { ...x, state: failed ? "error" : "done", note: failed ? String((result as Record<string, unknown>).error) : x.note } : x,
                ),
              }
            : t,
        );
        break;
      }
      case "artifact_start":
        setDraft({ artifactId: String(data.artifact_id), text: "" });
        break;
      case "artifact_delta":
        setDraft((d) => (d && d.artifactId === data.artifact_id ? { ...d, text: d.text + String(data.text ?? "") } : d));
        break;
      case "artifact":
        setDraft(null);
        setSelected({ artifactId: String(data.artifact_id), version: Number(data.version) });
        void load();
        break;
      case "error":
        setError(String(data.message ?? "The agent failed"));
        break;
    }
  }, [load]);

  /**
   * Follow a run's event log. The worker writes events to Postgres; this
   * replays them from `seq` 0 (or resumes after a dropped connection from the
   * last seq seen), so nothing is lost or shown twice.
   */
  const followRun = useCallback(async (runId: string) => {
    setLive((t) => (t ? { ...t, runId } : t));
    const controller = new AbortController();
    abortRef.current = controller;
    let seq = 0;
    let finished = false;
    let attempt = 0;
    while (!finished && !controller.signal.aborted) {
      try {
        await streamSse(
          `${API}/studio/runs/${runId}/events?after_id=${seq}`,
          { method: "GET" },
          (event, data) => {
            const payload = data as Record<string, unknown>;
            if (typeof payload.seq === "number") seq = Math.max(seq, payload.seq);
            attempt = 0;
            if (event === "done") finished = true;
            else handleEvent(event, payload);
          },
          controller.signal,
        );
      } catch (e) {
        if (controller.signal.aborted) return;
        if (++attempt > 6) {
          setError(e instanceof Error ? e.message : String(e));
          break;
        }
      }
      if (!finished && !controller.signal.aborted) await new Promise((r) => setTimeout(r, Math.min(8000, 500 * 2 ** attempt)));
    }
    if (controller.signal.aborted) return;
    setDraft(null);
    await load();
    setLive(null);
  }, [handleEvent, load]);

  useEffect(() => {
    let cancelled = false;
    get<ThreadDetail>(`/studio/threads/${threadId}`)
      .then((detail) => {
        if (cancelled) return;
        setThread(detail);
        const run = detail.active_run;
        if (run) {
          // Resume watching a turn that is still running (reload, second tab).
          const user = detail.messages.find((m) => m.id === run.message_id);
          setLive({ runId: run.id, user: user?.content ?? "", assetIds: user?.asset_ids ?? [], reply: "", tools: [] });
          void followRun(run.id);
        }
        if (!detail.artifacts.length) return;
        const wanted = search.get("artifact");
        const artifact = detail.artifacts.find((a) => a.id === wanted) ?? detail.artifacts[detail.artifacts.length - 1];
        setSelected({ artifactId: artifact.id, version: artifact.current_version });
      })
      .catch((e) => !cancelled && setError(e instanceof Error ? e.message : String(e)));
    return () => {
      cancelled = true;
    };
  }, [threadId, search, followRun]);

  // Leaving the chat only stops *watching*: the turn runs on its own Modal
  // worker and is picked up again (from its event log) on the next visit.
  useEffect(() => () => abortRef.current?.abort(), []);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [thread?.messages.length, live?.reply, live?.tools.length]);

  async function attach(files: FileList | File[]) {
    const images = Array.from(files).filter((f) => f.type.startsWith("image/"));
    for (const file of images) {
      setUploading((n) => n + 1);
      try {
        const form = new FormData();
        form.append("file", file);
        form.append("thread_id", threadId);
        const asset = await upload<Asset>("/studio/assets", form);
        setPending((p) => [...p, asset]);
      } catch (e) {
        setError(`Upload failed for ${file.name}: ${e instanceof Error ? e.message : e}`);
      } finally {
        setUploading((n) => n - 1);
      }
    }
  }

  async function send() {
    const text = input.trim();
    if (!text || live) return;
    const assetIds = pending.map((a) => a.id);
    setInput("");
    setPending([]);
    setError(null);
    setLive({ runId: null, user: text, assetIds, reply: "", tools: [] });
    try {
      const started = await post<{ run: AgentRun }>(`/studio/threads/${threadId}/runs`, { content: text, asset_ids: assetIds });
      await followRun(started.run.id);
    } catch (e) {
      // 409: a turn is already running here (another tab?) — watch that one instead.
      const runId = e instanceof ApiError && e.status === 409 ? (e.body as { error?: { run_id?: string } })?.error?.run_id : null;
      if (runId) {
        await followRun(runId);
      } else {
        setError(e instanceof Error ? e.message : String(e));
        setLive(null);
      }
    }
  }

  async function stop() {
    const runId = live?.runId;
    if (!runId) return;
    try {
      await post(`/studio/runs/${runId}/cancel`);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  if (!thread) {
    return error ? <ErrorNote message={error} /> : <p className="text-sm text-muted">Loading chat…</p>;
  }

  return (
    <div className="grid h-[calc(100vh-4rem)] gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.15fr)]">
      {/* ---------------------------------------------------------------- chat */}
      <section className="flex min-h-0 flex-col rounded-xl border border-line bg-surface-1">
        <header className="flex items-center justify-between gap-3 border-b border-line px-4 py-3">
          <div className="min-w-0">
            <Link href="/studio" className="text-xs text-muted hover:text-accent-ink">← All chats</Link>
            <h1 className="truncate text-sm font-semibold">{thread.title}</h1>
          </div>
        </header>

        <div ref={scrollRef} className="min-h-0 flex-1 space-y-4 overflow-y-auto px-4 py-4">
          {thread.messages.length === 0 && !live && <Starter onPick={setInput} />}
          {thread.messages.map((m) => (
            <MessageBubble key={m.id} message={m} />
          ))}
          {live && (
            <>
              <MessageBubble message={{ id: "live-user", role: "user", content: live.user, created_at: "", tool_calls: [], asset_ids: live.assetIds, model: null }} />
              <div className="space-y-2">
                {live.tools.map((tool) => (
                  <ToolChip key={tool.id} name={tool.name} state={tool.state} note={tool.note} />
                ))}
                <div className="max-w-[90%] whitespace-pre-wrap rounded-2xl rounded-tl-sm bg-surface-2 px-4 py-2.5 text-sm text-ink">
                  {live.reply || <span className="text-muted">{live.tools.some((t) => t.state === "running") ? "Working…" : "Thinking…"}</span>}
                </div>
              </div>
            </>
          )}
        </div>

        <div
          className="border-t border-line p-3"
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault();
            void attach(e.dataTransfer.files);
          }}
        >
          <ErrorNote message={error} />
          {(pending.length > 0 || uploading > 0) && (
            <div className="mb-2 flex flex-wrap gap-2">
              {pending.map((a) => (
                <div key={a.id} className="relative">
                  {/* eslint-disable-next-line @next/next/no-img-element -- authenticated proxy URL */}
                  <img src={`${ASSET_BASE}/${a.id}/raw`} alt={a.filename} className="size-14 rounded-lg border border-line object-cover" />
                  <button
                    type="button"
                    aria-label={`Remove ${a.filename}`}
                    onClick={() => setPending((p) => p.filter((x) => x.id !== a.id))}
                    className="absolute -right-1.5 -top-1.5 size-5 rounded-full border border-line bg-surface-1 text-xs"
                  >
                    ×
                  </button>
                </div>
              ))}
              {uploading > 0 && <div className="flex size-14 items-center justify-center rounded-lg border border-dashed border-line text-xs text-muted">…</div>}
            </div>
          )}
          <div className="flex items-end gap-2">
            <label className="cursor-pointer rounded-lg border border-line px-3 py-2 text-sm text-ink-2 hover:bg-surface-2" title="Attach images">
              <span aria-hidden>＋</span>
              <span className="sr-only">Attach images</span>
              <input type="file" accept="image/png,image/jpeg,image/webp,image/gif" multiple className="hidden" onChange={(e) => e.target.files && attach(e.target.files)} />
            </label>
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onPaste={(e) => {
                if (e.clipboardData.files.length) void attach(e.clipboardData.files);
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
                  e.preventDefault();
                  void send();
                }
              }}
              rows={2}
              placeholder={live ? "The agent is working…" : "Ask for an article, e.g. “Write about agentic AI in Thai retail for CMOs, and design it with my photo”"}
              className="min-h-[2.75rem] flex-1 resize-none rounded-lg border border-line bg-surface-1 px-3 py-2 text-sm focus:border-accent focus:outline-none"
              disabled={!!live}
            />
            {live ? (
              <Button onClick={stop} disabled={!live.runId}>Stop</Button>
            ) : (
              <Button variant="primary" onClick={send} disabled={!input.trim() || uploading > 0}>Send</Button>
            )}
          </div>
          <p className="mt-1 text-[11px] text-muted">Enter to send · Shift+Enter for a new line · drop or paste images</p>
        </div>
      </section>

      {/* ------------------------------------------------------------- artifact */}
      <ArtifactPanel
        artifacts={thread.artifacts}
        selected={selected}
        onSelect={setSelected}
        draft={draft}
        onPublished={load}
      />
    </div>
  );
}

function Starter({ onPick }: { onPick: (text: string) => void }) {
  const ideas = [
    "Write a medium article on agentic AI in Southeast Asian retail for C-suite readers.",
    "Draft a short piece on tokenised real-world assets for Thai banking leaders.",
    "Write about circular supply chains after the tariff shock, then design it as a page.",
  ];
  return (
    <div className="py-8 text-center">
      <p className="mb-4 text-sm text-ink-2">What should we write? Attach images with ＋ to use them on the page.</p>
      <div className="flex flex-col items-center gap-2">
        {ideas.map((idea) => (
          <button key={idea} type="button" onClick={() => onPick(idea)} className="max-w-md rounded-lg border border-line px-3 py-2 text-left text-sm text-ink-2 hover:border-accent hover:text-ink">
            {idea}
          </button>
        ))}
      </div>
    </div>
  );
}

function MessageBubble({ message }: { message: ChatMessage }) {
  if (message.role === "user") {
    return (
      <div className="flex flex-col items-end gap-1.5">
        {message.asset_ids.length > 0 && (
          <div className="flex gap-1.5">
            {message.asset_ids.map((id) => (
              // eslint-disable-next-line @next/next/no-img-element -- authenticated proxy URL
              <img key={id} src={`${ASSET_BASE}/${id}/raw`} alt="" className="size-16 rounded-lg border border-line object-cover" />
            ))}
          </div>
        )}
        <div className="max-w-[85%] whitespace-pre-wrap rounded-2xl rounded-tr-sm bg-accent px-4 py-2.5 text-sm text-white">{message.content}</div>
      </div>
    );
  }
  return (
    <div className="space-y-2">
      {message.tool_calls.map((call: ToolCallRecord, i) => (
        <ToolChip key={call.id ?? i} name={call.name} state={toolFailed(call) ? "error" : "done"} />
      ))}
      <div className="max-w-[90%] whitespace-pre-wrap rounded-2xl rounded-tl-sm bg-surface-2 px-4 py-2.5 text-sm text-ink">{message.content}</div>
      {message.model && <div className="text-[11px] text-muted">{message.model}</div>}
    </div>
  );
}

function toolFailed(call: ToolCallRecord): boolean {
  try {
    return Boolean(call.result && "error" in JSON.parse(call.result));
  } catch {
    return false;
  }
}

function ToolChip({ name, state, note }: { name: string; state: LiveTool["state"]; note?: string }) {
  const icon = state === "running" ? "◌" : state === "done" ? "✓" : "✕";
  const tone = state === "running" ? "text-accent-ink animate-pulse" : state === "done" ? "text-good" : "text-critical";
  return (
    <div className="inline-flex max-w-full items-center gap-2 rounded-full border border-line px-3 py-1 text-xs text-ink-2">
      <span className={tone} aria-hidden>{icon}</span>
      <span className="font-medium text-ink">{TOOL_LABEL[name] ?? name}</span>
      {note && <span className="truncate text-muted">· {note}</span>}
    </div>
  );
}

function ArtifactPanel({
  artifacts,
  selected,
  onSelect,
  draft,
  onPublished,
}: {
  artifacts: Artifact[];
  selected: { artifactId: string; version: number } | null;
  onSelect: (s: { artifactId: string; version: number }) => void;
  draft: { artifactId: string; text: string } | null;
  onPublished: () => Promise<unknown>;
}) {
  const [tab, setTab] = useState<"preview" | "markdown">("preview");
  const [content, setContent] = useState<{ key: string; html: string; markdown: string } | null>(null);
  const [publishing, setPublishing] = useState(false);
  // Publish outcome is tagged with the version it belongs to, so switching
  // versions shows no stale "Live at…" / error without resetting in an effect.
  const [outcome, setOutcome] = useState<{ key: string; page?: PublishedPage; error?: string } | null>(null);
  const artifact = artifacts.find((a) => a.id === selected?.artifactId) ?? null;
  const key = selected ? `${selected.artifactId}:${selected.version}` : "";
  const published = outcome?.key === key ? (outcome.page ?? null) : null;
  const error = outcome?.key === key ? (outcome.error ?? null) : null;
  const html = content?.key === key ? content.html : "";
  const markdown = content?.key === key ? content.markdown : "";

  useEffect(() => {
    if (!selected) return;
    let cancelled = false;
    const tag = `${selected.artifactId}:${selected.version}`;
    const query = new URLSearchParams({ version: String(selected.version), asset_base: ASSET_BASE });
    Promise.all([
      getText(`/studio/artifacts/${selected.artifactId}/preview?${query}`).catch(() => ""),
      get<{ markdown: string | null }>(`/studio/artifacts/${selected.artifactId}/versions/${selected.version}`)
        .then((v) => v.markdown ?? "")
        .catch(() => ""),
    ]).then(([nextHtml, nextMarkdown]) => {
      if (!cancelled) setContent({ key: tag, html: nextHtml, markdown: nextMarkdown });
    });
    return () => {
      cancelled = true;
    };
  }, [selected]);

  const liveUrl = useMemo(() => {
    const page = published ?? artifact?.published.find((p) => p.status === "published") ?? null;
    return page ? `${typeof location === "undefined" ? "" : location.origin}/p/${page.slug}` : null;
  }, [published, artifact]);

  async function publish() {
    if (!selected) return;
    const tag = key;
    setPublishing(true);
    try {
      const page = await post<PublishedPage>(`/studio/artifacts/${selected.artifactId}/publish`, { version: selected.version });
      setOutcome({ key: tag, page });
      await onPublished();
    } catch (e) {
      setOutcome({ key: tag, error: e instanceof Error ? e.message : String(e) });
    } finally {
      setPublishing(false);
    }
  }

  if (draft) {
    return (
      <section className="flex min-h-0 flex-col rounded-xl border border-line bg-surface-1">
        <header className="flex items-center gap-2 border-b border-line px-4 py-3 text-sm">
          <span className="animate-pulse text-accent-ink" aria-hidden>●</span>
          <span className="font-semibold">Writing with the fine-tuned model…</span>
        </header>
        <pre className="min-h-0 flex-1 overflow-y-auto whitespace-pre-wrap px-5 py-4 font-sans text-sm leading-relaxed text-ink" aria-live="polite">
          {draft.text || "Waiting for the first tokens (a cold GPU can take a minute or two)…"}
        </pre>
      </section>
    );
  }

  if (!artifact || !selected) {
    return (
      <section className="flex min-h-0 items-center justify-center rounded-xl border border-dashed border-line bg-surface-1 p-8 text-center text-sm text-muted">
        Articles and pages the agent creates appear here — preview, compare versions, and publish.
      </section>
    );
  }

  return (
    <section className="flex min-h-0 flex-col rounded-xl border border-line bg-surface-1">
      <header className="flex flex-wrap items-center gap-2 border-b border-line px-4 py-3">
        <select
          className="max-w-[16rem] truncate rounded-lg border border-line bg-surface-1 px-2 py-1.5 text-sm"
          value={artifact.id}
          aria-label="Artifact"
          onChange={(e) => {
            const next = artifacts.find((a) => a.id === e.target.value);
            if (next) onSelect({ artifactId: next.id, version: next.current_version });
          }}
        >
          {artifacts.map((a) => <option key={a.id} value={a.id}>{a.title}</option>)}
        </select>
        <select
          className="rounded-lg border border-line bg-surface-1 px-2 py-1.5 text-sm"
          value={selected.version}
          aria-label="Version"
          onChange={(e) => onSelect({ artifactId: artifact.id, version: Number(e.target.value) })}
        >
          {artifact.versions.map((v) => (
            <option key={v.version} value={v.version}>
              v{v.version}{v.note ? ` — ${v.note}` : ""}
            </option>
          ))}
        </select>
        <div className="ml-auto flex items-center gap-1 rounded-lg bg-surface-2 p-0.5 text-xs" role="tablist">
          {(["preview", "markdown"] as const).map((t) => (
            <button key={t} type="button" role="tab" aria-selected={tab === t} onClick={() => setTab(t)} className={`rounded-md px-2.5 py-1 font-medium ${tab === t ? "bg-surface-1 text-ink shadow-sm" : "text-ink-2"}`}>
              {t === "preview" ? "Preview" : "Markdown"}
            </button>
          ))}
        </div>
        <Button variant="primary" onClick={publish} busy={publishing}>{liveUrl ? "Re-publish" : "Publish"}</Button>
      </header>
      {(error || liveUrl) && (
        <div className="border-b border-line px-4 py-2 text-xs">
          {error ? (
            <span className="text-critical">✕ {error}</span>
          ) : (
            <span className="text-ink-2">
              <span className="text-good" aria-hidden>✓ </span>Live at{" "}
              <a href={liveUrl!} target="_blank" rel="noreferrer" className="font-medium text-accent-ink underline">{liveUrl}</a>
            </span>
          )}
        </div>
      )}
      <div className="min-h-0 flex-1">
        {tab === "preview" ? (
          // No allow-scripts: agent HTML stays inert (it is also sanitised server-side).
          // allow-same-origin only so preview images load through the authenticated proxy.
          <iframe title="Page preview" sandbox="allow-same-origin" srcDoc={html} className="h-full min-h-[60vh] w-full rounded-b-xl" />
        ) : (
          <pre className="h-full overflow-y-auto whitespace-pre-wrap px-5 py-4 font-mono text-xs leading-relaxed text-ink">{markdown}</pre>
        )}
      </div>
    </section>
  );
}
