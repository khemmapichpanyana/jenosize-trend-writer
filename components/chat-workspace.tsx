"use client";

import { useSearchParams } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ArrowUp, Check, ChevronDown, Copy, Eye, FileCode2, LoaderCircle, PanelRight, PenLine, Plus, RotateCcw, Square, X } from "lucide-react";
import {
  Message as AiMessage,
  MessageContent as AiMessageContent,
  MessageResponse,
} from "@/components/ai-elements/message";
import { Conversation, ConversationContent } from "@/components/ai-elements/conversation";
import { PromptInput, PromptInputBody, PromptInputTextarea } from "@/components/ai-elements/prompt-input";
import { openPalette } from "@/components/app-sidebar";
import { AgentSteps, parseResult, stepsFromRecords, type AgentStep } from "@/components/agent-steps";
import { ModelBadge } from "@/components/model-badge";
import { ModelStatusControl } from "@/components/model-status";
import { Button, ErrorNote } from "@/components/ui";
import { LoadingState } from "@/components/loading-state";
import { API, ApiError, get, getText, post, upload } from "@/lib/api";
import { invalidate } from "@/lib/hooks";
import { markModelReady, useModelStatus, wakeModel } from "@/lib/model-status";
import { takePendingBrief } from "@/lib/pending-brief";
import { streamSse } from "@/lib/sse";
import type { AgentRun, Artifact, Asset, ChatMessage, PublishedPage, ThreadDetail } from "@/lib/types";

const ASSET_BASE = `${API}/studio/assets`;
// Streamdown renders plain markdown elements at browser/library default sizes;
// these child selectors force the reply text down to the compact scale the
// rest of the console uses, regardless of what Streamdown sets internally.
const COMPACT_PROSE =
  "[&_p]:my-1.5 [&_p]:text-[13px] [&_p]:leading-5 [&_li]:text-[13px] [&_li]:leading-5 [&_ul]:my-1.5 [&_ol]:my-1.5 [&_h1]:mb-1 [&_h1]:mt-2 [&_h1]:text-[15px] [&_h1]:font-semibold [&_h2]:mb-1 [&_h2]:mt-2 [&_h2]:text-[14px] [&_h2]:font-semibold [&_h3]:mb-1 [&_h3]:mt-1.5 [&_h3]:text-[13px] [&_h3]:font-semibold [&_pre]:text-[12px] [&_code]:text-[12px]";


interface LiveTurn {
  runId: string | null;
  user: string;
  assetIds: string[];
  reply: string;
  tools: AgentStep[];
}

/** Keep the chat renderable while older API deployments are rolling forward. */
function normalizeThread(detail: ThreadDetail): ThreadDetail {
  return {
    ...detail,
    messages: (Array.isArray(detail.messages) ? detail.messages : []).map((message) => ({
      ...message,
      tool_calls: Array.isArray(message.tool_calls) ? message.tool_calls : [],
      asset_ids: Array.isArray(message.asset_ids) ? message.asset_ids : [],
    })),
    artifacts: (Array.isArray(detail.artifacts) ? detail.artifacts : []).map((artifact) => ({
      ...artifact,
      versions: Array.isArray(artifact.versions) ? artifact.versions : [],
      published: Array.isArray(artifact.published) ? artifact.published : [],
    })),
    assets: Array.isArray(detail.assets) ? detail.assets : [],
  };
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
  const [canvasOpen, setCanvasOpen] = useState(true);
  const abortRef = useRef<AbortController | null>(null);
  const sendingRef = useRef(false);
  const followingRunIdRef = useRef<string | null>(null);
  const modelStatus = useModelStatus();

  const load = useCallback(async () => {
    try {
      const detail = normalizeThread(await get<ThreadDetail>(`/studio/threads/${threadId}`));
      setThread(detail);
      return detail;
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      return null;
    }
  }, [threadId]);

  const handleEvent = useCallback((event: string, data: Record<string, unknown>) => {
    switch (event) {
      case "thread":
        // First brief renames a "New chat" thread; show it without waiting for the turn to end.
        if (typeof data.title === "string") {
          const title = data.title;
          setThread((t) => (t ? { ...t, title } : t));
          invalidate("/studio/threads");
        }
        break;
      case "token":
        markModelReady();
        setLive((t) => (t ? { ...t, reply: t.reply + String(data.text ?? "") } : t));
        break;
      case "tool_start":
        setLive((t) =>
          t && !t.tools.some((x) => x.id === String(data.id))
            ? {
                ...t,
                tools: [
                  ...t.tools,
                  {
                    id: String(data.id),
                    name: String(data.name),
                    state: "running",
                    args: (data.args as Record<string, unknown>) ?? {},
                    notes: [],
                    startedAt: typeof data.started_at === "string" ? Date.parse(data.started_at) : Date.now(),
                  },
                ],
              }
            : t,
        );
        break;
      case "tool_progress":
        setLive((t) =>
          t
            ? {
                ...t,
                // Notes go to the latest running call of that tool (parallel searches share a name).
                tools: (() => {
                  const message = String(data.message ?? "");
                  const target = t.tools.findLast((x) => x.name === data.tool && x.state === "running");
                  if (!target || !message || target.notes.at(-1) === message) return t.tools;
                  return t.tools.map((x) => (x === target ? { ...x, notes: [...x.notes, message] } : x));
                })(),
              }
            : t,
        );
        break;
      case "tool_end": {
        const result = parseResult(data.result);
        const failed = Boolean(result && "error" in result);
        setLive((t) =>
          t
            ? {
                ...t,
                tools: t.tools.map((x) =>
                  x.id === String(data.id)
                    ? { ...x, state: failed ? "error" : "done", result, durationMs: typeof data.duration_ms === "number" ? data.duration_ms : x.startedAt ? Date.now() - x.startedAt : undefined }
                    : x,
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
        setCanvasOpen(true);
        void load();
        break;
      case "asset":
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
    // Guards against ever tailing the same run's event log twice at once
    // (e.g. a racing resume-on-mount and a fresh send), which double-applies
    // every token/tool event into the shared `live` state.
    if (followingRunIdRef.current === runId) return;
    followingRunIdRef.current = runId;
    try {
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
        if (!finished && !controller.signal.aborted) await new Promise((r) => setTimeout(r, Math.min(8777, 500 * 2 ** attempt)));
      }
      if (controller.signal.aborted) return;
      setDraft(null);
      invalidate("/studio/threads");
      await load();
      setLive(null);
    } finally {
      followingRunIdRef.current = null;
    }
  }, [handleEvent, load]);

  useEffect(() => {
    let cancelled = false;
    get<ThreadDetail>(`/studio/threads/${threadId}`)
      .then((detail) => {
        if (cancelled) return;
        const normalized = normalizeThread(detail);
        setThread(normalized);
        const run = normalized.active_run;
        if (run) {
          // Resume watching a turn that is still running (reload, second tab).
          const user = normalized.messages.find((m) => m.id === run.message_id);
          setLive({ runId: run.id, user: user?.content ?? "", assetIds: user?.asset_ids ?? [], reply: "", tools: [] });
          void followRun(run.id);
        }
        if (!normalized.artifacts.length) return;
        const wanted = search.get("artifact");
        const artifact = normalized.artifacts.find((a) => a.id === wanted) ?? normalized.artifacts[normalized.artifacts.length - 1];
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

  // Esc stops the running turn — unless a dialog (the ⌘K palette) is open and
  // the key is meant for it. ⌘K itself lives in the app sidebar.
  const liveRunId = live?.runId ?? null;
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key !== "Escape" || !liveRunId || document.querySelector('[role="dialog"]')) return;
      void post(`/studio/runs/${liveRunId}/cancel`).catch((err) => setError(err instanceof Error ? err.message : String(err)));
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [liveRunId]);

  // A brief typed on Home arrives here via sessionStorage: send it once the
  // thread has loaded (and isn't already mid-turn). `send` is read through a
  // ref so this runs once per thread, not on every render.
  const sendRef = useRef<(text: string) => Promise<void>>(async () => {});
  useEffect(() => {
    sendRef.current = (text) => send(text);
  });
  const threadLoaded = thread?.id === threadId;
  const hasActiveRun = Boolean(thread?.active_run);
  useEffect(() => {
    if (!threadLoaded || hasActiveRun) return;
    const brief = takePendingBrief(threadId);
    if (brief) void sendRef.current(brief);
  }, [threadLoaded, hasActiveRun, threadId]);

  async function uploadImages(files: FileList | File[]): Promise<Asset[]> {
    const images = Array.from(files).filter((f) => f.type.startsWith("image/"));
    const uploaded: Asset[] = [];
    for (const file of images) {
      setUploading((n) => n + 1);
      try {
        const form = new FormData();
        form.append("file", file);
        form.append("thread_id", threadId);
        const asset = await upload<Asset>("/studio/assets", form);
        setPending((p) => [...p, asset]);
        uploaded.push(asset);
      } catch (e) {
        setError(`Upload failed for ${file.name}: ${e instanceof Error ? e.message : e}`);
      } finally {
        setUploading((n) => n - 1);
      }
    }
    return uploaded;
  }

  async function attach(files: FileList | File[]) {
    await uploadImages(files);
  }

  async function send(textOverride?: string, assetOverride?: string[]) {
    const text = (textOverride ?? input).trim();
    // `live` is React state (updates lag a render), so two calls fired in the
    // same tick — e.g. Enter's form-submit racing the Send button — can both
    // read it as null. This ref is set synchronously and closes that gap.
    if (!text || live || sendingRef.current) return;
    sendingRef.current = true;
    const assetIds = assetOverride ?? pending.map((a) => a.id);
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
    } finally {
      sendingRef.current = false;
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
    return <div className="mx-auto max-w-3xl px-4 py-6">{error ? <ErrorNote message={error} /> : <LoadingState label="Loading conversation" />}</div>;
  }

  // While a turn is live, its user message is already saved server-side, so a
  // mid-run reload (e.g. on the "artifact" event) brings it back in
  // thread.messages. The live block renders it, so drop the saved copy — the
  // trailing user message(s) after the last assistant reply.
  const shown = live ? dropTrailingUser(thread.messages) : thread.messages;
  const lastAssistantId = [...thread.messages].reverse().find((m) => m.role === "assistant")?.id ?? null;
  const lastUserText = [...thread.messages].reverse().find((m) => m.role === "user")?.content ?? null;

  // The preview card only exists once there is something to show (a draft
  // being written or a saved artifact); until then the chat has the page.
  const hasCanvas = Boolean(draft) || thread.artifacts.length > 0;
  const showCanvas = hasCanvas && (canvasOpen || Boolean(draft));

  return (
    // Full-bleed like Claude: the chat *is* the page (no card, no background of
    // its own); the artifact preview is the one card, inset on the right.
    // Below lg the 3rem mobile top bar sits above, and the card stacks under the chat.
    <div className="lg:flex lg:h-dvh">
      {/* ---------------------------------------------------------------- chat */}
      <section className="flex h-[calc(100dvh-3rem)] min-w-0 flex-col lg:h-auto lg:flex-1">
        <header className="flex h-12 shrink-0 items-center justify-between gap-2 px-2 sm:px-3">
          {/* Like Claude's chat title: the title is the switcher (opens ⌘K). */}
          <button
            type="button"
            onClick={openPalette}
            title="Switch conversation (⌘K)"
            className="group flex min-w-0 items-center gap-1 rounded-md px-1.5 py-1 transition-colors hover:bg-surface-2"
          >
            <span key={thread.title} className="t-reveal truncate text-[13px] font-semibold tracking-tight text-ink">{thread.title}</span>
            <ChevronDown size={13} className="shrink-0 text-muted group-hover:text-ink" aria-hidden />
          </button>
          <div className="flex shrink-0 items-center gap-1">
            <ModelStatusControl />
            {hasCanvas && (
              <button
                type="button"
                onClick={() => setCanvasOpen((open) => !open)}
                title={showCanvas ? "Hide preview" : "Show preview"}
                aria-pressed={showCanvas}
                className={`hidden size-8 items-center justify-center rounded-lg transition-colors lg:flex ${showCanvas ? "bg-accent-wash text-accent-ink" : "text-ink-2 hover:bg-surface-2 hover:text-ink"}`}
              >
                <PanelRight size={16} aria-hidden />
                <span className="sr-only">{showCanvas ? "Hide preview" : "Show preview"}</span>
              </button>
            )}
          </div>
        </header>

        <Conversation className="min-h-0 flex-1">
          <ConversationContent className="mx-auto w-full max-w-3xl space-y-5 px-4 pb-6 pt-2 sm:px-6">
          {thread.messages.length === 0 && !live && <Starter onPick={setInput} />}
          {shown.map((m) => (
            <MessageBubble
              key={m.id}
              message={m}
              onRetry={!live && m.id === lastAssistantId && lastUserText ? () => void send(lastUserText) : undefined}
            />
          ))}
          {live && (
            <>
              <MessageBubble message={{ id: "live-user", role: "user", content: live.user, created_at: "", tool_calls: [], asset_ids: live.assetIds, model: null }} />
              <AiMessage from="assistant">
                <AgentSteps steps={live.tools} />
                <AiMessageContent className={`w-full px-0.5 py-0 text-[13.5px] text-ink ${COMPACT_PROSE}`}>
                  {live.reply ? <MessageResponse>{live.reply}</MessageResponse> : (
                    <span className="t-shimmer inline-flex items-center gap-2 font-medium">
                      {modelStatus === "warming" && !live.tools.length
                        ? "Waking the model — the first reply after a quiet spell can take 1–3 minutes…"
                        : live.tools.some((t) => t.state === "running") ? "Working on your brief…" : "Thinking through the angle…"}
                    </span>
                  )}
                </AiMessageContent>
              </AiMessage>
            </>
          )}
          </ConversationContent>
        </Conversation>

        <div className="mx-auto w-full max-w-3xl shrink-0 px-4 pb-3 sm:px-6">
          <ErrorNote message={error} />
          {modelStatus === "warming" && !live && (
            <p className="mb-2 flex items-center gap-1.5 px-1 text-[11px] text-ink-2" role="status">
              <span className="size-1.5 animate-pulse rounded-full bg-warning" aria-hidden />
              Waking the model — you can send now; the first reply may take 1–3 minutes.
            </p>
          )}
          {modelStatus === "offline" && !live && (
            <p className="mb-2 flex items-center gap-1.5 px-1 text-[11px] text-ink-2" role="status">
              <span className="size-1.5 rounded-full bg-critical" aria-hidden />
              The model is asleep or didn&rsquo;t answer in time.
              <button type="button" onClick={wakeModel} className="font-semibold text-accent-ink underline-offset-2 hover:underline">
                Wake it
              </button>
              — or just send; the agent retries while it starts.
            </p>
          )}
          {(pending.length > 0 || uploading > 0) && (
            <div className="mb-2 flex flex-wrap gap-2">
              {pending.map((a) => (
                <div key={a.id} className="relative">
                  {/* eslint-disable-next-line @next/next/no-img-element -- authenticated proxy URL */}
                  <img src={`${ASSET_BASE}/${a.id}/raw`} alt={a.filename} className="size-12 rounded-lg border border-line object-cover shadow-sm" />
                  <button
                    type="button"
                    aria-label={`Remove ${a.filename}`}
                    onClick={() => setPending((p) => p.filter((x) => x.id !== a.id))}
                    className="absolute -right-1.5 -top-1.5 flex size-5 items-center justify-center rounded-full border border-line bg-surface-1 text-xs shadow-sm transition-transform hover:scale-110"
                  >
                    <X size={11} />
                  </button>
                </div>
              ))}
              {uploading > 0 && <div className="flex size-12 items-center justify-center rounded-lg border border-dashed border-line text-xs text-muted"><LoaderCircle size={14} className="animate-spin" /></div>}
            </div>
          )}
          <div className="rounded-2xl border border-line bg-surface-1 shadow-[0_6px_20px_rgba(7,19,38,0.05)] transition-colors focus-within:border-accent/45 focus-within:shadow-[0_0_0_3px_rgba(36,87,214,0.08)]">
            <PromptInput
              className="w-full"
              accept="image/*"
              multiple
              maxFiles={8}
              onSubmit={async ({ text, files }) => {
                if (live) return;
                let assetIds: string[] | undefined;
                if (files.length > 0) {
                  const uploaded = await uploadImages(
                    await Promise.all(
                      files.filter((file) => file.url).map(async (file) => {
                        const response = await fetch(file.url!);
                        const blob = await response.blob();
                        return new File([blob], file.filename || "attachment.png", { type: file.mediaType || blob.type || "image/png" });
                      }),
                    ),
                  );
                  assetIds = uploaded.map((asset) => asset.id);
                }
                await send(text, assetIds);
              }}
              onError={(issue) => setError(issue.message)}
            >
              <PromptInputBody>
                <PromptInputTextarea
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  rows={1}
                  placeholder={live ? "The agent is working…" : "Describe an article or page to create…"}
                  className="min-h-[42px] max-h-32 overflow-y-auto border-0 bg-transparent px-3.5 pb-0 pt-3 text-[13.5px] leading-6 shadow-none outline-none placeholder:text-muted focus-visible:ring-0"
                  disabled={!!live}
                />
              </PromptInputBody>
            </PromptInput>
            <div className="flex items-center justify-between px-2 pb-2 pt-1">
              <label className="flex size-8 cursor-pointer items-center justify-center rounded-full text-ink-2 transition-colors hover:bg-surface-2 hover:text-accent-ink" title="Attach images">
                <Plus size={18} aria-hidden />
                <span className="sr-only">Attach images</span>
                <input type="file" accept="image/png,image/jpeg,image/webp,image/gif" multiple className="hidden" onChange={(e) => e.target.files && attach(e.target.files)} />
              </label>
              {live ? (
                <button
                  type="button"
                  onClick={stop}
                  disabled={!live.runId}
                  title="Stop"
                  className="flex size-8 items-center justify-center rounded-full bg-ink text-white transition-opacity hover:opacity-90 disabled:opacity-40"
                >
                  <Square size={12} fill="currentColor" />
                </button>
              ) : (
                <button
                  type="button"
                  onClick={() => void send()}
                  disabled={!input.trim() || uploading > 0}
                  title="Send"
                  className="flex size-8 items-center justify-center rounded-full bg-accent text-white transition-opacity hover:opacity-90 disabled:opacity-30"
                >
                  <ArrowUp size={16} strokeWidth={2.4} />
                </button>
              )}
            </div>
          </div>
          <p className="mt-1.5 px-1 text-[10.5px] text-muted">Enter to send · Shift+Enter for a new line · drop or paste images</p>
        </div>
      </section>

      {/* ------------------------------------------------------------- artifact */}
      {showCanvas && (
        <div className="h-[85dvh] p-2 lg:h-auto lg:w-[52%] lg:shrink-0 lg:pl-0">
          <ArtifactPanel
            artifacts={thread.artifacts}
            selected={selected}
            onSelect={setSelected}
            draft={draft}
            onPublished={load}
            onClose={() => setCanvasOpen(false)}
          />
        </div>
      )}
    </div>
  );
}

function CloseButton({ onClose, className = "" }: { onClose: () => void; className?: string }) {
  return (
    <button
      type="button"
      onClick={onClose}
      title="Hide preview"
      className={`hidden size-7 items-center justify-center rounded-md text-ink-2 transition-colors hover:bg-surface-2 hover:text-ink lg:flex ${className}`}
    >
      <X size={14} aria-hidden />
      <span className="sr-only">Hide preview</span>
    </button>
  );
}

function Starter({ onPick }: { onPick: (text: string) => void }) {
  const ideas = [
    "Write a medium article on agentic AI in Southeast Asian retail for C-suite readers.",
    "Draft a short piece on tokenised real-world assets for Thai banking leaders.",
    "Write about circular supply chains after the tariff shock, then design it as a page.",
  ];
  return (
    <div className="t-stagger is-shown py-7 text-center sm:py-9">
      <h2 className="t-stagger-line t-stagger-line--1 text-[13px] font-semibold tracking-tight text-ink">Start with a brief</h2>
      <p className="t-stagger-line t-stagger-line--2 mx-auto mb-4 mt-1.5 max-w-sm text-xs leading-5 text-ink-2">Give the agent a topic and audience. It will write with the fine-tuned model, then turn the draft into a branded page.</p>
      <div className="t-stagger-line t-stagger-line--2 flex flex-col items-center gap-1.5">
        {ideas.map((idea) => (
          <button key={idea} type="button" onClick={() => onPick(idea)} className="group flex w-full max-w-md items-start justify-between gap-3 rounded-lg border border-line/80 bg-surface-1 px-3 py-2.5 text-left text-[12.5px] leading-5 text-ink-2 transition-[border-color,box-shadow,transform,color] duration-250 hover:-translate-y-px hover:border-accent/40 hover:text-ink hover:shadow-[0_8px_24px_rgba(36,87,214,0.08)]">
            <span className="min-w-0">{idea}</span>
            <span className="shrink-0 text-accent-ink opacity-0 transition-opacity group-hover:opacity-100" aria-hidden>→</span>
          </button>
        ))}
      </div>
    </div>
  );
}

function dropTrailingUser(messages: ChatMessage[]): ChatMessage[] {
  let end = messages.length;
  while (end > 0 && messages[end - 1].role === "user") end -= 1;
  return end === messages.length ? messages : messages.slice(0, end);
}

function MessageBubble({ message, onRetry }: { message: ChatMessage; onRetry?: () => void }) {
  const assetIds = Array.isArray(message.asset_ids) ? message.asset_ids : [];
  const toolCalls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  if (message.role === "user") {
    return (
      <AiMessage from="user" className="items-end gap-1.5">
        {assetIds.length > 0 && (
          <div className="flex gap-1.5">
            {assetIds.map((id) => (
              // eslint-disable-next-line @next/next/no-img-element -- authenticated proxy URL
              <img key={id} src={`${ASSET_BASE}/${id}/raw`} alt="" className="size-12 rounded-md border border-line object-cover" />
            ))}
          </div>
        )}
        <AiMessageContent className="!max-w-[80%] !rounded-2xl !bg-accent !px-3.5 !py-2 !text-[13.5px] !text-white">{message.content}</AiMessageContent>
      </AiMessage>
    );
  }
  return (
    <AiMessage from="assistant">
      <AgentSteps steps={stepsFromRecords(toolCalls)} />
      <AiMessageContent className={`w-full px-0.5 py-0 text-[13.5px] text-ink ${COMPACT_PROSE}`}><MessageResponse>{message.content}</MessageResponse></AiMessageContent>
      <MessageActions text={message.content} model={message.model} onRetry={onRetry} />
    </AiMessage>
  );
}

/** Copy (and, on the latest reply, Retry) under an assistant message. */
function MessageActions({ text, model, onRetry }: { text: string; model: string | null; onRetry?: () => void }) {
  const [copied, setCopied] = useState(false);
  async function copy() {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard blocked — the text is still selectable */
    }
  }
  const action = "inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[10.5px] font-medium text-muted transition-colors hover:bg-surface-2 hover:text-ink";
  return (
    <div className="-ml-1.5 flex items-center gap-0.5">
      <button type="button" onClick={() => void copy()} className={action} aria-label="Copy reply">
        {copied ? <Check size={11} className="text-good" /> : <Copy size={11} />} {copied ? "Copied" : "Copy"}
      </button>
      {onRetry && (
        <button type="button" onClick={onRetry} className={action} aria-label="Send the last brief again">
          <RotateCcw size={11} /> Retry
        </button>
      )}
      {model && <ModelBadge model={model} />}
    </div>
  );
}




function ArtifactPanel({
  artifacts,
  selected,
  onSelect,
  draft,
  onPublished,
  onClose,
}: {
  artifacts: Artifact[];
  selected: { artifactId: string; version: number } | null;
  onSelect: (s: { artifactId: string; version: number }) => void;
  draft: { artifactId: string; text: string } | null;
  onPublished: () => Promise<unknown>;
  onClose: () => void;
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
      invalidate("/studio/generated");
      await onPublished();
    } catch (e) {
      setOutcome({ key: tag, error: e instanceof Error ? e.message : String(e) });
    } finally {
      setPublishing(false);
    }
  }

  if (draft) {
    // The raw markdown streams in behind the scenes (`draft.text`), but showing
    // it token-by-token read as noisy/slow — a quiet skeleton plus a live word
    // count reads as "working" without the jitter, and the finished artifact
    // then mounts through the reveal transition below.
    const words = draft.text.trim() ? draft.text.trim().split(/\s+/).length : 0;
    return (
      <section className="flex h-full flex-col overflow-hidden rounded-xl border border-line/80 bg-surface-1 shadow-[0_8px_28px_rgba(7,19,38,0.06)]">
        <header className="flex items-center gap-2 border-b border-line/80 bg-accent-wash/30 px-3 py-2.5 text-xs">
          <span className="flex size-5 items-center justify-center rounded-md bg-accent text-white" aria-hidden><PenLine size={11} /></span>
          <span className="t-shimmer font-semibold">Writing with the fine-tuned model…</span>
          {words > 0 && <span className="ml-auto text-[11px] font-medium tabular-nums text-muted">{words} words so far</span>}
          <CloseButton onClose={onClose} className={words > 0 ? "" : "ml-auto"} />
        </header>
        <div className="min-h-0 flex-1 overflow-hidden px-4 py-4" aria-live="polite" aria-label="The agent is writing the article">
          <div className="t-skeleton mb-3 h-6 w-3/4 rounded-lg" />
          <div className="t-skeleton mb-2 h-3 w-full rounded-md" />
          <div className="t-skeleton mb-2 h-3 w-full rounded-md" />
          <div className="t-skeleton mb-5 h-3 w-2/3 rounded-md" />
          <div className="t-skeleton mb-2 h-3 w-full rounded-md" />
          <div className="t-skeleton mb-2 h-3 w-5/6 rounded-md" />
          <div className="t-skeleton mb-2 h-3 w-full rounded-md" />
          <div className="t-skeleton h-3 w-3/5 rounded-md" />
        </div>
      </section>
    );
  }

  if (!artifact || !selected) {
    return (
      <section className={`${"flex h-full flex-col overflow-hidden rounded-xl border border-line/80 bg-surface-1 shadow-[0_8px_28px_rgba(7,19,38,0.06)]"} items-center justify-center`}><LoadingState label="Loading preview" rows={4} /></section>
    );
  }

  return (
    <section key={key} className="t-reveal flex h-full flex-col overflow-hidden rounded-xl border border-line/80 bg-surface-1 shadow-[0_8px_28px_rgba(7,19,38,0.06)]">
      <header className="flex flex-wrap items-center gap-1.5 border-b border-line/80 bg-surface-1/90 px-3 py-2 backdrop-blur sm:px-4">
        <select
          className="max-w-[14rem] truncate rounded-md border border-line bg-surface-0/60 px-2 py-1.5 text-[11px] font-medium text-ink outline-none transition-[border-color,box-shadow] focus:border-accent focus:ring-4 focus:ring-accent/10"
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
          className="rounded-md border border-line bg-surface-0/60 px-2 py-1.5 text-[11px] font-medium text-ink outline-none transition-[border-color,box-shadow] focus:border-accent focus:ring-4 focus:ring-accent/10"
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
        <div className="ml-auto flex items-center gap-0.5 rounded-full bg-surface-2 p-0.5 text-[11px]" role="tablist">
          {(["preview", "markdown"] as const).map((t) => (
            <button key={t} type="button" role="tab" aria-selected={tab === t} onClick={() => setTab(t)} className={`inline-flex items-center gap-1 rounded-full px-2 py-1 font-medium transition-colors ${tab === t ? "bg-surface-1 text-ink shadow-sm" : "text-ink-2 hover:text-ink"}`}>
              {t === "preview" ? <Eye size={12} /> : <FileCode2 size={12} />}{t === "preview" ? "Preview" : "Markdown"}
            </button>
          ))}
        </div>
        <Button className="h-8 px-3 text-xs" variant="primary" onClick={publish} busy={publishing}>{liveUrl ? "Re-publish" : "Publish"}</Button>
        <CloseButton onClose={onClose} />
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
      <div className="min-h-0 flex-1 bg-white">
        {tab === "preview" ? (
          // No allow-scripts: agent HTML stays inert (it is also sanitised server-side).
          // allow-same-origin only so preview images load through the authenticated proxy.
          <iframe title="Page preview" sandbox="allow-same-origin" srcDoc={html} className="h-full w-full" />
        ) : (
          <pre className="h-full overflow-y-auto whitespace-pre-wrap px-5 py-4 font-mono text-xs leading-relaxed text-ink">{markdown}</pre>
        )}
      </div>
    </section>
  );
}
