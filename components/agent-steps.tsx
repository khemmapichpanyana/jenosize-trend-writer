"use client";

import { useEffect, useState, type ReactNode } from "react";
import { Check, ChevronDown, ChevronRight, CircleAlert, ExternalLink, Globe, ImageIcon, LayoutTemplate, ListChecks, LoaderCircle, PenLine, Wrench } from "lucide-react";
import type { ToolCallRecord } from "@/lib/types";

/** One agent tool call, live (from the SSE stream) or saved (from the message). */
export interface AgentStep {
  id: string;
  name: string;
  state: "running" | "done" | "error";
  args?: Record<string, unknown>;
  result?: Record<string, unknown> | null;
  notes: string[];
  /** Epoch ms, for the live elapsed timer. */
  startedAt?: number;
  durationMs?: number;
}

type Json = Record<string, unknown>;

const LABEL: Record<string, string> = {
  web_search: "Searched the web",
  write_article: "Wrote with the fine-tuned model",
  design_page: "Designed the branded page",
  list_images: "Checked your images",
  get_artifact: "Read the article",
  generate_image: "Generated an image",
};
const RUNNING_LABEL: Record<string, string> = {
  web_search: "Searching the web",
  write_article: "Writing with the fine-tuned model",
  design_page: "Designing the branded page",
  list_images: "Checking your images",
  get_artifact: "Reading the article",
  generate_image: "Generating an image",
};
const ICON: Record<string, typeof Globe> = {
  web_search: Globe,
  write_article: PenLine,
  design_page: LayoutTemplate,
  generate_image: ImageIcon,
  list_images: ImageIcon,
  get_artifact: ListChecks,
};

export function parseResult(raw: unknown): Json | null {
  if (raw && typeof raw === "object") return raw as Json;
  if (typeof raw !== "string") return null;
  try {
    const value = JSON.parse(raw);
    return value && typeof value === "object" ? (value as Json) : { text: raw };
  } catch {
    return { text: raw };
  }
}

export function stepsFromRecords(calls: ToolCallRecord[]): AgentStep[] {
  return calls.map((call, i) => {
    const result = parseResult(call.result);
    return {
      id: call.id ?? String(i),
      name: call.name,
      state: result && "error" in result ? "error" : "done",
      args: call.args,
      result,
      notes: Array.isArray(call.notes) ? call.notes : [],
      durationMs: typeof call.duration_ms === "number" ? call.duration_ms : undefined,
    };
  });
}

function formatMs(ms: number): string {
  const s = Math.max(0, Math.round(ms / 1000));
  return s < 60 ? `${s}s` : `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, "0")}s`;
}

/** Re-renders once a second while something is running, for live elapsed times. */
function useNow(active: boolean): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!active) return;
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [active]);
  return now;
}

const str = (v: unknown): string | null => (typeof v === "string" && v.trim() ? v : null);

/** The one-line gist shown next to a step's label. */
function gist(step: AgentStep): string | null {
  const a = step.args ?? {};
  const r = step.result ?? {};
  if (step.state === "error") return str(r.error);
  switch (step.name) {
    case "web_search": {
      const n = Array.isArray(r.results) ? r.results.length : null;
      return [str(a.query) && `“${a.query}”`, n !== null && `${n} result${n === 1 ? "" : "s"}`].filter(Boolean).join(" · ") || null;
    }
    case "write_article":
      return str(r.title) ?? str(a.topic);
    case "design_page":
      return r.layout ? `${r.layout} layout${r.images_used ? ` · ${r.images_used} image${r.images_used === 1 ? "" : "s"}` : ""}` : null;
    case "generate_image":
      return str(a.prompt);
    default:
      return str(r.title);
  }
}

/**
 * The agent's working, step by step (like Claude's tool trace): each tool call
 * with its inputs, live progress notes, what it returned and how long it took.
 * Live steps stream in over SSE; saved turns render the same record.
 */
export function AgentSteps({ steps }: { steps: AgentStep[] }) {
  const running = steps.some((s) => s.state === "running");
  const [manuallyOpen, setManuallyOpen] = useState<boolean | null>(null);
  const open = manuallyOpen ?? running;
  const now = useNow(running);

  if (!steps.length) return null;

  const total = steps.reduce((sum, s) => sum + (s.durationMs ?? (s.startedAt ? now - s.startedAt : 0)), 0);
  const current = steps.findLast((s) => s.state === "running");
  const errors = steps.filter((s) => s.state === "error").length;
  const title = current
    ? `${RUNNING_LABEL[current.name] ?? current.name}…`
    : `${steps.length} step${steps.length === 1 ? "" : "s"}${errors ? ` · ${errors} failed` : ""}`;

  return (
    <div className="not-prose w-full max-w-[92%]">
      <button
        type="button"
        onClick={() => setManuallyOpen(!open)}
        aria-expanded={open}
        className="group inline-flex items-center gap-1.5 rounded-md py-0.5 text-[12px] text-ink-2 transition-colors hover:text-ink"
      >
        {current ? <LoaderCircle size={12} className="animate-spin text-accent-ink" /> : <Wrench size={12} className="text-muted" />}
        <span className={current ? "t-shimmer font-medium" : "font-medium"}>{title}</span>
        {total > 0 && <span className="tabular-nums text-muted">· {formatMs(total)}</span>}
        <ChevronDown size={12} className={`text-muted transition-transform ${open ? "rotate-180" : ""}`} />
      </button>
      {open && (
        <ol className="mt-1.5 space-y-0.5 border-l border-line pl-3">
          {steps.map((step) => (
            <StepRow key={step.id} step={step} now={now} />
          ))}
        </ol>
      )}
    </div>
  );
}

function StepRow({ step, now }: { step: AgentStep; now: number }) {
  const [open, setOpen] = useState(false);
  const Icon = ICON[step.name] ?? Wrench;
  const elapsed = step.durationMs ?? (step.startedAt ? now - step.startedAt : undefined);
  const line = gist(step);
  const tone = step.state === "running" ? "text-accent-ink" : step.state === "error" ? "text-critical" : "text-muted";
  return (
    <li className="t-reveal">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="group flex w-full min-w-0 items-center gap-1.5 rounded-md px-1 py-1 text-left text-[12px] transition-colors hover:bg-surface-2"
      >
        <span className={`shrink-0 ${tone}`} aria-hidden>
          {step.state === "running" ? <LoaderCircle size={13} className="animate-spin" /> : step.state === "error" ? <CircleAlert size={13} /> : <Icon size={13} />}
        </span>
        <span className="shrink-0 font-medium text-ink">{(step.state === "running" ? RUNNING_LABEL : LABEL)[step.name] ?? step.name}</span>
        {line && <span className={`min-w-0 truncate ${step.state === "error" ? "text-critical" : "text-muted"}`}>{line}</span>}
        <span className="ml-auto flex shrink-0 items-center gap-1 pl-2 text-[11px] tabular-nums text-muted">
          {elapsed !== undefined && formatMs(elapsed)}
          {step.state === "done" && <Check size={11} className="text-good" aria-hidden />}
          <ChevronRight size={12} className={`transition-transform ${open ? "rotate-90" : ""}`} aria-hidden />
        </span>
      </button>
      {/* A running step shows its latest progress note without being opened. */}
      {!open && step.state === "running" && step.notes.length > 0 && (
        <p className="truncate pb-1 pl-6 text-[11px] text-muted">{step.notes[step.notes.length - 1]}</p>
      )}
      {open && <StepDetail step={step} />}
    </li>
  );
}

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="grid grid-cols-[5.5rem_minmax(0,1fr)] gap-2">
      <dt className="text-muted">{label}</dt>
      <dd className="min-w-0 break-words text-ink-2">{children}</dd>
    </div>
  );
}

function hostname(url: string): string {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return url;
  }
}

function StepDetail({ step }: { step: AgentStep }) {
  const a = step.args ?? {};
  const r = step.result ?? {};
  const results = Array.isArray(r.results) ? (r.results as Json[]) : [];
  const shownArgs = Object.entries(a).filter(([, v]) => v !== null && v !== undefined && v !== "" && !(Array.isArray(v) && !v.length));

  return (
    <div className="mb-2 ml-6 mt-0.5 space-y-2 rounded-lg border border-line/80 bg-surface-1 p-2.5 text-[11.5px] leading-5">
      {step.name === "web_search" ? (
        <>
          {str(a.query) && <Field label="Query">{String(a.query)}</Field>}
          {str(r.answer) && <Field label="Summary">{String(r.answer)}</Field>}
          {results.length > 0 && (
            <ul className="space-y-1.5 border-t border-line/70 pt-2">
              {results.map((hit, i) => (
                <li key={`${hit.url}-${i}`} className="min-w-0">
                  <a href={String(hit.url)} target="_blank" rel="noreferrer" className="group/link inline-flex max-w-full items-center gap-1 font-medium text-accent-ink hover:underline">
                    <span className="truncate">{str(hit.title) ?? hostname(String(hit.url))}</span>
                    <ExternalLink size={10} className="shrink-0 opacity-60" aria-hidden />
                  </a>
                  <p className="text-[10.5px] text-muted">{hostname(String(hit.url))}</p>
                  {str(hit.snippet) && <p className="line-clamp-2 text-ink-2">{String(hit.snippet)}</p>}
                </li>
              ))}
            </ul>
          )}
        </>
      ) : (
        shownArgs.length > 0 && (
          <dl className="space-y-0.5">
            {shownArgs.map(([key, value]) => (
              <Field key={key} label={key.replace(/_/g, " ")}>
                {Array.isArray(value) ? value.join(", ") : typeof value === "object" ? JSON.stringify(value) : String(value)}
              </Field>
            ))}
          </dl>
        )
      )}

      {step.notes.length > 0 && (
        <div className="border-t border-line/70 pt-2">
          <p className="mb-0.5 text-muted">Progress</p>
          <ol className="space-y-0.5 text-ink-2">
            {step.notes.map((note, i) => (
              <li key={i} className="flex gap-1.5">
                <span className="text-muted" aria-hidden>›</span>
                {note}
              </li>
            ))}
          </ol>
        </div>
      )}

      {step.state === "error" && str(r.error) && <p className="border-t border-line/70 pt-2 text-critical">{String(r.error)}</p>}

      {step.state === "done" && step.name !== "web_search" && <ResultFields name={step.name} result={r} />}
      {step.state === "running" && !step.notes.length && <p className="text-muted">Working…</p>}
    </div>
  );
}

function ResultFields({ name, result }: { name: string; result: Json }) {
  const rows: [string, ReactNode][] = [];
  if (name === "write_article") {
    if (str(result.title)) rows.push(["Title", String(result.title)]);
    if (typeof result.word_count === "number") rows.push(["Words", result.word_count]);
    if (typeof result.quality_passed === "boolean")
      rows.push(["Quality", result.quality_passed ? <span className="text-good">Passed the quality gate</span> : <span className="text-critical">Did not pass</span>]);
    if (Array.isArray(result.quality_warnings) && result.quality_warnings.length) rows.push(["Warnings", (result.quality_warnings as string[]).join(" · ")]);
    if (result.version) rows.push(["Saved as", `version ${result.version}`]);
  } else {
    for (const [key, value] of Object.entries(result)) {
      if (key === "excerpt" || key === "text" || value === null || value === undefined) continue;
      rows.push([key.replace(/_/g, " "), typeof value === "object" ? JSON.stringify(value) : String(value)]);
    }
  }
  if (!rows.length) return null;
  return (
    <dl className="space-y-0.5 border-t border-line/70 pt-2">
      <p className="mb-0.5 text-muted">Result</p>
      {rows.map(([label, value]) => (
        <Field key={label} label={label}>{value}</Field>
      ))}
    </dl>
  );
}
