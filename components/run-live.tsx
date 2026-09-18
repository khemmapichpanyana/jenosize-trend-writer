"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { LineChart } from "@/components/line-chart";
import { StatusBadge } from "@/components/status";
import { Button, Card, Empty, ErrorNote, Meter, StatTile } from "@/components/ui";
import { API, get, post } from "@/lib/api";
import { duration, fixed } from "@/lib/format";
import { streamSse } from "@/lib/sse";
import type { JobRun, ProgressPoint } from "@/lib/types";

const TERMINAL = new Set(["succeeded", "failed", "cancelled"]);

export function RunLive({ id }: { id: string }) {
  const [run, setRun] = useState<JobRun | null>(null);
  const [points, setPoints] = useState<ProgressPoint[]>([]);
  const [connection, setConnection] = useState<"connecting" | "live" | "reconnecting" | "closed">("connecting");
  const [error, setError] = useState<string | null>(null);
  const [cancelling, setCancelling] = useState(false);
  const lastId = useRef(0);

  // Follow the run's event stream; on a dropped connection, resume from the last
  // progress row received (after_id) so the chart has no gaps or duplicates.
  useEffect(() => {
    const controller = new AbortController();
    let stopped = false;
    (async () => {
      let attempt = 0;
      while (!stopped) {
        try {
          await streamSse(
            `${API}/runs/${id}/events?after_id=${lastId.current}`,
            { method: "GET" },
            (event, data) => {
              setConnection("live");
              attempt = 0;
              if (event === "status") {
                setRun((prev) => ({ ...(prev ?? {}), ...(data as JobRun), stages: prev?.stages }));
                void get<JobRun>(`/runs/${id}`).then(setRun).catch(() => undefined);
              } else if (event === "progress") {
                const point = data as ProgressPoint;
                lastId.current = Math.max(lastId.current, point.id);
                setPoints((prev) => (prev.some((p) => p.id === point.id) ? prev : [...prev, point]));
              } else if (event === "done") {
                stopped = true;
                setConnection("closed");
              }
            },
            controller.signal,
          );
          if (stopped) break;
        } catch (e) {
          if (controller.signal.aborted) return;
          setError(e instanceof Error ? e.message : String(e));
        }
        if (stopped) break;
        setConnection("reconnecting");
        attempt += 1;
        await new Promise((r) => setTimeout(r, Math.min(15_000, 1000 * 2 ** attempt)));
      }
      setConnection("closed");
      void get<JobRun>(`/runs/${id}`).then(setRun).catch(() => undefined);
    })();
    return () => {
      stopped = true;
      controller.abort();
    };
  }, [id]);

  const training = points.filter((p) => p.phase === "training" && p.step !== null);
  const latest = points[points.length - 1];
  const lossPoints = useMemo(() => training.filter((p) => p.loss !== null).map((p) => ({ x: p.step!, y: p.loss! })), [training]);
  const lrPoints = useMemo(() => training.filter((p) => p.learning_rate !== null).map((p) => ({ x: p.step!, y: p.learning_rate! })), [training]);
  const gpuPoint = [...points].reverse().find((p) => p.gpu_util !== null);
  const step = latest?.step ?? 0;
  const total = latest?.total_steps ?? training.at(-1)?.total_steps ?? 0;
  const pct = total ? Math.min(100, (step / total) * 100) : 0;
  const eta = useMemo(() => {
    if (training.length < 3 || !total) return null;
    const first = training[0];
    const last = training[training.length - 1];
    const secs = (new Date(last.created_at).getTime() - new Date(first.created_at).getTime()) / 1000;
    const perStep = secs / Math.max(1, (last.step ?? 0) - (first.step ?? 0));
    return Math.round(perStep * (total - (last.step ?? 0)));
  }, [training, total]);

  async function cancel() {
    setCancelling(true);
    try {
      setRun(await post<JobRun>(`/runs/${id}/cancel`));
    } finally {
      setCancelling(false);
    }
  }

  if (!run) return <Empty>{connection === "connecting" ? "Connecting…" : "Loading run…"}</Empty>;
  const isTrain = run.kind === "train";

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <h1 className="text-2xl font-semibold tracking-tight">
            {run.kind} {run.params.version ? String(run.params.version) : ""}
          </h1>
          <StatusBadge status={run.status} />
          <span className="text-xs text-muted" aria-live="polite">
            {connection === "live" ? "● live" : connection === "reconnecting" ? "reconnecting…" : connection === "closed" ? "stream ended" : "connecting…"}
          </span>
        </div>
        {!TERMINAL.has(run.status) && <Button variant="danger" onClick={cancel} busy={cancelling}>Cancel run</Button>}
      </div>
      <ErrorNote message={run.error ?? (connection !== "live" ? error : null)} />

      {isTrain && (
        <>
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
            <StatTile label="Step" value={total ? `${step} / ${total}` : step || "—"} hint={latest?.phase} />
            <StatTile label="Latest loss" value={fixed(training.at(-1)?.loss)} hint={lossPoints.length > 1 ? `started at ${fixed(lossPoints[0].y)}` : undefined} />
            <StatTile label="Elapsed" value={duration(run.started_at, TERMINAL.has(run.status) ? run.finished_at : null)} />
            <StatTile label="Time left" value={eta !== null && !TERMINAL.has(run.status) ? `~${Math.max(0, Math.round(eta / 60))}m` : "—"} />
          </div>
          <div className="h-2 overflow-hidden rounded-full bg-accent-wash" role="progressbar" aria-valuenow={Math.round(pct)} aria-valuemin={0} aria-valuemax={100} aria-label="Training progress">
            <div className="h-full rounded-full bg-accent transition-[width] duration-500" style={{ width: `${pct}%` }} />
          </div>

          <div className="grid gap-6 xl:grid-cols-3">
            <Card title="Training loss" className="xl:col-span-2">
              <LineChart points={lossPoints} xLabel="step" yLabel="loss" emptyText={latest?.message ?? "Waiting for the first logged step…"} />
            </Card>
            <Card title="GPU (live)">
              <div className="space-y-4">
                <Meter label="Utilisation" value={gpuPoint?.gpu_util} max={100} unit="%" />
                <Meter label="Memory" value={gpuPoint?.gpu_mem_used_gb} max={Math.round(gpuPoint?.gpu_mem_total_gb ?? 24)} unit=" GB" warnAt={0.9} dangerAt={0.97} />
                <dl className="grid grid-cols-2 gap-2 text-xs">
                  <dt className="text-muted">Epoch</dt><dd className="tabular-nums">{fixed(latest?.epoch, 2)}</dd>
                  <dt className="text-muted">Grad norm</dt><dd className="tabular-nums">{fixed(training.at(-1)?.grad_norm)}</dd>
                  <dt className="text-muted">Phase</dt><dd>{latest?.phase ?? "—"}</dd>
                </dl>
                {latest?.message && <p className="text-xs text-ink-2">{latest.message}</p>}
              </div>
            </Card>
          </div>
          <Card title="Learning rate">
            <LineChart points={lrPoints} xLabel="step" yLabel="learning rate" formatY={(v) => v.toExponential(1)} height={160} />
          </Card>
        </>
      )}

      {run.stages && run.stages.length > 0 && (
        <Card title="Stages">
          <table className="w-full text-left text-sm">
            <tbody>
              {run.stages.map((s) => (
                <tr key={s.stage + s.started_at} className="border-t border-line first:border-t-0">
                  <td className="py-2 font-medium">{s.stage}</td>
                  <td className="py-2 text-ink-2">{s.status}</td>
                  <td className="py-2 font-mono text-xs text-ink-2">
                    {Object.entries(s.stats).filter(([k]) => k !== "seconds").map(([k, v]) => `${k}=${v}`).join("  ")}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      )}

      <div className="grid gap-6 lg:grid-cols-2">
        <Card title="Parameters"><pre className="overflow-auto text-xs text-ink-2">{JSON.stringify(run.params, null, 2)}</pre></Card>
        <Card title="Result">
          {run.result ? <pre className="overflow-auto text-xs text-ink-2">{JSON.stringify(run.result, null, 2)}</pre> : <Empty>{TERMINAL.has(run.status) ? "No result." : "Running…"}</Empty>}
        </Card>
      </div>
    </div>
  );
}
