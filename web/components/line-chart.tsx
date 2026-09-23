"use client";

import { useEffect, useId, useMemo, useRef, useState } from "react";

/**
 * Single-series line chart (training loss, learning rate).
 *
 * Dataviz spec: 2px line with round joins, >=8px end dot, value labelled at the
 * line end, hairline recessive grid, clean ticks, one axis. The crosshair snaps
 * to the nearest step (the reader aims at a step, not at a 2px line), the same
 * readout is reachable by keyboard (arrow keys), and every value is also in the
 * table view, so the tooltip never gates information. One series -> no legend;
 * the card title names it.
 */

export interface Point {
  x: number;
  y: number;
}

interface Props {
  points: Point[];
  xLabel: string;
  yLabel: string;
  formatY?: (v: number) => string;
  height?: number;
  emptyText?: string;
}

const PAD = { top: 12, right: 64, bottom: 28, left: 52 };

function niceTicks(min: number, max: number, count = 4, integer = false): number[] {
  if (min === max) {
    const pad = Math.abs(min) * 0.1 || 1;
    min -= pad;
    max += pad;
  }
  const raw = (max - min) / count;
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  let step = [1, 2, 2.5, 5, 10].map((m) => m * magnitude).find((s) => s >= raw) ?? raw;
  // Steps are counts: an x axis of training steps must never show "2.5".
  if (integer) step = Math.max(1, Math.ceil(step));
  const start = Math.floor(min / step) * step;
  const ticks: number[] = [];
  for (let t = start; t <= max + step * 0.5; t += step) ticks.push(Number(t.toPrecision(12)));
  return ticks;
}

export function LineChart({ points, xLabel, yLabel, formatY = (v) => v.toFixed(3), height = 220, emptyText = "No data yet" }: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(600);
  const [active, setActive] = useState<number | null>(null);
  const [showTable, setShowTable] = useState(false);
  const titleId = useId();

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => setWidth(Math.max(280, entry.contentRect.width)));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const geo = useMemo(() => {
    if (!points.length) return null;
    const xs = points.map((p) => p.x);
    const ys = points.map((p) => p.y);
    const yTicks = niceTicks(Math.min(...ys), Math.max(...ys));
    const xMin = Math.min(...xs);
    const xMax = Math.max(...xs) === xMin ? xMin + 1 : Math.max(...xs);
    const yMin = yTicks[0];
    const yMax = yTicks[yTicks.length - 1];
    const innerW = width - PAD.left - PAD.right;
    const innerH = height - PAD.top - PAD.bottom;
    const sx = (x: number) => PAD.left + ((x - xMin) / (xMax - xMin)) * innerW;
    const sy = (y: number) => PAD.top + (1 - (y - yMin) / (yMax - yMin || 1)) * innerH;
    const path = points.map((p, i) => `${i ? "L" : "M"}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join("");
    const integerX = xs.every(Number.isInteger);
    const xTicks = niceTicks(xMin, xMax, Math.max(2, Math.min(6, Math.floor(innerW / 90))), integerX).filter((t) => t >= xMin && t <= xMax);
    return { sx, sy, path, yTicks, xTicks, innerW, innerH };
  }, [points, width, height]);

  if (!points.length || !geo) {
    return (
      <div ref={wrapRef} className="flex items-center justify-center rounded-lg bg-surface-2 text-sm text-muted" style={{ height }}>
        {emptyText}
      </div>
    );
  }

  const last = points[points.length - 1];
  const hovered = active !== null ? points[active] : null;

  function nearest(clientX: number, svg: SVGSVGElement) {
    const rect = svg.getBoundingClientRect();
    const x = clientX - rect.left;
    let best = 0;
    let bestD = Infinity;
    points.forEach((p, i) => {
      const d = Math.abs(geo!.sx(p.x) - x);
      if (d < bestD) {
        bestD = d;
        best = i;
      }
    });
    return best;
  }

  return (
    <div ref={wrapRef} className="relative">
      <svg
        width={width}
        height={height}
        role="img"
        aria-labelledby={titleId}
        tabIndex={0}
        className="block touch-none select-none outline-none"
        onPointerMove={(e) => setActive(nearest(e.clientX, e.currentTarget))}
        onPointerLeave={() => setActive(null)}
        onFocus={() => setActive(points.length - 1)}
        onBlur={() => setActive(null)}
        onKeyDown={(e) => {
          if (e.key === "ArrowLeft") setActive((i) => Math.max(0, (i ?? points.length - 1) - 1));
          if (e.key === "ArrowRight") setActive((i) => Math.min(points.length - 1, (i ?? 0) + 1));
        }}
      >
        <title id={titleId}>{`${yLabel} by ${xLabel}, latest ${formatY(last.y)}`}</title>
        {geo.yTicks.map((t) => (
          <g key={`y${t}`}>
            <line x1={PAD.left} x2={PAD.left + geo.innerW} y1={geo.sy(t)} y2={geo.sy(t)} stroke="var(--grid)" strokeWidth={1} />
            <text x={PAD.left - 8} y={geo.sy(t)} dy="0.32em" textAnchor="end" fontSize={11} fill="var(--text-muted)" className="tabular-nums">
              {formatY(t)}
            </text>
          </g>
        ))}
        {geo.xTicks.map((t) => (
          <text key={`x${t}`} x={geo.sx(t)} y={height - 8} textAnchor="middle" fontSize={11} fill="var(--text-muted)" className="tabular-nums">
            {t}
          </text>
        ))}
        <text x={PAD.left + geo.innerW} y={height - 8} textAnchor="end" dx={56} fontSize={11} fill="var(--text-muted)">
          {xLabel}
        </text>

        <path d={geo.path} fill="none" stroke="var(--series-1)" strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
        <circle cx={geo.sx(last.x)} cy={geo.sy(last.y)} r={4} fill="var(--series-1)" stroke="var(--surface-1)" strokeWidth={2} />
        <text x={geo.sx(last.x) + 8} y={geo.sy(last.y)} dy="0.32em" fontSize={12} fontWeight={600} fill="var(--text-primary)" className="tabular-nums">
          {formatY(last.y)}
        </text>

        {hovered && (
          <g pointerEvents="none">
            <line x1={geo.sx(hovered.x)} x2={geo.sx(hovered.x)} y1={PAD.top} y2={PAD.top + geo.innerH} stroke="var(--text-muted)" strokeWidth={1} />
            <circle cx={geo.sx(hovered.x)} cy={geo.sy(hovered.y)} r={5} fill="var(--series-1)" stroke="var(--surface-1)" strokeWidth={2} />
          </g>
        )}
      </svg>

      {hovered && (
        <div
          className="pointer-events-none absolute top-1 rounded-lg border border-line bg-surface-1 px-3 py-2 text-xs shadow-sm"
          style={{ left: Math.min(geo.sx(hovered.x) + 12, width - 150) }}
          aria-live="polite"
        >
          <div className="text-sm font-semibold tabular-nums text-ink">{formatY(hovered.y)}</div>
          <div className="flex items-center gap-1.5 text-ink-2">
            <span className="inline-block h-0.5 w-3 rounded" style={{ background: "var(--series-1)" }} aria-hidden />
            {yLabel} · {xLabel} {hovered.x}
          </div>
        </div>
      )}

      <button type="button" onClick={() => setShowTable((s) => !s)} className="mt-2 text-xs font-medium text-accent-ink hover:underline">
        {showTable ? "Hide table" : "View as table"}
      </button>
      {showTable && (
        <div className="mt-2 max-h-56 overflow-auto rounded-lg border border-line">
          <table className="w-full text-left text-xs">
            <thead className="sticky top-0 bg-surface-2 text-ink-2">
              <tr>
                <th className="px-3 py-1.5 font-medium">{xLabel}</th>
                <th className="px-3 py-1.5 font-medium">{yLabel}</th>
              </tr>
            </thead>
            <tbody className="tabular-nums text-ink">
              {points.map((p) => (
                <tr key={p.x} className="border-t border-line">
                  <td className="px-3 py-1">{p.x}</td>
                  <td className="px-3 py-1">{formatY(p.y)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
