"use client";

import { useSyncExternalStore } from "react";
import { post } from "./api";

export type ModelStatus = "idle" | "warming" | "ready" | "offline";

// Module-level so every component (header badge, composer hint) sees the same
// state and a route change doesn't fire a second warmup while one is pending.
let status: ModelStatus = "idle";
let checkedAt = 0;
const listeners = new Set<() => void>();
const RECHECK_MS = 5 * 60_000; // a warm Modal GPU scales down after idle; re-probe after this

function set(next: ModelStatus) {
  status = next;
  listeners.forEach((notify) => notify());
}

/** Wake the model if it isn't known to be warm. Safe to call often. */
export function ensureModelWarm() {
  if (status === "ready" && Date.now() - checkedAt < RECHECK_MS) return;
  wakeModel();
}

/** Start a warmup now (e.g. the user clicked "Wake"); no-op if one is already running. */
export function wakeModel() {
  if (status === "warming") return;
  set("warming");
  post<{ status: string }>("/studio/warmup")
    .then((r) => {
      checkedAt = Date.now();
      set(r.status === "error" ? "offline" : "ready");
    })
    .catch(() => {
      checkedAt = Date.now();
      set("offline");
    });
}

/** A turn just streamed tokens, so the model is demonstrably warm. */
export function markModelReady() {
  checkedAt = Date.now();
  if (status !== "ready") set("ready");
}

export function useModelStatus(): ModelStatus {
  return useSyncExternalStore(
    (notify) => {
      listeners.add(notify);
      return () => listeners.delete(notify);
    },
    () => status,
    () => "idle",
  );
}
