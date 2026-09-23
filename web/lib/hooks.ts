"use client";

import { useCallback, useEffect, useState, useSyncExternalStore } from "react";
import { get } from "./api";

interface PollState<T> {
  path: string | null;
  data: T | null;
  error: string | null;
}

// Last good response per path, shared across mounts: navigating back to a page
// shows its previous data instantly while the refetch runs (stale-while-revalidate).
const cache = new Map<string, unknown>();
// One request per path at a time, however many components poll it.
const inflight = new Map<string, Promise<unknown>>();

function fetchShared<T>(path: string): Promise<T> {
  const pending = inflight.get(path);
  if (pending) return pending as Promise<T>;
  const request = get<T>(path)
    .then((data) => {
      cache.set(path, data);
      return data;
    })
    .finally(() => inflight.delete(path));
  inflight.set(path, request);
  return request;
}

/** Drop cached data for paths starting with `prefix`, e.g. after a write. */
export function invalidate(prefix: string) {
  for (const key of cache.keys()) if (key.startsWith(prefix)) cache.delete(key);
}

/**
 * Fetch a studio path now and every `intervalMs`, pausing while the tab is hidden
 * and refetching as soon as it becomes visible again. State is tagged with the
 * path it belongs to, so switching paths never shows another path's data.
 */
export function usePoll<T>(path: string | null, intervalMs = 5000) {
  const [state, setState] = useState<PollState<T>>({ path: null, data: null, error: null });

  const refresh = useCallback(async () => {
    if (!path) return;
    inflight.delete(path);
    try {
      const data = await fetchShared<T>(path);
      setState({ path, data, error: null });
    } catch (e) {
      const error = e instanceof Error ? e.message : String(e);
      setState((prev) => ({ path, data: prev.path === path ? prev.data : null, error }));
    }
  }, [path]);

  useEffect(() => {
    if (!path) return;
    let cancelled = false;
    const tick = () =>
      fetchShared<T>(path)
        .then((data) => !cancelled && setState({ path, data, error: null }))
        .catch((e) => {
          if (cancelled) return;
          const error = e instanceof Error ? e.message : String(e);
          setState((prev) => ({ path, data: prev.path === path ? prev.data : null, error }));
        });
    void tick();
    const timer = intervalMs
      ? setInterval(() => {
          if (document.visibilityState === "visible") void tick();
        }, intervalMs)
      : undefined;
    const onVisible = () => {
      if (document.visibilityState === "visible") void tick();
    };
    document.addEventListener("visibilitychange", onVisible);
    return () => {
      cancelled = true;
      if (timer) clearInterval(timer);
      document.removeEventListener("visibilitychange", onVisible);
    };
  }, [path, intervalMs]);

  const current = state.path === path;
  // Not fetched on this mount yet, but seen before: show the cached copy.
  const cached = !current && path !== null && cache.has(path);
  return {
    data: current ? state.data : cached ? (cache.get(path) as T) : null,
    error: current ? state.error : null,
    loading: path !== null && !current && !cached,
    refresh,
  };
}

/** A string persisted in localStorage for this browser only (never on the server). */
export function useLocalSetting(key: string): [string, (value: string) => void] {
  const subscribe = useCallback(
    (notify: () => void) => {
      const onStorage = (e: StorageEvent) => {
        if (e.key === key) notify();
      };
      window.addEventListener("storage", onStorage);
      window.addEventListener(`local-setting:${key}`, notify);
      return () => {
        window.removeEventListener("storage", onStorage);
        window.removeEventListener(`local-setting:${key}`, notify);
      };
    },
    [key],
  );
  const value = useSyncExternalStore(
    subscribe,
    () => {
      try {
        return localStorage.getItem(key) ?? "";
      } catch {
        return "";
      }
    },
    () => "",
  );
  const set = useCallback(
    (next: string) => {
      try {
        localStorage.setItem(key, next);
      } catch {
        /* storage unavailable (private mode) — the value just won't persist */
      }
      window.dispatchEvent(new Event(`local-setting:${key}`));
    },
    [key],
  );
  return [value, set];
}
