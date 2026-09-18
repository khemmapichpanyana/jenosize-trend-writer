"use client";

import { useCallback, useEffect, useState, useSyncExternalStore } from "react";
import { get } from "./api";

interface PollState<T> {
  path: string | null;
  data: T | null;
  error: string | null;
}

/**
 * Fetch a studio path now and every `intervalMs`, pausing while the tab is hidden.
 * State is tagged with the path it belongs to, so switching paths never shows
 * the previous path's data and "loading" is derived rather than set in an effect.
 */
export function usePoll<T>(path: string | null, intervalMs = 5000) {
  const [state, setState] = useState<PollState<T>>({ path: null, data: null, error: null });

  const refresh = useCallback(async () => {
    if (!path) return;
    try {
      const data = await get<T>(path);
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
      get<T>(path)
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
    return () => {
      cancelled = true;
      if (timer) clearInterval(timer);
    };
  }, [path, intervalMs]);

  const current = state.path === path;
  return {
    data: current ? state.data : null,
    error: current ? state.error : null,
    loading: path !== null && !current,
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
