import { get, post } from "./api";
import type { Thread, ThreadDetail } from "./types";

/** The browser's "current" conversation. New tabs/visits resume it until the user starts a new one. */
export const DEMO_THREAD_KEY = "jenosize.demo.thread-id";

/** The backend replaces this with the first brief, so conversations name themselves. */
const PLACEHOLDER_TITLE = "New chat";

/** Create a fresh conversation and make it the browser's current one. */
export async function createThread(title = PLACEHOLDER_TITLE): Promise<Thread> {
  const thread = await post<Thread>("/studio/threads", { title });
  try {
    window.localStorage.setItem(DEMO_THREAD_KEY, thread.id);
  } catch {
    /* storage unavailable (private mode) — the thread still works, just isn't pinned */
  }
  return thread;
}

/** Resume the browser's pinned conversation, or start one if there isn't a live one. */
export async function openOrCreateThread(): Promise<Thread> {
  let existing: string | null = null;
  try {
    existing = window.localStorage.getItem(DEMO_THREAD_KEY);
  } catch {
    /* ignore */
  }
  if (existing) {
    try {
      return await get<ThreadDetail>(`/studio/threads/${existing}`);
    } catch {
      try {
        window.localStorage.removeItem(DEMO_THREAD_KEY);
      } catch {
        /* ignore */
      }
    }
  }
  return createThread();
}
