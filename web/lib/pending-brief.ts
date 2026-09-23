// Hand a brief typed on Home to the Studio page that opens next, which sends it
// as the conversation's first message. sessionStorage: this tab only, one use.
const key = (threadId: string) => `jenosize.pending-brief.${threadId}`;

export function setPendingBrief(threadId: string, text: string) {
  try {
    sessionStorage.setItem(key(threadId), text);
  } catch {
    /* storage unavailable — the user can retype it */
  }
}

export function takePendingBrief(threadId: string): string | null {
  try {
    const text = sessionStorage.getItem(key(threadId));
    if (text) sessionStorage.removeItem(key(threadId));
    return text;
  } catch {
    return null;
  }
}
