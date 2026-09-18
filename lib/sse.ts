/**
 * Server-sent events over fetch.
 *
 * `EventSource` only does GET and can't be aborted cleanly mid-turn; the agent
 * chat needs POST, so both streams go through this one reader.
 */

export type SseHandler = (event: string, data: unknown) => void;

export async function streamSse(
  url: string,
  init: RequestInit,
  onEvent: SseHandler,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(url, { ...init, signal, cache: "no-store" });
  if (!response.ok || !response.body) {
    let message = `HTTP ${response.status}`;
    try {
      const body = await response.json();
      message = body?.error?.message ?? message;
    } catch {
      /* not JSON */
    }
    throw new Error(message);
  }
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;
    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const frame = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      let event = "message";
      const data: string[] = [];
      for (const line of frame.split("\n")) {
        if (line.startsWith("event: ")) event = line.slice(7);
        else if (line.startsWith("data: ")) data.push(line.slice(6));
      }
      if (data.length) {
        const raw = data.join("\n");
        let parsed: unknown = raw;
        try {
          parsed = JSON.parse(raw);
        } catch {
          /* plain text payload */
        }
        onEvent(event, parsed);
      }
      boundary = buffer.indexOf("\n\n");
    }
  }
}
