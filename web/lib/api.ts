/** Browser-side client for the studio API, always via this app's /api/studio proxy. */

export const API = "/api/studio/v1";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly code?: string,
    readonly body?: unknown,
  ) {
    super(message);
  }
}

async function parse<T>(response: Response): Promise<T> {
  const text = await response.text();
  const body = text ? safeJson(text) : null;
  if (!response.ok) {
    const error = (body as { error?: { message?: string; code?: string } } | null)?.error;
    throw new ApiError(error?.message ?? `HTTP ${response.status}`, response.status, error?.code, body);
  }
  return body as T;
}

function safeJson(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

export async function get<T>(path: string): Promise<T> {
  return parse<T>(await fetch(`${API}${path}`, { cache: "no-store" }));
}

export async function post<T>(path: string, body?: unknown): Promise<T> {
  return parse<T>(
    await fetch(`${API}${path}`, {
      method: "POST",
      headers: body === undefined ? undefined : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    }),
  );
}

export async function del(path: string): Promise<void> {
  await parse(await fetch(`${API}${path}`, { method: "DELETE" }));
}

export async function upload<T>(path: string, form: FormData): Promise<T> {
  return parse<T>(await fetch(`${API}${path}`, { method: "POST", body: form }));
}

export async function getText(path: string): Promise<string> {
  const response = await fetch(`${API}${path}`, { cache: "no-store" });
  if (!response.ok) throw new ApiError(`HTTP ${response.status}`, response.status);
  return response.text();
}
