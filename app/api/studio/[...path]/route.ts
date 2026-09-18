import type { NextRequest } from "next/server";
import { studioConfig } from "@/lib/server-config";

/**
 * Backend-for-frontend proxy to the studio API on Modal.
 *
 * The browser calls /api/studio/v1/...; this handler adds the API key and
 * streams the response straight through, so SSE (agent chat, live training)
 * arrives token by token instead of buffered. The key never reaches the client.
 */

// Response headers that describe the upstream connection, not the body we relay.
const HOP_BY_HOP = new Set([
  "connection",
  "keep-alive",
  "transfer-encoding",
  "content-encoding",
  "content-length",
]);

async function forward(request: NextRequest, ctx: RouteContext<"/api/studio/[...path]">) {
  const { path } = await ctx.params;
  // Only the studio API's own routes; encoding each segment blocks traversal.
  if (path[0] !== "v1") {
    return Response.json({ error: { code: "not_found", message: "Unknown route" } }, { status: 404 });
  }
  const { url, key } = studioConfig();
  const target = `${url}/${path.map(encodeURIComponent).join("/")}${request.nextUrl.search}`;

  const headers = new Headers({ "X-API-Key": key });
  for (const name of ["content-type", "accept", "x-request-id"]) {
    const value = request.headers.get(name);
    if (value) headers.set(name, value);
  }
  const hasBody = !["GET", "HEAD"].includes(request.method);
  let upstream: Response;
  try {
    upstream = await fetch(target, {
      method: request.method,
      headers,
      body: hasBody ? request.body : undefined,
      // Required by Node's fetch to stream a request body (multipart uploads).
      ...(hasBody ? { duplex: "half" } : {}),
      cache: "no-store",
      redirect: "manual",
    } as RequestInit);
  } catch (error) {
    const message = error instanceof Error ? error.message : "unreachable";
    return Response.json(
      { error: { code: "upstream_unreachable", message: `Studio API unreachable: ${message}` } },
      { status: 502 },
    );
  }
  const out = new Headers();
  upstream.headers.forEach((value, name) => {
    if (!HOP_BY_HOP.has(name)) out.set(name, value);
  });
  if (out.get("content-type")?.includes("text/event-stream")) {
    out.set("Cache-Control", "no-cache, no-transform");
    out.set("X-Accel-Buffering", "no");
  }
  return new Response(upstream.body, { status: upstream.status, headers: out });
}

export { forward as DELETE, forward as GET, forward as PATCH, forward as POST, forward as PUT };
