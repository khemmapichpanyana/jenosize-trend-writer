import { studioConfig } from "@/lib/server-config";

/** Images of published pages (the backend refuses anything unpublished). */
export async function GET(_request: Request, ctx: RouteContext<"/p/assets/[id]">) {
  const { id } = await ctx.params;
  const { url } = studioConfig();
  const upstream = await fetch(`${url}/p/assets/${encodeURIComponent(id)}`, { cache: "no-store" });
  const headers = new Headers();
  for (const name of ["content-type", "cache-control"]) {
    const value = upstream.headers.get(name);
    if (value) headers.set(name, value);
  }
  return new Response(upstream.body, { status: upstream.status, headers });
}
