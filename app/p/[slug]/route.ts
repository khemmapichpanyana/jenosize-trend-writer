import { studioConfig } from "@/lib/server-config";

/** Public share link: the published page, served from this domain. */
export async function GET(_request: Request, ctx: RouteContext<"/p/[slug]">) {
  const { slug } = await ctx.params;
  const { url } = studioConfig();
  const upstream = await fetch(`${url}/p/${encodeURIComponent(slug)}`, { cache: "no-store" });
  const headers = new Headers({ "Content-Type": "text/html; charset=utf-8" });
  for (const name of ["content-security-policy", "cache-control"]) {
    const value = upstream.headers.get(name);
    if (value) headers.set(name, value);
  }
  return new Response(upstream.body, { status: upstream.status, headers });
}
