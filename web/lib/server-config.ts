import "server-only";

/** Server-side configuration. Importing this from a client component fails the build. */
export function studioConfig(): { url: string; key: string } {
  const url = process.env.STUDIO_API_URL?.replace(/\/+$/, "");
  const key = process.env.STUDIO_API_KEY;
  if (!url || !key) {
    throw new Error("STUDIO_API_URL and STUDIO_API_KEY must be set (see .env.example)");
  }
  return { url, key };
}
