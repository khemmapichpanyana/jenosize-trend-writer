import { timingSafeEqual } from "node:crypto";
import { NextResponse, type NextRequest } from "next/server";

/**
 * HTTP Basic auth in front of the console.
 *
 * The console can start GPU training and publish content, so it fails closed in
 * production when no password is configured. Shared pages (/p/*) stay public —
 * that is the point of sharing them.
 */
export function proxy(request: NextRequest) {
  const password = process.env.CONSOLE_PASSWORD;
  const user = process.env.CONSOLE_USER ?? "jenosize";
  if (!password) {
    if (process.env.NODE_ENV === "production") {
      return new NextResponse("Console is locked: set CONSOLE_PASSWORD.", { status: 503 });
    }
    return NextResponse.next();
  }
  const header = request.headers.get("authorization") ?? "";
  const [scheme, encoded] = header.split(" ");
  if (scheme === "Basic" && encoded) {
    const [givenUser, ...rest] = Buffer.from(encoded, "base64").toString("utf8").split(":");
    if (safeEqual(givenUser, user) && safeEqual(rest.join(":"), password)) {
      return NextResponse.next();
    }
  }
  return new NextResponse("Authentication required.", {
    status: 401,
    headers: { "WWW-Authenticate": 'Basic realm="Jenosize console", charset="UTF-8"' },
  });
}

function safeEqual(a: string, b: string): boolean {
  const left = Buffer.from(a);
  const right = Buffer.from(b);
  // Compare equal-length buffers so the timing doesn't leak the length either.
  const same = left.length === right.length;
  return timingSafeEqual(same ? left : right, right) && same;
}

export const config = {
  // Everything except static assets and public share pages.
  matcher: ["/((?!_next/static|_next/image|favicon.ico|icon.png|apple-icon.png|jenosize-icon.png|jenosize-logo.svg|p/).*)"],
};
