"use client";

import { usePathname } from "next/navigation";
import type { ReactNode } from "react";

/**
 * The content column next to the sidebar. Most pages sit in a padded,
 * max-width frame; Studio is a full-bleed workspace (chat fills the page,
 * the artifact preview is the only card), so it gets no padding or frame.
 */
export function AppMain({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  if (pathname.startsWith("/studio")) {
    return <main className="min-w-0 flex-1">{children}</main>;
  }
  return (
    <main className="brand-grid min-w-0 flex-1 px-4 py-6 lg:px-8">
      <div className="mx-auto w-full max-w-[1400px]">{children}</div>
    </main>
  );
}
