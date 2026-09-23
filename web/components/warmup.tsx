"use client";

import { usePathname } from "next/navigation";
import { useEffect } from "react";
import { ensureModelWarm } from "@/lib/model-status";

/** Best-effort serverless model warmup; deduped, so route changes don't pile up requests. */
export function WarmupOnVisit() {
  const pathname = usePathname();

  useEffect(() => {
    // Fire-and-forget: a cold GPU must never block the page shell.
    ensureModelWarm();
  }, [pathname]);

  return null;
}
