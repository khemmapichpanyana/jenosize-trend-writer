"use client";

import { useRouter } from "next/navigation";
import { useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";
import { ErrorNote } from "@/components/ui";
import { LoadingState } from "@/components/loading-state";
import { get } from "@/lib/api";
import { openOrCreateThread } from "@/lib/thread";

function StudioIndex() {
  const router = useRouter();
  const search = useSearchParams();
  const [status, setStatus] = useState<"starting" | "ready" | "error">("starting");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function openDemoThread() {
      try {
        const artifact = search.get("artifact");
        if (artifact) {
          const owner = await get<{ thread_id: string }>(`/studio/artifacts/${artifact}`);
          if (!cancelled) router.replace(`/studio/${owner.thread_id}?artifact=${artifact}`);
          return;
        }

        const thread = await openOrCreateThread();
        if (!cancelled) router.replace(`/studio/${thread.id}`);
      } catch (e) {
        if (cancelled) return;
        setStatus("error");
        setError(e instanceof Error ? e.message : String(e));
      }
    }

    void openDemoThread();
    return () => {
      cancelled = true;
    };
  }, [router, search]);

  return (
    <div className="mx-auto max-w-3xl px-4 py-6">
      <ErrorNote message={error} />
      {status === "starting" && <LoadingState label="Opening your latest conversation" rows={4} />}
    </div>
  );
}

export default function StudioPage() {
  return (
    <Suspense>
      <StudioIndex />
    </Suspense>
  );
}
