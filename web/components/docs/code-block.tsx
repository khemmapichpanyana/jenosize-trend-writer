"use client";

import { useState } from "react";
import { Check, Copy } from "lucide-react";

export function CodeBlock({ children, lang = "bash" }: { children: string; lang?: string }) {
  const [copied, setCopied] = useState(false);

  async function copy() {
    try {
      await navigator.clipboard.writeText(children);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard unavailable — the code is still selectable by hand */
    }
  }

  return (
    <div className="my-4 max-w-[62ch] overflow-hidden rounded-lg border border-line">
      <div className="flex items-center justify-between border-b border-line bg-surface-2/70 px-3 py-1.5">
        <span className="text-[10.5px] font-medium uppercase tracking-[0.08em] text-muted">{lang}</span>
        <button
          type="button"
          onClick={() => void copy()}
          className="inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[11px] font-medium text-ink-2 transition-colors hover:bg-surface-1 hover:text-accent-ink"
        >
          {copied ? <Check size={11} className="text-good" /> : <Copy size={11} />}
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <pre className="overflow-x-auto bg-surface-2/40 p-3 font-mono text-[12px] leading-[1.6] text-ink">{children}</pre>
    </div>
  );
}
