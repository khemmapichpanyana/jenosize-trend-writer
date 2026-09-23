import type { ReactNode } from "react";

/**
 * Long-form doc typography. No @tailwindcss/typography — just enough child
 * selectors to read comfortably, at the same restrained scale as the rest of
 * the console (no oversized editorial headings).
 */
export function Prose({ children, className = "" }: { children: ReactNode; className?: string }) {
  return (
    <div
      className={`max-w-[62ch] text-[13.5px] leading-[1.7] text-ink-2
        [&>h1]:mb-3 [&>h1]:mt-0 [&>h1]:text-[22px] [&>h1]:font-semibold [&>h1]:tracking-[-0.02em] [&>h1]:text-ink
        [&>h2]:mb-2.5 [&>h2]:mt-8 [&>h2]:text-[16px] [&>h2]:font-semibold [&>h2]:tracking-[-0.01em] [&>h2]:text-ink [&>h2]:first:mt-0
        [&>h3]:mb-2 [&>h3]:mt-6 [&>h3]:text-[14px] [&>h3]:font-semibold [&>h3]:text-ink
        [&>p]:my-3 [&>p]:leading-[1.7]
        [&>ul]:my-3 [&>ul]:list-disc [&>ul]:space-y-1.5 [&>ul]:pl-5
        [&>ol]:my-3 [&>ol]:list-decimal [&>ol]:space-y-1.5 [&>ol]:pl-5
        [&_li]:leading-[1.6]
        [&_strong]:font-semibold [&_strong]:text-ink
        [&_code]:rounded [&_code]:bg-surface-2 [&_code]:px-1.5 [&_code]:py-0.5 [&_code]:font-mono [&_code]:text-[12px] [&_code]:text-ink
        [&_a]:font-medium [&_a]:text-accent-ink [&_a]:underline [&_a]:underline-offset-2
        [&>blockquote]:my-4 [&>blockquote]:border-l-2 [&>blockquote]:border-accent [&>blockquote]:pl-3.5 [&>blockquote]:text-ink-2
        [&_table]:my-4 [&_table]:w-full [&_table]:border-collapse [&_table]:text-[12.5px] max-sm:[&_table]:block max-sm:[&_table]:overflow-x-auto
        [&_th]:border-b [&_th]:border-line [&_th]:py-1.5 [&_th]:pr-3 [&_th]:text-left [&_th]:font-semibold [&_th]:text-ink
        [&_td]:border-b [&_td]:border-line/70 [&_td]:py-1.5 [&_td]:pr-3 [&_td]:align-top
        [&>pre]:my-4 [&>pre]:overflow-x-auto [&>pre]:rounded-lg [&>pre]:border [&>pre]:border-line [&>pre]:bg-surface-2 [&>pre]:p-3 [&>pre]:font-mono [&>pre]:text-[12px] [&>pre]:leading-[1.6] [&>pre]:text-ink
        [&>hr]:my-8 [&>hr]:border-line
        ${className}`}
    >
      {children}
    </div>
  );
}

/** A callout box for evidence/notes that should stand out from the prose flow. */
export function Callout({ tone = "note", title, children }: { tone?: "note" | "evidence" | "warn"; title: string; children: ReactNode }) {
  const styles = {
    note: "border-line bg-surface-2/60",
    evidence: "border-good/25 bg-good/5",
    warn: "border-warning/30 bg-warning/8",
  }[tone];
  return (
    <div className={`my-4 max-w-[62ch] rounded-lg border px-3.5 py-3 text-[12.5px] leading-6 text-ink-2 ${styles}`}>
      <p className="mb-1 font-semibold text-ink">{title}</p>
      {children}
    </div>
  );
}
