import Link from "next/link";
import { ArrowLeft, ArrowRight } from "lucide-react";
import { DOCS_LINKS } from "./docs-links";

export function DocsPager({ current }: { current: string }) {
  const index = DOCS_LINKS.findIndex((l) => l.href === current);
  const prev = index > 0 ? DOCS_LINKS[index - 1] : null;
  const next = index >= 0 && index < DOCS_LINKS.length - 1 ? DOCS_LINKS[index + 1] : null;
  if (!prev && !next) return null;

  return (
    <nav aria-label="Docs pages" className="mt-10 flex max-w-[62ch] items-stretch gap-3 border-t border-line pt-5">
      {prev ? (
        <Link href={prev.href} className="group flex flex-1 flex-col rounded-lg border border-line px-3 py-2.5 transition-colors hover:border-accent/35 hover:bg-accent-wash">
          <span className="inline-flex items-center gap-1 text-[10.5px] font-medium text-muted"><ArrowLeft size={11} /> Previous</span>
          <span className="mt-0.5 text-[13px] font-semibold text-ink group-hover:text-accent-ink">{prev.label}</span>
        </Link>
      ) : <div className="flex-1" />}
      {next ? (
        <Link href={next.href} className="group flex flex-1 flex-col items-end rounded-lg border border-line px-3 py-2.5 text-right transition-colors hover:border-accent/35 hover:bg-accent-wash">
          <span className="inline-flex items-center gap-1 text-[10.5px] font-medium text-muted">Next <ArrowRight size={11} /></span>
          <span className="mt-0.5 text-[13px] font-semibold text-ink group-hover:text-accent-ink">{next.label}</span>
        </Link>
      ) : <div className="flex-1" />}
    </nav>
  );
}
