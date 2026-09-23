"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { DOCS_LINKS } from "./docs-links";

export function DocsNav() {
  const pathname = usePathname();
  return (
    <nav aria-label="Docs" className="-mx-1 flex gap-1 overflow-x-auto px-1 pb-1 lg:mx-0 lg:flex-col lg:gap-0.5 lg:overflow-visible lg:px-0 lg:pb-0">
      {DOCS_LINKS.map((link) => {
        const active = link.href === "/docs" ? pathname === "/docs" : pathname.startsWith(link.href);
        const Icon = link.icon;
        return (
          <Link
            key={link.href}
            href={link.href}
            aria-current={active ? "page" : undefined}
            className={`flex shrink-0 items-start gap-2.5 rounded-lg px-2.5 py-2 text-[13px] transition-colors ${
              active ? "bg-accent-wash text-accent-ink" : "text-ink-2 hover:bg-surface-2 hover:text-ink"
            }`}
          >
            <Icon size={14} className="mt-0.5 shrink-0" strokeWidth={active ? 2.2 : 1.8} aria-hidden />
            <span className="min-w-0">
              <span className="block font-semibold">{link.label}</span>
              <span className="hidden text-[11px] text-muted lg:block">{link.hint}</span>
            </span>
          </Link>
        );
      })}
    </nav>
  );
}
