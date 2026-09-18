"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/", label: "Overview", icon: "◧" },
  { href: "/data", label: "Data", icon: "⛁" },
  { href: "/training", label: "Training", icon: "◔" },
  { href: "/models", label: "Models", icon: "◇" },
  { href: "/studio", label: "Studio", icon: "✎" },
  { href: "/content", label: "Published", icon: "↗" },
];

export function Nav() {
  const pathname = usePathname();
  return (
    <nav className="flex gap-1 overflow-x-auto md:flex-col md:overflow-visible" aria-label="Console">
      {LINKS.map((link) => {
        const active = link.href === "/" ? pathname === "/" : pathname.startsWith(link.href);
        return (
          <Link
            key={link.href}
            href={link.href}
            aria-current={active ? "page" : undefined}
            className={`flex shrink-0 items-center gap-2.5 rounded-lg px-3 py-2 text-sm font-medium transition ${
              active ? "bg-accent-wash text-accent-ink" : "text-ink-2 hover:bg-surface-2 hover:text-ink"
            }`}
          >
            <span aria-hidden className="w-4 text-center">{link.icon}</span>
            {link.label}
          </Link>
        );
      })}
    </nav>
  );
}
