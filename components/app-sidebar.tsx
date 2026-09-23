"use client";

import Image from "next/image";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useState, type ReactNode } from "react";
import {
  Activity,
  BookMarked,
  Bot,
  ChevronDown,
  Database,
  FileText,
  House,
  Layers3,
  LoaderCircle,
  Menu,
  MessageSquareText,
  PanelLeft,
  Plus,
  Search,
  X,
} from "lucide-react";
import { ConversationPalette } from "@/components/conversations";
import { ModelStatusControl, ModelStatusIcon } from "@/components/model-status";
import { LoadingState } from "@/components/loading-state";
import { invalidate, useLocalSetting, usePoll } from "@/lib/hooks";
import { createThread } from "@/lib/thread";
import type { Thread } from "@/lib/types";

const PRIMARY = [
  { href: "/", label: "Home", icon: House },
  { href: "/studio", label: "Studio", icon: MessageSquareText },
  { href: "/content", label: "Articles", icon: FileText },
  { href: "/docs", label: "Docs", icon: BookMarked },
];
const PIPELINE = [
  { href: "/system", label: "System", icon: Activity },
  { href: "/data", label: "Data", icon: Database },
  { href: "/training", label: "Training", icon: Layers3 },
  { href: "/models", label: "Models", icon: Bot },
];

/** Anything (e.g. the Studio title) can open the ⌘K palette with this event. */
export const OPEN_PALETTE_EVENT = "jenosize:open-palette";
export function openPalette() {
  window.dispatchEvent(new Event(OPEN_PALETTE_EVENT));
}

function isActive(pathname: string, href: string) {
  return href === "/" ? pathname === "/" : pathname === href || pathname.startsWith(`${href}/`);
}

export function AppSidebar() {
  const pathname = usePathname();
  const router = useRouter();
  const [collapsedSetting, setCollapsed] = useLocalSetting("jenosize.sidebar.collapsed");
  const collapsed = collapsedSetting === "1";
  const [mobileOpen, setMobileOpen] = useState(false);
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [startingNew, setStartingNew] = useState(false);
  const threads = usePoll<Thread[]>("/studio/threads?limit=40", 30_000);
  const currentThread = pathname.startsWith("/studio/") ? pathname.split("/")[2] : "";

  useEffect(() => {
    const open = () => setPaletteOpen(true);
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setPaletteOpen((o) => !o);
      }
    };
    window.addEventListener(OPEN_PALETTE_EVENT, open);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener(OPEN_PALETTE_EVENT, open);
      window.removeEventListener("keydown", onKey);
    };
  }, []);

  async function newConversation() {
    if (startingNew) return;
    setStartingNew(true);
    try {
      const t = await createThread();
      invalidate("/studio/threads");
      setMobileOpen(false);
      router.push(`/studio/${t.id}`);
    } finally {
      setStartingNew(false);
    }
  }

  const body = (isCollapsed: boolean, close?: () => void) => (
    <SidebarBody
      collapsed={isCollapsed}
      pathname={pathname}
      threads={threads.data}
      threadsLoading={threads.loading}
      currentThread={currentThread}
      startingNew={startingNew}
      onNew={() => void newConversation()}
      onSearch={() => setPaletteOpen(true)}
      onNavigate={close}
      onToggle={() => setCollapsed(isCollapsed ? "" : "1")}
    />
  );

  return (
    <>
      {/* Desktop */}
      <aside
        className={`sticky top-0 hidden h-screen shrink-0 flex-col border-r border-line/80 bg-surface-1 transition-[width] duration-250 ease-[cubic-bezier(0.22,1,0.36,1)] lg:flex ${collapsed ? "w-[3.25rem]" : "w-60"}`}
        aria-label="Primary"
      >
        {body(collapsed)}
      </aside>

      {/* Mobile: slim top bar + drawer */}
      <div className="sticky top-0 z-40 flex h-12 items-center gap-2 border-b border-line/80 bg-surface-1/95 px-3 backdrop-blur lg:hidden">
        <button type="button" onClick={() => setMobileOpen(true)} className="flex size-8 items-center justify-center rounded-md text-ink-2 hover:bg-surface-2" aria-label="Open menu">
          <Menu size={17} />
        </button>
        <Link href="/" aria-label="Jenosize AI Content home">
          <Image src="/jenosize-logo.svg" alt="Jenosize" width={147} height={37} unoptimized className="h-[22px] w-auto" />
        </Link>
        <div className="ml-auto">
          <ModelStatusControl compact />
        </div>
      </div>
      {mobileOpen && (
        <div className="fixed inset-0 z-50 lg:hidden">
          <button type="button" aria-label="Close menu" className="absolute inset-0 bg-ink/25" onClick={() => setMobileOpen(false)} />
          <aside className="t-reveal absolute inset-y-0 left-0 flex w-72 flex-col border-r border-line bg-surface-1 shadow-xl">
            <button type="button" onClick={() => setMobileOpen(false)} className="absolute right-2 top-2.5 flex size-8 items-center justify-center rounded-md text-ink-2 hover:bg-surface-2" aria-label="Close menu">
              <X size={16} />
            </button>
            {body(false, () => setMobileOpen(false))}
          </aside>
        </div>
      )}

      <ConversationPalette
        open={paletteOpen}
        onOpenChange={setPaletteOpen}
        threads={threads.data}
        currentId={currentThread}
        onNew={() => void newConversation()}
      />
    </>
  );
}

function SidebarBody({
  collapsed,
  pathname,
  threads,
  threadsLoading,
  currentThread,
  startingNew,
  onNew,
  onSearch,
  onNavigate,
  onToggle,
}: {
  collapsed: boolean;
  pathname: string;
  threads: Thread[] | null;
  threadsLoading: boolean;
  currentThread: string;
  startingNew: boolean;
  onNew: () => void;
  onSearch: () => void;
  onNavigate?: () => void;
  onToggle: () => void;
}) {
  const pipelineActive = PIPELINE.some((l) => isActive(pathname, l.href));
  const [pipelineOpen, setPipelineOpen] = useState(pipelineActive);
  const showPipeline = pipelineOpen || pipelineActive;

  return (
    <>
      {/* Brand + collapse */}
      <div className={`flex h-13 shrink-0 items-center ${collapsed ? "justify-center" : "justify-between pl-3.5 pr-2"}`}>
        {!collapsed && (
          <Link href="/" onClick={onNavigate} aria-label="Jenosize AI Content home" className="transition-opacity hover:opacity-80">
            <Image src="/jenosize-logo.svg" alt="Jenosize" width={147} height={37} priority unoptimized className="h-[24px] w-auto" />
          </Link>
        )}
        <button type="button" onClick={onToggle} className="hidden size-8 items-center justify-center rounded-md text-muted transition-colors hover:bg-surface-2 hover:text-ink lg:flex" aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"} title={collapsed ? "Expand sidebar" : "Collapse sidebar"}>
          <PanelLeft size={16} />
        </button>
      </div>

      {/* New + search */}
      <div className={`space-y-0.5 ${collapsed ? "px-1.5" : "px-2"}`}>
        <Row
          collapsed={collapsed}
          icon={startingNew ? <LoaderCircle size={15} className="animate-spin" /> : <Plus size={15} />}
          label="New conversation"
          onClick={onNew}
          className="bg-accent-wash font-semibold text-accent-ink hover:bg-accent/15"
        />
        <Row collapsed={collapsed} icon={<Search size={15} />} label="Search" onClick={onSearch} trailing={<kbd className="rounded border border-line px-1 text-[10px] text-muted">⌘K</kbd>} />
      </div>

      {/* Sections */}
      <nav className={`mt-3 space-y-0.5 ${collapsed ? "px-1.5" : "px-2"}`} aria-label="Sections">
        {PRIMARY.map((l) => (
          <NavRow key={l.href} collapsed={collapsed} href={l.href} label={l.label} icon={<l.icon size={15} />} active={isActive(pathname, l.href)} onNavigate={onNavigate} />
        ))}
        {collapsed ? (
          PIPELINE.map((l) => <NavRow key={l.href} collapsed href={l.href} label={l.label} icon={<l.icon size={15} />} active={isActive(pathname, l.href)} onNavigate={onNavigate} />)
        ) : (
          <>
            <button
              type="button"
              onClick={() => setPipelineOpen((o) => !o)}
              aria-expanded={showPipeline}
              className="flex w-full items-center gap-2.5 rounded-lg px-2.5 py-1.5 text-[13px] text-ink-2 transition-colors hover:bg-surface-2 hover:text-ink"
            >
              <ChevronDown size={15} className={`transition-transform ${showPipeline ? "" : "-rotate-90"}`} />
              Pipeline
            </button>
            {showPipeline && (
              <div className="t-reveal ml-3 space-y-0.5 border-l border-line/80 pl-1.5">
                {PIPELINE.map((l) => (
                  <NavRow key={l.href} collapsed={false} href={l.href} label={l.label} icon={<l.icon size={14} />} active={isActive(pathname, l.href)} onNavigate={onNavigate} />
                ))}
              </div>
            )}
          </>
        )}
      </nav>

      {/* Conversations */}
      {collapsed ? (
        <div className="flex-1" />
      ) : (
        <div className="mt-4 flex min-h-0 flex-1 flex-col">
          <p className="px-4.5 pb-1 text-[11px] font-medium text-muted">Conversations</p>
          <div className="min-h-0 flex-1 space-y-px overflow-y-auto px-2 pb-2">
            {threads?.length ? (
              threads.map((t) => {
                const active = t.id === currentThread;
                return (
                  <Link
                    key={t.id}
                    href={`/studio/${t.id}`}
                    onClick={onNavigate}
                    aria-current={active ? "page" : undefined}
                    title={t.title}
                    className={`block truncate rounded-lg px-2.5 py-1.5 text-[13px] transition-colors ${active ? "bg-surface-2 font-medium text-ink" : "text-ink-2 hover:bg-surface-2 hover:text-ink"}`}
                  >
                    {t.title}
                  </Link>
                );
              })
            ) : threadsLoading ? (
              <div className="px-2.5"><LoadingState label="Loading conversations" rows={5} /></div>
            ) : (
              <p className="px-2.5 py-2 text-xs text-muted">No conversations yet.</p>
            )}
          </div>
        </div>
      )}

      {/* Footer */}
      <div className={`flex shrink-0 items-center border-t border-line/80 ${collapsed ? "justify-center py-2" : "justify-between px-3.5 py-2.5"}`}>
        {collapsed ? (
          <ModelStatusIcon />
        ) : (
          <>
            <span className="text-[11px] text-muted">Model</span>
            <ModelStatusControl />
          </>
        )}
      </div>
    </>
  );
}

function Row({ collapsed, icon, label, onClick, trailing, className = "" }: { collapsed: boolean; icon: ReactNode; label: string; onClick: () => void; trailing?: ReactNode; className?: string }) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={collapsed ? label : undefined}
      aria-label={collapsed ? label : undefined}
      className={`flex w-full items-center rounded-lg text-[13px] text-ink-2 transition-colors hover:bg-surface-2 hover:text-ink ${collapsed ? "justify-center py-2" : "gap-2.5 px-2.5 py-1.5"} ${className}`}
    >
      {icon}
      {!collapsed && <span className="min-w-0 flex-1 truncate text-left">{label}</span>}
      {!collapsed && trailing}
    </button>
  );
}

function NavRow({ collapsed, href, label, icon, active, onNavigate }: { collapsed: boolean; href: string; label: string; icon: ReactNode; active: boolean; onNavigate?: () => void }) {
  return (
    <Link
      href={href}
      onClick={onNavigate}
      aria-current={active ? "page" : undefined}
      title={collapsed ? label : undefined}
      aria-label={collapsed ? label : undefined}
      className={`flex items-center rounded-lg text-[13px] transition-colors ${collapsed ? "justify-center py-2" : "gap-2.5 px-2.5 py-1.5"} ${
        active ? "bg-surface-2 font-medium text-ink [&_svg]:text-accent-ink" : "text-ink-2 hover:bg-surface-2 hover:text-ink"
      }`}
    >
      {icon}
      {!collapsed && <span className="truncate">{label}</span>}
    </Link>
  );
}
