"use client";

import { useRouter } from "next/navigation";
import { MessageSquare, Plus } from "lucide-react";
import { CommandDialog, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList, CommandShortcut } from "@/components/ui/command";
import { ago } from "@/lib/format";
import type { Thread } from "@/lib/types";

/** ⌘K palette: search conversations or start a new one. */
export function ConversationPalette({
  open,
  onOpenChange,
  threads,
  currentId,
  onNew,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  threads: Thread[] | null;
  currentId: string;
  onNew: () => void;
}) {
  const router = useRouter();
  return (
    <CommandDialog open={open} onOpenChange={onOpenChange} title="Conversations" description="Search conversations or start a new one">
      <CommandInput placeholder="Search conversations…" />
      <CommandList>
        <CommandEmpty>No matching conversations.</CommandEmpty>
        <CommandGroup>
          <CommandItem
            value="new conversation"
            onSelect={() => {
              onOpenChange(false);
              onNew();
            }}
          >
            <Plus /> New conversation
          </CommandItem>
        </CommandGroup>
        <CommandGroup heading="Recent">
          {(threads ?? []).map((t) => (
            <CommandItem
              key={t.id}
              value={`${t.title} ${t.id}`}
              onSelect={() => {
                onOpenChange(false);
                router.push(`/studio/${t.id}`);
              }}
            >
              <MessageSquare />
              <span className="min-w-0 truncate">{t.title}</span>
              {t.id === currentId ? <CommandShortcut>current</CommandShortcut> : <CommandShortcut>{ago(t.updated_at)}</CommandShortcut>}
            </CommandItem>
          ))}
        </CommandGroup>
      </CommandList>
    </CommandDialog>
  );
}
