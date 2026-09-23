import { PageHeader } from "@/components/ui";
import { DocsNav } from "@/components/docs/docs-nav";

export default function DocsLayout({ children }: LayoutProps<"/docs">) {
  return (
    <>
      <PageHeader
        title="Docs"
        subtitle="How it works, how it was fine-tuned, and the evidence."
      />
      <div className="grid gap-5 lg:grid-cols-[13rem_minmax(0,1fr)]">
        <aside className="lg:sticky lg:top-6 lg:h-fit">
          <DocsNav />
        </aside>
        <div className="min-w-0 rounded-xl border border-line bg-surface-1 px-5 py-6 shadow-[0_6px_20px_rgba(7,19,38,0.03)] sm:px-7">
          {children}
        </div>
      </div>
    </>
  );
}
