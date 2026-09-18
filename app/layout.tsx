import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import Link from "next/link";
import { Nav } from "@/components/nav";
import "./globals.css";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  title: { default: "Jenosize AI Content", template: "%s · Jenosize AI Content" },
  description: "Console for the Jenosize content agent: data, fine-tuning, models, studio and publishing.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}>
      <body className="min-h-full">
        <div className="flex min-h-screen flex-col md:flex-row">
          <aside className="border-b border-line bg-surface-1 p-3 md:sticky md:top-0 md:h-screen md:w-56 md:shrink-0 md:border-b-0 md:border-r md:p-4">
            <Link href="/" className="mb-0 flex items-center gap-2 px-2 pb-3 md:mb-6 md:pb-0">
              <span className="text-lg font-extrabold tracking-tight text-ink">
                Jeno<span className="text-brand">size</span>
              </span>
              <span className="rounded bg-accent-wash px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-accent-ink">
                AI Content
              </span>
            </Link>
            <Nav />
          </aside>
          <main className="min-w-0 flex-1 px-4 py-6 md:px-8 md:py-8">{children}</main>
        </div>
      </body>
    </html>
  );
}
