import type { Metadata } from "next";
import { Suspense } from "react";
import { Geist, Geist_Mono, Inter } from "next/font/google";
import { AppMain } from "@/components/app-main";
import { AppSidebar } from "@/components/app-sidebar";
import { WarmupOnVisit } from "@/components/warmup";
import { TooltipProvider } from "@/components/ui/tooltip";
import "./globals.css";
import { cn } from "@/lib/utils";

const inter = Inter({subsets:['latin'],variable:'--font-sans'});

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  title: { default: "Jenosize AI Content", template: "%s · Jenosize AI Content" },
  description: "Console for the Jenosize content agent: data, fine-tuning, models, studio and publishing.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={cn("h-full", "antialiased", geistSans.variable, geistMono.variable, "font-sans", inter.variable)}>
      <body className="min-h-full">
        <TooltipProvider>
        <WarmupOnVisit />
        {/* App shell: left sidebar (drawer + slim top bar on mobile), content to the right. */}
        <div className="min-h-screen bg-surface-0 lg:flex">
          <Suspense>
            <AppSidebar />
          </Suspense>
          <AppMain>{children}</AppMain>
        </div>
        </TooltipProvider>
      </body>
    </html>
  );
}
