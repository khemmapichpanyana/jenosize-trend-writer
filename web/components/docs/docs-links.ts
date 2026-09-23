import { BookOpen, Cpu, FlaskConical, Rocket, Sparkles } from "lucide-react";

export const DOCS_LINKS = [
  { href: "/docs", label: "Overview", hint: "What this is, how to use it", icon: BookOpen },
  { href: "/docs/how-it-works", label: "How it works", hint: "Architecture & data pipeline", icon: Cpu },
  { href: "/docs/fine-tuning", label: "Fine-tuning", hint: "Model choice & training", icon: FlaskConical },
  { href: "/docs/results", label: "Results & evidence", hint: "Evaluation, adapters, weights", icon: Sparkles },
  { href: "/docs/api", label: "API & deployment", hint: "Endpoints, how to test", icon: Rocket },
];
