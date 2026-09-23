import Link from "next/link";
import { Prose, Callout } from "@/components/docs/prose";
import { DocsPager } from "@/components/docs/docs-pager";

export const metadata = { title: "Overview" };

export default function DocsOverviewPage() {
  return (
    <>
    <Prose>
      <h1>Jenosize Trend Writer</h1>
      <p>
        A content generation tool that takes a topic, industry, audience and SEO keywords, and produces a
        business-trend article in Jenosize Ideas&rsquo; voice — written by a model fine-tuned on Jenosize&rsquo;s own
        published articles, not a generic LLM prompt.
      </p>
      <p>
        Everything here — the chat agent, the data pipeline dashboards, the training runs, the model
        registry and the published-article library — is one Next.js app talking to one Python backend.
        There is no separate admin tool: what you use to review the work is the same surface a real editor
        would use day to day.
      </p>

      <h2>What you can do here</h2>
      <ul>
        <li><strong>Overview</strong> (this app&rsquo;s home page) — a quick chat entry point plus a live pulse of the
          corpus, training jobs and recent conversations.</li>
        <li><strong>Studio</strong> — the full chat workspace. Describe an article or branded page; the agent
          researches (when web search is configured), writes with the fine-tuned model, and can lay the
          result out as a Jenosize-branded page you preview and publish.</li>
        <li><strong>Data</strong> — the scraped/cleaned/labelled article corpus the training set is built from.</li>
        <li><strong>Training</strong> — fine-tuning runs, live loss and GPU telemetry while a job is running.</li>
        <li><strong>Models</strong> — every trained LoRA adapter, its metrics, and which one is currently active.</li>
        <li><strong>Articles</strong> — every article the agent has generated, draft or published, with a link
          back to its conversation.</li>
      </ul>

      <h2>Using the chat agent</h2>
      <p>Open <strong>Studio</strong> and describe what you want, e.g.:</p>
      <blockquote>
        &ldquo;Write a medium article on agentic AI in Southeast Asian retail for C-suite readers.&rdquo;
      </blockquote>
      <p>The agent then, in order:</p>
      <ol>
        <li>researches the web for current facts when the brief needs them (see <Link href="/docs/fine-tuning">Fine-tuning</Link> for how facts and style are kept separate),</li>
        <li>calls the fine-tuned writer model to draft the article — you&rsquo;ll see a &ldquo;writing&rdquo; state while
          this happens, then the finished draft opens in the panel on the right,</li>
        <li>can lay the article out as a branded page with a hero image on request,</li>
        <li>waits for you to hit <strong>Publish</strong> — the agent never publishes on its own.</li>
      </ol>
      <p>
        Use <strong>New</strong> in the Studio header to start a fresh conversation per article, and
        <strong> Conversations</strong> to switch back to an earlier one — every conversation and every
        article it produced stays listed under Overview and Articles.
      </p>

      <Callout tone="note" title="No accounts needed to try it">
        The backend has a zero-account mode (a deterministic mock model and agent) for trying it without
        cloud credentials. See <Link href="/docs/api">API &amp; deployment</Link> for exact commands.
      </Callout>

      <h2>Where to go next</h2>
      <ul>
        <li><Link href="/docs/how-it-works">How it works</Link> — architecture and the data pipeline.</li>
        <li><Link href="/docs/fine-tuning">Fine-tuning</Link> — model choice, dataset, and training setup.</li>
        <li><Link href="/docs/results">Results &amp; evidence</Link> — evaluation numbers, adapters, and weights.</li>
        <li><Link href="/docs/api">API &amp; deployment</Link> — endpoints and how to run it yourself.</li>
      </ul>
    </Prose>
    <DocsPager current="/docs" />
    </>
  );
}
