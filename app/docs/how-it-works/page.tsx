import { Prose, Callout } from "@/components/docs/prose";
import { DocsPager } from "@/components/docs/docs-pager";

export const metadata = { title: "How it works" };

export default function HowItWorksPage() {
  return (
    <>
    <Prose>
      <h1>How it works</h1>
      <p>
        Two services share one database and one file bucket. The <strong>article API</strong> is CPU-only and
        does the product work: it generates articles and never scrapes, labels, or trains. The{" "}
        <strong>jobs/studio API</strong> does everything GPU-shaped and long-running: scraping, labelling,
        publishing datasets, training LoRA adapters, evaluating them, and running the chat agent.
      </p>
      <pre>{`Browser ──► this console (Next.js)
              ├─ /api/studio/*  proxy → the jobs/studio API (adds the key server-side)
              └─ /p/[slug]      public share links

this console's backend proxy ──► Jobs + Studio API (FastAPI, on Modal)
                                    ├─ scrape → clean → label → build_dataset → train → evaluate
                                    ├─ LangChain agent (write_article, design_page, web_search, ...)
                                    ├─ Postgres (Supabase)   — records
                                    └─ Cloudflare R2         — raw HTML, datasets, assets
                                              ▲
                                    Modal vLLM server (Qwen3-4B-Instruct-2507 + LoRA adapter)
                                    scale-to-zero, OpenAI-compatible

Separately: an article-only FastAPI service (CPU only, deployable to Vercel)
exposes POST /api/v1/articles for direct topic → article generation, using
the same writer model and prompt contract as the agent's write_article tool.`}</pre>

      <h2>Core design principle</h2>
      <p>
        <strong>Fine-tuning teaches style; retrieval and search supply facts.</strong> The adapter is never
        asked to memorise a statistic — it learns Jenosize&rsquo;s tone, structure and the article&rsquo;s output
        contract (title, standfirst, H2 sections, a closing next-steps block). Anything factual has to arrive
        in the prompt as a retrieved passage or a web-search result; the system prompt explicitly forbids
        inventing figures when none were supplied.
      </p>
      <p>
        The payoff: a small 4B base model is enough, LoRA converges in minutes on one GPU, and a wrong fact
        is a retrieval/search problem to fix, not a reason to retrain the model.
      </p>

      <h2>Request flow</h2>
      <p>Both the direct API and the agent&rsquo;s <code>write_article</code> tool run the same pipeline:</p>
      <ol>
        <li><strong>Normalize</strong> — clean and canonicalise the brief (topic, industry, audience, keywords, length).</li>
        <li><strong>Record</strong> — insert a queued row so every generation is tracked, even ones that fail.</li>
        <li><strong>Ground</strong> — if a source URL/document was supplied, fetch and extract it, then retrieve the
          most relevant passages with BM25; the agent can also call <code>web_search</code> first to gather sources.</li>
        <li><strong>Generate</strong> — build the prompt and call the fine-tuned model (streamed as SSE for the
          direct API; streamed as agent tokens in the console).</li>
        <li><strong>Quality-check</strong> — four deterministic checks (word count, H2 section count, SEO keyword
          coverage, title length); one retry with feedback if it fails.</li>
        <li><strong>Persist</strong> — write the markdown to storage and close the row as succeeded or failed.</li>
      </ol>

      <h2>Data engineering pipeline</h2>
      <p>
        The training set is not hand-written — it is built from Jenosize Ideas&rsquo; own published articles, so
        the model learns the brand&rsquo;s real voice rather than an approximation of it.
      </p>
      <ol>
        <li><strong>Discover &amp; scrape</strong> — sitemap discovery, then fetch at roughly one request per
          second, respecting <code>robots.txt</code>; raw HTML is archived so cleaning can be re-run without
          re-scraping.</li>
        <li><strong>Clean</strong> — strip navigation, CTAs, bylines, dates and reference lists; normalise
          headings; reject outliers outside 300–4000 words; mark near-duplicates.</li>
        <li><strong>Reverse-label</strong> — an LLM reconstructs a plausible brief (topic, industry, audience,
          keywords) from each finished article, since the original editorial brief doesn&rsquo;t exist.</li>
        <li><strong>Build the dataset</strong> — each (brief, article) pair is rendered through the exact same
          prompt-builder functions used at inference time, validated against the output contract, and split
          deterministically (SHA-256 of the source URL) into an immutable train/eval set. A quality filter
          then excludes examples that don&rsquo;t meet the article contract before publishing a version.</li>
      </ol>

      <Callout tone="note" title="Handles any business topic, not just the training categories">
        The pipeline&rsquo;s output is a prompt contract (system + rendered brief + article), not a fixed topic
        list — the same builder accepts any industry, audience or keyword set at inference time, which is
        what lets one adapter generalise beyond the ~159 articles it was trained on.
      </Callout>

      <h2>Pluggable backends</h2>
      <p>
        Three seams, each a small interface selected by an environment variable — this is what makes a
        zero-account local run a <em>real</em> end-to-end test rather than a stub:
      </p>
      <table>
        <thead><tr><th>Seam</th><th>Env var</th><th>Options</th></tr></thead>
        <tbody>
          <tr><td>Model</td><td><code>MODEL_PROVIDER</code></td><td><code>mock</code> (deterministic, streams too) · <code>openai_compatible</code> (the Modal vLLM server)</td></tr>
          <tr><td>Persistence</td><td><code>PERSISTENCE</code></td><td><code>none</code> (in-memory) · <code>supabase</code></td></tr>
          <tr><td>Storage</td><td><code>STORAGE</code></td><td><code>local</code> (<code>./.data/</code>) · <code>r2</code></td></tr>
        </tbody>
      </table>

      <h2>Constraints that shaped the design</h2>
      <ul>
        <li><strong>A CPU-only deployment target&rsquo;s bundle-size limit.</strong> No torch/transformers/vllm in the
          article API&rsquo;s dependencies — heavy ML packages are declared only inside the GPU job images, and CI
          greps for violations.</li>
        <li><strong>No background workers on that target.</strong> Generation happens inside the request (with a
          long timeout and SSE) rather than in a queue.</li>
        <li><strong>Scale-to-zero GPU.</strong> Cold starts run 1–3 minutes, so there is an explicit warmup
          endpoint and an SSE heartbeat so proxies don&rsquo;t drop the connection while it starts.</li>
        <li><strong>No end-user auth.</strong> There are no product accounts yet: row-level security denies
          everything except the server-side key, and the console has no login of its own yet — put it behind one before deploying it publicly.</li>
      </ul>
    </Prose>
    <DocsPager current="/docs/how-it-works" />
    </>
  );
}
