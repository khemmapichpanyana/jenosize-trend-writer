import { Prose, Callout } from "@/components/docs/prose";
import { CodeBlock } from "@/components/docs/code-block";
import { DocsPager } from "@/components/docs/docs-pager";

export const metadata = { title: "API & deployment" };

export default function ApiPage() {
  return (
    <>
    <Prose>
      <h1>API &amp; deployment</h1>
      <p>
        The article generator is a FastAPI service; the chat agent you use in this console calls the same
        writer model through one of its own tools. Both are reachable directly for testing.
      </p>

      <h2>Endpoints</h2>
      <table>
        <thead><tr><th>Method &amp; path</th><th>What it does</th></tr></thead>
        <tbody>
          <tr><td><code>GET /api/v1/health</code></td><td>Liveness check</td></tr>
          <tr><td><code>POST /api/v1/model/warmup</code></td><td>Best-effort wake for a scaled-to-zero GPU</td></tr>
          <tr><td><code>POST /api/v1/articles</code></td><td>Topic/parameters in → a generated article, blocking</td></tr>
          <tr><td><code>POST /api/v1/articles/stream</code></td><td>Same generation, as Server-Sent Events (<code>status | token | heartbeat | result | error</code>)</td></tr>
          <tr><td><code>GET /docs</code></td><td>Interactive OpenAPI docs for the running instance</td></tr>
        </tbody>
      </table>

      <h2>Run it locally — 4 commands, zero cloud accounts</h2>
    </Prose>
    <CodeBlock>{`uv sync --group dev          # 1. install (Python 3.12)
cp .env.example .env         # 2. defaults are already the zero-account mode
make dev                     # 3. reads PORT from .env (defaults to 8777)
./scripts/smoke_test.sh      # 4. health + generate + stream, in another shell`}</CodeBlock>
    <Prose>
      <p>This runs with <code>MODEL_PROVIDER=mock</code> — a deterministic stand-in that exercises the full
        pipeline (normalize → retrieve → generate → quality-check → persist) without a GPU, an API key, or an
        account of any kind. That is a deliberate design choice, not a shortcut: it means the orchestration
        code is exercised identically on a laptop, in CI, and in production.</p>

      <h2>Generate an article</h2>
    </Prose>
    <CodeBlock>{`curl -s localhost:8777/api/v1/articles -H 'Content-Type: application/json' -d '{
  "topic": "Agentic AI in Southeast Asian retail",
  "category": "Futurist",
  "industry": "ecommerce",
  "audience": "C-suite executives",
  "keywords": ["agentic ai", "retail media", "personalization"],
  "length": "short"
}' | jq '{title, quality_report}'`}</CodeBlock>
    <Prose>
      <h2>Stream it (SSE)</h2>
    </Prose>
    <CodeBlock>{`curl -N localhost:8777/api/v1/articles/stream -H 'Content-Type: application/json' \\
  -d '{"topic":"The future of embedded finance","industry":"fintech","length":"short"}'`}</CodeBlock>
    <Prose>
      <h2>Test against the real fine-tuned model</h2>
      <p>
        With Modal secrets configured (<code>make modal-secrets</code>) and <code>.env</code> pointed at the
        deployed vLLM server, the same endpoints call <code>jeno-lora</code> for real instead of the mock:
      </p>
    </Prose>
    <CodeBlock>{`set -a; source .env; set +a
curl -sS $JOBS_BASE_URL/v1/adapters -H "X-API-Key: $JOBS_API_KEY" | jq   # lists every trained version
curl -sS -X POST $JOBS_BASE_URL/v1/studio/warmup -H "X-API-Key: $JOBS_API_KEY"`}</CodeBlock>
    <Prose>
      <h2>Deployment model</h2>
      <ul>
        <li><strong>Article API</strong> — CPU only, deployable to a serverless Python target (e.g. Vercel).
          No ML dependency is allowed in its bundle; CI enforces this with a grep.</li>
        <li><strong>Jobs/Studio API + vLLM server</strong> — deployed to Modal with a single command:
          <code> modal deploy modal/deploy.py</code>. This provisions the FastAPI jobs/studio app, the
          scale-to-zero vLLM GPU server, and the training/eval GPU functions in one shot.</li>
        <li><strong>This console</strong> — a Next.js app; its server-side <code>/api/studio/*</code> route
          proxies every browser request to the jobs/studio API and injects the API key, so the key never
          reaches the browser bundle.</li>
      </ul>

      <Callout tone="note" title="Auth model">
        The product has no end-user accounts yet (it&rsquo;s a single-tenant console, not a multi-tenant product):
        Postgres row-level security denies everything except the server-side key, and an optional
        <code> X-API-Key</code> guards the write routes. The console itself is behind one shared login
        (<code>CONSOLE_USER</code> / <code>CONSOLE_PASSWORD</code>); published <code>/p/</code> pages stay public.
      </Callout>

      <h2>Scaling &amp; optimisation notes</h2>
      <ul>
        <li>The vLLM server defaults to <code>--enforce-eager</code> (skips CUDA-graph compilation) to favour
          faster cold starts over peak throughput — flip it once a warm, continuously-running container makes
          throughput the priority instead.</li>
        <li><code>VLLM_MAX_MODEL_LEN</code> can be lowered from its 8192-token default when prompts never need
          the longer retrieval context, trading context length for faster startup and lower memory.</li>
        <li>Every trained adapter version is registered simultaneously in vLLM, so switching the active model
          is an alias change, not a redeploy or a new container.</li>
        <li>Generation retries transient <code>502</code>/<code>503</code>/<code>504</code> responses with
          bounded backoff, so a cold GPU start degrades to extra latency rather than a hard failure.</li>
      </ul>
    </Prose>
    <DocsPager current="/docs/api" />
    </>
  );
}
