# Assignment demo deployment

This is the shortest path from the trained adapter to a reviewer-facing demo.
It keeps the backend key server-side and uses the fine-tuned `jeno-lora`
adapter for article writing.

## 1. Deploy the AI services

From `../ai-services`, configure the existing Modal secrets and deploy the
current code (including transient vLLM retries and the studio warmup route):

```bash
uv run modal deploy modal/deploy.py
# Copy the printed vLLM URL (ending in /v1) into ai-services/.env as MODEL_BASE_URL,
# then refresh the worker secret and deploy once more so Studio can use the model:
make modal-secrets
uv run modal deploy modal/deploy.py
```

The command prints two URLs. Set the Jobs API URL as `STUDIO_API_URL` and use
the same `JOBS_API_KEY` value as `STUDIO_API_KEY`. The Modal environment must
contain:

```text
MODEL_PROVIDER=openai_compatible
MODEL_BASE_URL=https://<workspace>--jenosize-trend-writer-vllmserver.us-east.modal.direct/v1
MODEL_NAME=jeno-lora
MODEL_API_KEY=<same value as VLLM_API_KEY>
AGENT_MODEL=Qwen/Qwen3-4B-Instruct-2507
JOBS_API_KEY=<server-only key>
# Optional: enables the agent's server-side GPT Image 2 tool.
OPENAI_API_KEY=<server-only key>
IMAGE_MODEL=gpt-image-2
```

`MODEL_NAME` is the adapter served by vLLM. Do not put the key in a browser
environment variable. `min_containers=0` remains the default; the backend
retries transient cold-start failures, and opening Studio sends a best-effort
warmup request while the user is reading the empty state.

When a brief asks for an original hero image, the agent calls the optional
`generate_image` tool, stores the returned image as a private thread asset, and
passes its id to `design_page`. The OpenAI key stays in the Modal secret and is
never exposed to the console.

## 2. Configure and run the web console

```bash
cp .env.example .env.local
```

Set these values in `.env.local` (server-only variables are intentionally not
prefixed with `NEXT_PUBLIC_`):

```text
STUDIO_API_URL=https://<workspace>--jenosize-trend-writer-jobs-api.modal.run
STUDIO_API_KEY=<same JOBS_API_KEY as Modal>
```

Then verify the production bundle locally:

```bash
npm run lint && npm run build
npm run start
```

Open `http://localhost:3030/studio`. The console asks for one shared login
(HTTP Basic auth in `proxy.ts`, from `CONSOLE_USER` / `CONSOLE_PASSWORD`),
since it can start GPU jobs. In production it refuses to serve at all if
`CONSOLE_PASSWORD` is unset. The public
`/p/<slug>` share pages are meant to stay accessible. The first browser creates
one demo thread and stores only that thread id in local storage, so separate
browsers do not accidentally merge conversations.

## 3. Reviewer smoke test

1. Open `/studio` and authenticate.
2. Ask for a Jenosize trend article with a topic, audience, keywords, and
   target length.
3. Confirm that tool events and the article stream appear in the workspace.
4. Inspect the artifact preview, then explicitly click **Publish**.
5. Open the generated `/p/<slug>` link in a private window.

The v2 held-out evaluation measured a 76.9% quality-gate pass rate versus
69.2% for base, but that is a one-example difference across 13 briefs. Only
five blind-judge verdicts were parseable (3 v2 wins, 2 base wins), and v2 had
lower SEO keyword coverage. Keep these limitations in the report rather than
claiming general superiority.

## What is intentionally not automated here

Real multi-user accounts and organization-level row ownership require a chosen
Supabase Auth project, JWT verification, and a migration/RLS rollout. The
demo has no console login and uses browser-scoped threads. Do
not expose the service key or claim tenant isolation until those external
credentials and policies are configured.
