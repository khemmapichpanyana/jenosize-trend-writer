# Jenosize AI Content — console

Part of the [Jenosize Trend Writer monorepo](../README.md).

The operator console for the Jenosize Trend Writer. From here you can:
- pull and review the article corpus
- fine-tune the model and **watch training live**, with the loss curve and GPU gauges
- manage and evaluate the adapters
- **chat with the content agent**: it writes with the fine-tuned model, lays the
  article out as a Jenosize-branded page with your images, and you publish it to
  a shareable link

The chat surface uses AI Elements for the conversation, markdown responses,
collapsible agent steps, and attachment-aware prompt input. The transport stays
on the existing resumable Studio SSE API, so the UI does not need a second chat
backend or an exposed model key.

All AI and data work happens in the Python backend (`../ai-services`: FastAPI,
LangChain, Modal). This app is a thin, typed UI plus a server-side proxy.

```
Browser ──► this app (Next.js 16)
              ├─ pages: /, /data, /training, /runs/[id], /models, /studio, /content
              ├─ /api/studio/*   proxy → studio API; adds the key, streams SSE through
              └─ /p/[slug]       public share links (+ /p/assets/[id] images)
                       │
                       ▼
            studio API on Modal (ai-services: pipeline/api + studio/)
              ├─ LangChain agent: orchestrator on Modal vLLM, fallback LLM
              │    └─ write_article → the fine-tuned jeno-lora (streams into the artifact panel)
              ├─ jobs: scrape · label · datasets · train (L4) · eval · publish to HF
              └─ Postgres (Supabase) + Cloudflare R2 (private bucket)
```

## Run it

```bash
npm install
cp .env.example .env.local     # STUDIO_API_URL, STUDIO_API_KEY
npm run dev                    # http://localhost:3030
```

**No accounts at all:** start the backend with its mocks, then point this app
at it:

```bash
# in ../ai-services (Postgres needed; see its README for a disposable local one)
MODEL_PROVIDER=mock AGENT_PROVIDER=mock JOBS_API_KEY=dev make jobs-dev   # :8001
# here
STUDIO_API_URL=http://localhost:8001 STUDIO_API_KEY=dev npm run dev
```

| Variable | Where | Purpose |
|---|---|---|
| `STUDIO_API_URL` | server | The studio API (`make deploy-modal` prints it) |
| `STUDIO_API_KEY` | server | = `JOBS_API_KEY`. Never sent to the browser |

Set `PUBLIC_SHARE_BASE_URL` on the backend to this app's public URL, so shared
links point at `https://<console>/p/<slug>`.

## What's where

| Path | What it does |
|---|---|
| `app/api/studio/[...path]/route.ts` | Backend-for-frontend proxy: adds `X-API-Key`, forwards only `/v1/*`, streams bodies both ways (uploads and SSE) |
| `app/p/[slug]`, `app/p/assets/[id]` | Public share routes; the backend decides what is published |
| `components/chat-workspace.tsx` | Chat plus artifact panel: live tokens, tool steps, the article streaming in, versions, preview, publish |
| `components/run-live.tsx` | Live run view: SSE with resume (`after_id`), loss and learning-rate charts, GPU meters, ETA, cancel |
| `components/line-chart.tsx` | Hand-built SVG line chart (no chart library) |
| `lib/sse.ts` | SSE over `fetch`, because `EventSource` can't POST, and chat needs POST |

## Design notes

- **UI foundation is intentionally small.** The shadcn/ui base-nova preset provides
  the accessible primitives, while the console keeps its Jenosize teal tokens and
  uses the transitions.dev `t-stagger` and `t-shimmer` patterns for brief reveals
  and live-writing states. Both motion patterns include reduced-motion fallbacks.
- **Charts follow a validated spec.** The series colour is a teal step of the
  Jenosize brand accent (`#0a93a3` light, `#0aa3b3` dark), checked for contrast,
  lightness and chroma on each surface. The brand's own `#00bcce` is only 2.3:1
  on white, so it's used for brand moments, not data. Charts have one axis, a
  2px line, an end label, a crosshair that also works by keyboard, and a table
  view. Status colours are reserved and always paired with an icon and label.
- **Agent output is untrusted.** The backend sanitises page HTML to an
  allowlist, and published pages carry a no-script CSP. The preview iframe
  allows no scripts (`allow-same-origin` only, so preview images load through
  the authenticated proxy).
- **Publishing is a person's decision.** The agent can write and design, but
  only the Publish button makes a page public.
- **Brand context is a placeholder.** Edit `ai-services/studio/brand/jenosize.md`
  (voice and layout rules the agent follows) and `theme.css` (colours and type)
  once the official guidelines are available.

## Checks

```bash
npm run lint && npm run build
```

For the assignment deployment variables, Modal rollout command, and reviewer
smoke test, see [`DEPLOYMENT.md`](DEPLOYMENT.md). The phase status and the
explicit multi-user-auth boundary are documented in [`PHASES.md`](PHASES.md).
