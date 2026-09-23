# Jenosize AI Content Web phases

## Phase 1 — assignment demo (complete)

Goal: demonstrate the real fine-tuned writer and agent without exposing backend
secrets or pretending this is a multi-tenant product.

- Next.js console with HTTP Basic auth in production.
- One persisted `thread_id` per browser session; conversations are not merged.
- Server-side `/api/studio/*` proxy adds `X-API-Key`.
- Studio runs use the real `/v1/studio` API and stream events to the UI.
- The backend is configured with `MODEL_NAME=jeno-lora` so article writing uses
  the active LoRA adapter.

Acceptance check:

1. Open `/studio`.
2. The browser creates or resumes one demo thread.
3. Send a request and see tool events plus the generated article.
4. The API key is absent from browser requests and network-visible JavaScript.

## Phase 2 — real accounts and organization separation

- Add Supabase Auth (or the selected identity provider).
- Add `owner_id` and `org_id` to threads, messages, assets, artifacts and runs.
- Enforce ownership in every query and with database RLS.
- Replace the single demo-thread redirect with a per-user thread list.

Status: intentionally deferred for the assignment. The current demo has
server-side Basic auth and browser-scoped thread isolation. Supabase Auth/JWT
and RLS need a real project configuration; see `DEPLOYMENT.md`.

## Phase 3 — production reliability (implemented)

- Add model warmup on high-intent UI entry, not anonymous landing-page loads.
- Retry transient vLLM cold-start `502/503/504` responses in the backend worker.
- Add request/run telemetry, rate limits and user-visible progress states.
- Keep `min_containers=0` by default; schedule a warm pool only when latency
  requirements justify the GPU cost.

The OpenAI-compatible provider now retries transient connection, 408/409/425,
429, and 5xx failures with bounded exponential backoff. Studio also exposes a
guarded warmup route, and the console calls it when a user opens Studio.

## Phase 4 — submission polish (complete in repo; deploy externally)

- Deploy the web console and API with separate secret boundaries.
- Verify the fine-tuned model path end-to-end from the public UI.
- Add a short report linking the data pipeline, training metrics, evaluation
  result and known limitations.
- Capture a clean demo path and public share link for the assignment.

See `DEPLOYMENT.md` for the exact Modal/web environment and reviewer smoke test.
