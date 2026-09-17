#!/usr/bin/env bash
# End-to-end smoke test against a running deployment.
#
#   ./scripts/smoke_test.sh                              # local (default)
#   ./scripts/smoke_test.sh https://trend-writer.workser.app
#
# Exits non-zero on the first failure so it can gate a deploy in CI.
set -euo pipefail

BASE="${1:-http://localhost:8000}"
API="${BASE%/}/api/v1"
# `set -u` treats an empty array expansion as unbound on bash < 4.4 (macOS ships
# 3.2), so the array always holds at least one harmless element.
KEY_HEADER=(-sS)
[[ -n "${API_KEY:-}" ]] && KEY_HEADER+=(-H "X-API-Key: ${API_KEY}")

pass() { printf '  \033[32mok\033[0m   %s\n' "$1"; }
fail() { printf '  \033[31mFAIL\033[0m %s\n' "$1"; exit 1; }

echo "smoke testing ${BASE}"

# 1. health ------------------------------------------------------------------
health="$(curl -fsS "${API}/health")" || fail "health endpoint unreachable"
echo "${health}" | grep -q '"status"' || fail "health returned no status: ${health}"
status="$(echo "${health}" | sed -n 's/.*"status" *: *"\([a-z]*\)".*/\1/p')"
[[ "${status}" == "ok" || "${status}" == "degraded" ]] || fail "unexpected status: ${health}"
pass "health -> ${status}"
echo "       ${health}"

# 2. openapi -----------------------------------------------------------------
curl -fsS "${BASE%/}/openapi.json" >/dev/null || fail "/openapi.json unreachable"
pass "/openapi.json"

# 3. warmup ------------------------------------------------------------------
warm="$(curl -fsS -X POST "${API}/model/warmup" "${KEY_HEADER[@]}")" || fail "warmup failed"
pass "model/warmup -> ${warm}"

# 4. blocking generation -----------------------------------------------------
body='{"topic":"Smoke test: agentic AI in retail","industry":"ecommerce","keywords":["agentic ai"],"length":"short"}'
article="$(curl -fsS -X POST "${API}/articles" \
  -H 'Content-Type: application/json' "${KEY_HEADER[@]}" -d "${body}")" || fail "generation failed"
echo "${article}" | grep -q '"status":"succeeded"' || fail "generation not succeeded: ${article:0:300}"
id="$(echo "${article}" | sed -n 's/.*"id" *: *"\([0-9a-f-]*\)".*/\1/p')"
pass "POST /articles -> ${id}"

# 5. read back ---------------------------------------------------------------
curl -fsS "${API}/articles/${id}" >/dev/null || fail "could not read generation ${id}"
pass "GET /articles/${id}"

# 6. streaming ---------------------------------------------------------------
# Written to a file rather than piped into `head`: closing the pipe early makes
# curl exit 23, which `pipefail` would report as a stream failure.
sse="$(mktemp)"
trap 'rm -f "${sse}"' EXIT
curl -fN --max-time 300 -X POST "${API}/articles/stream" \
  -H 'Content-Type: application/json' "${KEY_HEADER[@]}" -d "${body}" -o "${sse}" \
  || fail "stream failed"
grep -q 'event: token' "${sse}" || fail "no token events in stream"
grep -q 'event: result' "${sse}" || fail "stream ended without a result event"
pass "POST /articles/stream ($(grep -c 'event: token' "${sse}") token events)"

printf '\n\033[32mall checks passed\033[0m\n'
