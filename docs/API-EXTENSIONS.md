# Lightbulb-specific API extensions

Lightbulb's `/v1/chat/completions` and `/v1/completions` are OpenAI-compatible. This document
lists the extensions that are **not** part of that compatibility surface -- deliberately kept
apart from the request/response JSON schemas documented on those structs, so that a stock
OpenAI client (one that only knows the standard fields) is never affected by any of them.

## `X-Lightbulb-Priority` request header (M4 slice 5.4c, Option B)

**What it does:** sets the admission priority of the request against Lightbulb's
`MemoryAwareScheduler`. Under memory pressure (the scheduler's `eviction_pressure_threshold`
exceeded), a request below `Priority::High` can be rejected outright rather than queued; a
`High` or `Critical` request in the identical state is queued instead of rejected. Below that
threshold, priority currently has no observable effect -- see
`MemoryAwareScheduler::allocate_pending_requests` (`src/engine/memory_aware_scheduler.rs`) for
the exact rule.

**How to send it:** an HTTP header on the request, not a field in the JSON body:

```
POST /v1/chat/completions
X-Lightbulb-Priority: high
Content-Type: application/json

{"model": "...", "messages": [...]}
```

**Accepted values** (case-insensitive): `low`, `normal`, `high`, `critical`. Absent, empty, or
any other value defaults to `normal` -- silently, never a 400. See
`crate::api::openai::chat::priority_from_headers` for the parser.

**Why a header and not a body field:** OpenAI's chat/completions schema has no priority field at
all. A non-standard top-level JSON field (e.g. `{"priority": "high", ...}`) would be silently
dropped by a stock OpenAI client that never sends it (harmless), but risks a strict
client-side schema validator rejecting an *unknown field on a response that echoes the request
back* -- none of Lightbulb's response shapes do that today, but relying on "probably fine" is
weaker than a checked guarantee. A header never touches the request body schema at all, so it
carries zero compatibility risk for any OpenAI-shaped client or validator, at the cost of being
less discoverable than a body field. See `ROADMAP.md`'s M4 slice plan, slice 5.4c, for the full
tradeoff writeup and the PM's ruling (2026-10-21: Option B).

## `lightbulb.output_contract` (and the rest of the `lightbulb` request object)

A top-level `lightbulb` object on the `/v1/chat/completions` JSON body (NOT a header) carrying
Lightbulb's structured-output contract extension (`output_contract`, `max_attempts`,
`fallback_contracts`). See `ChatCompletionRequest::lightbulb`
(`src/api/openai/chat.rs`) for the field definitions. Listed here only to make clear it is a
*different* kind of extension (body field, not header) from the one above -- it predates this
document and is not itself new.
