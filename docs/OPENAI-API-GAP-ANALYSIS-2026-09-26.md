# OpenAI API gap analysis — 2026-09-26

Answers board item 71 (CireSnave, relayed by the PM): *"Lightbulb should support the entirety of
the OpenAI API. That includes handling tool calls in some fashion whether that is internally or by
passing the tool calls themselves to something like OverMind (likely the better option)."*

**Gap analysis only. No code written.** Method: OpenAI's own current API reference, read today
(`developers.openai.com/api/reference/`, 2026-09-26 — this is the provider's live reference; "the
entirety" is dated because it is a moving target, and this list will rot exactly like every other
undated capability claim this portfolio has been burned by). Cross-checked against Lightbulb's
actual route table and request/response structs on `main` at `85f0d60f6fe94e47fb34818b95de1c2d2331d421`,
not from memory or the earlier compat checklist.

## Endpoint-level surface: what OpenAI's API comprises today, dated 2026-09-26

Full resource index (`developers.openai.com/api/reference/llms.txt`), grouped by whether Lightbulb
has *any* route for it:

**Lightbulb has a route:**
- Chat Completions (`/v1/chat/completions`) — has a route, missing most parameters (below)
- Completions (`/v1/completions`, legacy) — has a route, missing most parameters (below)
- Models (`GET /v1/models` per spec) — Lightbulb registers this **`POST`-only** (`src/api/openai/mod.rs:19`), a real endpoint-shape defect independent of this audit (already known, see PR #95's recorded defect #1)

**Lightbulb has zero route, confirmed by reading the whole route table (`src/api/openai/mod.rs`, `src/api/mod.rs`) — only 3 OpenAI-shaped routes exist plus `/health`:**
- **Responses** — OpenAI's own current recommendation for new integrations ("supported indefinitely" applies to Chat Completions, but Responses is what OpenAI steers new work toward). Absent entirely.
- **Embeddings** (`/v1/embeddings`) — absent.
- **Images** (generate, edit, variation) — absent.
- **Audio** (speech/TTS, transcriptions, translations, voices, voice consents) — absent.
- **Files**, **Uploads** — absent.
- **Batch** — absent.
- **Fine-tuning** (jobs, checkpoints, permissions) — absent.
- **Moderations** — absent.
- **Vector Stores** (create/search/file batches) — absent.
- **Evals**, **Graders** — absent.
- **Containers** (code-execution sandboxes) — absent.
- **Conversations** (stateful, backs the Responses API) — absent.
- **Realtime** (WebRTC/WebSocket/SIP voice, calls) — absent.
- **Webhooks** — absent.
- **Administration/Organization/Projects** (users, invites, roles, API keys, audit logs — account management, not inference) — absent; almost certainly out of scope for what "the entirety of the OpenAI API" means for an inference server, flagging rather than assuming.
- **Beta**: Agents, Chatkit, Assistants/Threads/Runs (older stateful agent API, largely superseded by Responses+Conversations) — absent.
- **Live** (fork/primary/sideband WebSocket, appears to be a newer real-time variant) — absent.

**Scale, stated plainly:** this is roughly 20 resource groups, most with 4-10 sub-methods each. "The
entirety of the OpenAI API" is not a small target — it is closer to a full clone of OpenAI's public
surface, most of which (audio, images, realtime voice, fine-tuning, vector stores) has nothing to do
with what a single local text-generation model can actually produce. **This list is the size CireSnave
needs to see before sequencing it himself, per the PM's instruction — not a recommendation on my
part about which of these matter.**

## Chat Completions: every request parameter, present / absent / ignored

The one endpoint audited to full parameter depth, since CireSnave named tool calls specifically and
chat completions is where they live. Current OpenAI parameter list (same read, dated 2026-09-26)
against `src/api/openai/chat.rs`'s `ChatCompletionRequest` (lines ~19-49) and `completions.rs`'s
equivalent:

| Parameter | OpenAI has it | Lightbulb chat/completions | Lightbulb /v1/completions | Status |
|---|---|---|---|---|
| `model` | yes | field, used | field, used | present, real |
| `messages` | yes | field, used | n/a (`prompt` instead) | present, real |
| `temperature` | yes | field, used (confirmed live: temp 0.0 → deterministic output, measured 2026-09-24) | field, **declared, never read** | ⚠️ present-but-ignored on `/v1/completions` |
| `max_tokens`/`max_completion_tokens` | yes (renamed, `max_tokens` legacy) | field, used | field, used | present, real |
| `stream` | yes | field, used (real SSE path, `create_chat_stream`) | absent | present on chat, absent on completions |
| `stream_options` | yes | absent | absent | absent |
| `n` | yes | **field declared (`chat.rs:42`), never read anywhere — response `choices` is always `vec![single]` regardless of value** | **same defect, `completions.rs:34`** | ⚠️⚠️ **present-but-ignored, both endpoints** |
| `stop` | yes | **field declared (`chat.rs:46`), never read anywhere — no stop-sequence enforcement at all** | **same defect, `completions.rs:38`** | ⚠️⚠️ **present-but-ignored, both endpoints** |
| `top_p` | yes | **absent from struct entirely** | **field declared (`completions.rs:31`), never read** | absent on chat; present-but-ignored on completions |
| `presence_penalty` / `frequency_penalty` | yes | absent | absent | absent |
| `logit_bias` | yes | absent | absent | absent |
| `logprobs` / `top_logprobs` | yes | absent from request | request: absent; response field exists but is hardcoded `None` always (`completions.rs:142,181`) | absent as input on both; completions' response field is a permanent null, not a real absence but not real output either |
| `seed` | yes | absent | absent | absent |
| `tools` | yes | **absent — accepted-and-silently-dropped, per last night's compat checklist and design doc (confirmed still true tonight, unchanged)** | absent | ⚠️ **the item this whole board ruling is about** |
| `tool_choice` | yes | absent | absent | absent |
| `parallel_tool_calls` | yes | absent | absent | absent |
| `functions`/`function_call` (deprecated) | yes, deprecated | absent | absent | absent, and correctly so — no reason to build a deprecated surface |
| `response_format` (structured outputs / JSON mode) | yes | absent as the standard parameter — Lightbulb has its **own**, non-standard mechanism (`lightbulb.output_contract`, `chat.rs:86-89`) that does something similar (wrap prompt, parse, retry) but is not what a standard OpenAI client sends | absent | absent (standard shape); a parallel bespoke feature exists, worth knowing about before building the real thing so the two don't collide |
| `service_tier`, `store`, `metadata`, `prediction`, `prompt_cache_key`/`options`/`retention`, `safety_identifier`, `user`, `modalities`, `audio`, `verbosity`, `reasoning_effort`, `web_search_options`, `moderation` | yes (current spec) | absent | absent | absent — none of these are ignored, they're simply not in the struct, so serde silently drops unknown JSON fields (verified behavior, not assumed: confirmed last night that sending an unknown field like `tools` doesn't error, consistent with how all of these behave) |
| `echo` (completions-only, not in chat) | yes (completions-only) | n/a | **field, used** (`completions.rs:167`) | present, real — the one legacy-endpoint parameter that IS implemented |

**Ranked finding, per the PM's instruction that ignored-and-accepted parameters rank above new
surface area:** `n` and `stop` are accepted-and-ignored on **both** endpoints, and `top_p` is
accepted-and-ignored on `/v1/completions`. A caller sending `stop: ["</s>"]` or `n: 3` gets a 200
response that looks correct and is silently wrong — no error, no truncation at the boundary. This
is worse than the tool-calling absence in one specific way: tool-calling absence is at least
*consistent* (nothing ever comes back), while `n`/`stop`/`top_p` looks like it worked. **Recommend
fixing these before or alongside any new surface area**, per the instruction that this class of
defect ranks first.

## Response shape

`ChatCompletionResponse`/`ChatMessage` (chat.rs) has no `tool_calls` field anywhere — confirmed
again tonight, unchanged since last night's design doc and PR #95's audit. `FinishReason`
(`model_runner.rs:17-19`) still has exactly two variants (`Stop`, `Length`) — OpenAI's spec also
defines `tool_calls`, `content_filter`, and `function_call` (deprecated) as finish reasons, none
constructible today.

## Tool-call seam: what OverMind can actually accept today, verified directly

Per instruction, asked OverMind's lane directly rather than designing against an assumed capability.
**Answer, from OverMind's own lane (`b8rktmxy`), re-verified against their code the same message,
not from memory:**

> The total block OverMind reported is **not** in its receiving/consumption path. `ProviderClient.chat()`
> builds its result generically from `choices[0].message` for any provider, with zero vendor-specific
> gating (`providers.py:461-462`); `agent.py:310` reads `result.tool_calls` the same way regardless of
> which provider answered; `_parse_arguments` tolerates the `arguments` field as either a JSON string
> or a dict; `executor.execute()` — "THE ONLY PATH TO AN EFFECT" — runs inside the per-call loop,
> reachable whenever `calls` is non-empty, from any provider. **If Lightbulb emitted a real response
> shaped `{"choices":[{"message":{"role":"assistant","content":null,"tool_calls":[{"id":"call_1",
> "type":"function","function":{"name":"...","arguments":"{...json...}"}}]}}]}`, OverMind's existing
> code — zero changes on their side — would receive it, execute the tool, and continue correctly.**

This means **the seam is exactly the standard OpenAI tool-calling response shape and nothing more**
— no OverMind-specific protocol, no separate channel, no additional negotiation. The entire design
question is producing that shape correctly on Lightbulb's side (the hard part, per last night's
`docs/TOOL-CALLING-DESIGN-2026-09-25.md`: one parser per model family for extracting a tool call out
of generated text, since there is no universal wire format for how a model expresses a tool call in
its own output). OverMind's lane was explicit that this is "code reading says yes," not "measured" —
they have not received a real Lightbulb tool-calling response, only traced what would happen. A real
round-trip test is the natural first integration proof once Lightbulb can emit the shape at all.

**This also directly answers CireSnave's stated lean.** Passing tool calls out to OverMind (or
anything else speaking this same standard shape) costs Lightbulb *nothing extra* beyond emitting the
spec-correct response — there is no OverMind-specific adapter to build. Building an internal executor
instead would be the *more* expensive path: Lightbulb would need its own permission model, its own
tool registry, its own execution sandboxing — all of which OverMind already has. The externalizing
lean is the cheaper one, confirmed rather than assumed.

## Sizing and sequencing, for CireSnave to decide, not a recommendation from this lane

1. **Cheapest, ranks first per the ignored-parameter rule:** fix `n`, `stop`, `top_p` being silently
   dropped. Small, contained, no design questions, done before or alongside anything else.
2. **The named item — tool calls out to OverMind:** per last night's design doc (still accurate, this
   session's audit found nothing invalidating it): a `tools` context key into the chat template, one
   hand-written parser per model family to read a tool call back out of generated text (the real cost,
   no generic shortcut), the `FinishReason::ToolCalls` variant landing with its first real producer, not
   alone. Streaming's tool-call delta shape is a **second, separate design gap** last night's audit
   found and this one confirms is still unaddressed — OpenAI's streamed tool_calls arrive incrementally
   (first delta: `id`+`type`+`function.name`; later deltas append to `function.arguments`), a
   genuinely different producer than the non-streaming path.
3. **Everything else in the endpoint list** is new surface area with no existing partial
   implementation to build from — embeddings, images, audio, files, batch, etc. Each is its own
   project-sized decision (does a local Llama-shape text model even produce embeddings/images/audio in
   any form Lightbulb could serve?), which is why this document stops at listing them rather than
   sizing each one — that's real work this document intentionally does not do, per the instruction to
   show the actual size rather than a plan built on this lane's guess at priorities.

## What this lane did NOT do

No code. No implementation of the tool-call seam. No internal tool executor (explicitly not built,
per CireSnave's stated lean toward externalizing — building one "just in case" was the specific
mistake the task called out to avoid). No sizing of the ~17 absent resource groups beyond naming
them — that sequencing decision is named as CireSnave's to make once he can see the list.
