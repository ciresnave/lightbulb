# Runner result metadata and `/v1/completions` — design

**Date:** 2026-08-06
**Status:** approved, ready for an implementation plan
**Sibling spec:** chat-template resolution is designed separately
(`2026-08-06-chat-template-resolution-design.md`). Independent; either can land
first.

---

## Summary

Lightbulb's OpenAI-compatible API reports four things it does not know:
`finish_reason`, and all three fields of `usage`. It reports them anyway, with
plausible values. One endpoint returns a hardcoded placeholder with HTTP 200.

All of it traces to one cause: **the model runner's response channel carries a
bare `String`.** There is nowhere to put "how did this end" or "how many
tokens", so the handlers invent both.

This spec widens that channel and wires the dead endpoint.

---

## Context: what is actually wrong

**[verified 2026-08-06 on `main`]**

| # | Defect | Evidence |
| --- | --- | --- |
| 1 | `/v1/completions` returns a hardcoded placeholder with HTTP 200 | `completions.rs:103` takes `_state` and never touches `inference_tx`; `:125` returns *"This is a placeholder completion. Inference engine integration pending."* |
| 2 | `finish_reason` is always `"stop"` | `chat.rs:221`, `:255`, `:376` (non-stream) and `:491` (stream terminal chunk) |
| 3 | `completion_tokens` is always `0` | `chat.rs:225`, `:259` |
| 4 | `prompt_tokens` is a **word count**, not a token count | `chat.rs:192` is `prompt.split_whitespace().enumerate().map(\|(i,_)\| i as u32)` under a `TODO: Use proper tokenizer` |

Defect 2 is the one with client-visible consequences beyond cosmetics. `"stop"`
means "the model finished"; `"length"` means "it was cut off and continuing is
meaningful". Reporting `"stop"` for a truncated generation tells a client not to
continue when it should.

Defect 4 means `usage` is wrong on **every** field, since `total_tokens` is
derived from it.

---

## Scope

**In:** the response type, both endpoints, both response modes, and every
construction site the type change reaches.

**Out:** prompt construction / chat templates (sibling spec). Streaming *usage*
— OpenAI omits `usage` from stream chunks by default, so there is nothing to
correct there; only the terminal `finish_reason` is wrong.

---

## §1 — The response type

```rust
// src/engine/model_runner.rs

/// Why generation stopped. Serialises to OpenAI's vocabulary.
pub enum FinishReason {
    /// The model emitted an end-of-sequence token.
    Stop,
    /// `max_new_tokens` was reached with no EOS. Continuing is meaningful.
    Length,
}

pub struct CompletionResult {
    pub text: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub finish_reason: FinishReason,
}
```

`ResponseMode::Complete` carries `oneshot::Sender<Result<CompletionResult>>`
instead of `Result<String>`.

**Why a struct rather than a tuple or separate channels:** the four values are
produced together by one loop iteration and consumed together by one response
builder. Splitting them invites a caller to have three of the four.

## §2 — Where the values come from

**They are observed, not derived.** This is the point of the change: today the
handler guesses because the runner never tells it.

- **`finish_reason`** — `run_jobs` already distinguishes the two exits. A
  request whose state reaches `Completed` stopped on EOS
  (`ctx.complete()` is called from the EOS check in both backends). A request
  that leaves the loop with `state == Decoding` and
  `tokens_generated >= max_new_tokens` hit the cap. That distinction exists in
  `RequestContext` today and is discarded.

  **These two are exhaustive for a `CompletionResult`, and only because the
  other exits do not produce one.** A mid-decode error, a `decode_text` failure,
  or a dropped receiver leaves the loop via `Err` or `break` and resolves the
  channel to `Err` — so there is no third case needing a `FinishReason`. Stating
  this because the mapping looks partial otherwise, and a future exit path that
  *does* return successfully would need a variant rather than defaulting to
  `Stop`.
- **`completion_tokens`** — `ctx.tokens_generated`, already maintained.
- **`prompt_tokens`** — a new `RequestContext.prompt_tokens: usize`, set by
  whichever backend tokenised the prompt during prefill. Same pattern as the
  `temperature` field. **From the tokenizer the model actually used** — a second
  tokenizer in the API layer would drift from what the model saw, which is how
  `prompt_tokens` becomes wrong in a way nobody notices.

## §3 — Streaming

`ResponseMode::Streaming` carries `UnboundedSender<Result<String>>` — one item
per token, so there is no terminal slot for a finish reason. `chat.rs:491`
therefore emits a hardcoded `"stop"` chunk after the stream ends.

The item type becomes an enum:

```rust
pub enum StreamItem {
    Token(String),
    Done { finish_reason: FinishReason },
}
```

`Done` carries **only** the finish reason. It would be natural to add
`completion_tokens` alongside it, and this spec deliberately does not: OpenAI
omits `usage` from stream chunks by default, so nothing would consume it. A
field that exists because it seemed symmetrical is a field the next reader has
to work out is unused.

`run_jobs` sends `Done` exactly once, when the loop exits, with the same
observed values as the complete path. `chat.rs` builds its terminal chunk from
it instead of assuming.

**Why include streaming rather than defer it:** a streaming client sees the same
lie, and leaving one mode honest and the other not is worse than fixing both —
it makes the API's truthfulness depend on which mode you called.

## §4 — Construction sites

**[verified]** The type change reaches three sites, one of them outside `api/`:

- `src/api/openai/chat.rs:280` — `Complete`
- `src/api/openai/chat.rs:408` — `Streaming`
- `src/contracts/executor.rs:61` — `Complete`

`contracts/executor.rs` is the one worth naming: it is not an HTTP handler and
will not be found by looking at `api/`. It uses only the text, so it takes
`.text` and is otherwise unaffected.

## §5 — `/v1/completions`

`create_completion` takes `state` rather than `_state` and dispatches through
`inference_tx` exactly as `run_inference_once` does. Behaviour:

- **No chat template.** `/v1/completions` is OpenAI's raw-text endpoint; the
  prompt is used as given. Templating belongs to `/v1/chat/completions` and to
  the sibling spec.
- `PromptInput::Multiple` keeps its existing `join("\n")` semantics.
- `echo` keeps prefixing the prompt to the returned text.
- No runner configured → the same informational response the chat path already
  returns for that case, so the two endpoints degrade identically.

## §6 — Error handling

| Condition | Behaviour |
| --- | --- |
| No runner configured | Informational 200 with a body naming the absence, matching chat's existing fallback. Not an error — the server is up, the model is not. |
| Runner returns `Err` | HTTP 500 with the message, as chat does today. |
| Runner channel closed (thread died) | HTTP 500. The `oneshot` resolves to `Err` on drop, so this is already covered by the arm above. |

## §7 — Testing

Each test names the production change that would make it fail.

- **`finish_reason` is `Length` at the cap.** A request with `max_tokens` low
  enough to truncate must report `"length"`. **Fails today** — this is the
  defect. Would fail again if `run_jobs` stopped distinguishing the exits.
- **`finish_reason` is `Stop` on EOS.** A prompt whose greedy continuation
  terminates must report `"stop"`. Guards the inverse error of the above; a fix
  that hardcoded `"length"` would pass the first test and fail this one.
- **`completion_tokens` equals tokens actually generated.** Asserted against a
  known-length greedy generation, not against `> 0`, which `1` would satisfy
  while still being wrong.
- **`prompt_tokens` matches the model's tokenizer.** Compared against
  `tokenizer.encode(prompt)` on the same checkpoint — not against a word count,
  which is the bug.
- **`/v1/completions` returns model output.** Asserts the response does **not**
  contain the placeholder string, the same provenance assertion
  `tests/fuel_engine_http.rs` uses. Without it the test passes when the endpoint
  is still a stub.
- **Streaming's terminal chunk carries the observed reason**, not a constant.

Checkpoint-dependent tests are `#[ignore]`d and **fail rather than skip** when
the checkpoint is absent, consistent with the existing gates. `TINYLLAMA_DIR`
overrides the path.

---

## §8 — Risks

| Risk | Assessment |
| --- | --- |
| **Type change reaches unfound sites** | Three are enumerated above, and the compiler finds the rest — this is a type error, not a silent failure. The risk is low precisely because the change is not clever. |
| **`prompt_tokens` on a cache hit** | `ParallelModelManager`'s prefix-cache path can skip tokenising the full prompt. The count must still reflect the whole prompt, not the uncached remainder, or `usage` under-reports. Set it from the full tokenisation before the cache lookup. |
| **`FinishReason` grows** | OpenAI also defines `content_filter` and `tool_calls`. Not modelled: nothing produces them today, and an enum with unconstructible variants is worse than one that grows when a producer appears. |
| **Streaming enum churn** | Widening `StreamItem` touches the SSE assembly in `chat.rs`. Mechanical, but it is the least-tested path in the file. |

---

## Out of scope

- Chat templates and prompt construction — sibling spec.
- `usage` in streaming chunks — OpenAI omits it by default.
- `logprobs`, `n > 1`, `best_of` — unimplemented today and unrelated to this
  defect class.
- Retiring the `lightbulb_result` extension field on `ChatCompletionResponse`.
