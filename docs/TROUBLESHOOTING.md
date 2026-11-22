# Troubleshooting

What the server answers when something goes wrong, what that means, and what to do. Every message below is quoted
from the source (the file is named under each entry), and every status code is one a test in that file or in
`tests/` exercises. Environment variables are in [`CONFIGURATION.md`](CONFIGURATION.md); not everything that can go
wrong is here, only what has a distinct answer.

**Checked against the source of `lane-2026-11-22-logprobs` on 2026-11-22; not re-run against a release.**

## A request is refused before any generation

### 400 `model 'X' is not loaded on this server; the loaded model is 'Y'`

The request's `model` is not a model this server can serve. With a models directory (`LIGHTBULB_MODELS_DIR`) the
name must be the file name of a servable `.gguf` directly under it; without one it must equal the one loaded model.
A name is a single path component: no `/`, no `..`, no leading dot.

Do: `GET /v1/models` lists what the directory holds (`lightbulb_loaded` says which are in memory now). A `.gguf`
the server cannot serve (unreadable, or an architecture it has no loader for) is left out of that list rather than
listed to fail.

Source: `src/api/model_select.rs` (`not_loaded`), `src/api/openai/models.rs`. Test: `tests/gguf_synthetic_models_dir.rs`.

### 503 with `Retry-After: 5`: `not enough device memory for model 'X': needs N bytes, M free, K evictable`

The model is in the directory but loading it needs more device memory than is free, and the models that could be
evicted to make room are all serving a request. Nothing was evicted. Retry after the header's 5 seconds, or wait for
the busy requests to finish. An idle model is evicted (least recently used first) instead of refusing.

Source: `src/api/model_select.rs` (`busy`, `answer`). Test: `no_room_is_a_503_with_retry_after_and_the_busy_model_survives`.

### 503 `inference queue is full; try again shortly` (`"type": "server_busy"`)

The pending-request queue is at its depth. Safe to retry. The depth is `LIGHTBULB_MODEL_MAX_QUEUE_DEPTH` (default 32).
On a streaming request this arrives as a plain 503 before any SSE frame, not as a frame inside a 200.

Source: `src/api/openai/chat.rs`. Test: `chat_completions_returns_503_when_the_inference_queue_is_full`.

### 413

The request body is larger than `LIGHTBULB_MAX_REQUEST_BODY_BYTES` (default 2 MiB). Test: the body-limit tests in `src/api/mod.rs`.

### 400 on `logprobs` / `top_logprobs`

`top_logprobs` needs `logprobs: true`; its range is 0 to 20 (a completions `logprobs: N` has the same range);
`logprobs` is refused with `response_format`, with `lightbulb.output_contract`, and, on a stream, with `tools`;
completions refuses it with `echo`. The text of each refusal says which.

Source: `src/api/openai/logprobs.rs`, `src/api/openai/chat.rs` (`validate_chat_logprobs`).

## A model does not load

### 500 `model load failed: ...`

The runner started but the model could not be built. The text after the colon is the reason; two seen in tests:

- `gguf token_embd.weight: Q8_1 dequantization is unimplemented upstream (GAP-125)`: a GGUF with `Q8_1` tensors.
  This is a Fuel gap, not a configuration problem; use a file quantised to another type (`Q8_0`, `Q4_0` serve).
  Test: `a_q8_1_gguf_is_refused_by_name_not_served`; row B12 in `ROADMAP.md`.
- A preflight refusal, next entry.

### `model needs at least X but the device reports Y ...  (set LIGHTBULB_ALLOW_OVERSIZE_MODEL=1 to load anyway)`

The memory preflight found that the weights cannot fit. On a GPU this refuses the load, because VRAM is a hard
limit; on the CPU it only logs a warning and loads, because free RAM is soft. `LIGHTBULB_ALLOW_OVERSIZE_MODEL`
(see `CONFIGURATION.md` for the accepted spellings) loads it anyway on a GPU.

Source: `src/model_fuel/preflight.rs` (`refuses_on_wont_fit`, `enforce_with`).

### A model that loaded once answers every request with the runner's error

Fixed: a model whose runner thread has exited is unloaded and loaded afresh on the next request, so it works again
once the cause (memory, a replaced file) is gone. If it keeps failing, each request retries the load.

## A request fails during generation

### 502 `tool_choice` or `response_format` not met

For `tool_choice: "required"` or a named function, and for `response_format` (`json_object`, `json_schema`), the
server asks the model by prompt, checks the answer, and regenerates up to 3 attempts in all before answering 502. This is not
constrained decoding: a small model may fail all three. Use a model that follows instructions, or relax the request.

Source: `src/api/openai/chat.rs` (`queue_error_response`, `ToolChoiceUnmet`, `FormatUnmet`).

### 504 on a non-streaming request; an error frame on a stream

`inference request timed out`: the request ran past `LIGHTBULB_REQUEST_TIMEOUT_MS` (default 300000). The job may
still be running, so retrying is not obviously safe. A stream times out per chunk, not per response, and ends with a
frame `{"error": {"message": "inference request timed out", "type": "timeout_error"}}`.

Source: `src/api/openai/chat.rs`. Test: `chat_completions_streaming_times_out_when_the_runner_stalls`.

## 501 on `/v1/lightbulb/{knowledge,reasoning,state,tools}/*`

Reserved and not implemented: every one answers `501 Not Implemented` and never a fake success.

Source: `src/api/lightbulb.rs`. Test: `every_reserved_route_is_501_and_never_a_fake_success`.
