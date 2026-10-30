# What Lightbulb still lacks to replace Ollama and Cherry Studio (census)

**Date:** 2026-10-30. **Ref:** every code claim was read at `origin/main` 7ccf91f8 (lightbulb 0.23.2) unless it says otherwise. **Why:** CireSnave's 2026-10-28 milestone (PM [TASK] 2026-10-30): with dtype and the prefix benchmark done, say what Lightbulb on Fuel still lacks to replace Ollama and Cherry Studio for him.

**Scope and honesty.** The feature list below is the PM's list of what CireSnave uses (model pull/list/run, chat UI protocol and OpenAI-compatible endpoints, multi-model, embeddings, tool/MCP use, system prompts/templates, quant formats, GPU+CPU). I did **not** inspect Ollama or Cherry Studio themselves, so nothing here says what either one sends on the wire; where a gap depends on that, the row says so. "Have" means code plus a test that exercises it was found; "partial" names what is missing; "missing" means a `git grep` for the capability found nothing, with the control named. Nothing in this table was run today except where it says **measured**; the rest is read from source and tests, which show intent and coverage, not that a given model works.

## Table

| # | Feature | Status | Evidence | ROADMAP |
|---|---|---|---|---|
| 1 | List models | **have** | `GET /v1/models` and `GET /v1/models/{model}` list exactly the loaded model, empty when none (`src/api/openai/models.rs`, route `openai/mod.rs:23-27`; tests in `models.rs`). Also reports `lightbulb_memory.weights_bytes_on_disk` (0.23.0). | A1 done (#232, 0.19.21) |
| 2 | Pull / download a model | **missing** | `src/hub.rs` was dead and deleted (#235); no download code remains (`git grep -il "hf_hub\|huggingface\.co" -- src` finds two files: a comment in `model_fuel/loader.rs:4` and the test-only HF cache locator in `src/test_models.rs`; neither downloads anything). Network fetch is parked as a future `mlmf-source-hf` crate (mlmf lane, via PM). Models are files placed in `LIGHTBULB_MODELS_DIR`. | F1 |
| 3 | Run a model | **partial** | The server loads **one** model named by `LIGHTBULB_DEFAULT_MODEL` from `LIGHTBULB_MODELS_DIR` at start (`src/main.rs`, `src/api/mod.rs:731`). There is no `lightbulb run <model>`; `lightbulb-cli` is an HTTP chat client for a running server (`src/bin/lightbulb-cli.rs`). | F2 |
| 4 | Multi-model (several loaded, or switch at runtime) | **missing** | One `ModelRunner` per server; a request for any other id is a 400 "not loaded" (`models.rs` header, `chat.rs`). `git grep -n "load_model\|switch_model\|/models/load" -- src/api` finds nothing (control: `git grep -n "fn loaded_models" -- src/api` finds `models.rs:55`). | F2 |
| 5 | OpenAI-compatible chat (`/v1/chat/completions`), streaming | **have** | Route `openai/mod.rs:21`; SSE at `chat.rs:1710,1751`. | - |
| 6 | OpenAI `/v1/completions` | **have** | Route `openai/mod.rs:22`. | - |
| 7 | OpenAI request parameters Cherry-style clients send | **partial** | `n>1` is a 400 (`validate_n`, `chat.rs:825`); `top_p != 1` is a 400 (per ROADMAP A5; field defaults at `chat.rs:52`); `logprobs` always `None`, no `seed`, `logit_bias`, penalties, or `stream_options.include_usage` (A5). `git grep -n "response_format" -- src/api` finds **nothing**, so standard JSON mode is missing (the `lightbulb_result` extension is not it). | A5 (open) |
| 8 | Embeddings (`/v1/embeddings`) | **missing** | Only three OpenAI routes exist (`openai/mod.rs:21-27`). The PM listed embeddings among the features used, which meets A6's own condition ("embeddings only when the Cherry census needs them"). Needs an embedding model path through Fuel; not scoped here. | A6 (un-defer, see F3) |
| 9 | Tool / function calling, non-streaming | **have (narrow)** | Tools are described to the model and `<tool_call>` replies are parsed into `tool_calls` (`chat.rs:776,971`; unit tests `chat.rs:3841-3895`). Output-format parsing is one family's convention, not per-family. | A4 remainder |
| 10 | Tool calling with `stream: true` | **missing** | `chat.rs:416` returns a 400 "`tools` with `stream: true` is not yet implemented". Cherry-style clients stream by default, so this likely blocks tool use from a chat UI (not verified against Cherry). | A4 |
| 11 | `tool_choice` required / named function | **missing** | 400 today (`validate_tool_choice`, `chat.rs:801`); `auto` and `none` accepted. | A4 |
| 12 | MCP | **n/a at the server (client-side in a chat UI)** | `git grep -il mcp -- src` finds nothing. In a UI like Cherry the UI calls MCP tools and relies on the model endpoint returning OpenAI `tool_calls`, so MCP use depends on rows 9-11. Not verified against Cherry. | via F4 |
| 13 | System prompts / chat templates | **have** | Jinja templates via `minijinja` (`Cargo.toml:39`), resolved per model including the template stored in GGUF metadata (`src/api/chat_template.rs:1229 resolve_for_serving`, `chat_template/registry.rs`); `lightbulb-cli --system`. The `render` entry has no `tools` parameter (A4). | - |
| 14 | Quantized formats (GGUF) | **have, measured by test** | `tests/gguf_e2e_weight_load_census.rs`: `KNOWN_FAILURES` is empty (`:216`); CHANGELOG 0.19.x records 11 of 11 weighted corpus files loading, including IQ-coded files. I did not re-run it today (needs `C:\Models\gguf-corpus` and a long build). Q8_1 is a typed error on Fuel (B12, unverified on 0.25.x). | B12 |
| 15 | Model architectures | **partial** | GGUF: `llama` and `qwen3` only; anything else is refused naming the string (`src/gguf/mod.rs:1742-1747`). SafeTensors: Llama-shaped only (Fuel loaders hard-gate to Llama). Mistral/Gemma/Phi-family checkpoints that are not `llama`-tagged will not load. | F5 |
| 16 | SafeTensors BF16 / F16 weights | **missing** | f32 loader only; waits on Fuel #489 (0.25.13+). | B1 |
| 17 | Sharded checkpoints | **partial** | Sharded SafeTensors done (0.20.4). Sharded GGUF not started; waits on `mlmf-gguf` split helpers (mlmf#187, in PM gate). | B5 (rest) |
| 18 | GPU (CUDA) and CPU | **have** | Features `cuda`/`fuel-cuda` (`Cargo.toml:15,22`), device chosen in `model_fuel/device.rs`; falls back to CPU deliberately. **Measured 2026-10-30:** the CUDA build ran the prefix-sharing bench end to end (exit 0, bit-identical output; #227 comment). | - |
| 19 | Vulkan GPU | **partial** | Feature `vulkan = ["fuel/vulkan"]` (`Cargo.toml:24`); no lightbulb test or measurement exercises it (`git grep -n -i vulkan -- tests` finds two comments, no test). | F6 |
| 20 | Apple Metal GPU | **missing** | No `metal` Cargo feature (`Cargo.toml` features list); Fuel has a Metal backend (`fuel::metal_backend`). Not CireSnave's hardware (Windows, NVIDIA). | F6 |
| 21 | Memory safety on load | **have** | Pre-load fit check against the backend (0.21.2); GPU refuses an oversize model unless `LIGHTBULB_ALLOW_OVERSIZE_MODEL`. | - |
| 22 | Concurrent requests | **have** | Batched paged scheduler (`LIGHTBULB_MODEL_MAX_BATCH_SIZE`, queue depth 503). Prefix sharing exists but is **off** by default; GPU and CPU runs show a large TTFT cut with identical output (n=1 each); flip waits on n>=3 per arm. | #227 |
| 23 | Ollama-native HTTP API (`/api/chat`, `/api/generate`, `/api/tags`, `/api/pull`, `/api/show`, `/api/embed`) | **missing** | No such route. Not needed by clients that speak OpenAI-compatible endpoints; needed only by tools that speak Ollama's own protocol. PM: no Ollama-API code without a go. | F7 (WAIT) |
| 24 | Vision / multimodal input | **missing** | 0 hits for `image_url` in `src/api`. PM ruling 2026-10-25: defer. | A12 (defer) |
| 25 | Local auth for a desktop UI | **have** | `--no-auth` flag and API keys (`src/main.rs:24,57`); configurable CORS (`LIGHTBULB_CORS_ORIGINS`, `api/mod.rs:324`) so a browser-based UI can call it. | - |
| 26 | One-step install (single binary / installer) | **unknown** | `git ls-files` shows no installer, MSI or Dockerfile; whether a release binary is wanted is a decision, not a gap I can size. Lightbulb has never been published (patent-sweep record). | F8 |

## What blocks "replace Ollama and Cherry for CireSnave" (in my order)

1. **Streaming + tools (rows 10, 11)** and **embeddings (row 8)**, because a chat UI uses them by default or for its knowledge features; both are inside the OpenAI-compatible surface, so they need no Ollama-API code.
2. **Model management (rows 2-4):** no pull, one model per process, no `run`. These are the daily-use parts of Ollama.
3. **Architecture breadth (row 15)**, which decides whether the models he actually runs load at all. **I do not know which models he runs**; that is the first thing to ask him (ask through the PM; `CIRESNAVE-EXPECTATIONS.md` first).
4. Parameter gaps (row 7), then Vulkan/Metal (rows 19-20) and the Ollama-native API (row 23, WAIT).

## New ROADMAP rows

See ROADMAP section F (this PR). Each cites the table row above.
