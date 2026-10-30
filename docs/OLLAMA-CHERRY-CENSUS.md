# What Lightbulb still lacks to replace Ollama and Cherry Studio (census)

**Date:** 2026-10-30. **Ref:** every code claim was read at `origin/main` 7ccf91f8 (lightbulb 0.23.2) unless it says otherwise. **Why:** CireSnave's 2026-10-28 milestone (PM [TASK] 2026-10-30): with dtype and the prefix benchmark done, say what Lightbulb on Fuel still lacks to replace Ollama and Cherry Studio for him.

**Scope and honesty.** The feature list below is the PM's list of what CireSnave uses (model pull/list/run, chat UI protocol and OpenAI-compatible endpoints, multi-model, embeddings, tool/MCP use, system prompts/templates, quant formats, GPU+CPU). I did **not** inspect Ollama or Cherry Studio themselves. Two sources were added on 2026-10-30: (1) `ollama list` / `ollama show` on this box (read-only) for the installed model set, and (2) Cherry Studio's published docs (`docs.cherry-ai.com`, which 301-redirects to `docs.cherryai.com.cn`; I followed that redirect, a server-supplied location I did not independently verify). **Cherry's docs say nothing about the wire protocol** (tool_calls, streaming, `/v1/embeddings`), so every claim about what Cherry sends is unverified and rows say so. "Have" means code plus a test that exercises it was found; "partial" names what is missing; "missing" means a `git grep` for the capability found nothing, with the control named. Nothing in this table was run today except where it says **measured**; the rest is read from source and tests, which show intent and coverage, not that a given model works.

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
| 8 | Embeddings (`/v1/embeddings`) | **missing** | Only three OpenAI routes exist (`openai/mod.rs:21-27`). The PM listed embeddings among the features used (Ollama serves them). **Cherry's knowledge-base page says an embedding model is optional** ("没有嵌入模型也能使用知识库，此时主要依靠 BM25 关键词检索", i.e. it works on BM25 keyword search without one) and does not say which API it calls or whether a local/Ollama model works; so for Cherry alone this is *not* a blocker, and for CireSnave it matters only if he uses Ollama embeddings elsewhere. Needs an embedding-model path through Fuel; not scoped here. | A6, F3 (low until he confirms use) |
| 9 | Tool / function calling, non-streaming | **have (narrow)** | Tools are described to the model and `<tool_call>` replies are parsed into `tool_calls` (`chat.rs:776,971`; unit tests `chat.rs:3841-3895`). Output-format parsing is one family's convention, not per-family. | A4 remainder |
| 10 | Tool calling with `stream: true` | **missing** | `chat.rs:416` returns a 400 "`tools` with `stream: true` is not yet implemented". Whether Cherry streams when it uses tools is **unverified** (its docs do not say); chat UIs generally stream, so it is a plausible blocker, not a demonstrated one. | A4 |
| 11 | `tool_choice` required / named function | **missing** | 400 today (`validate_tool_choice`, `chat.rs:801`); `auto` and `none` accepted. | A4 |
| 12 | MCP | **n/a at the server (client-side in a chat UI)** | `git grep -il mcp -- src` finds nothing. Cherry's MCP docs (`/advanced-basic/extensions/mcp`) cover only MCP-server transport (SSE / streamable HTTP) and say nothing about the model endpoint. Third-party Cherry tutorials say MCP needs a model Cherry marks as function-calling-capable (a wrench icon); that marking is a Cherry-side setting per model, so MCP use needs the endpoint to return usable `tool_calls`, i.e. rows 9-11. **Unverified against Cherry's code.** | via F4 |
| 13 | System prompts / chat templates | **have** | Jinja templates via `minijinja` (`Cargo.toml:39`), resolved per model including the template stored in GGUF metadata (`src/api/chat_template.rs:1229 resolve_for_serving`, `chat_template/registry.rs`); `lightbulb-cli --system`. The `render` entry has no `tools` parameter (A4). | - |
| 14 | Quantized formats (GGUF) | **have, measured by test** | `tests/gguf_e2e_weight_load_census.rs`: `KNOWN_FAILURES` is empty (`:216`); CHANGELOG 0.19.x records 11 of 11 weighted corpus files loading, including IQ-coded files. I did not re-run it today (needs `C:\Models\gguf-corpus` and a long build). Q8_1 is a typed error on Fuel (B12, unverified on 0.25.x). | B12 |
| 15 | Model architectures | **partial** | GGUF: `llama` and `qwen3` only; anything else is refused naming the string (`src/gguf/mod.rs:1742-1747`). SafeTensors: Llama-shaped only. Share of the most-downloaded GGUF repos this covers: 6.5% (table below). Serving more architectures is blocked on **Fuel** (`PagedDecodeModel` is implemented only for Llama and Qwen3 families, `fuel-inference/src/multi_session.rs:427-591`; Lightbulb cannot add impls); priority one is Qwen3.8-27B, a Gated DeltaNet hybrid that Fuel cannot run yet (`forward` is a typed refusal, GAP-368). Plan: `docs/superpowers/specs/2026-10-30-qwen38-27b-plan.md`. | F5, F10 |
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

## Popular model families (Hugging Face Hub, read 2026-10-30)

**Method (so it can be repeated):** the Hub search tool, `repo_types=[model]`, filters `text-generation` + `gguf`, sorted by `downloads`, limit 50, run 2026-10-30. "Downloads" is the figure the Hub reports for each repo; the tool does not say over what window, and each row is one quantisation repo, not one model, so a popular model that is split across many repos is counted in pieces. Family labels are my grouping from repo names and tags; **one family, Ornith, is untagged on six of its seven repos, so its architecture is unknown to me** (one sibling is tagged `qwen3_5_moe`). 50 repos, 45.1M listed downloads.

| Family | Repos | Downloads | Share | Served by Lightbulb today |
|---|---|---|---|---|
| Qwen3.5 / 3.8 hybrid (`qwen3_5`, Gated DeltaNet; dense and MoE) | 18 | 15.3M | 34.0% | no |
| Ornith (ornith-ai; architecture not tagged on 6 of 7 repos) | 6 | 14.9M | 33.0% | no |
| DeepSeek V4 (MoE) | 3 | 3.2M | 7.1% | no |
| Qwen2.5 (`qwen2`) | 5 | 2.7M | 5.9% | no |
| GLM-5.3 (`glm5_next` / `glm_moe_dsa`, MoE) | 4 | 2.3M | 5.1% | no |
| Qwen3 dense (`qwen3`) | 4 | 2.2M | 4.9% | **yes** |
| Spark-X2.5 (`spark2_5`) | 3 | 1.5M | 3.3% | no |
| LFM2.5 (Liquid) | 2 | 0.8M | 1.8% | no |
| Llama 3.x-based (`llama`) | 2 | 0.7M | 1.6% | **yes** |
| gpt-oss | 1 | 0.6M | 1.3% | no |
| Qwen3 MoE (`qwen3moe`) | 1 | 0.5M | 1.0% | no |
| Speculative-decoding draft model (not a chat model) | 1 | 0.4M | 0.9% | no |

**Reading it:** Lightbulb serves GGUF `llama` and `qwen3` only (`src/gguf/mod.rs:1742-1747`), which covers 3.0M of 45.1M (6.5%) of these downloads. The Qwen3.5/3.8 hybrid family is the largest identifiable one (15.3M) and is CireSnave's first target. The installed-model list (`ollama list`) used in the first version of this census is **dropped as a priority signal**: CireSnave says he uses none of those (2026-10-30, relayed by the PM).

## What blocks "replace Ollama and Cherry for CireSnave" (in my order)

1. **Architecture breadth (row 15)**, with **Qwen3.8-27B first** (CireSnave: "The first I really want to run is Qwen 3.8 27B on my desktop rig"). It needs work from Fuel (Gated DeltaNet forward, a quantized `qwen35` model, `PagedDecodeModel` for a hybrid, pipeline-parallel on Vulkan) and from Lightbulb (Vulkan and multi-device selection, a per-device B9 plan); see the plan note. After it, the families in the popularity table, in that order.
2. **Streaming + tools (rows 10, 11)**: inside the OpenAI-compatible surface (no Ollama-API code), but whether Cherry needs streaming tool calls is unverified. **Embeddings (row 8)** drop to low: Cherry's knowledge base works without an embedding model.
3. **Model management (rows 2-4):** no pull, one model per process, no `run`. These are the daily-use parts of Ollama.
4. Parameter gaps (row 7), then Vulkan/Metal (rows 19-20) and the Ollama-native API (row 23, WAIT).

## New ROADMAP rows

See ROADMAP section F (this PR). Each cites the table row above.
