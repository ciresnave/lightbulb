# Glossary

> ## ⚠️ PARTLY SUPERSEDED — some of it describes code that no longer exists
>
> Some entries refer to `src/cache/kv_compression.rs` and `src/model/chunked_prefill.rs`, which are not in
> the tree (removed with the candlelight migration, PRs #195/#196, 2026-10-22) (checked
> against `origin/main` 526d800 on 2026-10-29). Any "complete" or "in progress" status
> below is historical and has **not** been re-validated. Current status is
> `ROADMAP.md`. Kept as provenance; deleting it is a separate decision.


Domain terms used across this repo's docs. **A term appearing here does not mean the feature it
names is wired into the serving path** — `ROADMAP.md` repeatedly distinguishes "implemented" from
"implemented and exercised," and this glossary preserves that distinction per-term rather than
flattening it.

## Backends and formats

- **candlelight** — *Historical.* Lightbulb's former in-tree fork of Hugging Face Candle. Removed
  entirely (PR #195, 0.16.0, 2026-10-22); nothing in the tree depends on it.
- **Fuel** (`src/model_fuel/`) — The inference backend: Lightbulb serves requests through Fuel
  (Cargo features `cuda`/`vulkan` select its GPU backends). The `fuel-engine` feature is an empty
  legacy flag, formerly the opt-in switch from candlelight to Fuel.
- **GGUF** — The quantized checkpoint file format that bundles tensors plus tokenizer metadata in
  one file. Lightbulb has its own zero-copy/mmap reader (`src/gguf/parser.rs`), separate from
  candle's loader. "Our reader can open it" (metadata-only) is a different claim from "the library
  can load it" (tensor dtypes supported) — see `ROADMAP.md`.
- **GGUF dtype codes** — Numeric tensor-quantization type IDs stored per tensor in a GGUF file.
  candle's loader only accepts a specific subset; a file mixing in an unsupported code fails to
  build a model even though its metadata and tokenizer read fine.
- **AWQ** (Activation-aware Weight Quantization) — A 4-bit weight-quantization format/loader. CUDA
  only today; bails on CPU.
- **GPTQ** — A sibling 4-bit post-training quantization format, sharing the Marlin CustomOp kernel
  path with AWQ in this repo.
- **Marlin kernel** — The CustomOp GPU kernel wrapper used to run GPTQ- or AWQ-quantized 4-bit
  weights efficiently.

## Chat templates

- **Chat template resolution tiers (0–4)** — A five-tier, cheapest-first fallback chain for
  picking a model's prompt template: a sidecar JSON file, `tokenizer_config.json`'s
  `chat_template` field, vocabulary-signature detection, a family registry keyed on the model
  directory/config (heuristic), and an empirical probe (generate-and-score, never runs
  automatically). Resolution stops at the first tier that produces a template; which tier fired is
  recorded.
- **`lightbulb-probe`** — The CLI binary implementing resolution tier 4: generates text with
  candidate templates against a real checkpoint and scores them by EOS-fire rate.
- **EOS-fire rate** — A rolling ratio, over recent completions, of how many ended via
  `FinishReason::Stop` (the model emitted its end-of-sequence token) versus `Length` (hit
  `max_tokens`). Used to flag at runtime that the chosen chat template is probably wrong;
  observation only — it never auto-switches templates.
- **`FinishReason`** — The enum distinguishing why generation stopped (`Stop` vs `Length`), which
  the EOS-fire-rate tracking above depends on.

## Serving and scheduling

- **`ParallelModelManager`** — The production model-serving component; owns the live
  `PrefixKvCache` instance and is the one actual model loader/decoder in the default path.
- **Chunked prefill** — Splitting a long prompt's prefill pass into multiple chunks so a very long
  prompt doesn't block the batch (`src/model/chunked_prefill.rs`).
- **Prefix cache / `PrefixKvCache`** — A hash-keyed cache of per-layer KV tensors keyed by shared
  prompt prefix, to cut time-to-first-token by reusing KV state across requests that share a
  prefix. `ParallelModelManager`'s copy is live in the serving path.
- **`MemoryAwareScheduler`** — A scheduler component (admission control, priority queue, memory
  tracking) that is fully implemented and exported, but — as of the 2026-10-14 M4 slice plan in
  `ROADMAP.md` — not yet wired into the serving path: it exists as a disconnected instance nothing
  submits a request to.
- **`SlotPool`** — The continuous-batching component meant to let the model-runner pull multiple
  in-flight requests into one batched forward pass. Implemented but currently unused — the live
  job loop (`run_jobs`) processes one job at a time instead.
- **Continuous batching** — Dynamically adding and removing requests from an in-flight batch each
  decode step, rather than batching only at request start. Not yet live end to end here; the plan
  is to deliver it by wiring `SlotPool` into the job loop.
- **Speculative decoding** — Using a small "draft" model to propose several tokens ahead, which a
  larger "target" model verifies in one batched pass. Implemented today as
  `src/engine/speculative.rs`'s `SpeculativeModel` trait.

## KV-cache compression and eviction

- **KV cache compression** — A family of strategies for shrinking the stored key/value cache
  (`src/cache/kv_compression.rs`), applied via a trait-based, policy-driven interface rather than
  automatically by the cache builder.
- **KIVI** — A specific KV-cache compression strategy: per-channel quantization of K/V to 2 or 4
  bits.
- **H2O (Heavy Hitters Oracle)** — A KV-cache eviction policy tracking cumulative per-token
  attention scores and evicting the lowest-attention tokens to bound cache size.
- **R-KV** — A KV-cache compression policy using importance/redundancy scoring, listed alongside
  H2O/KIVI.
- **PagedAttention / paged KV cache** — Storing the KV cache in fixed-size, non-contiguous blocks
  rather than one contiguous per-request buffer (the vLLM-style design).
- **TTFT (time-to-first-token)** — The latency metric prefix caching and chunked prefill are
  explicitly optimizing for.
