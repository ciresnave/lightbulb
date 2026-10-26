# Patent sweep: lightbulb 0.19.x @ 24b595c (IN PROGRESS)

Date: 2026-10-26; Run by: lightbulb lane, Claude Sonnet 5.5; Method: AI-assisted search and read; **NOT legal advice; NOT a freedom-to-operate opinion.**
Process: `C:\Projects\PATENT-SWEEP-PROCESS.md`. Manifest line covered: **0.19.x** (`Cargo.toml` 0.19.31 at 24b595c). Lightbulb has never been published (crates.io holds only a 0.0.0 placeholder).

**STATUS: INCOMPLETE. This record is NOT a clearance.** Step 1 (ideas) is done and step 3 (claim reads) is done for two patents only. Step 2 (searches per idea) has not been run, so no idea below has been searched. Do not publish on the strength of this file.

## Ideas swept (what lightbulb does that might be unusual; module in brackets)
1. A new request splices a live donor request's KV blocks for a shared, block-aligned prompt prefix instead of re-prefilling it. [`engine::model_runner` prefix sharing, `LIGHTBULB_PREFIX_SHARING`, default off]
2. Hash-and-reuse of prefill KV state for common prompt prefixes keyed by a hash of the token ids. [`cache::prefix_cache`]
3. Continuous batching with a slot pool where requests join and leave independently and each step batches whatever slots are ready. [`engine::slot_pool`]
4. Priority-based admission to slots under a tracked memory budget, with an eviction-pressure metric and a slot limit that adjusts to available memory. [`engine::memory_aware_scheduler`]
5. Segment-level KV eviction scored by the model's own attention weights, raised to a configurable power so "relied upon" outweighs "glanced at". [`cache::segmented_eviction_policy`]
6. Attention-sink plus sliding-window KV eviction (StreamingLLM) and heavy-hitter (H2O) eviction. [`cache::streaming_policy`, `cache::h2o_policy`] (published methods; swept for completeness)
7. Output contracts for small models: a single enum choice from a fixed list, XML-like tagged fields, and sentinel-delimited commit blocks, each with a forgiving parser and a progressively stricter re-prompt on retry. [`contracts`]
8. A rolling end-of-sequence rate used as a signal that a heuristically chosen chat template is wrong, and a probe that generates under each candidate template to pick one. [`engine::eos_monitor`, `bin/lightbulb-probe`]
9. Knowledge-aware iterative decomposition that simplifies a problem from structural to atomic operations as facts accumulate in a knowledge base. [`engine::decomposition`]
10. An NLI-based check that rejects a new fact contradicting the knowledge base before insertion. [`engine::consistency_checking`]
11. Capacity-aware Mixture-of-Experts routing with token-drop and expanded-drop policies. [`engine::moe_router`]
12. Speculative decoding with a draft model and memory estimation that accounts for it. [`engine::speculative`, `memory::speculative`]
13. Estimating model plus KV memory from quantized sizes to choose batch size and slot count. [`memory::estimate`, `hardware::batch_sizing`]
14. Weight pruning by mask applied to GGUF tensors. [`pruning`, `quantization::gguf_ops`]

Not swept as lightbulb ideas: the attention, quantized-matmul and KV-block-pool kernels, which run in Fuel (Fuel lane's own sweep).

## Queries (source, query, date)
None run yet. (Google Patents returned 503 to `tools/patent-claims.py` on 2026-10-25 ~22:55Z after the portfolio's burst; pacing is ~1 query/minute.)

## Hits
| # | patent | title | assignee | status | claim(s) read | overlap | our difference | verdict |
|---|---|---|---|---|---|---|---|---|
| 1 | US12045307B2 | per-vector scaled quantization (Nvidia) | Nvidia | granted | independent 1, 14, 17, 20 (USPTO PDF, page images) | every independent claim requires a per-vector scale factor, a reduced-bitwidth multiplier dot-product per vector, and scaled accumulation into an output tensor | lightbulb contains no such computation: `src/quantization` extracts and masks tensors; `src/backend/marlin_ffi.rs` is an FFI binding to external kernels | clear for lightbulb's own code; **Fuel's kernels are the open question** (Fuel lane charting) |
| 2 | US12412082B2 | per-vector scaled quantization with a coarse scale (Nvidia) | Nvidia | granted | independent 1, 16, 19 (USPTO PDF, page images) | as #1, plus a coarse scale factor shared by at least two vectors | as #1 | clear for lightbulb's own code; Fuel-dependent |

Dependent claims were skimmed, not charted. The PDFs have no text layer, so the claim text was read from rendered page images.

## Needs review (to the PM)
None yet.

## Limitations
AI-assisted; search not yet run; reads limited to two patents; the first two hits were routed by the PM from the Fuel lane's work, not found by searching our ideas. Not legal advice.

## To finish
Run paced searches for ideas 1-14 (priority: 1, 4, 5, 7, 8; the rest are mostly published methods); read up to 5 closest hits per idea with `tools/patent-claims.py` (or USPTO PDFs when Google 503s); fill the table; any hit whose claims could not be read stays `unread`, never `clear`.
