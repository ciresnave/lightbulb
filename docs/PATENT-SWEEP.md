# Patent sweep: lightbulb 0.19.x @ 24b595c (IN PROGRESS)

Date: 2026-10-26; Run by: lightbulb lane, Claude Sonnet 5.5; Method: AI-assisted search and read; **NOT legal advice; NOT a freedom-to-operate opinion.**
Process: `C:\Projects\PATENT-SWEEP-PROCESS.md`. Manifest line covered: **0.19.x** (`Cargo.toml` 0.19.31 at 24b595c). Lightbulb has never been published (crates.io holds only a 0.0.0 placeholder).

**STATUS: INCOMPLETE. This record is NOT a clearance.** Step 1 (ideas) is done. Step 2 (searches) is done for ideas 4, 5, 7 and 8 only (2026-10-26), plus ideas 1-2 by cross-reference to the Fuel lane's record; ideas 3, 6, 9-14 are unsearched. Step 3 (claim reads) covers the five patents in the table. Do not publish on the strength of this file.

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

## Queries (source, query, date 2026-10-26)
Source `G` = WebSearch restricted to patents.google.com (+ patents.justia.com where marked). **This is a general web search limited to patent domains, not a patent search engine: it mostly returns papers, so "no patent found" below is weak evidence.** Control: the same kind of query does surface patents (rows 3-5 of the hits table), so the tool can return them; it does not prove it ranks the relevant one. Paced about one query a minute; one 503 from `tools/patent-claims.py` on 2026-10-25 ~22:55Z (USPTO PDFs used instead).

- **Idea 1, 2:** not searched by me. The Fuel lane's record (`fuel/docs/PATENT-SWEEP.md`, idea 11) searched paged KV blocks with prefix sharing (G, "key-value cache blocks shared prefix reference count copy-on-write large language model serving patent claim \"block table\"") and found one patent, CN 122019408 B, whose claim 1 (read by the Fuel lane, not by me) needs a fragment-hash index with logical-to-physical alias mapping. Lightbulb's `cache::prefix_cache` keys a whole prefix by a token hash and has no alias map. Same search method as ours, so this is not independent confirmation.
- **Idea 4:** G+Justia "patent scheduling inference requests language model priority admission memory budget kv cache slots evict preempt lower priority request continuous batching claims". Result: papers only, zero patents.
- **Idea 5:** G+Justia "patent key-value cache eviction segments of context scored by attention weights relevance score evict lowest scoring segment language model inference". Result: papers, one patent (CN120952055B).
- **Idea 7:** G "patent language model output contract single choice from fixed list retry stricter re-prompt parse tagged fields small language model structured output" (tangential patents only); G+Justia "patent method validating large language model response against schema, upon validation failure re-prompting model with corrective instruction, retry until valid output" (three candidate patents).
- **Idea 8:** G+Justia "patent determine chat template prompt format for language model by generating with candidate templates and measuring end-of-sequence token rate finish reason". Result: papers and unrelated patents, none about prompt-format detection.
- **Idea 1 (own query):** G "patent LLM inference prefix sharing KV cache blocks reuse shared prompt prefix between concurrent requests claims": papers only.

## Hits
| # | patent | title | assignee | status | claim(s) read | overlap | our difference | verdict |
|---|---|---|---|---|---|---|---|---|
| 1 | US12045307B2 | per-vector scaled quantization (Nvidia) | Nvidia | granted | independent 1, 14, 17, 20 (USPTO PDF, page images) | every independent claim requires a per-vector scale factor, a reduced-bitwidth multiplier dot-product per vector, and scaled accumulation into an output tensor | lightbulb contains no such computation: `src/quantization` extracts and masks tensors; `src/backend/marlin_ffi.rs` is an FFI binding to external kernels | clear for lightbulb's own code; **Fuel's kernels are the open question** (Fuel lane charting) |
| 2 | US12412082B2 | per-vector scaled quantization with a coarse scale (Nvidia) | Nvidia | granted | independent 1, 16, 19 (USPTO PDF, page images) | as #1, plus a coarse scale factor shared by at least two vectors | as #1 | clear for lightbulb's own code; Fuel-dependent |
| 3 | US12198030B1 | validating vector constraints of model outputs | Citibank | granted | claims 1-6 (Google Patents text via `tools/patent-claims.py`); independent claim 1 read in full | claim 1 needs three trained models, a vector-alignment threshold test, an aggregate model and a GUI | contracts do none of this: plain text instruction, forgiving parser, re-prompt | clear (idea 7); other independent claims not read |
| 4 | US20250322047A1 | prompt refinement to improve accuracy of model outputs | Citibank | published application | independents 1, 7, 14 read | all three require running a second prompt built from rule-based/structured pseudocode and comparing the two outputs against an accuracy threshold | no second pseudocode prompt, no output-to-output comparison | clear (idea 7); application, claims may change |
| 5 | CN120952055B | KV cache compression and sparse attention (Beihang Univ. Hangzhou) | Beihang Univ. Hangzhou | granted (China) | independents 1 and 8 read | PCA projection of keys, key-only cache with V = K.W, top-k selection by approximate attention | `cache::segmented_eviction_policy` scores whole segments by the model's own attention weights (power-boosted) and evicts; no projection, no key-only cache | clear (idea 5); China only, no US family looked for |

Candidates seen in search results but **not read**: US20260133964A1 (LLM text-to-SQL verification via directional graphs), US12292915B1 (security for generative models using attention analysis). Verdict `unread`; they were judged off-topic from titles only.

Dependent claims were skimmed, not charted. The PDFs have no text layer, so the claim text was read from rendered page images.

## Needs review (to the PM)
None yet.

## Limitations
The search surface is weak (see Queries). Ideas 3, 6, 9-14 unsearched. Hits 3-5 were found by the same weak search; a real patent database search could find closer patents.
AI-assisted; reads limited to five patents; hits 1-2 were routed by the PM from the Fuel lane's work. Not legal advice.

## To finish
Search ideas 3, 6, 9-14 (and 1-2 ourselves); prefer a real patent search surface if one is available to the portfolio; read up to 5 closest hits per idea; any hit whose claims could not be read stays `unread`, never `clear`.
