# Feasibility: a synthetic checkpoint/GGUF fixture for the ignore-census paydown

Written per HANDOFF item 2, before any implementation. Answers one question:
**which of the checkpoint/GGUF-gated `#[ignore]` tests can run in the normal CI
gate against a tiny synthetic (random-weight) fixture, instead of needing the
real, multi-GB TinyLlama checkpoint that only exists locally?**

## Verdict

Of the tests `#[ignore]`d for needing a local checkpoint/GGUF, the large
majority (43 of 62 real ones) check architecture, shape, metadata, API
contract, or self-consistency -- none of that depends on the weights being
*correct*, only on the file being the *right shape*. Those 43 are real
candidates for a synthetic fixture. The remaining 15 check specific output
values (generated text, pinned vocab hashes, measured token counts) that a
random-weight model will never reproduce, and 4 are special-cased (see
below) -- building a synthetic fixture does not help those at all.

**Recommendation: build the fixture, start with the 7 `model_fuel_golden.rs`
tests (zero additional design work beyond having a fixture at all), then the
metadata-only GGUF tests.** Treat the EOS-timing-dependent API tests and the
multi-file corpus sweeps as a second phase -- they need a fixture engineered
for specific behavior, not just a valid file.

## Methodology

Current `main` has 129 real `#[ignore]` attributes (`grep -rn "#\[ignore"`
over `git ls-files '*.rs'`, minus matches that are doc-comment prose like
`` `#[ignore]`d `` or string-literal panic messages that merely mention the
attribute -- both kinds of false positive are present and were excluded by
hand). Filtering to ones whose reason mentions a checkpoint/GGUF/TinyLlama
gives 67 lines; 4 of those are the same string-literal false positive
pattern (panic messages reading "...these tests are `#[ignore]`d...", not an
actual attribute), leaving **63** -- matching the HANDOFF estimate. One of
those 63 (`gguf_metadata_harness.rs:415`) only matched because its *file
name* contains "gguf"; its actual reason is `"compares two dumps; set
LIGHTBULB_DUMP_A and LIGHTBULB_DUMP_B"` and it never touches a checkpoint,
so it's excluded from the classification below. **62 tests classified.**

Each of the 62 was read in full (not judged from its name or `#[ignore]`
reason string alone) by a dedicated review pass, checking what the test's
assertions actually depend on.

## Classification

### GOLDEN (15) -- needs real trained weights or a real external reference artifact

A synthetic fixture cannot help these; they are either pinned against a
known hash/value from the real model, or assert the literal generated text
("contains Paris/Tokyo").

| File:line | Test |
|---|---|
| `src/gguf/mod.rs:2451` | `derived_merges_are_set_identical_to_a_real_declared_list` |
| `src/gguf/mod.rs:2775` | `baichuan_derived_order_agrees_with_the_converters_own_scores` |
| `src/gguf/mod.rs:2864` | `every_raw_violation_on_the_validated_file_is_a_sentinel_artefact` |
| `src/gguf/mod.rs:2936` | `phi3_extends_the_llama_vocabulary_and_inherits_its_oracle` |
| `src/gguf/mod.rs:2992` | `phi3_agrees_with_the_oracle_through_the_production_path` |
| `src/gguf/mod.rs:3091` | `the_v1_toy_order_agrees_over_a_small_but_clean_population` |
| `src/gguf/mod.rs:3181` | `the_derivation_over_generates_on_byte_level_bpe` |
| `src/gguf/mod.rs:3234` | `the_allowlisted_digest_belongs_to_llama_spm` |
| `src/gguf/mod.rs:3324` | `llama_spm_agrees_with_the_oracle_through_the_production_path` |
| `src/model_fuel/batched.rs:2236` | `tinyllama_two_prompts_in_one_batch` |
| `src/model_fuel/generate.rs:210` | `generates_coherent_text_end_to_end` |
| `src/model_fuel/loader_gguf.rs:260` | `quantized_llama_gguf_serves_a_coherent_completion` -- **also independently blocked** by an unrelated upstream `fuel` dequant bug on this exact file (filed 2026-09-24); a fixture wouldn't unblock it even if content-correctness weren't the issue |
| `tests/chat_template_render.rs:2539` | `probe_binary_refuses_tinyllama_because_all_three_candidates_fire` |
| `tests/fuel_engine_http.rs:136` | `fuel_runner_serves_a_coherent_completion_over_http` |
| `tests/gguf_serving_e2e.rs:334` | `a_gguf_is_served_a_coherent_answer` |

### PLUMBING (43) -- architecture/shape/metadata/contract/self-consistency

No dependency on trained values. Ranked within each tier by estimated fixture effort.

**Tier 1 -- works with any valid tiny synthetic checkpoint, no special engineering (13):**

| File:line | Test | Note |
|---|---|---|
| `tests/model_fuel_golden.rs:2615,2724,2828,2922,2988,3088,3145` (7) | determinism / cache-capacity invariance / position correctness / tie-break agreement | **All self-referential** -- each compares the fixture against a captured snapshot of *itself*, or two code paths on the *same* weights. None needs an externally-correct value. Despite the file's name, this is the single best conversion: have `capture_golden_fixture` point at the synthetic checkpoint and all 7 run as-is. |
| `src/model_fuel/decoder.rs:142` | `stepwise_logits_match_the_reference_loop_exactly` | Differential: two code paths, same weights, `maxdiff == 0` |
| `src/model_fuel/loader_f32.rs:252` | `f32_and_stock_loaders_agree` | Differential: two loaders, same weights |
| `tests/enhanced_correctness_tests.rs:88` | `test_batch_vs_sequential_single_token` | Differential: batch vs sequential, same weights. (This file has 7 more `#[ignore]`d siblings at lines 148/219/254/302/327/385/439 that weren't caught by the checkpoint/GGUF keyword filter and so aren't in this note's 62-test count, but are identically-shaped differential tests -- same conversion applies, free bonus beyond the scope counted here.) |
| `tests/fuel_engine_http.rs:290` | `fuel_runner_serves_a_default_temperature_completion` | Explicitly documented in the test as shape/provenance-only, not content |
| `tests/integration_local_model.rs:20` | `local_model_smoke` | Only checks `out.len() > 0` |
| `tests/paged_plan_once.rs:151` | `plan_once_vs_replan` | Already uses synthetic token ids; timing/non-vacuity only |

**Tier 2 -- metadata-only, no tensor data needed at all (7):**

| File:line | Test | Note |
|---|---|---|
| `tests/gguf_architecture_refusal.rs:61,128` (2) | architecture-refusal-by-name | File's own doc comment: a vocabulary-only GGUF with zero tensors is a complete fixture |
| `tests/gguf_bpe_tokenizer_fidelity.rs:447` | refusal-by-name (unverified `pre`) | Metadata only |
| `tests/gguf_metadata_harness.rs:387,563` (2) | dump emission / determinism | Metadata only |
| `tests/gguf_metadata_sibling_sweep.rs:241` | sibling-key detector | Needs one file deliberately carrying sibling keys (as real `command-r.gguf` does with multiple `chat_template` variants) |
| `src/gguf/mod.rs:1897` | `our_metadata_agrees_with_candles_on_every_file_both_can_read` | Any valid GGUF; cross-parser metadata agreement only |

**Tier 3 -- needs a matched (GGUF, tokenizer) pair or specific tensor dtype (6):**

| File:line | Test | Note |
|---|---|---|
| `tests/gguf_bpe_tokenizer_fidelity.rs:359` | tokenizer fidelity | Needs a synthetic GGUF + its matching `tokenizer.json` |
| `tests/gguf_reference_admissibility.rs:134` | vocab-prefix-identity | Same matched-pair requirement |
| `tests/gguf_tokenizer_fidelity.rs:62` | tokenizer fidelity | Same |
| `tests/gguf_dtype_census.rs:242` | dtype-code-not-index check | Needs a tensor actually encoded in an exotic/unsupported quant format -- the *dtype*, not the values |
| `tests/gguf_serving_census.rs:237` | load-success census | Needs weight tensors present in several dtypes, values irrelevant |
| `tests/gguf_serving_e2e.rs:193` | template resolution + HTTP shape | Explicitly does not assert completion text; needs the fixture's own known `chat_template` metadata |

**Tier 4 -- needs a fixture engineered for specific timing/behavior, not just validity (8):**

| File:line | Test | Note |
|---|---|---|
| `tests/api_result_metadata.rs:226,250,434,490,568` (5) | truncation/usage/stop/streaming/completions | Usage-accounting and API-contract checks, but need a fixture that reliably emits EOS *late* (for the length/truncation tests) or *early* (for the stop test) -- deterministic timing, not pure noise |
| `tests/chat_template_e2e.rs:158,183` (2) | `*_stops_on_eos` | Same EOS-timing requirement |
| `src/model_fuel/batched.rs:2371` | `b_sweep_per_sequence_cost` | Already uses synthetic token ids; own docstring says "shape-driven, not value-driven" -- listed here only because it's cost/timing-sensitive, not value-sensitive |

**Tier 5 -- needs multiple distinct synthetic files, not one (2):**

| File:line | Test | Note |
|---|---|---|
| `tests/gguf_corpus_sweep.rs:147` | corpus-wide rebuild/refuse sweep | Needs several small files covering different `tokenizer.ggml.model`/`pre` combinations |
| `tests/gguf_corpus_vocab_census.rs:342` | vocab-vs-file-count census | Tied to real-corpus quirks (GGUF v1/v2 width differences, duplicate vocabs across quantizations) -- more elaborate to replicate synthetically than Tier 1-4 |

**Also PLUMBING but needing config-value updates, not just a fixture (3):**

| File:line | Test | Note |
|---|---|---|
| `src/model_fuel/loader.rs:217` | `loads_tinyllama_from_local_dir` | Asserts specific dim/layer/head/vocab counts -- hardcoded to TinyLlama's; needs updating to the synthetic fixture's own (presumably much smaller) architecture |
| `src/model_fuel/loader_f32.rs:203` | `every_weight_is_f32` | Dtype/shape check only |
| `src/model_fuel/loader_f32.rs:293` | `loader_surfaces_eos_and_tokenizer` | Asserts a specific EOS id -- needs the fixture's own |
| `src/gguf/mod.rs:2901,3149` (2) | `baichuan_rebuilds_now_and_would_not_without_its_allowlist_entry`, `the_v1_toy_rebuilds_through_the_production_path` | Smoke tests (non-empty rebuild+encode), no value check |
| `tests/c1_prefill_vs_decode.rs:145,164` (2) | token-count preconditions | Tokenizer self-consistency; needs a matching synthetic tokenizer |

### SPECIAL (4) -- excluded from the fixture plan, flagged for separate action

- **`src/model/batched_llama.rs:159`** (`test_batched_llama_single_request`) --
  vacuous stub: body is entirely commented out, just `Ok(())`. Not actually
  blocked by needing model files right now; worth a separate look at
  whether it should be deleted, filled in, or left as a placeholder.
- **`tests/c1_prefill_vs_decode.rs:188`** (`h2_flatness_across_prompt_lengths`) --
  zero assertions, a pure `eprintln!` diagnostic probe. Nothing for a
  fixture to unlock; it doesn't pass or fail today either way.
- **`tests/corpus_readme_figures.rs:71`** -- doesn't need trained weights at
  all. Needs a fake corpus *directory* plus a `README.md` whose stated file
  count matches it. A different, much simpler kind of fixture (directory +
  text file), unrelated to model weights -- can be done independently of
  this plan.
- **`tests/gpu_paged_vs_contiguous.rs:1205`** -- primarily CUDA-gated
  (`CudaDevice::new(0)` fails first on a machine with no GPU). A synthetic
  checkpoint doesn't unblock it; GPU absence is the binding constraint.
  Belongs with the ~25-test GPU bucket, not this one.

## Recommended fixture design

A single tiny checkpoint (same architecture family as TinyLlama -- Llama-style
decoder -- but with drastically reduced dimensions: a handful of layers,
small hidden size, small vocab) covers every Tier 1-3 test. Concretely:

1. **Architecture**: match whatever `src/model_fuel/loader.rs` and
   `src/gguf/mod.rs` actually read off a GGUF/checkpoint (tensor names,
   `n_layer`/`n_embd`/`n_head`/`n_vocab` metadata keys) -- small values are
   fine, the tests that care about specific numbers (Tier "config-value
   updates" above) just need updating to whatever the fixture declares.
2. **Weights**: random (any fixed seed), generated at build/fixture-creation
   time, not committed as a large binary -- a few KB at these dimensions.
3. **Format**: both a plain checkpoint (for `model_fuel::loader`/`loader_f32`
   tests) and a `.gguf` encoding of the same architecture (for the
   `gguf_*` tests) -- two small artifacts, not one.
4. **Tokenizer**: a tiny matching vocab/merges for the Tier 3 tests that need
   a (GGUF, tokenizer.json) pair.

This single fixture (plus, later, the Tier 4/5 variants) unlocks all of Tier
1-3 (26 tests) with no further design work; Tier 4 (8 tests) needs the same
fixture but tuned so EOS fires at specific, deliberate token counts; Tier 5
(2 tests) needs 2-3 more such files with varied metadata.

## What this note does not decide

- Exactly how the fixture is generated (a build script? committed bytes? a
  helper in a shared test-support crate?) and where it lives.
- Whether to update the "config-value" tier's hardcoded expectations (exact
  dim/layer/vocab counts, EOS id, token counts) now, in one pass, or
  individually as each test is wired up.
- The GPU bucket (~25 tests) and the already-closed Postgres bucket are out
  of scope for this note; GPU tests need fuel's own runtime-skip-with-cause
  pattern, not a fixture.

Those are implementation-PR questions, not feasibility ones -- raised here
only so they aren't silently decided by whoever picks this up next.
