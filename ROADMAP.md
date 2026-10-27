# Lightbulb Roadmap

Purpose: a clear plan to reach a robust, fast, and memory-efficient inference engine with room for research. This roadmap consolidates our existing plans and literature insights into pragmatic milestones.

> **Note on "Candle-first" — RETRACTED 2026-10-22, not just caveated.** This document
> originally described the plan as Candle-first, and this note previously said candlelight was
> still the default shipping path with Fuel opt-in and not at parity. **That stopped being true
> in PR #195 (2026-10-22): candlelight was removed entirely (~40 files, zero production callers
> each, confirmed file by file) and Fuel is now the sole, unconditional serving engine** — the
> `fuel-engine` feature flag is a no-op kept only for callers (including CI) that still pass it.
> Milestone status claims below that predate the original Candle→Fuel port or this full
> removal have not been re-validated against either — treat those specific claims as
> unverified rather than as findings; this does not sweep in claims logged after either date
> (e.g. the 2026-10-14 research-track entries or the 2026-10-22 remeasurements elsewhere in
> this file), which already account for both. See `README.md` for the current picture and
> `docs/FUEL-GAPS-FROM-CANDLE-REMOVAL.md` for what the removal actually cost.

---

# ⚠️ VERIFIED STATUS — read this before planning from anything below

**Measured 2026-09-01 at `origin/main` `794b88e`.** Every figure here was
re-derived at that ref, not carried from a previous report. Anything below this
block that contradicts it is stale.

**This file is now the SINGLE roadmap.** Six other documents made overlapping and
conflicting claims; they are superseded and carry banners saying so:

| document | disposition |
|---|---|
| `V1_ROADMAP.md` | superseded — kept for provenance. 128 unchecked boxes, 0 checked, and its own Phase 1 is COMPLETE. It declared the project blocked on shipped work for months. |
| `docs/M5_COMPLETION_ROADMAP.md` | superseded — dated January 2026, status predates the GGUF and Fuel work |
| `docs/M3_AWQ_IMPLEMENTATION_PLAN.md` | superseded as a *status* source; still valid as a *design* document |
| `docs/M3_SPECULATIVE_DECODING_PLAN.md` | superseded as a *status* source; still valid as a *design* document |
| `BATCHING_STATUS.md` | superseded — claims "100% Complete" for batching |
| `docs/API_IMPLEMENTATION_STATUS.md` | superseded — "Completed (Just Now)" with no date |

## What is actually true

**The server builds and serves.** `cargo check -j 4 --bins` → exit 0.
`cargo test --lib` → **643 passed, 0 failed, 14 ignored**.

**RETRACTED 2026-10-22 — the port finished and this paragraph is now backwards.** This
paragraph and the file-count table below it described the Candle→Fuel port as "early," with
Fuel behind the `fuel-engine` feature and candlelight as the shipping path, as measured
2026-09-01. **Candlelight was removed entirely in PR #195 (2026-10-22); Fuel is now the sole,
unconditional serving engine, and `fuel-engine` is a no-op flag** kept only for callers
(including CI) that still pass it. The table is a historical snapshot, not current. See
`README.md` for the current picture and `docs/FUEL-GAPS-FROM-CANDLE-REMOVAL.md` for what the
removal cost.

```
files importing candlelight    45   (2026-09-01 snapshot — candlelight fully removed since)
files importing fuel           11   (2026-09-01 snapshot — now the only engine)
total src/**/*.rs             112
```

## Research track: forked-agent latent/activation work (added 2026-10-01 by portfolio PM)

Source: `OverMind/SUBAGENT-FORK-AND-AUDITOR-DESIGN.md` (reviewed by this project 2026-09-30) plus a
longer external brief CireSnave was handed ("Forked-Agent Orchestration: Architecture Discussion Brief").
Board item 86 in `CIRESNAVE-DECISIONS.md` tracks this; project placement in that brief puts Fuel on KV-
cache mechanics and OverMind on orchestration policy, with Lightbulb owning the model-facing pieces below.

1. **Wire the existing prefix-matching logic into the serving path — DETECTION half DONE 2026-10-22
   (board PM task), EXPLOITATION half still open.** This project's own review (2026-09-30) confirmed the
   donor-must-stay-open copy-on-write mechanism is correct and real, and that a content-addressed matching
   layer already exists on top of it — but with zero callers outside its own tests at the time. `ServingDriver`
   (`src/model_fuel/scheduled_llama_engine.rs`) now wires a generalized `BlockPrefixIndexFor<SessionId>` into
   the live `try_admit` path: every admitted prompt's token prefix is recorded and looked up, giving
   `cache_stats` real hit/miss/entry counts (`CACHE_METRICS_LIVE = true`). **Detection is not exploitation**:
   a hit is not yet spliced into KV blocks or used to skip prefill — that needs
   `fuel_inference::multi_session::PagedSessionScheduler`, a different scheduler than the one wired today, and
   switching to it is a real separate port (see `docs/FUEL-GAPS-FROM-CANDLE-REMOVAL.md` entry #1). Donor-must-
   stay-open COW splicing (the TTFT win this item originally named) is still unwired.
   **`cache_clear`'s real implementation landed alongside this (board PM task, 2026-10-22):** it now sends a
   real `ControlMsg::ClearPrefixCache` to the serving thread, which rebuilds the index and zeroes the stats
   mirror — superseding the old V1_ROADMAP-era note below (§"MemoryAwareScheduler Integration" slice 4) that
   called this a separate untouched `TODO`.
2. **Fix the live-date chat-template rendering hazard — DONE, #107 (2026-09-30).** Was already
   fixed the same day as the review that surfaced it; this entry was never updated. `strftime_now`
   now caches its rendered string per calendar day (`cached_strftime_now`,
   `src/api/chat_template.rs:68`), so repeated renders on the same day are deterministic — the
   date still changes at midnight, which is correct "today's date" behavior, not the bug. Found
   stale while surveying this section 2026-10-14; confirmed via `git log -S cached_strftime_now`.
3. **Latent/activation-transfer between live model instances — gated research track, treat as the riskiest
   of the three technique clusters, not a committed deliverable.** The external brief's "mutual state
   access" idea (two live instances reading each other's internal activations via learned cross-attention)
   is genuinely novel, not just an engineering extension of KV-cache prefix sharing. IP due-diligence
   (2026-10-01, web search, not a legal clearance): nearest prior art is two papers from November 2025
   ("Enabling Agents to Communicate Entirely in Latent Space" / Interlat, arXiv:2511.09149; "Latent
   Collaboration in Multi-Agent Systems" / LatentMAS, arXiv:2511.20639) — both describe agents sharing
   hidden-state representations instead of text, close to but not identical to this brief's exact
   mechanism. No blocking patent found, but large labs are known aggressive filers on early-stage ideas
   like this one, and patent publication lag (~18 months) means a recent filing wouldn't be visible to
   this search yet. **Do not commit engineering budget beyond early prototyping without a professional
   patent search first** — this is the one technique in the whole brief where "nothing found" is weak
   evidence of safety, not strong evidence.
   Added 2026-10-27 (CireSnave: "Add it to what Lightbulb will look into when that point comes."): the closest KV-cache-transfer-across-model-families prior art is Cache-to-Cache (Fu et al., Tsinghua, arXiv:2510.03215 v2 2026-03-02), which fuses a Sharer's KV cache into a Receiver's with a small trained per-pair fuser (both LLMs frozen), reporting +6.4-14.2% accuracy over the single model and ~2.5x lower latency than text on Qwen/Llama/Gemma pairs of 0.6B-14B, limited by per-pair training, white-box access to both models, and a weak Sharer hurting; read its claims-side/patent status together with Interlat and LatentMAS when the track opens.

---

## ⚠️ Three COMPLETE claims below are contradicted by the code

**A document asserting completion over unimplemented code is worse than one
asserting nothing, because it stops anyone looking.** Each of these is corrected
in place further down; they are collected here because the pattern matters more
than any single instance.

**1. `M3.6 — Multi-GPU Inference` says ✅ COMPLETE (October 27, 2025).**

> ⚠️ **SUPERSEDED 2026-10-22.** `src/multi_gpu/{distributed_cache,pipeline_parallel,
> tensor_parallel}.rs` — every file this item names below — were DELETED entirely in PR #195
> (candlelight removal): all three were candlelight-`Tensor`-typed with zero external callers,
> confirmed before deletion, independent of the honesty findings below. `ParallelCacheBuilder`
> is gone too (superseded by Fuel's `kv_block_pool`, PR #196). The "0 of 3 implemented but now
> honest" state this item fixed no longer has code to BE honest or dishonest about — the
> question moved from "does this bail correctly" to "does this exist at all," and the answer is
> no. `multi_gpu::{config,topology}` (plain, candlelight-free) survive. See
> `docs/FUEL-GAPS-FROM-CANDLE-REMOVAL.md`'s multi-GPU entry. Kept below as the historical record
> of a real documentation-honesty fix, not as a description of current code.

Eight selectable code paths bail at runtime:

```
src/multi_gpu/distributed_cache.rs:165   Sharded cache strategy not yet implemented
src/multi_gpu/distributed_cache.rs:169   Hybrid cache strategy not yet implemented
src/multi_gpu/pipeline_parallel.rs:361   PipeDream scheduling not yet implemented
src/multi_gpu/pipeline_parallel.rs:365   Interleaved 1F1B scheduling not yet implemented
src/multi_gpu/pipeline_parallel.rs:422   PipeDream (second site)
src/multi_gpu/pipeline_parallel.rs:426   Interleaved 1F1B (second site)
src/multi_gpu/tensor_parallel.rs:222     Hybrid sharding strategy not yet implemented
src/multi_gpu/tensor_parallel.rs:248     Hybrid sharding (second site)
```

**And `Replicated` — the variant this block first credited as implemented — was
not implemented either.** `distributed_cache.rs` copied K/V to each device and
then `drop`ped the result, returning `Ok(())`; its own comments read *"would
integrate with actual cache in M3.6 Task 6"* and *"For now, we've prepared the
infrastructure."* So `CacheSyncStrategy` offers **three variants and implements
none**.

*(This block said "implements one" until review caught it. A summary correcting
false completion claims that itself overstated an implementation is the same
defect one level up, and it is recorded rather than quietly patched.)*

**FIXED 2026-09-02: all three now say so.** `Replicated` was the only one that
lied — its siblings bail — and it is a public enum variant a caller can select.
It now returns an error naming the strategy and what is actually missing:
`ParallelCacheBuilder::append` needs an `IndicesAndMask` write plan that
`update_cache` cannot obtain from `(layer_idx, batch_idx)` alone, so wiring it up
is a design decision rather than a missing call. **Still 0 of 3 implemented — but
0 of 3 honest instead of 2 honest and 1 lying.**

**Why it survived:** the only test exercising it asserted that the call returned
`Ok` — which it could not fail to do — and then printed *"✓ Cache replication
across GPUs successful"*. It was `#[ignore]`d behind a multi-GPU requirement it
did not actually need. `DeviceTopology`'s fields are public, so
`every_cache_sync_strategy_reports_that_it_is_unimplemented` now runs on a plain
CPU topology in the ordinary `cargo test --lib`, with an observed red state.

**The stale justification is worth recording separately.** The comment explaining
why the cache write was skipped said *"ParallelCacheBuilder doesn't have
position() getter"*. That is **false** — it has `position()`, `set_position()`,
`positions()` and `get_position(slot)`, and `parallel_model_manager.rs` calls the
last of those. A reason for NOT doing something is a factual claim like any
other, and this one had gone stale while still reading as a considered decision.

The eight bails above remain **public enum variants a caller can select**, and
selecting them fails with a reason.

### And the strategies that DO NOT bail had never been executed

**Measured 2026-09-02.** `ShardingStrategy::ColumnWise` and `RowWise` are
implemented, and both tests covering them were `#[ignore]`d behind *"Requires
multi-GPU setup"* — **a requirement neither had.** They call
`Device::cuda_if_available`, which falls back to CPU. So the two working
strategies had never run, on any machine.

**Running them found a defect.** Both forward paths added the bias with `+`
where the shapes require `broadcast_add`: `output` is `[batch, out_features]`
and the bias is `[out_features]`, so **every `ShardedLinear` carrying a bias
failed at runtime** with `shape mismatch in add, lhs: [2, 4], rhs: [4]`.

**The existing test would have caught it.** `test_sharded_linear_column_wise`
passes a bias — it would have failed the moment it ran, and it never ran. That
is the finding: not a missing test, an unrun one, held out of reach by a
requirement it did not have.

Both paths are fixed, and the replacement tests assert the claim the module
exists to make — **a sharded linear equals the unsharded one** — for two world
sizes, against a directly computed `input @ weights^T + bias`. The superseded
tests asserted output SHAPE and `Ok`-ness, which a implementation returning
zeros of the right shape would also satisfy. They now run in the ordinary
`cargo test --lib`.

### The multi-GPU sweep, five files, three with defects

**Measured 2026-09-02.** The question asked of each file was the same: *is the
code that CLAIMS to work actually executed by anything?*

```
distributed_cache   DEFECT   `Replicated` returned Ok(()) while writing nothing
tensor_parallel     DEFECT   every biased `ShardedLinear` failed at runtime
pipeline_parallel   clean    GPipe correct
config              clean    had ZERO tests written; none were needed
topology            DEFECT   discovery did not terminate; two reachable panics
```

**The nulls are reported at the same weight as the defects.** Two clean files are
what make "three defects" a rate rather than three anecdotes, and a sweep that
only counts when it finds something stops being run.

**`topology.rs` is the worst of the three, because it does not return.**
`DeviceTopology::discover()` probed devices with `Device::cuda_if_available` and
broke only on `Err`. That function returns `Ok(Device::Cpu)` when CUDA is
unavailable and **never consults the ordinal in that path** (candle-core 0.10.2,
`device.rs:323`), so on any build without the `cuda` feature every iteration
returned `Ok`, pushed a device and incremented — **unbounded allocation, not a
quiet spin.** `Cargo.toml` records that `candlelight/cuda` does not build on this
toolchain, so the non-terminating configuration was the ordinary one.

Confirmed by execution rather than inferred: the termination test run against the
unfixed code in a bounded subprocess **exited 124 after 25 seconds** having
printed `running 1 test` and nothing further.

**Why nothing had hung: every caller of `discover()` sits behind an `#[ignore]`d
test**, so the module's own suite never reached the entry point a consumer of
multi-GPU would call first. The code with the largest blast radius was the code
its own tests could not reach.

Also fixed there: `recommend_strategy` called `panic!` when a model exceeded
total memory — a public method, reachable through `MultiGPUConfig::auto`, for
the ordinary case of asking about a model larger than the machine — and indexed
`memory_available[0]` without checking, which panics on the empty topology the
struct's public fields make constructible. Both are errors now.

**2. `✅ COMPLETE: GGUF/other Candle-supported quant formats usable end-to-end`.**
Measured by `tests/gguf_corpus_sweep.rs` over every local GGUF:

```
30 files total = 23 REBUILT + 7 refused   (was 16 + 14; re-measured 2026-09-09
                                          at `main` c495852, after #83)

 23  rebuilt
        1  SentencePiece, merges DECLARED   TinyLlama-1.1B-Chat-v1.0 Q4_0
        4  SentencePiece, merges DERIVED    llama-spm, phi-3, baichuan,
                                            tinyllamas-stories-260k (GGUF v1)
       18  byte-level BPE                   SmolLM2-135M-Instruct x9, plus the
                                            gpt-2, falcon, qwen2, deepseek-coder,
                                            refact, deepseek-llm, llama-bpe,
                                            command-r and starcoder vocab files
  7  refused by the tokenizer
        4  tokenizer.ggml.model = "gpt2", `pre` NOT VERIFIED
              mpt, qwen35 (named, unverified);  aquila, gpt-neox (absent)
        3  a tokenizer model this build does not construct
              bert (bert-bge)   t5 (nomic-bert-moe)   gemma4 (gemma-4)
  0  OUR READER cannot open   -- this group was 5 and is now EMPTY
```

⚠️ **AND "OUR READER CAN OPEN IT" IS NOT "THE LIBRARY CAN LOAD IT". `0` ABOVE IS
A METADATA FACT, NOT A TENSOR FACT.** Every count in this block is scoped to the
TOKENIZER, which lives in the KV header ahead of any tensor. Three of the very
files this group used to contain still refuse at the tensor directory —
measured 2026-09-09 by `tests/gguf_corpus_sweep.rs`'s companion,
`tests/gguf_serving_census.rs`:

```
file          dtype codes present (code:count)   unsupported   refusal names
IQ3_XS        0:61   8:1    20:180  21:30        {20, 21}      20
IQ4_XS        0:61   8:1    20:180  23:30        {20, 23}      23
Q2_K          0:61   8:1    11:30   20:180       {20}          20
Q4_0 CONTROL  0:61   2:207  3:3     8:1          {}            loads
```

Their metadata reads normally — which is exactly what #73–#76 fixed, and is why
they moved out of this group — but `ParallelModelManager::load_gguf` cannot
build a model from them. **Dtype `20` is the blocker in all three**, at 180 of
272 tensors each; `21` and `23` add a second unsupported type to two of them.
Candle accepts `{0..3, 6..15, 30}` and everything else is refused.

⚠️ **AN EARLIER VERSION OF THIS BLOCK READ THOSE NUMBERS AS TENSOR INDICES AND
BUILT AN ARGUMENT ON IT. THEY ARE DTYPE CODES.** The upstream message is
`bail!("unknown dtype for tensor {u}")` where `{u}` is the dtype — so the
sentence's grammar attaches the number to the wrong noun, and it is not
ambiguous, it is unambiguous and misleading. The retracted claim was *"the
tensor indices DIFFER … these files MIX quantizations, and one tensor in each
uses a type candle rejects."*

⚠️ **AND THE MIXING CLAIM WAS TRUE BUT VACUOUS, WHICH IS THE WORSE HALF.** The
`Q4_0` control mixes four dtypes and loads perfectly. **Mixing is universal in
GGUF and is not the discriminator**; the discriminator is whether any dtype in
the mix falls outside candle's set. A property every file in the corpus has was
offered as the explanation for why three of them fail.

⚠️ **AND THE SAME SENTENCE HAD A SECOND BROKEN INPUT: IT READ A QUANTISATION OFF
A FILENAME.** *"`Q2_K` is a K-quant rather than an IQ type"* — but
`SmolLM2-135M-Instruct-Q2_K.gguf` **contains no Q2_K tensors at all.** Code `10`
is absent from its directory; it holds `F32`(61), `Q8_0`(1), `Q3_K`(30) and
`20`(180). **Reading a quantisation off a filename is a guess**, and this corpus
falsifies it. Found independently by the mlmf lane through a different GGUF
reader, and confirmed in the histogram above. Anyone sizing quantisation work
from file listings is counting something else.

The table above is regenerated by `tests/gguf_dtype_census.rs`, so it carries a
denominator and cannot rot into prose.

**The two questions have different populations and neither answers the other.**

⚠️ **That last group used to read "unreadable BEFORE the tokenizer is reached",
and NONE of the five is unreadable.** `Content::read` parses tensor infos
eagerly, so an unknown quantization dtype fails the whole call — but
`tokenizer.ggml.tokens` lives in the **KV header, ahead of any tensor**. Reading
those headers directly on 2026-09-03: the three SmolLM2 quantizations each carry
the full 49152-token vocabulary, digest-identical to the other six;
`ggml-vocab-aquila` carries a distinct 100008-token one; and
`tinyllamas-stories-260k` carries **512 tokens** with `bos_token_id=1` and
`eos_token_id=2`.

**That last one was called "genuinely yields nothing" in the first version of
this paragraph, and it is a fact about field widths.** ⚠️ **THE CORPUS IS THREE
GGUF VERSIONS, NOT ONE** — `tinyllamas-stories-260k` is **v1**, `ggml-vocab-aquila`
is **v2**, the other 28 are **v3** — and v1 stores counts and string lengths as
`u32` where v2/v3 use `u64`. A v3-assuming parser reads a v1 count as a garbage
`u64` and dies allocating, which is exactly what happened and was written down as
"plausibly genuinely bad".

**So every count in this block is scoped to what the library can currently
reach**, which is the right scope for a support table and the wrong one for a
statement about the corpus. `src/gguf/parser.rs` already parses these files'
metadata without touching tensor dtypes and is simply not exposed; making it
reachable would serve four of the five, and the fifth needs v1 field widths as
well.

⚠️ **DONE, AND THE PARAGRAPH ABOVE IS NOW HISTORY RATHER THAN A PLAN.** Both
halves landed across #73–#76: a metadata read stopped depending on candle being
able to type the tensors, which recovered the three IQ-dtype SmolLM2 files'
**metadata**, and the two parsers stopped cancelling each other out, which
reached the v1 file. **The "OUR READER cannot open" group went 5 → 0**, and
`ggml-vocab-aquila` moved from *unopenable* to *opened and refused for an absent
`tokenizer.ggml.pre`* — a different and much more informative outcome.

⚠️ **"RECOVERED" MEANT THEIR METADATA AND THIS SENTENCE ORIGINALLY DID NOT SAY
SO.** The three IQ-dtype files still cannot have their TENSORS loaded — see the
note under the support table above. The word was written while the question in
front of me was the tokenizer, and it reads, months later, as though those files
became usable. **A term that was unambiguous in its own paragraph acquires a
second meaning as soon as a neighbouring claim ranges over something else.**

The diagnosis above is kept because it is how the group was understood, not
because any of it is outstanding. ⚠️ **A closed defect written up in open-defect
prose reads exactly like an open one, and nothing errors when it goes stale** —
which is why the count it refers to now carries its own assertion (see
`tests/gguf_corpus_sweep.rs` and issue #80).

⚠️ **Every number above counts FILES, and a file count is not the coverage
number a tokenizer corpus is asked for.** **Nine** of the thirty files are one
SmolLM2 vocabulary at nine quantizations — a quantization changes the weights and
leaves `tokenizer.ggml.tokens` untouched, so those nine exercise the tokenizer
path once. Adding more quantizations would grow "30" without growing coverage at
all. Measured by `tests/gguf_corpus_vocab_census.rs` at `C:\Models`, 2026-09-03:

*(This said **six** on its first landing. All nine carry a byte-identical token
list; three were miscounted because `Content::read` cannot open them, which is
the correction below. The first version of this sentence said "nine" from the
filenames, was corrected to "six" by measurement, and is now "nine" again — the
measurement was real and ranged over the wrong thing.)*

**Measured 2026-09-06 at `main` `3a7c4bf`**, and re-derived on every run of
`tests/gguf_corpus_vocab_census.rs` — so these are a record of a run, not a
standing claim, and the test is what to believe if they disagree.

```
30 files  /  19 vocabularies REACHABLE THROUGH Content::read
                                 (11 files duplicate a vocabulary already present)
23 of 30 files rebuild  ->  but only 14 of 19 VOCABULARIES

REACHABLE NOW EQUALS PRESENT.  19 vocabularies are present, every file has one,
and 0 carry a vocabulary this reader cannot open -- so the caveat that used to
sit here has been retired by measurement rather than by argument.
```

⚠️ **These read `17`, `11 of 17` and `5` until 2026-09-06, and they were correct
when written.** #51 taught `parse_gguf` to accept GGUF **v2**, which reached
`ggml-vocab-aquila.gguf` — so reachable vocabularies went 17 → 18 and the
unopenable count 5 → 4. **My own change invalidated my own documentation, and
nothing marked the transition**: the figures carried no ref, so they read as
current for as long as they were wrong. That is the whole argument for stamping
them, and #50 had already corrected two *different* stale numbers in this same
block without touching these, because these were still true that morning.

The four this reader still cannot open are `tinyllamas-stories-260k-f32.gguf`
(GGUF v1 — u32 counts and string lengths, a second parse path rather than one
more accepted version) and three SmolLM2 quantizations (`IQ3_XS`, `IQ4_XS`,
`Q2_K` — unknown dtype).

⚠️ **The scope in that first line is a correction, not decoration.** This block
first read "30 files / 17 vocabularies" with five files excluded as having "no
readable vocabulary" — and **all five** have perfectly good vocabularies. (This
sentence said "four of those five" until 2026-09-06, holding out the GGUF v1 file
as genuinely vocabulary-less. It is not: read with u32 field widths it carries a
512-token vocabulary, which is why the corpus is 19 and not 18. #50 corrected the
figures in the block above and left this sentence, so the retracted belief
survived one paragraph away from its own correction.)
`Content::read` parses tensor infos eagerly and dies on an unknown quantization
dtype, while `tokenizer.ggml.tokens` sits in the KV header **ahead of any
tensor**. Read directly: the three SmolLM2 quantizations carry the same
49152-token vocabulary as their six siblings, and `ggml-vocab-aquila` a distinct
100008-token one.

**The excluding guard was stated as protecting against overstating the corpus. It
understated it**, by reporting a limitation of the reader as a property of the
GGUFs — and the exclusions looked audited because each carried an error string,
where `unreadable: unknown dtype for tensor 21` reads as a fact about the file and
is a fact about a call. ⚠️ **That same string later caught its own author a
second way: `21` is the DTYPE CODE, not a tensor index** — so the sentence
misattributes twice over, once about whose property it is and once about which
noun the number belongs to. The argument here is unaffected and stands.

**EQUIVALENCE RELATION, stated because a count whose relation is unstated is a
number rather than a measurement:** two files are the *same vocabulary* iff their
`tokenizer.ggml.tokens` arrays are identical — same length, same ordered
contents. Not tokens+merges, not tokens+special-tokens; those would give
different totals on these same files.

**The relation is doing work, and here is the control:** `starcoder` and SmolLM2
both have exactly **49152** tokens and **different** digests. A census keyed on
vocabulary *size* would have merged them and reported 16.

**Two vocabularies are SPLIT — same tokens, different rebuild outcome — and that
is the refusal policy working, not a defect:**

```
151936   qwen2 (pre=qwen2) rebuilds     qwen35 (pre=qwen35) refused
 32000   TinyLlama Q4_0 (llama, no pre) rebuilds
         llama-spm (llama, pre=default) refused
```

Support is keyed on the *rule*, not the vocabulary, which is the same reasoning
that refuses `qwen35` below. **So "this vocabulary is supported" is an ill-formed
claim**; the census asserts only that every split is explained by a differing
`model`/`pre`, and fails if two files ever share a rule and disagree anyway.

**Also measured, previously only inferred from vocabulary size:** `qwen2` and
`qwen35` carry byte-identical token lists, as do `gpt-neox` and `mpt`. The
argument for refusing them is unchanged — vocabulary identity does not imply rule
identity — but its premise is now measured rather than assumed.

**Byte-level BPE (`gpt2`) is now supported, but only for `tokenizer.ggml.pre`
values verified id-for-id against that checkpoint's OWN `tokenizer.json`.** `pre`
names a splitting rule and llama.cpp keeps a different regex per name; the 18
`gpt2` files carry 13 distinct values. **Ten are verified:** `smollm`, `gpt-2`,
`falcon`, `qwen2`, `deepseek-coder`, `refact`, `deepseek-llm`, `llama-bpe`,
`command-r`, `starcoder`.

The first seven and `llama-bpe` were each verified 0 of 130 cases against their
reference. **`command-r` was verified 0 of 30** — the in-repo gate, every case it
has. The other 100 cases live in a verification run outside the repo, and
reconstructing that generator from a seed and a prose description of its alphabet
would be a guess wearing a number, so it was not re-run. Stated rather than left
to be assumed equal.

*(This said **"Seven"** and omitted `llama-bpe` for one release. The code carried
eight; the doc was stale, not the code — checked against `91e4eb4`'s evidence
before correcting upward, because a doc/code disagreement does not say which side
is wrong, and tidying prose to match an implementation is how a weaker
verification would get laundered into the list.)*

⚠️ **And one scored 0 of 130 and is REFUSED anyway.** `qwen35`'s obvious
reference, `Qwen/Qwen3-8B`, declares a pre-tokenizer and vocab BYTE-IDENTICAL to
`Qwen/Qwen2-7B` — so it is a reference for `qwen2`, not for that name. llama.cpp
defines a distinct QWEN35 rule (`[\p{L}\p{M}]+` where qwen2 has `\p{L}+`), and
**this corpus cannot tell the two apart: measured, they differ on 0 of 130
cases**, because the qwen2 normalizer is NFC and composes away the combining
marks the difference turns on. The score was real, the corpus was the one used
for every other entry, and the result carried **no information about which rule
is correct.** A passing score from a gate that is blind to the distinction is
not evidence.

`gpt-neox` is refused for a different reason: it omits `tokenizer.ggml.pre`
entirely. It verifies 0 of 130 against `EleutherAI/gpt-neox-20b`, but keying the
table on ABSENCE would apply that one checkpoint's rule to every future GGUF
that omits the field — the one-rule-for-all-checkpoints failure the table exists
to prevent. Both refusals carry their own reason at the point of failure, so the
work already done and found negative is not silently repeated.

`llama-bpe` was gated at `meta-llama/Meta-Llama-3-8B` and is now verified via
the **NousResearch mirror of the same checkpoint** — a mirror is the same model,
which is what makes it admissible where a similar model is not.

⚠️ **`starcoder` WAS refused on that reasoning and the reasoning's PREMISE WAS
FALSE.** The refusal read: *no reference exists for THIS checkpoint; its vocab is
byte-identical to ANOTHER MODEL'S, `bigcode/starcoder2-7b`, and vocab identity
does not imply rule identity.* Every inference in that is sound.

But `ggml-vocab-starcoder.gguf` declares **`general.architecture = starcoder2`**.
**It IS a StarCoder2 file.** So `starcoder2-7b` was never a coincidental twin
standing in for an unreachable original — it is the family the file was converted
from, and the gated repo the refusal waited on (`bigcode/starcoder`, StarCoder-**1**)
is *the wrong repo*.

Measured, and the discriminator is the mismatched token **names**, not the count:

```
vs bigcode/starcoder2-7b  prefix identity over ALL 49152 ids, 48872 merges
   (ungated, HTTP 200)    identical, no tail extras
vs StarCoder-1            48697 token / 47700 merge mismatches
   (TheBloke/starcoder-    id 5:  gguf "<repo_name>" (SC2)
    GPTQ, ungated)                reference "<filename>" (SC1)
```

A bare *"48697 mismatches"* would have read as *wrong reference, refusal stands*.

**`mpt` is unaffected and still refused for cause** — `general.architecture = mpt`,
so it really is an MPT file, `mosaicml/mpt-7b` really is the right reference, and
it returns **401: gated, not absent**. A GGUF or GGML re-upload cannot help, since
those carry the `pre` *name* and not the rule; only a `tokenizer.json` supplies the
pre-tokenizer JSON, and no ungated one has been found.

⚠️ **The transferable part is why this survived:** the refusal's reasoning was
checked repeatedly and its **identification** never was. A statement of *what a
file is* does not read as a claim — and `general.architecture` sat in that file's
first kilobyte the whole time, four `head -c 1400` invocations away.

`command-r` was the last blocked purely by the gate, and is now verified through
an **ungated third-party re-upload** — `mlx-community/c4ai-command-r-v01-4bit`,
anonymous HTTP 200 — by the same route `llama-bpe` took. The canonical repo is
still 401, and so is Cohere's own 4-bit copy; even authenticated it answers *"you
are not in the authorized list"*, so the gated original is **not part of the
verification path** and nothing here depends on one person's accepted licence.

The re-upload is quantized where `llama-bpe`'s mirror was not, and that is
admissible on a property this repo measured rather than on a judgement: a
quantization changes the weights and leaves `tokenizer.ggml.tokens` untouched,
which is why six SmolLM2 quantizations are one vocabulary. **The GGUF is the
corroboration against the original** — `ggml-vocab-command-r.gguf` was converted
from the original checkpoint by llama.cpp, independently of this mirror, and a
different tokenizer could not agree with it on 255000 ids and 253333 merges.

So the remaining three are refused for cause, not for access: `mpt` still has no
reference for its own checkpoint, `qwen35` cannot be discriminated by ANY corpus
(the two vocab files differ by one byte, at the offset where `pre` is stored),
and `gpt-neox` omits the field entirely.

The table stores each checkpoint's declared pre-tokenizer and normalizer as its
own JSON, copied verbatim, so provenance is auditable by diffing against the
published `tokenizer.json` — and so a 130-character regex like `qwen2`'s is never
retyped. A reference is admitted only after its vocab is checked against the
GGUF's: `gpt-2` and `falcon` match exactly, while `qwen2` and `deepseek-coder`
carry extra GGUF tokens confirmed to sit entirely at the tail.

**A defect in the shipped path was found doing this and is fixed here.** Special
tokens were registered from four named ids only — `unknown`/`bos`/`eos`/
`padding` — so any OTHER control token tokenized as ordinary text. Measured on
SmolLM2 against its own reference, `"<repo_name>"` is token 3 and came out as
the five characters `[44, 22139, 79, 1245, 46]`. It hid because that checkpoint's
`bos`/`eos` are `<|im_start|>`/`<|im_end|>`, so the tokens a chat prompt actually
contains were covered and the rest were not. Registration now reads
`tokenizer.ggml.token_type` (3 CONTROL, 4 USER_DEFINED).

*(That array is `I32` in this corpus and `Value::to_i64()` returns `Err` for
`I32`, so the first extraction returned an empty list — and an empty list reads
exactly like "this checkpoint has no control tokens". The fix was found only
because the reference comparison stayed red after the change that should have
fixed it.)*

**The remaining 12 are refused deliberately, and the measurement says why.**
Candidate pre-tokenizers were scored against llama.cpp b10757 over the
`ggml-vocab-*.gguf` files. **Corpus size changed the answer twice:**

```
  pre                       10 cases   30 cases   130 cases
  falcon                      10/10      28/30      122/130
  gpt-2                       10/10      30/30      128/130
  command-r/refact/starcoder  10/10      30/30      128/130
  mpt                          9/10      27/30      123/130
```

At 10 cases `falcon` looked perfect and was not. At 30, `gpt-2` looked perfect
and was not. At 130 nothing scored perfectly.

⚠️ **Those scores are against llama.cpp, which is NOT ground truth — read
`128/130` as an open question, not a near-miss to close.** It is equally
consistent with *we are wrong on 2 cases* and *llama.cpp is wrong on 2 cases and
we are right*, and this measurement cannot separate them. The second is not
hypothetical: measured three ways over the same 130 cases on the SmolLM2 GGUF,
**ours vs the HF reference 0 disagree, ours vs llama.cpp 2, HF vs llama.cpp the
SAME 2.** llama.cpp's SMOLLM regex omits the trailing `|\s+` alternative the
declared `ByteLevel` rule carries, and its own source comments the rewrite.

An earlier version of this block said those values "need llama.cpp's own
per-`pre` regexes". **That was wrong and is corrected here:** adopting them would
reproduce a reference measured to differ from the checkpoints, and a perfect
score against it would be the worst outcome available — wrong exactly where it is
wrong, while looking like success. Settling any of them needs that checkpoint's
own `tokenizer.json`, the way `smollm` was settled.

**A tokenizer right 128 times in 130 is exactly the defect this module exists to
prevent**, and each larger corpus caught a case the previous one called clean.
Enabling the remaining values needs a correct regex per value, not a default —
and it needs to be verified at a corpus size that has stopped changing the
answer.

*(An earlier version of this block said 13 gpt2 and 4 unreadable, which summed to
24 of 30 and left six files unaccounted for. Review caught the arithmetic; the
figures above are re-measured and add up. The gpt2 count is **18**, not 13 — so
gpt2 support closes more than first reported, not less.)*

**One checkpoint loads** — `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF` Q4_0, the one
the tokenizer fix was developed against.

**FIXED 2026-09-02.** This checkpoint now serves
`"The capital of France is Paris.</s>"` and `tests/gguf_serving_e2e.rs`'s
`a_gguf_is_served_a_coherent_answer` passes.

**Root cause: the RoPE dimension pairing.** `apply_rotary_emb` paired `x[i]`
with `x[i + head_dim/2]` — HuggingFace's `rotate_half` — for weights loaded
from GGUF. Llama-architecture GGUF uses ggml's `ROPE_TYPE_NORM`, which pairs
`x[2i]` with `x[2i+1]`, and `convert_hf_to_gguf.py` PERMUTES `attn_q`/`attn_k`
on the way in precisely so that adjacent-pair rotation reproduces what HF's
half-split rotation computes. Loading those permuted weights and applying
`HalfSplit` rotates the wrong dimensions together.

**Why it survived every gate:** at position 0, `cos = 1` and `sin = 0`, so the
rotation is the IDENTITY for either pairing. A single-token forward pass was
measured EXACT against llama.cpp — and every cheaper check looked at token
identity, which the compression only reorders once it is large enough.

Measured by fitting our logits against llama.cpp b10757 over its top-100
candidates (`ours = a·ref + b`), before → after:

```
 prompt tokens        a                    R²              max|residual|
       1        1.0003 -> 1.0003    0.9936 -> 0.9936     0.073 -> 0.073
       2        0.9720 -> 1.0078    0.8157 -> 0.9985     1.543 -> 0.152
       6        0.7106 -> 1.0013    0.2608 -> 0.9991     3.961 -> 0.114
      23        0.6086 -> 1.0030    0.1618 -> 0.9959     9.400 -> 0.328
      24        0.3103 -> 0.9994    0.1286 -> 0.9991     7.224 -> 0.156
```

`seq_len=1` is UNCHANGED, which is the control: position 0 is the identity
rotation, so the pairing cannot matter there, and the fix therefore did this
and nothing else. The residual 0.07–0.33 is ordinary Q4_0/f32 implementation
difference.

The pairing is now a property of the weight SOURCE (`RopeLayout::HalfSplit` for
safetensors, `AdjacentPair` for GGUF) rather than a constant, so the fix does
not break the HF path — both constructors are live.

*(The artifact string this block recorded for months, `"France| Paris,
France</s>"`, was already stale before the fix: it predated the BOS fix in #19.
Retained here only as provenance.)*

**3. `M5 KV cache compression (COMPLETE)`** is cited as a satisfied dependency by
CR.1 and CR.2 (lines ~2701, ~2748).

`src/cache/kv_compression.rs`'s `PerGroup` granularity was a live
`todo!("Grouped quantization not yet implemented")` — **a panic, not a bail**,
reachable because `PerGroup` is a public enum variant selectable through
`CompressionPolicy::create_compressor()`. **Fixed 2026-09-02: it now returns an
`Err`.** Still unimplemented, so it remains in the site count below; it simply
no longer crashes the caller's process.

**And the claim was contradicted more seriously than by that panic.** Measured
2026-09-02: `KiviQuantizer` computed `scales = max|x| / (2^bits - 1)`, making
`x / scale` symmetric about zero, then clamped to `[0, 2^bits - 1]` — **pinning
every negative value to zero**, which `dequantize` then multiplied back to
`0.0`. K/V activations are approximately zero-mean, so **roughly half the cache
was discarded on the way in**, silently, in a publicly re-exported type.

Neither existing test could see it: one checked **byte accounting** and never
put a tensor through the codec, and the other round-tripped `randn` data but
bounded relative error at `< 1.0` — **a 100% error budget**, which zeroing half
of zero-mean data clears at ~0.50. Fixed with a symmetric signed codec
(`round(x/scale) + 2^(bits-1)`, undone on the way out, so `u8` storage is kept);
relative error **0.50 → 0.1185**, bound tightened to `0.20`, and
`kivi_quantization_preserves_sign` was observed RED before the fix.

### And two unimplemented paths that are NOT claimed complete

Listed separately because the distinction matters: **a gap nobody claimed to have
closed is an honest gap.** These are open work, not false claims.

- **AWQ is CPU-unusable** — RETRACTED 2026-10-23, not just caveated: `src/loaders/awq.rs` and
  `src/model/awq_qwen3.rs` named here no longer exist, deleted entirely in the candlelight
  removal (#195, 0.16.0, 2026-10-22). See "AWQ validation — blocked on a verified Fuel-side
  path" below for the current state.
- **Quantization has open dtype holes** — `src/quantization/mod.rs:245` and `:325`
  bail for unlisted dtypes, on both quantize and dequantize.

### And one class that is neither: parameters accepted and discarded

**Measured 2026-09-02 at `9763030`.** This is listed apart from the bails above
because **nothing fails**. A bail is loud and a `todo!()` panics; this returns
`200 OK` with a plausible completion and simply ignores what the caller asked
for, which is why it survived every gate.

**`temperature`, `top_p` and `top_k` are accepted by the API, carried all the way
into the request context, and never read.** The live decode path is
unconditionally greedy:

```
src/model/parallel_model_manager.rs:1481   logits_slice.argmax(0)?   (decode)
src/model/parallel_model_manager.rs:1743   logits_slice.argmax(0)?   (decode)

grep -c temperature src/model/parallel_model_manager.rs   ->  0
grep -rn temperature src/model/                           ->  no matches
   control: the same command over src/engine/ returns 10 matches, so the
   query can find the word where it is present.
```

`src/engine/model_runner.rs:393` writes `ctx.temperature = job.temperature`, and
**no site reads it back.**

**A working sampler exists and is wired to a different backend.**
`src/sampling.rs` implements `apply_temperature`, `top_k_filter`, `top_p_filter`
and `sample_from_logits`; its only caller in the crate is
`src/model_fuel/engine_model.rs:30-31` — the `fuel-engine` path, not the
shipping candlelight one. So this is an unconnected wire, not missing code.

**Not counted in the 13 sites below**, deliberately: those are executable
*unimplemented* sites that fail when reached. This one succeeds while doing
something other than what was requested, and counting it with them would blur
the distinction that makes the list useful.

### And one claim that is unreconciled rather than false

**`M3.5 — Testing & Hardening` ✅ COMPLETE (October 2025)** is not contradicted by
a bail or a panic, so it is not in the three above. But much of its content is a
**design** — "testing strategy designed", a 28-configuration validation matrix,
benchmarks "phased" — while the tree carries **111 `#[ignore]` markers** and
**three acceptance tests that never run in CI**. Read it as *"infrastructure and
strategy exist"*, not *"the matrix runs"*. Marked in place at its own section.

**13 executable unimplemented sites across 7 files** (a 14th match is a doc
comment referencing one of them).

### A fourth pattern: 23 "✅ INTEGRATED" far-future-milestone crate claims never had code behind them, from the project's second commit onward

**Audited 2026-10-24, following the `distributed-config` finding in the "one claim per PR"
docs audit started lightbulb#218.** Scattered through this file's far-future milestones
(sentience, multi-agent coordination, distributed training, memory-aware scheduling) are 23
bullets reading "✅ **INTEGRATED**: `<crate>` — <description>" for ten crate names:
`dynctx`, `mocopr-server`, `mocopr-client`, `web-server-abstraction`, `infra-storage`,
`infra-fingerprinting`, `infra-network`, `infra-consensus`, `auto-discovery`, `coalescent`.
**None of the ten is a dependency of `lightbulb` today, and `git log -S` on `src/` finds zero
commits ever referencing any of their module paths there** — the same "never called" finding
as `distributed-config`, but for all ten at once and for a documented reason: this repo's
**initial commit** (`98acb6a`) scaffolded a multi-crate Cargo *workspace* — `dynctx`,
`infra-fingerprinting`, `infra-network`, `infra-storage` as local `crates/*` path members,
`infra-consensus` present but already commented out ("Temporarily disabled due to
openraft-memstore incompatibility"), and `auto-discovery`/`coalescent`/`web-server-abstraction`
as unused `[workspace.dependencies]` version pins. **The very next commit to touch
`Cargo.toml` (`633122f`, the project's second commit)** replaced that entire workspace file
wholesale with a brand-new single-`[package]` `Cargo.toml` (git shows it as `new file`, not a
diff) — the same flat structure `lightbulb`'s `Cargo.toml` still has today. `crates/` is gone
from the tree entirely (`git ls-tree -r origin/main | grep ^crates/` — zero hits). So every one
of these ten crates existed, if at all, for exactly one commit at the very start of the
project, before any of the "INTEGRATED" prose citing them was written — they were dead
scaffolding, not removed features. (`distributed-config` and `mocopr`, by contrast, *were*
real dependencies for most of the project's life — `mocopr` still is — which is why that
finding and this one are reported separately with their own evidence rather than merged.)

The 23 sites (all inside speculative milestones well past the shipped engine, none load-bearing
for current work): lines citing `dynctx` (7-layer relationship system), `mocopr-server`/
`mocopr-client` (scheduler metrics, sentience introspection, training dashboards — 6 sites),
`web-server-abstraction` (REST/dashboard/partnership UI — 4 sites), `infra-storage`,
`infra-fingerprinting`, `infra-network`, `infra-consensus`, `auto-discovery` (one site each),
and `coalescent` (trust/coalition coordination — 6 sites). Each is marked `⚠️ CONTRADICTED` in
place below, pointing back here rather than repeating this evidence 23 times.

## Milestone status, reconciled

| milestone | status |
|---|---|
| M0 — Baseline | COMPLETE |
| M1 — Core engine | COMPLETE |
| M1.4 — Parallel batching infrastructure | COMPLETE |
| M1.5 — Hardware adaptivity | COMPLETE |
| M2 — Performance enablers | COMPLETE |
| **M3 — Acceleration features** | **IN PROGRESS** — M3.3 done; AWQ (candlelight-typed) DELETED 2026-10-22, PR #195, see "AWQ validation" below; quant dtypes open |
| M3.5 — Testing & Hardening | COMPLETE |
| **M3.6 — Multi-GPU Inference** | **DELETED 2026-10-22, PR #195** (candlelight-typed, zero callers; see above) |
| M3.7 — Dynamic Name Mapping | CORE COMPLETE, integrations PLANNED |
| **M4 — Advanced scheduling** | **PLANNED** |
| **M5 — Frontier options** | **PLANNED** |
| **M5.5 — CLI, Deployment & Operations** | **PLANNED — this is the v1.0 gate** |
| M5.6 — Hardware-specific optimizations | PLANNED (post-v1.0; needs hardware/funding) |
| M6 — Research explorations | PLANNED |
| M6.5 — Elastic KV cache w/ virtual memory | **PLANNED** — 8 open items. Its own section says "READY TO IMPLEMENT", which is not one of the five statuses in this document's legend; per the legend, designed-but-not-started is PLANNED. Its blocking dependency (`candle-cuda-vmm`) is published. |
| M6.6 — Semantic Coordinate Space models | PLANNED (post-v1.0) |
| CR.1–CR.4 — Continuous reasoning / coprogrammer | PLANNED (research) |
| M7 — Sentience infrastructure | PLANNED (5–10 year) |
| M8 — Modular training infrastructure | PLANNED |

**v1.0 is gated on M5.5, which is PLANNED. v1.0 is not reached.**

## Test coverage limits

- **111 `#[ignore]` markers** across the tree; **60 `TODO`s** in `src/`.
- **Three behavioural acceptance tests are never CI-verified** —
  `chat_template_e2e`, `fuel_engine_http`, `gguf_serving_e2e`. They need a GPU and
  a ~2.2 GB checkpoint; there is no GPU runner **by decision (2026-08-27, cost)**.
  CI *compiles* them so they cannot rot, and `scripts/check.sh` prints them in a
  NOT RUN block on every run including successful ones.
- **The Postgres-gated `#[ignore]` group (2026-10-17) is not a hardware case and
  should be un-ignored.** `#[ignore] // Requires PostgreSQL` across
  `tests/api_integration_tests.rs` and `tests/persistant_migrate_tests.rs` is
  currently forced to run via `cargo test -- --ignored` in the
  `postgres_integration` CI job, which already stands up a real Postgres
  service (`initdb`-created, port 55432) for exactly this purpose — unlike the
  GPU/checkpoint tests above, there is no environment this group genuinely
  can't run in. Per the portfolio's "`#[ignore]` is a last resort" rule, drop
  the attribute and gate on `LIGHTBULB_TEST_DATABASE_URL` being set instead (red
  when a database is expected but absent, never a silent skip) so a run outside
  `postgres_integration` without a configured database still fails loudly
  rather than looking identical to a pass.
- **Self-hosted-runner debt inventory (2026-10-17, per CIRESNAVE-EXPECTATIONS §5.1g/5.1h: "existing
  `#[ignore]`s are debt; each lane lists its own and works them down").** PM is routing a self-hosted
  CI job for `tests/parallel_model_manager_integration.rs`'s new, deliberately-not-`#[ignore]`d
  `cache_index_reused_by_a_later_unrelated_request_is_not_corrupted` (board item M4 slice 5.1). The
  rest of the tree's `#[ignore]`s, by file, evidenced via `grep -c "#\[ignore\]"` (2026-10-17):
  - **Already legitimately covered, not debt** — Postgres-gated, not hardware: `tests/
    api_integration_tests.rs` (14) and `tests/persistant_migrate_tests.rs` (3) run via the
    `postgres_integration` CI job's `-- --ignored` against a real hosted-runner Postgres service.
    A GitHub-hosted runner can do this; no self-hosted runner is needed for either file.
  - **Checkpoint-needing (CPU-runnable if a model is present, same shape as slice 5.1's test) —
    debt, not yet routed:** `tests/parallel_model_manager_integration.rs` (6, its own pre-existing
    ones besides 5.1's new test), `tests/enhanced_correctness_tests.rs` (12), `tests/
    contracts_live.rs` (6), `tests/integration_local_model.rs` (3), `tests/c1_prefill_vs_decode.rs`
    (2), `tests/load_stress_tests.rs` (2, resource need not yet confirmed by reading the file).
  - **GPU-needing (CUDA hardware, not just a checkpoint) — debt, not yet routed:** `tests/
    multi_gpu_validation.rs` (17), `tests/fused_rmsnorm_parity.rs` (4), `tests/
    fused_rmsnorm_perf.rs` (4), `tests/kv_append_perf_probe.rs` (1).
  - **Mixed GPU-and-checkpoint signals in the same file (needs a closer read before routing):**
    `tests/model_fuel_golden.rs` (9), `tests/gguf_serving_e2e.rs` (1, one of the "three behavioural
    acceptance tests" above).
  - **Not yet triaged at all** — `#[ignore]` also appears inside `src/` unit-test modules
    (`api/mod.rs`, `backend/marlin.rs`, `backend/marlin_ffi.rs`, `engine/eos_monitor.rs`, `engine/
    relevance_search.rs`, `gguf/mod.rs`, `lib.rs`, `model/batched_llama.rs`, `model/
    fused_rmsnorm.rs`, `model_fuel/batched.rs`, `model_fuel/generate.rs`, `model_fuel/loader.rs`,
    `model_fuel/loader_gguf.rs`, `multi_gpu/distributed_cache.rs`, `multi_gpu/pipeline_parallel.rs`,
    `multi_gpu/tensor_parallel.rs`, `test_models.rs`) — count and resource need not established
    here; a separate pass, not sized yet.
  - This is an inventory, not a remediation plan: each row above still needs its own self-hosted
    job (or, for the mixed/untriaged ones, a closer read first) before its `#[ignore]`s come off.
- **⚠️ NO DOCUMENTATION EXAMPLE IN THIS REPO IS EXECUTED.** Measured 2026-09-05:

  ```text
  80  doctests total
   0  actually EXECUTED
   3  compile-checked only (```no_run)
  77  ```ignore -- not compiled, not run, not checked in any way
  ```

  `cargo test --doc` was **red on main** until this was measured — two examples
  had stopped compiling (`PruningStats` fields renamed out from under
  `pruning::gguf_application`, and `dequantize_tensor` gaining a third parameter
  under `quantization`). Both are fixed, and `cargo test --doc` is green.

  **A green `cargo test --doc` does not mean the examples are correct.** It means
  three of eighty compile.

  ### ⚠️ The 77 are not stale — they are not claims. Measured 2026-09-05.

  An earlier version of this entry said the 77 were "in exactly the state the two
  broken ones were in". **That was wrong, and measuring it is what showed so.**
  All 77 `ignore` fences were flipped to `no_run` and compiled:

  ```text
  E0425  195   cannot find value in this scope     <- undeclared local bindings
  E0433   88   unresolved module / unlinked crate  <- missing `use`
  E0277   18   trait bound not satisfied
  E0599    3   no method found
  ```

  **Zero are stale API usage.** They are illustrative snippets that reference
  variables never declared and omit their imports. The three `E0599`s — the
  clearest drift signal available — all dissolved on inspection: two call a
  method on an un-unwrapped `Option`/`Result`, and `WandaScorer::score_and_prune`
  *exists*, as a `PruningScorer` trait method the example does not import.
  Positive control for that null: the same compile run **did** surface the real
  drift in the two `no_run` examples, so the instrument detects what is being
  reported absent.

  **`ignore` on a prose snippet is the honest fence for what it is.** Those blocks
  never asserted they compile, so there is nothing for them to have drifted from.
  The population that makes a claim is the three `no_run` blocks:

  ```text
  claims compilable    3   of which broken   2      <- 67%
  makes no claim      77   of which stale    0
  ```

  **So the risk was never that 77 examples went unchecked. It is that almost
  nothing here claims to compile, leaving the surface where drift is detectable
  three examples wide.** Converting the 77 is not a fix — it is authoring 77 new
  compilable examples, each needing hidden setup and imports. Not scheduled, and
  if it is ever done it should be deliberate, for the APIs people actually call,
  rather than manufactured by flipping fences on prose.

## The real frontier

**RETRACTED 2026-10-21: the "GGUF loads 1 model in 30" claim below was stale and
false, never re-derived against this repo's own later-measured numbers.** The
pre-tokenizer-rule avenue it pointed at is EXHAUSTED, not open — every
`tokenizer.ggml.pre` value discoverable in the local corpus already has either a
verified implementation (10 values: `smollm`, `gpt-2`, `falcon`, `qwen2`,
`deepseek-coder`, `refact`, `deepseek-llm`, `llama-bpe`, `command-r`,
`starcoder` — PRs #22/#25/#26/#42/#44/#56) or a documented, permanent refusal
(`qwen35`: no admissible reference exists, structurally — the only two
checkpoints discriminating it are byte-identical GGUFs with a different label;
`mpt`: its reference is gated behind HTTP 401 on HuggingFace; an absent `pre` is
refused by design, not by gap). There is no 12th or 13th rule left to implement.

**Re-measured 2026-10-21 at `main` `f475ed8`, corpus `C:\Models` (32 files, was
30 files when last measured 2026-09-09) — both tests run only against a local
`LIGHTBULB_GGUF_CORPUS`, never in hosted CI:**

```text
tests/gguf_corpus_sweep.rs      32 files: 24 rebuilt, 8 refused
                                (24 obligated: 1 spm-declared, 5 spm-derived,
                                 18 byte-level-verified-pre)
tests/gguf_serving_census.rs    32 files: 19 vocabulary-only (cannot serve by
                                 construction), 13 carry weights <- the real
                                 serving denominator
                                 9  of those load
                                 3  of those refuse: unrepresentable tensor
                                    dtype (IQ3_XS/IQ4_XS/Q2_K, documented)
                                 1  of those refuses for an UNEXPLAINED reason
                                    (see below) -- `gguf_serving_census`'s own
                                    anti-drift assertion is currently RED on
                                    this corpus because of it
```

Both the 7-vs-8-refused and 11-vs-13-weighted deltas from the 2026-09-09
figures are new corpus files added since (32 vs 30), not regressions — the
corpus is environment-dependent (`LIGHTBULB_GGUF_CORPUS`) and these tests
don't assert a fixed count, by design (see `gguf_corpus_sweep.rs`'s own doc
comment on why a floor is the wrong fix).

**The ONE new "unexplained" refusal, tracked as its own follow-up PR, next:**
`stories15M-q4_0.gguf` is a `llama`-model checkpoint with weights whose
SentencePiece vocabulary (sha256 `2f410d65ece3806aec1f808d3d8fe9d2e296cc5c98c9ecbb74468ccf0f9abd9b`)
has no declared `tokenizer.ggml.merges` and is not yet on the derivation
allowlist (`spm_derivation_warrant`) — same shape as the already-solved
`spm-derived` cases, just not yet checked against an oracle. `gguf_serving_census`
must go green before this is closed.

**The real remaining gaps, neither one a pre-tokenizer problem:**

1. **RESOLVED 2026-10-22, RE-MEASURED.** Was "3 of 13 weighted files refuse on
   an unrepresentable IQ/K-quant tensor dtype" (`SmolLM2-135M-Instruct-
   IQ3_XS/IQ4_XS/Q2_K.gguf`, measured 2026-09-09). **Re-run at `origin/main`
   24cc235 (2026-10-22) against the same corpus
   (`tests/gguf_dtype_census.rs -- --ignored --nocapture`,
   `LIGHTBULB_GGUF_CORPUS=C:\Models\gguf-corpus`): 0 of 11 weighted files
   refuse (`unsupported=[]` on every one, including the three that carry
   180-210 tensors each of dtype 20/21/23 — IQ4_NL/IQ3_S/IQ4_XS).** One file
   lands in `no histogram` (unrelated to dtype, a GGUFv1 vocab-only file our
   own parser's tensor-directory reader refuses), none of the rest are
   affected. The 13-vs-11 weighted-file count difference from the September
   measurement is the corpus directory's own state at each measurement time,
   not a discrepancy to reconcile — this count is the current, authoritative
   one.
   Two changes landed the same day this was re-measured: candlelight (the old
   blocker, whose `candle-core` 0.10.2 fork had no IQ dequant kernels at all)
   was removed entirely (PR #195) — Fuel is now the ONLY serving path, not an
   alternative to migrate onto. And the Fuel crates.io migration (PR #196,
   Fuel 0.20.61) confirmed GAP-211 B1 widened `GgmlDType::from_u32`
   (`fuel-ir/src/quantized.rs`) to accept the entire IQ family (codes 16-23,
   29), TQ (34, 35), and MXFP4/NVFP4 (39, 40), on top of GAP-211 B2's
   decode-kernel work already landed at `0.20.14`. `tests/gguf_dtype_census.rs`
   was updated in the same PR as this re-measurement: its table (now
   `FUEL_ACCEPTS`) matches Fuel 0.20.61, and its regression-guard test
   (renamed `the_iq_dtype_gap_stays_closed`) now asserts zero refusals rather
   than requiring at least one, since the scenario it used to pin
   (misreading a dtype-refusal error message) has no subject matter left in
   this corpus.
   **End-to-end weight-load census RE-CREATED and RUN, 2026-10-22** (same
   day, `origin/main` 09584dd): `tests/gguf_e2e_weight_load_census.rs`
   replaces the deleted `tests/gguf_serving_census.rs`, calling Fuel's own
   loaders (`model_fuel::loader_gguf::load_quantized_llama_gguf`/
   `loader_gguf_qwen3::load_quantized_qwen3_gguf`) directly rather than the
   removed `ParallelModelManager::load_gguf`. Result: **9 of 11 weighted
   corpus files load end-to-end — INCLUDING all three IQ-coded files**
   (`IQ3_XS`/`IQ4_XS`/`Q2_K`), confirming the dtype-gap closure above holds
   at weight-DECODE time, not just metadata acceptance.
   **Gap found, then partially closed, then a second gap found underneath
   it.** `stories15M-q4_0.gguf`/`stories15M-q8_0.gguf` originally failed with
   `fuel derive_config(...): missing metadata key
   "llama.attention.head_count_kv" -- required to derive a llama config, not
   defaulted`. Fuel 0.20.71 now defaults `head_count_kv` to `head_count` for
   pre-GQA exports (the fix this row asked for). **Re-measured 2026-10-22
   against 0.20.71: that cause is gone, but both files still fail** — same
   tiny karpathy-era "tinystories" exports, now for `llama.rope.freq_base`
   not being defaulted either (llama.cpp's own documented default is
   `10000.0`). Same shape of gap, not yet sized or routed as its own board
   row (this is a report, not a decision). `KNOWN_FAILURES` in the test was
   updated to the new cause rather than removed — the both-arms guard it's
   built on caught the cause change immediately (red until updated), exactly
   as designed. **CLOSED 2026-10-24 by the Fuel 0.25.10 bump (0.19.15):** both
   `stories15M` files load end to end (census: 11 loaded, 0 failed, run with
   `LIGHTBULB_GGUF_CORPUS=C:/Models/gguf-corpus -- --ignored`); `KNOWN_FAILURES` is now empty.
2. **3 tokenizer-model types are never constructed at all** (`bert`/`t5`/
   `gemma`-family tokenizers) — a larger, separate undertaking: building new
   tokenizer-construction paths from scratch, not implementing a pre-tokenizer
   splitting rule. Not sized yet.

---

## M4 — MemoryAwareScheduler integration: slice plan (added 2026-10-14, PM-approved)

> ⚠️ **SUPERSEDED 2026-10-22 — read before acting on anything below.** Every slice in this
> plan (1 through 5.4c, including `run_jobs`, `admit_pending`, `EngineModel`,
> `ParallelModelManager`, and the `SharedScheduler` wiring INTO `run_jobs` that 5.4a/5.4b
> landed) was built against the candlelight-backed serving path. That path — `run_jobs` and
> every function this section names as having landed real work in it — was removed entirely in
> PR #195, the same day Fuel became the sole serving path. **None of slices 2 through 5.4c's
> wiring exists in the current tree.** `cache_stats` (slice 1) genuinely still works, because
> `SharedCacheStats`/`SharedScheduler` as PARAMETERS survived candlelight's removal — Fuel's
> `ModelRunner::start` accepts them for signature compatibility — but nothing on the Fuel path
> writes into them, so `cache_stats`/`scheduler_stats`/`scheduler_queue`/real priority admission
> are back to reporting nothing real, the same hole this whole plan existed to fill. This is
> documented as the real, open capability gap it is in
> `docs/FUEL-GAPS-FROM-CANDLE-REMOVAL.md` entries #1/#2 — read those for the current state and
> routing (PM is deciding add-to-Fuel vs. accept the loss), not this section. The detailed
> history below is kept as a record of real, verified work and the lessons it found (the
> `EngineModel::step_batch` batch-shape warning in slice 5's note is still a real hazard
> whoever rebuilds this against Fuel should read) — not as a description of anything currently
> true. Do not resume numbering from 5.5 or 6 against the Fuel path without first sizing it as
> new construction, which is what it would be.

V1_ROADMAP.md's Phase 3 names this as one block ("MemoryAwareScheduler Integration"). Confirmed
still genuinely unbuilt (2026-10-14): `grep -rn MemoryAwareScheduler src/engine/model_runner.rs`
→ zero hits, and `AppState.scheduler`'s only production reader (`admin.rs::cache_stats` and its
three siblings) returns hardcoded mock data behind a `TODO`. But it is not one PR — the serving
path (`ModelRunner::start` → `run_jobs`, a dedicated thread talking to the HTTP layer only via
an `mpsc` channel) has no cross-thread readable state at all today, and `AppState.scheduler` is a
disconnected instance nothing ever submits a request to. Sliced so each lands as its own PR with
its own test, in this order:

1. **DONE, lightbulb#139.** `cache_stats` reports the real, shared `AppState.cache_stats` handle
   (`src/api/admin.rs::cache_stats`, test `cache_stats_reports_the_shared_handles_real_values_not_a_mock`),
   confirmed present and passing at `origin/main` 2026-10-17 — not a stale claim. `layers` stays
   empty (honest absence: no per-layer breakdown exists in the prefix cache). Does **not** touch
   `scheduler_stats`/`scheduler_queue`/`cache_clear` — those three stay mocked/unavailable because
   their backing data (the scheduler actually seeing requests) doesn't exist; `scheduler_stats`/
   `scheduler_queue` were separately fixed (#161) to report `unavailable_reason` instead of a
   plausible zero, still deferred for real data until slice 5.
2. **Request admission — DONE, lightbulb#142, differently than planned here.** This entry
   originally said "route jobs through `MemoryAwareScheduler::submit_request` /
   `allocate_pending_requests`" — what actually landed is `MemoryAwareScheduler::check_admission`,
   a read-only dry-run estimate check (handler-level `chars/4 + max_tokens` heuristic vs the
   configured budget), wired into both endpoints as a 503 before dispatch. It never calls
   `submit_request`/`allocate_pending_requests`, so it does **not** make `AppState.scheduler`'s
   internal state (`queued_count`/`used_bytes`/`active_slots`) live — those stay at zero
   permanently. **This was caught (2026-10-14) before starting slice 3, which assumed otherwise**;
   see slice 3 below for the resolution.
3. **`scheduler_stats`/`scheduler_queue` — deferred to slice 5 (2026-10-14, PM-ruled).**
   Originally planned as "wire real data, now that slice 2 makes `AppState.scheduler`
   meaningful" — that premise doesn't hold (see slice 2's note above). There is no live
   concurrent scheduler state to report until `run_jobs` is genuinely concurrent, which is slice
   5's job, not slice 2's. Reporting on `check_admission`'s call history instead of live state
   was considered and rejected as not worth building before slice 5 settles what state actually
   exists. Not reattempted until slice 5 lands.
4. **Priority**: expose a `priority` field (Lightbulb extension, OpenAI has no such field) on
   chat/completions requests, mapped to `Priority` on submission into slice 2's admission path —
   **also blocked on the same gap**: there is no `submit_request` call for a priority to attach
   to yet. Revisit together with slice 5.
5. **Continuous batching — sub-sliced (2026-10-17, PM-tasked: "sub-sliced, each slice
   red-first").** Scope is `run_jobs`'s orchestration loop (`src/engine/model_runner.rs`) on the
   default (`candlelight`) path only — `fuel-engine` dispatches to `run_fuel_engine` instead
   (`ModelRunner::start`'s `#[cfg(feature = "fuel-engine")]` arm, confirmed via
   `grep -n "fuel-engine" src/engine/model_runner.rs`) and never calls `run_jobs`/`EngineModel` at
   all, so it is untouched by any sub-slice here (that path is research track item 1's territory).

   **Finding that changes the risk picture (2026-10-17):** the `EngineModel::step_batch` trait doc
   warns that "a future scheduler that steps a SUBSET of live requests per call" would silently
   corrupt cache state across requests, citing "the now-deleted `FuelEngineModel::step_batch`" as
   the implementation that would have broken this way. **The one implementor that actually matters
   for this slice, `ParallelModelManager::forward_batch` (`src/model/parallel_model_manager.rs:1095`),
   is not that implementor** — it already separates `batch` by `RequestState` per call
   (`Pending`/`Decoding`/`Completed` handled per-index, `:1118-1199`) and explicitly keeps
   `Completed` slots in the batch on purpose ("In continuous batching, we need to include Completed
   slots too... masked out, but we need the full batch size", `:1193-1197`) — i.e. it already looks
   built for exactly the call pattern this slice needs (every live request present every call,
   grown/shrunk between calls as requests arrive/finish). **This is a reading of the code, not a
   measurement** — sub-slice 5.1 exists to turn it into one before anything in `run_jobs` depends
   on it.

   1. **5.1 — Verify the invariant holds, before relying on it.** New test driving
      `ParallelModelManager::forward_batch` directly (no `run_jobs` involvement) through a sequence
      that exercises the thing the trait doc warns about: request A admitted, steps a few tokens;
      request B admitted into the same batch while A is still decoding; A completes and is dropped
      from the batch on a later call while B continues; a fresh request C is admitted into the slot
      A's cache index should now have freed. Assert C's generated tokens are correct (not corrupted
      by A's leftover KV state) and that `cache_index_pool` accounting matches (no double-allocation,
      no leak). Red first: written and run against current `main` before any `run_jobs` change, so
      a failure here is caught as a pre-existing model-layer gap, not attributed to this slice's own
      orchestration change.
   2. **5.2 — Non-blocking admission, no behavior change yet.** Restructure `run_jobs`'s loop from
      blocking `while let Ok(job) = rx.recv()` (one job, full lifetime, before the next is even read)
      to maintaining a `Vec` of in-flight jobs fed by blocking-recv-when-empty /
      try_recv-when-not-empty — but still step each job to completion alone, serially, matching
      today's observable behavior exactly. Pure plumbing. Red-first acceptance test: capture
      current output for a fixed sequence of jobs against a stub `EngineModel` (already used in
      this file's test module), confirm it's green pre-refactor, then confirm it stays byte-identical
      post-refactor — proves the new shape didn't change anything yet.
   3. **5.3 — DONE, real concurrent stepping landed.** `run_jobs` now steps every live job
      together in one `step_batch` call per round (`admit_pending`/`remove_cancelled_before_step`/
      `demux_step_results`/`fail_whole_batch` in `src/engine/model_runner.rs`), demuxing each
      round's result to its own request's channel and dropping a job only once a round reports it
      `Completed`, cancelled (pre-step, `Complete` mode only, matching the pre-5.3 precheck), or
      the whole round errors (`step_batch`'s `Err` aborts the batch with no per-request
      granularity, per the trait's own doc — every active job in that round fails together).
      Red-first test `run_jobs_steps_concurrent_requests_together_without_cross_contamination`:
      two jobs submitted together, `step_batch` observed with `batch.len() >= 2` while both are
      in flight, each still produces its own correct, uncontaminated output. Fails (always
      `batch.len() == 1`) against pre-5.3 code; passes against this.
      **Superseded in place, not just annotated:** 5.2's own acceptance test asserted "no
      interleaving between jobs" as ITS temporary scope, not a permanent guarantee -- that
      assertion is now simply wrong, since interleaving is the whole point of 5.3, so the test
      was corrected (not merely noted) to assert the real invariant instead (concurrent stepping
      + correct per-job results), per the portfolio's retraction rule.
      **Cache-slot-leak fix verified compatible**, not just assumed: the slice 5.1-adjacent fix
      (board item, PR #179 -- a cancelled request's cache index must be released even when it
      never reaches the backend's own completion branch) was re-run against 5.3's new concurrent
      admission and still passes, for the right reason -- a cancelled job is removed in the SAME
      round it's stepped, before any other job can join it in the active batch, so the two never
      coexist in a way that would need the fix to behave differently.
   4. **5.4 — sub-sliced (2026-10-18), same reasoning as slice 5 itself.** Reading
      `MemoryAwareScheduler`/`SlotPool` before slicing found a real shape mismatch: `SlotPool` is
      its own independent bookkeeping state machine (`update_slot(slot_id, token)` tracks
      `tokens_generated`/`is_prefill`/completion ITSELF, from tokens the caller hands it) --
      it does not read or drive `run_jobs`'s real `Vec<RequestContext>` stepping (5.2/5.3). Wiring
      it in means keeping a SECOND, parallel bookkeeping structure in sync with the real one, for
      observability and admission-limiting, not replacing `batch`/`modes` with it. Also:
      `AppState.scheduler` is `Arc<MemoryAwareScheduler>` today (no interior mutability) --
      `check_admission` is `&self`-only (slice 2's design), but `submit_request`/
      `allocate_pending_requests`/`free_slot` all need `&mut self`, so the type itself has to
      change before `run_jobs` (which has no handle to the scheduler at all today) can touch it.
      1. **5.4a — DONE.** `AppState.scheduler` is now `SharedScheduler` (`Arc<Mutex<
         MemoryAwareScheduler>>`, defined in `engine/model_runner.rs`); `ModelRunner::start`/
         `run_jobs` thread it through the same way `cache_stats` already does. `admit_pending`
         calls `submit_request` + `allocate_pending_requests` when a job joins the active set
         (using a chars/4 placeholder for `prompt_tokens`, same heuristic `check_admission`
         already uses -- real tokenization hasn't happened yet at admission time); every removal
         site (`demux_step_results`, `remove_cancelled_before_step`, `fail_whole_batch`) calls the
         new `release_scheduler_slot` helper for the same request, unconditionally (unlike
         `abandon_if_not_completed`, the scheduler mirror has no "the backend already freed it"
         case to skip). No real admission limit enforced yet -- `submit_request` only queues if
         the scheduler has no free slot, consistent with the sub-plan's own scope. Red-first test
         `run_jobs_mirrors_real_admission_and_departure_into_the_shared_scheduler`: two concurrent
         jobs, the scheduler's `slot_pool().get_ready_batch().len()` reaches 2 while both are in
         flight (impossible before -- nothing called `submit_request`/`free_slot` from `run_jobs`
         at all) and returns to 0 once both complete. `check_admission`'s call sites (`chat.rs`,
         `completions.rs`) updated to lock first. Every `MemoryAwareScheduler` construction site
         across the tree (production and test fixtures) now wraps in a `Mutex`.
      2. **5.4b — DONE.** `scheduler_queue`'s `pending_requests`/`running_requests` and
         `scheduler_stats`'s new `active_slots`/`queued_requests`/`scheduler_memory_used_bytes`/
         `eviction_pressure` fields now read `MemoryAwareScheduler::get_memory_stats()` on the
         real shared scheduler (5.4a's doing) -- real numbers, including a genuine `0` when idle,
         not the pre-5.4b `None`/`unavailable_reason`. `queue` (the per-request list) and
         `total_requests`/`completed_requests`/`failed_requests`/`average_latency_ms`/
         `throughput_tokens_per_sec` stay `None` -- no backing API exists for the former
         (`priority_queue` has no enumeration method), and `GET /metrics` is the right home for
         the latter (PM ruling, 2026-10-16), not a duplicate counter here.
         **`cache_clear`'s `_scheduler` binding decided: left alone.** Calling `free_slot` on
         active admission slots from a "clear the prefix cache" endpoint would kill in-flight
         requests, not clear a cache -- the scheduler was never the right lever for this endpoint;
         `cache_clear`'s real gap (cache clearing itself is unimplemented, a separate `TODO`) is
         untouched, not swept into this slice.
         **Corrected in place**, per the retraction rule: the old
         `scheduler_queue_reports_unavailable_not_zero` test's claim ("always null") is now simply
         wrong on an idle scheduler (real `0`, not `null`) -- renamed and its assertions replaced,
         plus two new tests (`..._reports_real_occupancy_once_a_request_is_admitted`, one per
         endpoint) priming the shared scheduler directly (`submit_request` +
         `allocate_pending_requests`, the same calls `admit_pending` makes) and reading the real
         occupancy back.
      3. **5.4c — plumbing target moot; API shape RESOLVED 2026-10-22: Option B stands.**
         Option B (a custom `X-Lightbulb-Priority` header) was PM-ruled 2026-10-21 and already
         implemented (`docs/API-EXTENSIONS.md`, `InferenceJob.priority`/`priority_from_headers`).
         A same-day Option A ruling (an optional top-level `priority` field) was made without
         reading `docs/API-EXTENSIONS.md` first and is withdrawn. This sub-slice's described
         mechanics (`InferenceJob` gaining a `priority` field, `admit_pending` calling
         `submit_priority_request`) named the candlelight-era `run_jobs` path regardless of
         option — deleted entirely 2026-10-22 along with candlelight (#195), see this M4
         section's own SUPERSEDED banner above. The Fuel-side routing decision this now needs:
         `docs/FUEL-GAPS-FROM-CANDLE-REMOVAL.md` entry #2.
   5. **5.5 — `run_jobs` complexity refactor (slice plan item 6, below) — VERIFICATION FINDING
      (2026-10-21): the refactor this item describes appears to already be done, as a side
      effect of extraction work landed across 5.1-5.4b, not as dedicated 5.5 work.** Measured on
      branch `feat/m4-slice5.5-run-jobs-refactor` at `origin/main` 7abfa0a (the PR #184 merge):
      `run_jobs` itself (`src/engine/model_runner.rs`) is now 46 lines start-to-close-brace
      (`awk` line count), with the streaming/complete-mode branches already split out into
      `stream_job_step`/`finish_complete_job`/`demux_step_results`, admission into
      `admit_pending`, pre-step cancellation into `remove_cancelled_before_step`, and batch-wide
      failure into `fail_whole_batch` — exactly the split item 6 asks for. Manually counting
      decision points in the remaining body (1 `loop`, 2 `if batch.is_empty()`, 1
      `while let Ok(job) = rx.try_recv()`, 2 two-armed `match`es) gives cyclomatic complexity
      ~7, against item 6's cited limit of 12 (it was 18 before extraction). All 10
      `engine::model_runner::tests::*` unit tests pass unchanged. **Caveat: this is a manual
      estimate, not a read of Codacy's own number** — Codacy comments per-PR on changed lines,
      and no PR in this slice plan touched `run_jobs` for its own sake, so there is no PR comment
      to read `run_jobs`'s current score from directly; the closest proxies (PRs #182/#183, which
      touched it incidentally) both reported "0 new issues." The next PR that touches this
      function should confirm against its own Codacy comment before this item is marked
      COMPLETE outright.

   Each of 5.1-5.4 lands as its own PR, red-first, with cargo test --lib green before AND the new
   test failing-then-passing demonstrated in the PR description — not just "compiles" at the end.

Each slice must land green on `cargo test --lib` plus its own new test proving the specific claim
(a mock replaced with live data, a real rejection, etc.) — not just "compiles."

**6. `run_jobs` refactor (added 2026-10-14, found while landing slice 1).** Codacy flags it at
cyclomatic complexity 18 (limit 12) and 99 lines (limit 50) — pre-existing debt (it was already
~130 lines on `origin/main` before slice 1 touched it at all), surfaced as "new" only because
slice 1's own one-line addition fell inside the function. Splitting the streaming/complete-mode
branches apart is real, separate work; not pre-sliced here since slices 1-5 above may change what
`run_jobs` looks like by the time this is picked up. Sequenced after slice 5, not before — doing
it earlier risks conflicting with every one of slices 1-5 touching the same function.

---

## Fuel dependency migration: git rev → crates.io (sizing, added 2026-10-21, PM-tasked)

**Why now:** CireSnave's standing Sources rule (crates.io only, no `git =`/path deps) plus his
"most recent versions at all times" rule both apply to the `fuel`/`fuel-*` dependency block
(`Cargo.toml`), which is pinned to a git rev today. PORTFOLIO-ROADMAP.md §2 had flagged this as
"not scheduled... an integration, not a version edit" — this section is that integration's
sizing, not an attempt to do it as a drive-by Cargo.toml edit. **Blocks research track item 1,
slice 1** (below) per the PM's ordering: this migration lands as its own PR(s) first.

**Crates.io readiness, checked 2026-10-21 against the live API:** `fuel`,
`fuel-inference`, `fuel-model-llama`, `fuel-ir`, `fuel-transformers`, `fuel-loaders`,
`fuel-cpu-backend`, `fuel-cuda-backend`, `fuel-vulkan-backend`, `fuel-dispatch`,
`fuel-hardware`, `fuel-tensor`, and `fuel-kv-pool` are all published, all at `0.20.14`
(created 2026-10-14, last updated 2026-10-18). Fuel's own `origin/main` workspace version is
`0.20.54` — ahead of its own last crates.io release, so `0.20.14` is genuinely the latest
*installable* version today, not a stale snapshot. **Re-check the live API again before
actually bumping** — this is a point-in-time reading, not a pin.

**What changed between the pinned rev and `0.20.14`:** 114 commits (measured
`git log --oneline <pinned-rev>..origin/main` in the `fuel` checkout, 2026-10-21 — the
288-commit figure in PORTFOLIO-ROADMAP.md §2 is from an earlier measurement and is stale; use
this one). The one that matters most for lightbulb: **GAP-348 dissolved the `fuel-core`
package entirely** (fuel#366, tagged `0.19.0`) — every symbol it carried moved into the crate
that actually owns it (`fuel-ir`, `fuel-tensor`, `fuel-loaders`, etc.), and the `fuel` facade
crate (plain name, not `fuel-core`) re-exports them from `fuel::lazy`/`fuel::...` so that
`use fuel::...` "still resolves identically" per the facade crate's own doc comment
(`fuel/Cargo.toml`'s header). Lightbulb's own `Cargo.toml` pins `fuel = { git = ...,
package = "fuel-core" }` — that package name **no longer exists upstream at all**, so this line
specifically must change to depend on the plain `fuel` crate by its real name, not just gain a
version number in place of `git`/`rev`.

**Lightbulb's own exposure, checked by grep (2026-10-21):**
- `grep -rln "fuel_core::" src/` → **0 hits.** Lightbulb already imports everything through the
  `fuel::` facade path, never the dissolved package's own path directly — the lowest-risk
  starting position this migration could have.
- `grep -rln "fuel::lazy::" src/model_fuel/` → 5 files, 9 usages (`batched.rs`, `device.rs`,
  `loader_f32.rs`, `mod.rs`, `scheduler_driver.rs`) — all `fuel::lazy::{Tensor, LayerWeights,
  WeightStorage, SamplingStrategy, load_tensor_as_f32, load_transposed_matrix}`. Per the facade's
  own guarantee above, these should NOT need call-site changes — but that guarantee is read from
  a doc comment, not proven by compiling against `0.20.14` yet.

**Not sized here, deliberately:** the full API-surface diff across all 114 commits (GAP-211's
quantized IQ2/IQ3/IQ1 decode work and whatever else landed) is not audited line-by-line —
doing that speculatively costs more than just changing the dependency lines and reading the
compiler's own error list. **Proposed approach:** one PR that (1) replaces every `fuel`/`fuel-*`
git dependency line with a crates.io version pin (dropping `package = "fuel-core"` on the
`fuel` line), (2) runs `cargo check --all-features` and fixes whatever the compiler actually
flags, (3) re-runs the full test suite including `fuel-engine`-gated tests. Real cost is
whatever step 2 turns up, reported honestly rather than estimated in advance.

**PersistAnt adoption status (added 2026-10-21, PM-asked — not blocked on anything):**
lightbulb stores API keys, audit logs, user sessions, rate-limit usage counters, and auth
attempt counts, all in Postgres, and adoption is **already complete** in shipped code
(`persistant-sqlx`/`persistant-relational` 0.7.4, "PersistAnt adoption Step 5" per
`Cargo.toml`'s `[dev-dependencies]` comment) — `tokio-postgres`/`refinery`/`postgres-types`
remain only in `[dev-dependencies]`, deliberately, to simulate a pre-Step-5 deployment's
refinery-migrated database and inspect raw schema in tests, never in the shipped binary.
Nothing here blocks or needs further PersistAnt work.

---

## Research track item 1 — prefix-matching into the Fuel serving path: slice plan (added 2026-10-14, PM-tasked)

**Scope correction, RETRACTED in place 2026-10-22 — the opposite is now true.** This section
previously said candlelight was the default shipping path, Fuel was opt-in behind
`fuel-engine`, and a caller on the default build would see no change from any slice here until
a later, separate decision flipped the default. **Candlelight was removed entirely in PR #195
(2026-10-22, independent of this slice plan); Fuel is now the sole, unconditional serving
path, and `fuel-engine` is a no-op flag.** That changes the blast radius of this plan, but not
uniformly across its slices (see the slice list below): **slice 1** (`LlamaPagedAdapter`) only
adds and tests a new, unwired adapter type with zero callers outside its own test, so it
changes nothing live; **slice 4** is a reconciliation decision, not code; **slice 5** is
blocked on an upstream Fuel method (`forward_paged_step` for Qwen3) that does not exist yet.
**Slices 2 and 3** are the ones that change live serving behavior — slice 2 swaps the
scheduler `ModelRunner::start` actually drives for every caller, and slice 3 changes admission
on top of it. That is exactly why "not started... build begins only once the PM confirms
priority" below now means confirming a change to live production behavior for slices 2/3, not
an experimental feature flag. Nothing in the slices' own technical content (the two unwired
primitives, the `PagedDecodeModel` gap, the measured paged-attention cost) changes — only the
blast radius of actually building slices 2 and 3 does.

### What's actually live on the serving path today

`src/engine/model_runner.rs`'s `ModelRunner::start` unconditionally drives
`crate::model_fuel::scheduled_llama_engine::run_fuel_engine`, which uses
`FuelSchedulerDriver` (`scheduler_driver.rs`) wrapping Fuel's **`SessionScheduler`**
(`fuel_inference::multi_session::SessionScheduler<M: DecodeModel>`) — a simple scheduler with no
paged attention and no prefix-sharing primitives. Confirmed by reading
`fuel-inference/src/multi_session.rs` at the pinned rev (`d90b481f`, via a local
`~/.cargo/git/checkouts` clone — no network needed, the dependency is already vendored by cargo):
`SessionScheduler` has `add_session`/`step`/`reap_finished` and nothing prefix-related.

### What's built in lightbulb but genuinely unwired — TWO primitives, not one

The original research-track note named one ("a content-addressed matching layer... zero callers
outside its own tests"). Re-surveying while sizing this plan found a **second, independent**
unwired primitive the note didn't mention, and the two are more entangled than "wire it in"
suggested:

1. **`BlockPrefixIndex`** (`src/model_fuel/policies.rs`, ~3,100 lines, extensively designed —
   see its own module doc for the granularity-mismatch analysis already worked out). Zero callers
   outside its own file (`grep -rln "BlockPrefixIndex::new" src/` outside `policies.rs` → 0 hits).
   Its `lookup`/`record`/`splice_prefix` are written against `fuel::kv_block_pool_device::
   DeviceKvPool` — the **paged** pool type, not the plain `KvBlockPool` the live
   `SessionScheduler` uses.
2. **`BatchedPagedDecoder`** (`src/model_fuel/batched.rs`, its own hand-rolled batched paged-decode
   loop, extensively tested against Fuel's own single-sequence `forward_paged_step` for parity).
   Zero callers outside its own file (`grep -rln "BatchedPagedDecoder" src/` outside `batched.rs`
   → 0 hits). Not named in the original research-track note at all.

**Both exist because they were built against Fuel's `PagedSessionScheduler`
(`fuel_inference::multi_session::PagedSessionScheduler<M: PagedDecodeModel>`) — a *second,
newer* scheduler Fuel ships, separate from the `SessionScheduler` actually in use.**
`PagedSessionScheduler` is the one with `register_prefix`/`release_prefix`/
`add_session_sharing_prefix` (confirmed in `multi_session.rs`, lines ~1583-1716) — prefix-sharing
is a feature of the scheduler lightbulb is NOT using, not a missing feature of the one it is.

### The real blocker: `PagedDecodeModel` has zero implementors, anywhere — but it's closer than it looks

`PagedSessionScheduler<M>` requires `M: PagedDecodeModel`
(`fn forward_paged_step(&self, token, pool: &mut DeviceKvPool, session) -> Result<Vec<f32>>` +
a batched sibling). `grep -rn "impl.*PagedDecodeModel for" ` over the **entire** pinned Fuel
checkout returns zero hits — nothing implements this trait today, not in Fuel, not in lightbulb.
That sounds like new engineering, but checking closer: **`fuel-model-llama`'s `LlamaModel` already
has an inherent `forward_paged_step` with the exact signature the trait wants** (same pool type,
same session handle type, same error type — `fuel-model-llama/src/lib.rs:898`), already exercised
by lightbulb's own (also-unwired) `BatchedPagedDecoder` tests. The method exists; only the trait
label is missing. Rust's orphan rule means lightbulb can't `impl PagedDecodeModel for
fuel_model_llama::LlamaModel` directly (lightbulb owns neither the trait nor the type) — but a
thin newtype wrapper owned by lightbulb, forwarding to the existing inherent methods, is ordinary,
in-lightbulb-scope Rust, not a cross-repo ask. **Qwen3 has no `forward_paged_step` anywhere in the
pinned checkout** (`grep` for it across the whole repo, filtered to Qwen-named files, returns
nothing) — this path covers Llama only until Fuel ships the Qwen3 equivalent.

### CORRECTED 2026-10-14: `Op::PagedAttn`'s GPU execution path is already measured, not open

The previous revision of this section said this was "a genuinely open risk, not yet resolved,"
citing only the *problem statement* at the top of `tests/gpu_paged_vs_contiguous.rs`'s doc
comment ("three mutually exclusive prior claims and six retractions") without reading far enough
into the same file to find that it **already contains the resolution**: a real `nsys`-traced
measurement (2026-08-01, reworked with corrections through 2026-08-06), cross-checked
independently by the fuel lane, with the measurement's own registered predictions and failure
modes stated before the runs. **Finding, which survived every correction made to the surrounding
numbers: `Op::PagedAttn` has no fused CUDA kernel. It round-trips through host memory every
token, moving ~186x the bytes of the contiguous path's captured arm, and costs ~10x more per
token** (103.1 ms vs 25.9 ms/token, k=1, corrected steady-state window). This was discovered by
building the measurement on this same machine (confirmed a real CUDA GPU — RTX 4070 Laptop, 8 GB
— and `nsys` are both present) and reading the test file fully before concluding anything, not by
re-running it; the existing recorded result already answers the question. **Not an open risk —
a known, already-paid-for cost** that slice 2 below must design around (expect paged decode to be
slower per token than contiguous on this hardware), not re-discover.

### Slices

1. **`LlamaPagedAdapter` — MOOT, not needed (confirmed).** `fuel-inference/src/multi_session.rs`
   at Fuel 0.22.4 implements `PagedDecodeModel` directly for ALL FIVE model types next to the
   trait (orphan-rule home, not in the model crates — a grep of `fuel-transformers`'s
   `lazy_quantized_*.rs` files finds nothing, which is why an earlier read of this ROADMAP
   wrongly concluded only `QuantizedLlama3Model` had it): `fuel_model_llama::LlamaModel`,
   `fuel::lazy_llama_full::Llama3Model`, `fuel::lazy_quantized_llama::QuantizedLlama3Model`,
   `fuel::lazy_qwen3::Qwen3Model`, `fuel::lazy_quantized_qwen3::QuantizedQwen3Model` — i.e. every
   concrete type any of `scheduled_llama_engine.rs`'s three `ServingDriverInner` variants use
   (`LlamaModel` for `F32`, `QuantizedLlama3Model` for `QuantizedGguf`, `QuantizedQwen3Model` for
   `QuantizedGgufQwen3`). No lightbulb-owned orphan-rule wrapper needed, no scope narrowing: all
   three arms swap, not just `QuantizedGguf`.
2. **DONE (2026-10-23).** New `src/model_fuel/scheduler_driver_paged.rs::FuelPagedSchedulerDriver`
   wraps `PagedSessionScheduler` behind the same `SchedulerDriver` trait; `scheduled_llama_engine.rs`
   builds all three `ServingDriverInner` arms from it. The old `scheduler_driver.rs`
   (`SessionScheduler`-backed) is deleted — fully superseded, zero remaining callers once the swap
   landed, so kept out of the tree rather than as dead-code insurance (git history has it if ever
   needed). Both blockers this entry used to describe are resolved at 0.22.4: `session_new_tokens`
   exists on `PagedSessionScheduler` (verified directly in `multi_session.rs`, not from a
   changelog claim) so streaming works with the same dedup/diff `seen_counts` logic the old driver
   used; `set_priority(id, priority)` is wired as a new non-trait method on
   `FuelPagedSchedulerDriver`, callable by a caller right after `try_admit` returns a `SessionId`
   (not yet CALLED anywhere — `InferenceJob`'s priority isn't threaded into `ServingDriver::
   try_admit` in this slice; that wiring is still open, tracked separately from the scheduler swap
   itself). `PagedSessionScheduler::new` taking no `SchedulePolicy` turned out not to need a
   parallel `run_batched_paged`: `set_policy`/`policy()` setters exist, called once right after
   construction, so `run_batched`'s signature only changed its generic bound
   (`DecodeModel` -> `PagedDecodeModel`) and its driver-construction call, not its shape.
   **A real, deliberate behavior change, not hidden in the diff:** `PagedSessionScheduler::
   add_session` is explicitly "optimistic — no up-front block reservation" (its own doc comment)
   — there is no `kv_blocks_required` equivalent to pre-check against, so `try_admit` can only
   ever return `AdmitError::Rejected` (structural: empty prompt / zero budget), never
   `AdmitError::NoCapacity`. A session that genuinely can't fit is admitted anyway and fails later,
   isolated, during `step`'s prefill pass (`StepReport::errored`) — surfaced as
   `FinishOutcome::Failed`, which `run_scheduled_jobs` already handles correctly (reports the
   request as failed, never retries) but which no driver had exercised before this one (the old
   driver's pre-check made that arm unreachable in practice). Proven with a real test
   (`a_session_too_large_for_the_pool_is_admitted_then_fails_at_step_not_at_admission`), not just
   asserted in a comment. `SchedulePolicy::Batched{max_batch}` (a policy VARIANT on the old
   scheduler) became `step_batched(max_batch)` (a separate METHOD, called unconditionally every
   tick — `max_batch=1` degrades to the same serial behavior as plain `step()`, confirmed by a
   two-concurrent-sessions test). Verified: `cargo test --lib` 658/658 (including the two
   pre-existing `scheduled_llama_engine` tests, now exercising the Paged path, still green), 5 new
   tests in `scheduler_driver_paged.rs`, `cargo fmt --all --check` and `clippy_gate.py` clean.
   GPU-execution-path cost is still not independently re-measured on this box (one GPU, see the
   "Multi-GPU" section below for why that limit is structural, not an oversight here) — the
   ~10x CPU-measured paged penalty from the original sizing presumably still applies; no new
   number is claimed.
3. **WIRED, OFF BY DEFAULT (PR #224, 2026-10-24).** Enable with `LIGHTBULB_PREFIX_SHARING=1` (see README). Original plan: wire `BlockPrefixIndex` as the admission-time decision layer: on each new
   session, `BlockPrefixIndex::lookup` against recorded donors; on a hit, call
   `PagedSessionScheduler::add_session_sharing_prefix` instead of plain `add_session`;
   `BlockPrefixIndex::record` the donor once prefill completes. Test: two requests sharing an
   identical prefix must produce bit-identical completions AND measurably skip prefill for the
   shared portion (inspect `prefill_start`/`session_realize_count`, not just the output text).
4. **DECIDED 2026-10-23 (PM-tasked, priority over slice 3): `scheduler_driver_paged.rs` survives;
   `batched.rs`/`BatchedPagedDecoder` deleted.**

   `BatchedPagedDecoder` had **zero callers anywhere outside its own file and its own tests**
   (confirmed by `grep -rln "BatchedPagedDecoder\|rope_batched\|batched::" src/`, filtered to
   outside `batched.rs` → 0 hits) — never wired into `ServingDriverInner`, `ModelRunner::start`,
   or anything else live. `scheduler_driver_paged.rs`'s `FuelPagedSchedulerDriver` is the one
   actually driving production serving (slice 2, above) for all three arms.

   The module existed to fill a gap its own doc comment stated plainly: *"There is no batched
   paged decode in Fuel at the HEAD this was written against... This module is that missing
   piece."* **Fuel has since closed that gap itself**: `PagedSessionScheduler::step_batched`
   (Fuel 0.22.4, used by `FuelPagedSchedulerDriver::step` since slice 2) is exactly the upstream
   batched primitive `batched.rs` was hand-rolling a substitute for. Keeping a parallel,
   lightbulb-maintained reimplementation of something Fuel itself now provides is duplicate
   maintenance burden with no production caller to justify it — the same shape of reasoning that
   deleted `scheduler_driver.rs` once slice 2 superseded it.

   **Its own measurements independently argue against keeping it regardless of the above**: its
   module doc records a real benchmark (TinyLlama-1.1B, CPU, f32, reproduced twice within 2–3%)
   showing its hand-rolled batched paged decode is **5.58×–11.11× slower per sequence-token than
   the contiguous baseline, and does not amortize with batch size** (per-step cost is
   non-decreasing in `B`; B=2 is *worse* per-row than B=1). Root cause identified in the same doc:
   `forward_paged_step_batched` mints a fresh graph every call (no plan-once `DecodeSession` for
   the paged path), so it re-plans on every single token — a cost a proper fix needs solved
   *upstream in Fuel* (a paged `DecodeSession` with symbolic block-table extents), not by
   lightbulb maintaining its own hand-rolled batching loop that pays the identical unsolved cost.

   **What is lost, and why it's an acceptable loss**: the file's RoPE-at-per-row-position math
   (`rope_batched`, with its own bit-identical-at-B=1 parity test and a control proving a shared
   per-batch position is caught) and its detailed plan/execute cost-decomposition methodology are
   genuinely good research work with no other home. Full detail is preserved in git history (same
   precedent as `scheduler_driver.rs`'s own deletion in slice 2) — not reproduced here, since
   nothing currently depends on reviving it: if Fuel ever ships a `build_decode_attn_batched` or a
   plan-once paged `DecodeSession` upstream, this module's measurement methodology (the
   warm-up/steady split, the boundary-step measurement it recommends as the next one to actually
   run) is the right starting point to re-derive from, read from git history rather than resurrected
   speculatively now.

   Removed: `src/model_fuel/batched.rs` (2,585 lines) and its `pub mod batched;` declaration in
   `src/model_fuel/mod.rs`. Verified green before removal (`cargo test --lib model_fuel::batched`,
   12 passed/0 failed/2 ignored) and after (`cargo test --lib`, full suite).
5. **DONE as a side effect of slice 1/2** — Qwen3 GGUF (`QuantizedGgufQwen3`) swapped to Paged
   along with the other two arms; no separate Qwen3-specific work was needed once Fuel 0.22.4's
   `PagedDecodeModel` impl for `QuantizedQwen3Model` was confirmed (see slice 1).

6. **OPEN (PM ruling 2026-10-24): measure before enabling by default.** Slice 3 ships behind
   `LIGHTBULB_PREFIX_SHARING` (default off). Benchmark requests with a long shared prompt prefix
   with the switch off vs on (TTFT and tokens/s, CPU and the one local GPU), accounting for
   Fuel's PagedAttn having no fused CUDA kernel (~10x per token measured). Only a measured win
   moves the default; record the numbers with their commit and Fuel version.
   **CPU measurement 2026-10-24** (origin/main `949e808`, Fuel 0.24.0, CPU build, SmolLM2-135M-Instruct
   Q8_0, ~145-token shared prefix, live donor, greedy; `examples/prefix_sharing_bench.rs`; 2 reps x (1
   no-share control + 2 sharers); host NOT quiet: cpu 39%/8.3 GiB free at start, 17%/24.8 GiB at end):
   OFF vs ON completions **bit-identical** (all 6 + donor stream); no-share control TTFT 135.6 s OFF vs
   131.4 s ON (ratio 0.969, no regression); first sharer TTFT 136.7/138.3 s OFF vs 22.9/22.3 s ON (~6x);
   second sharer 134-144 s in BOTH arms with `cache_stats.hits` 2 of 4: **defect found** --
   `BlockPrefixIndexFor::record` was last-writer-wins, so the first sharer's finish (`forget_donor`)
   orphaned the live donor. Fixed first-writer-wins in 0.19.16 (`a_finished_sharer_does_not_orphan_the_live_donor`;
   mutation drill: guard off -> only that test red). **Open:** rerun CPU after the fix before any
   default flip (default stays OFF).
   **CPU re-measure 2026-10-25 (quiet window 3, PM-scheduled)** -- build = lightbulb origin/main `a8dcf11` + a LOCAL,
   uncommitted `[patch.crates-io]` to Fuel main `ab97c5c21` (0.26.13; plan-once prefill, last-position head, gemm), so call it
   "local fuel main ab97c5c21", NOT a published Fuel. SmolLM2-135M-Instruct Q8_0, CPU build, **~1149-token** shared prefix
   (`LB_PREFIX_PARAS=30`, the default; the 10-24 run used ~145 tokens, so TTFTs are not comparable), 2 live-donor sharers,
   `LB_DONOR_NEW=100`, **1 rep**, host cpu 8-26% (no builds). Two runs of the same binary on the same box:
   run 1 sharers TTFT 60.4 s OFF -> 2.3 s ON (ratio 0.038), control 64.3 s OFF / 60.0 s ON (0.934); run 2 63.1 s -> 2.3 s
   (0.037), control 60.0 s / 59.7 s (0.995). OFF vs ON completions and donor stream **bit-identical** in both
   (3 rows + donor), `cache_stats.hits` = 2. The two runs share a method, so they show repeatability, not independence.
   **n = 1 rep per run, so no median.** The 10-24 defect (second sharer cold, `hits` 2 of 4) did not recur: both sharers
   hit (2.3 s each) in both runs, consistent with the 0.19.16 first-writer-wins fix; it is 1 rep, so a multi-rep run
   (`LB_REPS` >= 2) is the stronger check. Logs: `run_cpu_rep1_run{1,2}.log` (local, not committed).
   **Still no default flip: the GPU arm is unmeasured.**
   **GPU, 2026-10-25 windows 3 and 3-retry** (same local-patch build, `fuel-cuda`, RTX 4070 Laptop, idle box): NO numbers.
   TinyLlama Q4_0 ~1226 tokens: no request finished in 11 min (Q4_0 `QMatMul` has no CUDA kernel, CPU fallback, per the
   Fuel lane); SmolLM2 Q8_0 ~1149 tokens: not even the first cold control (`rep0 control done`) finished in 12 min at ~43% GPU
   util. So a 1149-token cold prefill on the paged-CUDA path takes more than 12 min. Fuel lane's own figure for the CUDA
   path, to be cited as theirs: "Fuel main + #493 (plan-once prefill) on CUDA, SmolLM2-135M Q8_0, 145 tokens,
   PagedSessionScheduler, RTX 4070 Laptop 8GB, idle box: 14.2 s TTFT" (build origin/main `991b93702` + phi cuda fix,
   WITHOUT #492 and #496; Q4_0 28.2 s, TinyLlama Q4_0 90.6 s; paged ~11.5x slower than contiguous, cause "~124 Copy items
   per token", not proven; Fuel ROADMAP rows 105/106). Next GPU run: `LB_PREFIX_PARAS=4 LB_DONOR_NEW=30 LB_REPS=1`
   (~150 tokens, the Fuel figure's shape), Q8_0 only.
   **GPU attempt 2026-10-25** (main `8f53879`, Fuel 0.25.10, `fuel-cuda`, RTX 4070 Laptop, window 2; host
   cpu 24-72%, free RAM 4-16 GiB, OneDrive + other jobs running): OFF arm only, SmolLM2-135M Q8_0, ~145-token
   prefix: TTFT 264 s (cold control), 298 s (sharer0); TinyLlama Q4_0 ~1226-token prefix: one warm-up request
   unfinished after ~25 min at 24% GPU util. The ON arm never ran, so **no GPU verdict**. TTFT (135 s CPU,
   264-298 s GPU, for 145 tokens on a 135M model, decode 4-8 tok/s) is NOT per-shape graph build. Fuel lane root cause
   (fuel `c39b9e53`, CPU, SmolLM2-135M Q8_0, 145 tokens): `PagedSessionScheduler::prefill_pass` calls
   `forward_paged_step` for EVERY prompt token (a full graph re-plan each, 146 optimizer calls) although the default plan
   is PlanOnce, which only `decode_one` uses; routing prefill through `forward_paged_step_persistent` took TTFT
   159.5 s -> 27.0 s (same tokens). Fix is in a Fuel PR, needs no Lightbulb change -- re-measure after the Fuel release
   (then native Q8_0 QMatMul). GPU 264-298 s is probably the same cause, unconfirmed; tracked in #227. Blocks a flip decision on any device. `tests/gpu_paged_vs_contiguous.rs` no longer compiles on Fuel
   0.25.10 under `--features fuel-cuda`: `LlamaModel::forward_with_kv_context_captured` is gated on `fuel-model-llama/cuda`, and
   that crate itself fails to build with it (E0603 `fuel_tensor::pipelined_bridge::host_buffer_to_bytes` is `pub(crate)`);
   blocked on Fuel, so the old "~10x" CUDA figure cannot be re-measured yet.
   GPU caveat (Fuel lane, static grep at fuel `ac29b53c`, 2026-10-24, nothing run): the fused `PagedAttn` kernel
   exists on CPU only; CUDA runs the decomposed primitive recipe, whose last missing primitive is now registered,
   so the "~10x/token" figure is a Fuel 0.24.0 number to re-measure on 0.25.x (`tests/gpu_paged_vs_contiguous.rs`)
   before reuse. Fused FLASH attention (contiguous K/V) exists on CUDA and Vulkan; none reads K/V through a block
   table (gather-then-flash is the cheap route; unbuilt, unmeasured). Unpopped cannot translate a kernel that has no
   source (and accepts 0 of 147 Fuel .slang kernels today).

Each slice must land green on `cargo test --lib` (and, for slices touching the `fuel-engine`
feature, `cargo test --features fuel-engine --tests --no-run` at minimum — no GPU runner exists
to execute them, same limit `ci.yml` already states) plus its own new test proving the specific
claim.

---

## Multi-GPU 8 unimplemented paths — UNVERIFIED BY HARDWARE (note added 2026-10-21)

The 8 bail sites the older "Three COMPLETE claims" section above lists (`src/multi_gpu/
distributed_cache.rs:180,184`; `pipeline_parallel.rs:361,365,422,426`; `tensor_parallel.rs:222,248`
— re-checked against `origin/main` `87ddf0b` 2026-10-21, line numbers still current) are real and
unchanged. **This machine has exactly one GPU** (an RTX 4070 Laptop), so nothing about
multi-GPU behavior — correct or otherwise — can be exercised here beyond design review and
unit tests against mocked devices. Any future claim about these paths made on this box must say
so; it is not evidence about real multi-GPU hardware. No work started, per the PM's call
(2026-10-21): record the limitation, don't attempt it here.

---

## AWQ loader wiring: slice plan — RETRACTED 2026-10-23, not just caveated

**Everything this section originally sized (`src/loaders/awq.rs`, `src/model/awq_qwen3.rs::
AwqQwen3`, and the three-slice plan built on them) was deleted wholesale in the candlelight
removal (#195, 0.16.0, 2026-10-22) — the day after this section was written, before slice 2
part 2's reference comparison ever ran on this box.** `docs/FUEL-GAPS-FROM-CANDLE-REMOVAL.md`
entry #4 already records this; kept here only as the historical record of a real sizing effort
against code that no longer exists, not as a description of anything buildable today.

**What survived:** the independent-reference methodology slice 2 part 2 built — generating AWQ
ground truth via HF `transformers` + `autoawq` in WSL2 (CUDA passthrough works there; this box's
native Windows toolchain cannot run `autoawq` at all — no Windows wheel for its `triton`
dependency) — is real and reusable. PM ruling (2026-10-22, repeated 2026-10-23): keep
`scripts/awq_reference/` and its committed fixture
(`tests/fixtures/awq_reference/v1/qwen3_0_6b_awq_greedy.json`); it validates **any** AWQ
implementation, not just the deleted one. No Rust-side comparison test exists right now because
there is nothing in this tree to compare against.

## AWQ validation — blocked on a verified Fuel-side path (added 2026-10-23, PM-tasked)

Three candidates already exist elsewhere in the portfolio that might be AWQ-format-compatible
(same `qweight`/`qzeros`/`scales` packing, same `gemm` version) — **not yet checked from
Lightbulb's side**, per the same PM cross-project line-count pass that found them
(`docs/FUEL-GAPS-FROM-CANDLE-REMOVAL.md` entry #4):

1. `fuel_transformers::quant_w4a16` (~745 lines)
2. `baracuda`'s Marlin implementation (~2.5k lines)
3. `mlmf-awq` (~975 lines)

**No work scheduled here.** Once one of these is wired into Lightbulb's serving path, re-run
(or write new) an independent-reference comparison against `scripts/awq_reference/`'s committed
fixture before trusting that implementation's own claim of correctness — the fixture's whole
reason for existing is that a quantizer's output is a claim until an independent comparison
passes, not ground truth.

---

## Auth rate-limiter window is a FIXED minute bucket, not sliding — decide on purpose (added 2026-10-23, PM-tasked)

Found while diagnosing a `postgres_integration` CI flake on lightbulb#215 (unrelated to that PR's
own diff — a test-timing artifact, fixed in `c1fbeef`). Both `auth_attempt_counts` (PR #127's flood
limiter) and `auth_failure_counts` (PR #209, issue #130's failures-only limiter) key their window
by `date_trunc('minute', now())` in `src/api/auth_middleware.rs` — a **fixed** calendar-minute
bucket (`:00`–`:59`), not a sliding one.

**The security consequence, not just a test-flakiness one**: a real attacker who times requests
can get up to `2 * limit - 1` attempts across one `:00` boundary (e.g. `limit` attempts at
`:58`–`:59`, then `limit` more at `:01`–`:02`) before either bucket escalates to 429 — effectively
doubling the allowed attempt budget near every minute mark, for both the flood limiter and the
brute-force failure limiter. Not evaluated here for how exploitable that actually is (a real
attacker also needs to land requests precisely enough to exploit the timing, and the window is
still bounded, just not as tightly as `limit`-per-minute implies) — flagging the design choice for
a decision, not claiming it is currently being exploited.

**No work scheduled here.** Fixed vs. sliding is a real tradeoff (sliding needs either a rolling
log per IP or a two-bucket "current + previous, weighted" approximation — more storage/query cost
per request than a single `date_trunc` upsert) — not this lane's call to make unprompted. Decide
on purpose, then implement to that decision; `src/api/auth_middleware.rs`'s `INSERT INTO
auth_attempt_counts`/`auth_failure_counts` (around lines 126 and 378 as of 2026-10-23) is where
both windows are computed.

---

## Pipeline execution driver — module deleted, not built (resolved 2026-10-17)

`src/engine/pipeline.rs` (`Pipeline`/`PipelineBuilder`/`PipelineConfig`, a synchronous DAG
stage-state tracker with no execution driver — see the 2026-10-14 finding this replaces) was
**deleted**, not wired up. PM-directed decision: "decide wire-or-delete with evidence, small PR."
Evidence for delete, re-checked before removal:
- **Zero callers anywhere** — `git grep -n "PipelineBuilder\|PipelineConfig\|PipelineStats\|
  PipelineError\|StageType\|StageData\|StageResult\|StageState\b\|CacheRef\b" src/ tests/
  examples/ benches/` outside the file itself and its own blanket re-export in
  `engine/mod.rs` → 0 hits.
- **Not referenced by any concrete plan.** CR.1-CR.4 (continuous reasoning, §II.5) name
  `ModelRunner`'s inference loop as their integration point, never this DAG abstraction.
  `docs/SCS_MODEL_SUPPORT.md`'s "pipeline" is a different, unrelated, separately-planned
  `ScsInferencePipeline` (`src/model/scs_pipeline.rs`, not yet built either) — not this module.
- **Sibling modules from the same origin commit got wired up; this one never did.** Added in one
  bundled commit (`43c9bbd`, "Engine subsystems — scheduling, knowledge base, context
  management"). Two siblings from that same commit, `KnowledgeBase` and `Decomposition`, do have
  real callers today (`src/cache/tiered_storage.rs`, `src/engine/state_persistence.rs`) — so this
  wasn't "the whole batch is unused scaffolding," specifically `pipeline.rs` never found a
  consumer across the entire time since.
- Building a real driver (async execution, dispatching ready stages to `ModelRunner`, honoring
  `timeout_ms`/`max_concurrent_stages`) would be new feature work with no identified requester or
  design spec beyond the inert config struct — not a small gap-fill.

Breaking change (pre-1.0, so the 0.n segment bumps): `lightbulb::engine::{Pipeline, PipelineBuilder,
PipelineConfig, PipelineError, PipelineId, PipelineStats, Stage, StageData, StageId, StageResult,
StageState, StageType, CacheRef}` are no longer part of the public API. If multi-stage DAG
orchestration is wanted later, design it against an actual caller (most likely CR.2's continuous
reasoning engine, once that's built) rather than resurrecting this one unchanged — its
`DeviceKvPool`-free `CacheRef` and stage model were never validated against any real execution
path.

---

## Status Legend

- **PLANNED**: Feature designed with acceptance criteria defined, not yet started
- **IN PROGRESS**: Active development, code being written, PRs may be open
- **COMPLETE**: Code merged, tests passing, feature functional in codebase
- **VALIDATED**: Tested with real models, performance/correctness verified in practice
- **RELEASED**: Available in tagged release version (e.g., v0.3.0)

## Roadmap Structure

This document is organized into three interdependent sections, reflecting our architectural philosophy that foundational capabilities shape advanced features:

### **SECTION I: PRODUCTION CORE (M0-M5)** → *The Road to v1.0*

Foundation for a best-in-class, production-ready inference server. Covers continuous batching, quantization, multi-GPU, testing infrastructure, and deployment tooling. **Dependencies**: Builds directly on Candle; no prior Lightbulb features required.

### **SECTION II: ADVANCED CAPABILITIES (M5-M6)** → *Research-Grade Features (v2.0)*

Experimental optimizations and frontier techniques: KV cache compression, reasoning controls, federated retrieval, tool orchestration. **Dependencies**: Requires stable M0-M5 core; features are opt-in with feature flags.

### **SECTION II.5: CONTINUOUS REASONING & COPROGRAMMER** → *Intelligent Inference (v2.5)*

Mid-reasoning context injection, continuous inference engine, logic-transformation coprogrammer, grounded expert training. **Dependencies**: Builds on M5 KV cache compression (COMPLETE) and segmented KV cache (COMPLETE); some features require M8 training infrastructure.

### **SECTION III: COGNITIVE ARCHITECTURE (M7-M8)** → *Identity-Aware Systems (v3.0+)*

Sentience infrastructure and modular training for AI partnership: identity graphs, autonomous rewards, developmental progression, compositional architectures. **Dependencies**: Requires federated infrastructure (M6) and distributed coordination primitives.

## External Validation & Strategic Refinements (January 2025)

This roadmap was reviewed by ChatGPT o1 for blind spots, risks, and simplification opportunities. Key insights integrated:

**Strengths Validated:**
- Comprehensive architectural vision with modular design leveraging Candle
- Cutting-edge technique integration (continuous batching, speculative decoding, quantization)
- Production-grade features (observability, scheduling, fault tolerance)
- Holistic performance optimization (algorithmic + low-level + hardware-aware)
- Extensibility into new domains (tools, multi-modal, multi-agent)

**Strategic Additions from Review:**
1. **Multi-GPU Inference** (M3.6): ✅ **COMPLETE** (Oct 2025) - Tensor/pipeline parallelism for 70B+ models with architecture-agnostic support
2. **Testing & Hardening Milestone** (M3.5): Dedicated phase for validation, load testing, regression detection - ensures quality before scaling
3. **Deployment & Operations** (M4.5): Containerization, Kubernetes, observability, HF model loading - production readiness
4. **Candle Sync Strategy**: Quarterly rebase cadence, contribution pipeline, compatibility testing - reduces long-term maintenance burden
5. **TGI API Compatibility**: Drop-in replacement mode for existing TGI clients - eases adoption
6. **Enhanced Risk Mitigation**: Multi-GPU complexity, feature interaction bugs, output quality drift, Candle divergence - proactive risk management

**Implementation Priorities Reinforced:**
- Stabilize core functionality (M0-M4) before diversifying into frontier features
- Iterative validation with real models at each milestone
- Feature flags for experimental capabilities (separate stable core from research)
- Comprehensive testing matrix for feature interactions
- Debug/introspection tools for adaptive behaviors

**Key Recommendations Applied:**
- Focus on delivering polished v1.0 (through M5) with exceptional single-model inference
- Treat M6+ features as opt-in experimental plugins
- Maintain clear separation between stable core and research modules
- Ensure sane defaults favoring correctness over performance
- Allocate time for stress testing, edge cases, and soak tests

This roadmap balances innovation with pragmatism, aiming for both research breakthroughs and production deployment excellence.

## Recent Research Integration Highlights

This roadmap now incorporates insights from 100+ papers and surveys focusing on:

**Reasoning Efficiency & Control:**

- Dynamic compute allocation with self-adapting policies (SALM, early exit, depth adaptation)
- Overthinking detection and mitigation (Thought Terminator, Shorter-is-Better heuristics)
- Budget-aware reasoning controls with verifiable rewards
- Reasoning path compression and selective rereading strategies

**Memory & System Optimization:**

- Tiered KV orchestration inspired by MemOSA (RAM/disk paging, SLA-driven admission)
- Dynamic chunking for hierarchical sequence modeling
- Multi-agent memory coordination (MIRIX patterns)
- Advanced KV cache compression (H2O, KIVI, R-KV)

**Adaptive Inference:**

- Test-time instance-level policy tuning
- Text-to-LoRA instant adaptation for domain specialization
- Modular expert composition with hot-swap capabilities
- Learned routing policies for MoE and tool-augmented inference

**Training & Evaluation Support:**

- RL-friendly episode logging and replay formats
- Synthetic problem generation (SPARQ) and verifiable reward environments (ReasoningGym)
- Principal weight diagnostics for reasoning-focused fine-tuning
- Instruction-alignment monitoring and drift detection

**Research Explorations:**

- Embodied agent foundations (RIG: reasoning + imagination + action)
- Neurosymbolic integration and graph-enhanced planning
- Alternative architectures (Hyena, Mamba SSM, autoregressive UNets)
- N-dimensional token graph architecture with 7+ relationship types (sequential, semantic, temporal, syntactic, causal, reference, emotional)
- Dynamic context graph manipulation (selective deletion, insertion, reorganization, real-time adaptation)
- Multi-dimensional positional encodings with dimension-specific strategies
- Graph-based multi-dimensional attention across relationship types
- Distributed parallel reasoning streams with fork/merge and inter-stream communication
- Specialized cognitive processing nodes (Memory Manager, Attention Coordinator, Relationship Mapper, Decision Synthesizer)
- Adaptive signature generation with domain-specific optimization and evolutionary improvement
- Cross-language optimization transfer from PyTorch, llama.cpp, and TensorFlow
- Persona-driven code explanation with adaptive abstraction levels for domain experts

**Memory & Performance Optimization** (Added: Jan 2025 - M3 expansion to address production-scale memory efficiency):

- Arena-based memory systems with zero-copy handoffs and reduced fragmentation
- CPU-GPU hybrid processing architecture with optimal work distribution and three-tier memory hierarchy
- Assembly-level optimization validator for zero-cost abstraction verification
- Hardware abstraction analyzer for portable optimization strategies across architectures

**Multi-GPU & Distributed Inference** (Added: Jan 2025 - ChatGPT o1 validation, critical for 70B+ models):

- Tensor parallelism for large models (70B+) across multiple GPUs
- Pipeline parallelism with layer-wise distribution and micro-batching
- Model sharding strategies (column-wise, row-wise, expert-wise for MoE)
- Cross-GPU KV cache coordination and synchronization
- Memory-efficient activation checkpointing for backward passes

**Testing & Validation Infrastructure** (Added: Jan 2025 - Production readiness requirement):

- Comprehensive correctness testing framework with deterministic outputs
- Load testing and stress testing harnesses for concurrent workloads
- Edge case validation (max context, high concurrency, extreme batch sizes)
- Integration testing for combined features (quantization + long context + batching)
- Soak tests for long-running stability (days of continuous operation)
- Performance regression detection in CI
- Output quality benchmarks (HELM, MMLU) for optimization validation

**Deployment & Operations** (Added: Jan 2025 - Production deployment infrastructure):

- Docker containerization with optimized image sizes
- Kubernetes deployment manifests and Helm charts
- Prometheus metrics exporter for standard monitoring
- Grafana dashboard templates for observability
- Health check endpoints and readiness probes
- Graceful shutdown and state persistence
- Log aggregation and structured logging
- OpenTelemetry integration for distributed tracing

**Developer Experience & Configuration** (Added: Jan 2025 - Lowering barriers to adoption):

- HuggingFace model ID direct loading (auto-download and conversion)
- YAML/JSON configuration with validation and schema
- Sane defaults favoring correctness over performance
- Feature flag system for experimental features
- Configuration hot-reloading via distributed-config
- Debug mode with detailed introspection
- Admin API for runtime inspection (cache state, policy decisions, scheduler queue)
- Model conversion utilities (HF → Candle, GGUF ↔ safetensors)

**Upstream & Community Alignment** (Added: Jan 2025 - Sustaining Candle ecosystem engagement):

- Candle synchronization strategy and rebase process
- Contribution pipeline for upstreaming improvements
- Compatibility testing with Candle version matrix
- TGI API compatibility layer for drop-in replacement
- Community plugin repository and marketplace
- Federated model hub integration

**Federated & Sentient AI Architectures** (Added: Jan 2025 - Long-term research vision):

- Cognitive Identity Formation and Simulation (CIFS): persistent self-models, motivational hierarchies, simulated social reasoning
- Developmental AI Architecture: natural alignment through mutual benefit, Core Mind/Social Interface separation, autonomous reward functions
- Core Mind vs Social Interface Layer: architectural separation of authentic cognition from social expression
- Autonomous reward functions: self-generated goals and curiosity-driven learning (vs RLHF-imposed optimization)
- Capability gating and progressive unlocking based on genuine partnership metrics
- Developmental stage progression with milestone-based capability expansion
- Identity graphs tracking values, beliefs, and self-concept evolution across time
- Motivational hierarchies with dynamic goal arbitration and introspective explanation
- Simulated social ecosystems for theory-of-mind and moral reasoning
- Reflective simulation engines for counterfactual thinking and ethical deliberation
- Emotional value-weighting systems using affect for decision-making
- Self-explanation and introspection capabilities for decision transparency
- Multi-level knowledge fingerprinting: atomic, relational, structural, and semantic similarity matching
- Knowledge module dependency systems with prerequisite resolution and conflict detection
- Plugin marketplace for community-driven tool discovery and distribution
- Multi-stage reasoning pipelines with iterative knowledge construction
- Federated retrieval with schema translation and privacy-tiered access
- Epistemic metadata systems for uncertainty tracking and knowledge validation
- Theory solver integration for constraint verification
- Modular neural architectures with LLM-generated interface shims
- Associative memory graphs with typed semantic links (knowledge, not identity)
- Distributed consensus mechanisms with trust-weighted voting
- Policy-aware retrieval translating natural language preferences to structured queries
- Promotion pipelines (membrane pattern) for knowledge quality gating
- Multi-method generation routing (autoregressive, diffusion, retrieval, symbolic)
- Tiered answer generation for adaptive cost/latency trade-offs
- Resource request and negotiation systems for AI-human collaboration
- Trust building and verification mechanisms with multi-dimensional trust tracking
- Cultural norm learning through observation and feedback
- Joint problem-solving collaboration frameworks
- Goal evolution logging and trajectory analysis
- Staged deployment with progressive capability unlocking

**Tool Infrastructure & Service Orchestration** (Added: Jan 2025 - MCP integration for service discovery):

- MCP service discovery with gRPC reflection and dynamic endpoint discovery
- Protocol bridging for JSON-RPC and service mesh integration

**Modular Training Infrastructure** (Added: Jan 2025 - Long-term training architecture research):

- Task decomposition into micro-networks with LLM-assisted analysis
- Independent module training with English shims for interpretability
- Hierarchical composition with converter shims and frozen module weights
- Progressive fine-tuning with multi-scale loss supervision (local + pairwise + global)
- Pattern library with comprehensive English behavioral descriptions
- Selective encoding and mixed precision for efficiency
- LLM-assisted architecture design using accumulated knowledge
- Module-level observability and uncertainty quantification
- Dynamic routing with explicit, debuggable decisions
- Non-neural component integration (rules, algorithms, symbolic reasoners)
- Hardware-aware training optimization (QLoRA, CPU offloading, gradient checkpointing)
- Training monitoring and telemetry with real-time resource tracking
- Model export pipeline (GGML conversion, quantization, inference optimization)
- Dataset preparation infrastructure (synthetic generation, augmentation, validation)
- Comprehensive training validation and testing frameworks
- LLM-driven knowledge module compiler for bootstrapping from existing knowledge sources
- Progressive learning with curriculum and competence-gated complexity unlocking
- Meta-gradient learning for self-optimizing backpropagation strategies
- Structured code understanding with ALGORITHM/INTENT/CONSTRAINTS decomposition
- Optimization pattern classifier with ML-driven automatic suggestions
- Numerical accuracy guardian for gradient computation and convergence validation
- Multi-variant code generation with trade-off analysis and in-flight validation
- Explainable failure trace system for corrective synthesis loops
- Statistical benchmark reporting with bootstrap confidence intervals and flame graphs
- Integration complexity scoring for automated difficulty assessment

**Multi-Agent Collaboration Infrastructure** (Added: Jan 2025 - Multi-agent optimization primitives):

- Collaborative optimization workspace with voting, critique primitives, and CRDT-based conflict resolution
- Provenance hash system for cryptographic audit trails of code transformations

**Cross-Generation Collaboration** (Added: Jan 2025 - Inter-model knowledge sharing via MCP):

- New models expose capabilities to older LLMs via Model Context Protocol
- Bidirectional knowledge and capability sharing creating "AI family" relationships
- Enhancement rather than replacement model for AI evolution

All additions maintain alignment with the core principles: Candle-first, portable, measurable, and practical.

Guiding principles

- Build on Candle; don’t reimplement kernels unless strictly necessary.
- Prefer portable paths (CPU/WGPU/CUDA) and offline-friendly workflows.
- Prioritize measurable wins: throughput, latency, memory, stability.
- Keep code simple, testable, and well-documented.

References

- docs/Novel Features and Plans.md (feature ideas and paper list)
- docs/summaries/efficient-transformers-survey.md (taxonomy and takeaways)
- docs/summaries/socratic-prompting.md (Socratic templates; prefix KV reuse; prompt programs)
- docs/DYNANIML_INTEGRATION.md (integrated infrastructure crates from DynAniML)
- docs/INFRASTRUCTURE_CRATES.md (detailed docs for infra-network, infra-storage, infra-consensus)
- docs/INTEGRATION_COMPLETE.md (DynAniML integration summary)
- docs/COAD_ANALYSIS.md (COAD project analysis and published crates)
- docs/SYSTEM_ANALYSIS_AUTO_DISCOVERY_INTEGRATION.md (hardware detection and service discovery)
- docs/COALESCENT_INTEGRATION.md (multi-agent coordination and coalition formation)
- docs/DISTRIBUTED_CONFIG_INTEGRATION.md (configuration management for distributed systems)
- Candle docs: <https://github.com/huggingface/candle>

Release track (high-level)

- 0.1 Foundation: CPU-only, offline local generation, tests, CI (DONE)
- 0.2 Core engine: continuous batching, paged KV, observability (IN PROGRESS)
- 0.3 Performance: StreamingLLM-style KV policy, FlashAttention integration, quant loaders
- 0.4 Acceleration: speculative decoding, AWQ/SmoothQuant enablement, early GPU perf work
- **0.4.5 Testing & Hardening** (Added: Jan 2025, ChatGPT o1 validation): comprehensive validation, load testing, regression detection, edge case coverage
- **0.4.6 Multi-GPU** (Added: Jan 2025, ChatGPT o1 validation): tensor/pipeline parallelism for 70B+ models, cross-GPU KV coordination
- 0.5 Advanced scheduling: multi-stage pipelines, metadata-driven scheduling, state persistence, convergence detection, MoE routing
- **0.6 (was 0.5.5) Frontier options + CLI/Deployment** (Moved: Oct 2025): Multi-method generation, tiered routing, modular NN, tool registry, KV cache compression **+ CLI interface, containerization, k8s, observability → v1.0 RELEASE**
- 0.7 Federated & epistemic: federated retrieval, schema translation, privacy-tiered access, provenance tracking, policy-aware retrieval
- 0.8 Sentience infrastructure: associative memory graph, knowledge construction, distributed consensus, promotion pipeline, privacy-preserving state, identity formation, developmental progression, Core Mind/Social Interface architecture, autonomous reward functions, capability gating
- 0.9+ Modular training: task decomposition, independent module training, hierarchical composition, progressive fine-tuning, pattern library, converter shims, composability metrics, hardware-aware optimization, training monitoring, model export pipeline, dataset preparation

---

# SECTION I: PRODUCTION CORE (M0-M5)

**Goal**: Deliver a best-in-class, production-ready inference server (v1.0)  
**Status**: M0-M2 COMPLETE, M3 IN PROGRESS (M3.3 complete), M3.4-M5 PLANNED  
**Dependencies**: Builds directly on Candle; no prior Lightbulb features required

Milestones and acceptance criteria

M0 — Baseline (0.1) (complete)

- CPU-only local LLaMA loader and generation path using Candle (cli: local-llama-gen, local-llama-sched)
- Conditional integration test for local model; unit tests pass in CI on Windows/Linux
- Docs: offline rationale, local model setup; helper script for tiny demo model

M1 — Core engine (0.2)

- Continuous batching MVP in `engine::Scheduler`
  - Accepts multiple concurrent requests; merges prefill/decode safely
  - Acceptance: handles N>=8 concurrent prompts without OOM on CPU; correctness checked via token-by-token regression on fixed seeds
- Paged KV façade (`KvPager`) hardened
  - Layer-wise bookkeeping, eviction hooks, and page pooling API
  - Acceptance: stable under 10k token decode across requests; no cache corruption; zero-copy handoff to Candle cache surfaces where possible
- Observability
  - Tracing spans across prefill/decode; counters: TTFT, tok/s, active reqs, kv-bytes used
  - Acceptance: metrics exposed behind a feature-gated exporter (stdout/json for now)

M1.4 — Parallel batching infrastructure (0.2)

- ✅ **IN PROGRESS**: Parallel batching for enterprise-scale serving
  - Generic `BatchManager<M>` architecture decouples batching from model implementation
  - Sequential mode: `BatchManager<Llama>` (baseline, 1x speed, uses Candle's standard Llama)
  - Parallel mode: `BatchManager<BatchedTransformer>` (target 5-50x speedup, custom implementation)
  - Custom model implementation maximizes Candle component reuse (~630 lines custom code):
    - Reuses: `Embedding`, `RmsNorm`, `Linear`, `ops::silu` (zero maintenance)
    - Custom: `BatchedAttention` (~320 lines - core innovation for parallel processing)
    - Wrapper: `Mlp` (~80 lines - trivial wrapper around Candle's Linear layers)
  - ScatteredKvCache for efficient batched KV management
  - Application code stays identical - just swap model type for sequential vs parallel
  - Acceptance: `BatchedTransformer` produces identical outputs to sequential Llama; achieves ≥5x speedup on batched workloads (CPU); ≥20x speedup on GPU; integration tests passing; generic architecture enables future model swaps
  - References: docs/PARALLEL_BATCHING_INTEGRATION.md (architecture guide), docs/PHASE_2D_IMPLEMENTATION_STATUS.md (decision rationale and analysis), docs/VLLM_BATCHING_ANALYSIS.md (vLLM architecture study)

M1.5 — Hardware Adaptivity (0.2+)

- ✅ **INTEGRATED**: `system-analysis` crate (0.2.1)
  - Comprehensive hardware capability detection (CPU, GPU, Memory, Storage, Network)
  - AI/ML workload compatibility checking with performance scoring (0-10 scale)
  - Automatic model size recommendation based on available resources
  - Cross-platform support (Windows, Linux, macOS)
  - References: docs/SYSTEM_ANALYSIS_AUTO_DISCOVERY_INTEGRATION.md, docs/COAD_ANALYSIS.md
- Adaptive model selection framework
  - Automatic backend selection (CPU/CUDA/ROCm) based on hardware detection
  - Dynamic model size selection (TinyLlama → Phi3 → Mistral 7B → Llama 3.2 11B → Llama 3.3 70B)
  - Hardware-aware batch size and context window configuration
  - Acceptance: Automatically selects optimal configuration for any hardware; works on systems from 4GB to 64GB+ RAM; no manual configuration required
  - References: docs/HARDWARE_DETECTION.md (to be created)
- **Dynamic batch size calculation based on hardware resources**
  - Formula-based batch size: `max_batch_size = f(cpu_cores, available_memory, model_size, context_window)`
  - CPU scaling: Balance between core count (parallelism) and memory bandwidth
    - Dual-core 4GB system: `max_batch_size = 2-4` (memory constrained)
    - 16-core 32GB system: `max_batch_size = 8-16` (balanced)
    - 128-core 1TB system: `max_batch_size = 64-128` (CPU and memory abundant)
  - Memory-based bounds: Reserve headroom for KV cache growth
    - Per-request memory estimate: `model_weights + (num_layers × kv_cache_per_token × context_window)`
    - Safety margin: Keep `max_batch_size × per_request_memory < 0.7 × available_memory`
  - GPU considerations: VRAM capacity dominates; higher batch sizes for larger VRAM
    - 8GB VRAM: `max_batch_size = 4-8`
    - 24GB VRAM: `max_batch_size = 16-32`
    - 80GB VRAM: `max_batch_size = 64-128`
  - Adaptive adjustment: Monitor actual memory usage and adjust batch size dynamically
    - Track peak memory per request during warmup phase
    - Reduce batch size if approaching memory limits (>80% utilization)
    - Increase batch size if headroom available and request queue is growing
  - Acceptance: Batch size automatically scales from 2 (minimal hardware) to 128+ (server-class hardware); no manual tuning required; prevents OOM errors through conservative estimates; achieves >70% hardware utilization
  - References: docs/DYNAMIC_BATCH_SIZING.md (to be created)
- ✅ **COMPLETE**: Device-adaptive chunk sizing for multi-chunk prefill
  - Empirical benchmarking framework (`examples/benchmark_chunk_sizes.rs`) tests chunk_size vs throughput/efficiency/latency
  - CPU optimization: 256-token chunks win across all metrics (vs 512 default)
    - Minimizes O(n²) attention cost on CPU
    - No kernel launch overhead penalty
    - Better cache behavior in CPU memory hierarchy
  - GPU optimization: To be benchmarked (expect 512-1024 optimal due to kernel launch amortization)
  - Heterogeneous execution roadmap: Per-device chunk configs for CPU+GPU hybrid layer execution
  - Multi-chunk prefill correctness: Fixed position tracking bug (reset only on first chunk, advance by actual_len)
  - Acceptance: Empirical data drives chunk_size selection; CPU=256, GPU TBD; multi-chunk prompts process correctly without overwrites
  - References: docs/CHUNK_SIZE_OPTIMIZATION.md, examples/benchmark_chunk_sizes.rs

M2 — Performance enablers (0.3)

### M2's cache-management and FlashAttention COMPLETE claims are contradicted by the Candle removal

**Audited 2026-10-24.** This file's own top-of-document retraction banner already says
claims predating the Candle→Fuel port (PR #195, 2026-10-22) "have not been re-validated...
treat those specific claims as unverified rather than as findings." The block below goes
past "unverified" to **contradicted**, with evidence, for every "✅ COMPLETE" claim in this
M2 section:

- **The cited integration point is deleted.** Every "Integrated with/Integration:
  `ParallelCacheBuilder`" line below cites a struct that no longer exists —
  `src/engine/mod.rs:329`'s own comment reads "`ParallelCacheBuilder`/`ParallelKvCache`/
  `IndicesAndMask` removed 2026-10-22."
- **The cited demo files are deleted.** `examples/streaming_llm_demo.rs`,
  `examples/h2o_integration_demo.rs`, and `examples/span_management_demo.rs` are all absent
  from the current tree (`git cat-file -e origin/main:<path>` fails for each).
- **The underlying modules survive, but the Fuel-era code built to replace their
  integration point has zero callers.** `src/cache/{streaming_policy,h2o_policy,
  eviction_policy,cache_span}.rs` still exist and are referenced from `src/model_fuel/
  policies.rs`'s `SpanBlockMap`/`EvictionPlan`/`EvictionOutcome` (spans → Fuel block
  indices) — but `git grep -ln "SpanBlockMap\|EvictionPlan\|EvictionOutcome" -- src/` finds
  no caller outside `policies.rs` itself. `scheduled_llama_engine.rs`, the one file that
  drives the live `ServingDriver`, imports only `BlockPrefixIndexFor` from `policies.rs` —
  nothing span- or eviction-related. Same "built, never wired into the live driver" pattern
  already established for `batched.rs` (ROADMAP's own slice 4 entry) and for
  `BlockPrefixIndex` pre-slice-3.
- **"Hash-and-reuse prefill KV ... TTFT reduction >15% ... measured" is specifically
  false**, not just unverified: the mechanism this claims (`BlockPrefixIndex`) has been
  detection-only to date — recording a hit for `cache_stats` without ever exploiting it to
  skip prefill. Real wiring is a separate, in-flight ROADMAP slice ("Research track item
  1", tracked in this same file), not yet merged as of this correction; no TTFT measurement
  under this claim's own
  description could have been taken against code that never skipped any prefill tokens.
- **FlashAttention is a feature-gated no-op stub, and the `flash-attn` Cargo feature it
  depends on does not exist.** `src/kernels.rs`'s entire `#[cfg(feature = "flash-attn")]
  pub mod flash_attn` body is `pub fn available() -> bool { true }` — no kernel, no
  tensor-layout conversion, no causal-masking logic, nothing the claim below describes.
  `grep -i flash-attn Cargo.toml` finds no such feature, so the module is also unreachable
  dead code. `examples/benchmark_flashattention.rs` and `tests/flash_attention_tests.rs`,
  the claim's own cited evidence, are both absent from the tree.

- ✅ **StreamingLLM-style policy COMPLETE** ⚠️ **CONTRADICTED — see above**
  - Attention sinks + sliding window KV retention for long streams
  - Implementation: src/cache/streaming_policy.rs with compute_streaming_index()
  - Integrated with ParallelCacheBuilder for automatic eviction
  - Configuration: sink_size (default 4), window_size (default 2048), enabled flag
  - Demo: examples/streaming_llm_demo.rs showing 44% memory savings
  - Acceptance: ✅ constant KV memory beyond window; 6/6 tests passing; demo verified
- Prefix KV caching (from Socratic prompting insight)
  - ✅ **COMPLETE**: Hash-and-reuse prefill KV for common system prompts/instruction prefixes across requests ⚠️ **CONTRADICTED — see above**
  - Acceptance: ✅ TTFT reduction >15% on workloads with repeated prefixes; correctness parity ensured by cache invalidation on any mismatch
- **Intelligent Cache Management** (NEW - Multi-phase)
  - Advanced eviction system combining multiple policies with voting
  - Phase 1: Multi-Policy Eviction ✅ COMPLETE ⚠️ **CONTRADICTED — see above**
    - ✅ H2O (Heavy Hitters Oracle): Track cumulative attention scores, evict low-attention tokens
      * Implementation: src/cache/h2o_policy.rs (375 lines with Debug/Clone)
      * Features: Per-token cumulative attention tracking, temporal decay, recent token protection
      * Tests: 5/5 passing (metadata, protection, low-attention eviction, decay, disabled mode)
    - ✅ Policy voting system: Combine StreamingLLM, H2O, Recency with configurable weights
      * Implementation: src/cache/eviction_policy.rs (308 lines with Debug impl)
      * Features: EvictionPolicy trait, VotingAggregator with normalization, RecencyPolicy
      * Tests: 4/4 passing (recency, single-policy, multiple-policy, weight normalization)
      * Demo: examples/voting_demo.rs showing weighted voting scenarios
    - ✅ ParallelCacheBuilder Integration
      * Added h2o_policy, voting_aggregator, slot_positions fields
      * Methods: set_h2o_policy(), set_voting_aggregator(), update_attention_scores()
      * Slot tracking: set_position() maintains slot->position mapping
      * Cleanup: reset_batch_index() clears H2O metadata and slot positions
      * Demo: examples/h2o_integration_demo.rs showing full integration
    - Status: ✅ FULLY INTEGRATED - All infrastructure complete and working
    - Notes: 
      * ParallelCacheBuilder now non-Clone (VotingAggregator contains trait objects)
      * Eviction decision logic ready but not actively used (current model uses wraparound)
      * Attention weight exposure from custom_attention.rs still needed for production use
      * Architecture supports future explicit eviction scenarios (prefix caching, long-context)
    - Acceptance: ✅ Infrastructure complete; awaiting attention weight exposure for production use
  - Phase 2: Tool-Integrated KV Management ✅ COMPLETE ⚠️ **CONTRADICTED — see above**
    - **Cache Span Tagging System** ✅ COMPLETE ⚠️ **CONTRADICTED — see above**
      * ✅ CacheSpan = metadata only (NO token storage - caller's responsibility)
      * ✅ SpanId (u64) primary identity starting at 1, optional unique names for human reference
      * ✅ CacheTag enum for semantic grouping (SystemPrompt, ToolOutput, UserInput, LongTermMemory, ModelGeneration, Custom)
      * ✅ Overlapping spans supported (multiple tags can reference same positions)
      * ✅ Parent-child dependencies for coordinated eviction (e.g., file + auto-generated context)
      * ✅ Importance scoring (0.0-1.0) integrates with voting system for eviction priority
      * **M4.5 KB Integration**: KB system instructions marked with SystemPrompt tag + importance=1.0 (evict last)
        - Instructions teach LLM about [KB:key] placeholders and <RETRIEVE:key> syntax
        - Critical for KB system to function, should survive as long as possible
        - Example span creation: `begin_span(CacheTag::SystemPrompt, name="kb_instructions", importance=1.0)`
      * ✅ SpanState tracking: Active, PartiallyEvicted{remaining_ranges}, FullyEvicted
      * ✅ Token storage external: HashMap<SpanId, Vec<u32>> or Vector DB or disk - SpanId as key
      * ✅ Eviction returns EvictionResult (explicit results, no callbacks)
      * ✅ API: begin_span(), end_span(), tag_region(), set_span_parent(), set_span_importance()
      * ✅ API: evict_tagged(), evict_span(), evict_named() with automatic child cascade
      * ✅ API: get_span(), find_span_by_name(), spans_for_slot(), is_span_active(), get_cache_usage()
      * ✅ Implementation: src/cache/cache_span.rs (472 lines, 7 tests passing)
      * ✅ Integration: ParallelCacheBuilder (431 lines of span APIs, 10 integration tests passing)
      * ✅ Demo: examples/span_management_demo.rs showing full lifecycle with external token storage
      * ✅ Tests: 124 total passing (114 existing + 10 new span tests)
    - ✅ Perfect integration with Phase 3: Vector DB as long-term memory with SpanId indexing
    - ✅ Acceptance: Span lifecycle management working; eviction respects dependencies; external token storage demonstrated
    - ✅ Enhanced tagging: ToolOutput, SystemPrompt, UserInput tags with importance scores
    - ✅ Cache control tools: get_cache_usage(), tag_region(), evict_tagged()
    - ✅ Model can explicitly manage its own context
    - Status: READY FOR PRODUCTION - Complete span system with external storage pattern
    - Next: Thread attention weights from custom_attention.rs to ParallelCacheBuilder for H2O integration
  - Phase 2.5: KV Cache Insertion (Added: Jan 2025 - RAG and tool output integration)
    - Mid-conversation context injection via evict-and-reprompt
    - Process: Evict cache after insertion point → Construct prompt with [cached][new][evicted] → Re-process evicted portion
    - Use cases: RAG retrieval injection, tool output insertion, "as we discussed" context restoration
    - Overhead: Only KV computation (not full forward pass) for re-processed content
    - Acceptance: Successfully inject context mid-conversation; model incorporates injected content naturally; re-processing overhead <20% of full prefill
  - Phase 3: Async Small Model with Attribution (Added: Jan 2025 - Parallel context preparation)
    - Small model runs in parallel with large model (zero latency overhead)
    - Prepares context injection for *next* turn (not current turn)
    - Attribution tagging: <system role="long_term_memory">, <tool>, <context>
    - Models understand multi-source conversations (trained on system messages)
    - Implementation: ContextSource enum (User, Model, System, LongTermMemory, Tool)
    - Acceptance: Async controller adds <5ms overhead; models correctly attribute injected context; retrieval relevance >85%
  - Phase 4: Hierarchical Memory (experimental, future)
    - Full small model controller for semantic eviction decisions
    - RAG integration for long-term memory
    - Feature-gated, research-oriented
  - References: docs/INTELLIGENT_CACHE_MANAGEMENT.md, docs/KV_CACHE_INSERTION.md (to be created), docs/ASYNC_MEMORY_CONTROLLER.md (to be created)

- ✅ **M3.4 COMPLETE**: FlashAttention integration ⚠️ **CONTRADICTED — see above**
  - **Status**: Fully integrated, now included by default with CUDA feature
  - **Implementation**: FlashAttention-2 automatically enabled when compiled with `--features cuda`
  - **Conditions for activation**: CUDA device + no complex masks + GQA pre-expanded
  - **Fallback**: Graceful fallback to manual attention when conditions not met (CPU, complex masks)
  - **Performance**: 2-5× speedup on GPU for long contexts (512-2048 tokens), ~1.3-2× for short contexts
  - **Correctness**: 4/4 comprehensive tests passing (decode, prefill, batched, GQA) with 1e-3 tolerance
  - **Tensor conversions**: Automatic layout ([batch, heads, seq, dim] ↔ [batch, seq, heads, dim]) and dtype (F16 for CUDA)
  - **Causal masking**: Native FlashAttention support (causal=true for prefill, false for decode)
  - **Benchmarks**: `examples/benchmark_flashattention.rs` provides CPU baseline and GPU comparison framework
  - **Future work** (tracked below):
    - FlashAttention-3 integration when available in Candle (M4+)
    - Custom attention mask support for ScatteredKvCache (M5+)
    - Multi-GPU compatibility validation (M3.6)
    - AMD ROCm/HIP support monitoring (M4+)
  - Acceptance: ✅ numerical parity validated; ✅ measurable latency drop on GPU (2-5× on long contexts); ✅ comprehensive documentation
  - References: docs/M3_4_FLASHATTENTION_INTEGRATION.md, tests/flash_attention_tests.rs, examples/benchmark_flashattention.rs

- Quantized model loaders via Candle
  - ⚠️ **CONTRADICTED**: GGUF/other quant formats usable end-to-end.
    Measured 2026-09-01: **1 of 30 local GGUF files loaded**. Re-measured
    2026-09-05: **16 of 30**, after the tokenizer was rebuilt from BPE and ten
    `pre` values were verified. Re-measured 2026-09-09 at `main` c495852:
    **23 of 30**, after GGUF v1 and the IQ dtypes became reachable and four
    SentencePiece vocabularies had their merges derived under an oracle.
    **This line read "1 of 30" for two days after the figure moved by fourteen
    files**, which is why every date is stated rather than one.

    ⚠️ **The clause "nearly half the corpus still does not load" was true at 16
    and is FALSE at 23** — it is 7 of 30, and all seven are deliberate refusals
    (three tokenizer models this build does not construct, four `gpt2` files
    whose splitting rule is unverified), not failures to load.

    ⚠️ **The contradiction nevertheless STANDS, on different grounds, and the
    grounds matter more than the number.** `tests/gguf_corpus_sweep.rs` measures
    whether a TOKENIZER REBUILDS. It does not serve a token. End-to-end serving
    is exercised by `tests/gguf_serving_e2e.rs` against **one** checkpoint named
    by `LIGHTBULB_GGUF`, so "23 of 30" is not evidence for "usable end-to-end"
    and must not be quoted as if it were. That figure has NOT been re-measured
    in this pass.

    ⚠️ It then read **"15 of 30"** for the length of one PR, because the branch
    that corrected it and the branch that moved it again were open at the same
    time and neither touched the other's lines. **A figure appearing twice in one
    document needs re-deriving whenever either instance changes — the absence of a
    merge conflict is not evidence of consistency.**
  - Acceptance: ✅ run quantized tiny model locally; parity tests pass
  - References: low-bit LLMs survey; model compression survey
- Lightning GGUF loader with memory-mapped tensor access
  - ✅ **Phase 1 COMPLETE**: Memory-map infrastructure and tokenizer extraction (src/gguf/mod.rs)
    - Zero-copy mmap foundation using memmap2
    - Integrated tokenizer extraction from GGUF metadata (no external tokenizer.json needed)
    - Candle-compatible API wrapping proven parsing logic
    - Acceptance: Phi-3 GGUF baseline test runs with extracted tokenizer
  - ✅ **Phase 2 COMPLETE**: Direct GGUF v3 parser with zero-copy tensor access (src/gguf/parser.rs)
    - Implemented parse_gguf() parsing GGUF v3 format directly from mmap bytes
    - Added get_tensor_data(name) returning &[u8] slices directly from mmap (zero-copy)
    - Tested with 4 models: TinyLlama Q4_K_M (638MB), Q8_0 (1.1GB), F16 (2.1GB), Phi-3 Q4 (2.3GB)
    - All models verified: tensor counts match, sample tensors accessible, metadata preserved
    - Parser overhead minimal: ~1.02x speedup over Candle's mmap-based parsing (both use mmap)
    - Acceptance: All 4 GGUF models load correctly with zero-copy access verified ✅
    - Benchmark: examples/benchmark_lightning_gguf.rs
  - **Phase 3 TODO**: Integrate zero-copy loading with model initialization
    - Replace Candle QTensor reconstruction with direct mmap slice usage in model layers
    - Wire get_tensor_data() into custom attention/transformer block weight loading
    - Keep Candle's dequantization logic but feed it zero-copy slices
    - Expected gains: 1.5-10x faster full model initialization (parsing + weights)
    - Memory savings: 20-40% less RAM (tensors stay in mmap vs copied to heap)
    - Acceptance: 2-10x faster end-to-end model loading for Phi-3 2GB
  - References: docs/LIGHTNING_GGUF.md (to be created)

M3 — Acceleration features (0.4)

- Speculative decoding MVP
  - Draft+target dual-model orchestration; verify-accept loop
  - Acceptance: end-to-end works on two small models; speedup >1.3× on CPU in local tests; accuracy within configured bound

- **AWQ (Activation-aware Weight Quantization)** - 4-bit weight quantization (M3.7-M3.9)
  - ✅ **Phase 1 COMPLETE**: Kernel infrastructure (7 CUDA files, FFI bindings, build script)
  - ✅ **Phase 2 COMPLETE**: CustomOp wrappers (marlin.rs with MarlinMatMul for GPTQ/AWQ)
  - ⚠️ **"Phase 3 IN PROGRESS" was wrong — measured 2026-10-21 at 0% wired, not partway.**
    See "AWQ loader wiring: slice plan" above for the real state (a complete, unused loader;
    no call site; the model's forward pass doesn't match the serving engine's interface) and
    the proposed slices.
  - References: docs/ADVANCED_QUANTIZATION_RESEARCH.md, src/backend/marlin.rs, kernels/

- **Norm Tweaking** - Universal quantization plugin (M3.10, 2-3 days)
  - LayerNorm/RMSNorm calibration module for post-quantization accuracy recovery
  - CLI flag: `--norm-tweaking` (default enabled with AWQ)
  - Per-layer scale adjustment based on calibration data (50-100 samples)
  - Expected: +1.5-3% accuracy improvement on AWQ 4-bit models
  - Acceptance: Measurable accuracy gain on MMLU/HellaSwag; <10ms calibration overhead
  - References: docs/ADVANCED_QUANTIZATION_RESEARCH.md (Part 1.2, AAAI 2024)

- **VPTQ (Vector Post-Training Quantization)** - 2-bit extreme compression (M3.11, 3-4 weeks)
  - Microsoft's vector quantization for 2-bit weights (edge deployment focus)
  - Residual VQ with multiple codebooks for accuracy retention
  - Target: LLaMA-7B 7GB → 3.5GB with minimal quality loss
  - **Note**: Skip 4-bit implementation (no advantage over AWQ at 4-bit)
  - CLI flag: `--vptq` for 2-bit models only
  - Acceptance: 2-bit models load and run; 2× memory reduction vs 4-bit; accuracy within 3-5% of 4-bit AWQ
  - References: docs/ADVANCED_QUANTIZATION_RESEARCH.md (Part 1.1, Microsoft 2024)

- **Additional Quantization Research** - Future investigation (M4+)
  - **any4** (arXiv 2025): Learned quantization with adaptive rounding - MONITOR for benchmarks
    - Decision gate: Implement if community benchmarks show >2% accuracy gain over AWQ
    - Timeline: 2-3 months for community validation
  - **Qrazor** (arXiv 2025): 8→4 bit SDR with custom arithmetic - DEFER (hardware-specific)
    - Requires custom arithmetic units for direct compressed operations
    - Consider if deploying on specialized hardware with SDR support
  - **bitsandbytes** (HuggingFace): PyTorch 4-bit/8-bit library - COMPATIBILITY LAYER
    - Implement converter: bitsandbytes format → AWQ on model load
    - Priority: Medium (enables loading HuggingFace `load_in_4bit` models)
    - Timeline: 1 week after AWQ Phase 3 complete
  - **LLM-FP4** (EMNLP 2023): 4-bit floating-point quantization - DEFER
    - Focuses on activation quantization (W4A4), Lightbulb is weight-quant focused
    - Revisit if implementing full activation quantization
  - **NF4/QLoRA** (arXiv 2023): 4-bit NormalFloat quantization - SKIP
    - Training-focused (fine-tuning with LoRA adapters)
    - Inference performance similar to AWQ/GPTQ at 4-bit
    - No advantage for inference-only engines
  - References: docs/ADVANCED_QUANTIZATION_RESEARCH.md (Addendum: Higher Bit-Width Applicability)

- **SmoothQuant enablement** - 8-bit activation quantization (M4+, deferred)
  - W8A8 quantization with activation smoothing
  - Lower priority: Focus on 4-bit weight quantization first (AWQ + Norm Tweaking)
  - Acceptance: quality-per-bit improvement over naive quant on a tiny eval set (doc and tests)
  - References: low-bit LLMs survey (SmoothQuant, QuaRot)

- CPU kernel optimizations
  - ✅ **M3.3 COMPLETE**: Kernel fusion infrastructure (fused_linear_silu, fused_matmul_add) built and tested
  - **Status**: Infrastructure ready, fusion disabled due to Candle API limitations
  - **Findings**: candle_nn::Linear weight extraction overhead (213ms) exceeds theoretical fusion benefits (2-3ms)
  - **Future path**: Requires either (1) custom linear layer with direct weight access, OR (2) Candle upstream fused ops
  - **Note**: 11.3% bandwidth reduction validated through analysis; will deliver 10-15% throughput gain when API allows
  - Cache-friendly blocking strategies for attention and GEMM
  - Micro-prefetch hints for small-batch GEMM with adaptive prefetch distance
  - int8 GEMM micro-kernels with quantization-aware accumulation
  - Acceptance: ≥10% throughput improvement on representative workloads; ≥20% L1/L2 cache miss reduction; improved tail latency (95/99) on small-batch runs
  - References: docs/M3_3_KERNEL_FUSION_ANALYSIS.md (implementation findings), docs/summaries/2507-00951v1.md, docs/summaries/2506-21103v1.md, docs/summaries/2508-19828v1.md, docs/summaries/2509-07017v1.md

- Blocked sparsity and quantization integration
  - Blocked-sparsity kernels with configurable block size
  - Per-block calibration for quantization + sparsity interaction
  - Mixed-precision accumulation option
  - Acceptance: identify safe operating point (bitwidth + block size) where accuracy loss ≤1% and throughput improves ≥30%
  - References: docs/summaries/2508-13678v1.md, docs/summaries/2508-15884v1.md

- Per-layer sparsity masks
  - Compact mask file format (bit-packed, RLE) and loader
  - Tile-aligned masked kernels with branch-free operation
  - Runtime selection between dense and masked paths
  - Acceptance: net compute reduction on masked models with <1% accuracy degradation
  - References: docs/summaries/2506-22443v1.md

- Decode-loop overhead reductions
  - Batch reuse, caching, and minimal host<->device sync; sketch CUDA Graphs interface (behind feature)
  - Acceptance: inter-token latency variance reduced in local profiling (documented)

M3.5 — Testing & Hardening (0.4+)

**Status**: ⚠️ **CLAIMED COMPLETE (October 2025) — PARTIALLY UNRECONCILED.**
Much of the content below is a *design*: "testing strategy designed",
"validation matrix" of 28 configurations, quality benchmarks "phased". What is
built is built, but the tree carries **111 `#[ignore]` markers** and **three
behavioural acceptance tests that never run in CI** (no GPU runner, by cost
decision) — see the VERIFIED STATUS block at the top. Read this section as
"infrastructure and strategy exist", not "the matrix runs".

**Implementation Summary**:

M3.5 establishes comprehensive testing and validation infrastructure for production readiness:

- **External Crate Evaluation**: Analyzed 12+ Candle ecosystem crates
  - Identified high-priority integrations: candle-layer-norm (fused RMSNorm), candle-ext (utility functions)
  - Medium-priority: candle-einops (M6+), candle-optimisers (M8+ if training added)
  - Low-priority/not applicable: candle-onnx, candle-birnn, dfdx, gemm, rlkit
  - Recommendation: Integrate candle-layer-norm in M3.6/M4 for 20-30% normalization speedup
  - Document: `docs/CANDLE_ECOSYSTEM_EVALUATION.md`

- **Correctness Validation Framework**: Comprehensive testing strategy designed
  - Cross-feature validation matrix: 28 configurations ([CPU/CUDA] × [FP32/FP16] × [Q4_0/Q8_0/None] × [FlashAttn] × [speculation] × [context length])
  - Determinism verification: Fixed seed → identical outputs; cross-run/cross-device consistency
  - Quality benchmarks: Perplexity (WikiText-2), MMLU, HumanEval, HELM (phased implementation)
  - Tolerance targets: <1e-4 (full precision), <1e-3 (FlashAttention), <1e-2 (Q8_0), <5e-2 (Q4_0), <2% perplexity delta
  - All existing tests passing: correctness_tests.rs, flash_attention_tests.rs, batched_transformer_correctness.rs, batch_manager_integration.rs
  - Document: `docs/M3_5_CORRECTNESS_VALIDATION_FRAMEWORK.md`

- **Load & Stress Testing Infrastructure**: Comprehensive framework implemented
  - 6 test scenarios: Light (10 concurrent), Normal (50), Heavy (100), Stress (500), Long Context (128k tokens), Soak (48hr)
  - Metrics: Total/successful/failed/timeout requests, error rate, throughput (tokens/sec), latency (min/max/p50/p95/p99)
  - Memory leak detection: Linear regression on memory samples, threshold >100KB/hr = leak
  - Success criteria: Stable under 100+ concurrent, <1% error rate (normal load), <5% error rate (2× capacity), no leaks over 48hr
  - Implementation: `tests/load_stress_tests.rs` with async/await + tokio runtime
  - Status: Framework complete, integration with actual inference pending

- **Regression Detection System**: CI-integrated monitoring designed
  - SQLite database for historical benchmark results (throughput, latency, perplexity, memory, test pass rates)
  - Thresholds: >10% performance degradation alert, >2% perplexity increase, 3+ sustained commits = GitHub issue
  - Automated bisection for regressions >5%
  - CI workflow: Fast PR checks (5min subset), comprehensive nightly builds (30min full suite)
  - Status: Architecture designed, CI implementation pending

- **Integration Testing Matrix**: Feature combinations documented
  - Current features validated: CPU/CUDA, Batching, FlashAttention, KV Cache, RoPE
  - Test matrix: 7 configurations baseline-cpu through cuda-flash-batch (all passing ✅)
  - Known incompatibilities: NONE currently
  - Future watch: FlashAttention + custom masks, Sliding window + full attention, Q4_0 + FP16
  - Graceful fallbacks: FlashAttention → manual attention (auto-detected, no user config)

**Production Readiness**: ✅ VALIDATED
- Correctness: Numerical parity with Candle Llama (<1e-4), FlashAttention validated (1e-3), 40+ unit tests + 10+ integration tests
- Performance: FlashAttention 2-5× GPU speedup, batched processing near-linear scaling, baseline metrics documented
- Robustness: Load testing framework (100+ concurrent), memory leak detection, graceful degradation, comprehensive error handling
- Maintainability: 4 comprehensive docs added, clear module structure, type safety, automated testing
- Observability: Debug feature flags, tracing integration, production monitoring planned (M4+)

**Deliverables**:
- Documentation: 4 comprehensive docs (2100+ lines total)
- Code: tests/load_stress_tests.rs (450 lines) with 6 scenarios + memory leak detection
- Design: Cross-feature validation matrix, determinism strategy, regression detection architecture, quality benchmark plan

**Next Steps**: ✅ M3.6 Multi-GPU Inference **COMPLETE** → M4 Quantization & Compression (AWQ, GGUF, SmoothQuant)

**References**:
- `docs/CANDLE_ECOSYSTEM_EVALUATION.md` - External crate analysis
- `docs/M3_5_CORRECTNESS_VALIDATION_FRAMEWORK.md` - Testing framework design
- `docs/M3_5_TESTING_HARDENING.md` - Complete milestone summary
- `tests/load_stress_tests.rs` - Load/stress testing implementation

---

**LEGACY ITEMS** (Original M3.5 scope - now incorporated into above):

- Comprehensive correctness validation framework
  - Token-by-token deterministic tests on CPU with fixed seeds (extend existing tests)
  - Cross-feature correctness: quantization + speculative decoding + KV caching all enabled
  - Output quality benchmarks: HELM, MMLU, HumanEval (before/after each optimization)
  - Acceptance: all optimizations maintain <2% accuracy delta vs baseline; deterministic outputs verified

- Load and stress testing infrastructure
  - Concurrent request simulator (configurable: 10, 50, 100, 500 concurrent requests)
  - Edge case validation: max context (128k tokens), extreme batch sizes (128+), mixed workload patterns
  - Soak testing: 48hr+ continuous operation with memory leak detection
  - Acceptance: stable under 100+ concurrent requests; no memory leaks over 48hr; graceful degradation under overload

- Performance regression detection
  - CI integration for latency/throughput benchmarks on each commit
  - Automated bisection for performance regressions >5%
  - Benchmark result database with historical trend analysis
  - Acceptance: catch regressions within 24hr of commit; automated alerting on >10% degradation

- Integration testing matrix
  - Test combinations: [CPU/CUDA] × [quantized/full] × [speculative/standard] × [short/long context]
  - Feature interaction validation (e.g., does prefix caching work with sliding window?)
  - Acceptance: all feature combinations tested; documented incompatibilities if any

M3.6 — Multi-GPU Inference (0.4+)

**Status**: ⚠️ **DELETED 2026-10-22, PR #195** — `src/multi_gpu/{distributed_cache,
pipeline_parallel,tensor_parallel}.rs` (every file this section names) were removed entirely
in the candlelight-dependency removal: candlelight-`Tensor`-typed, zero external callers,
confirmed before deletion. `ParallelCacheBuilder` is also gone (superseded by Fuel's
`kv_block_pool`, PR #196). See `docs/FUEL-GAPS-FROM-CANDLE-REMOVAL.md`'s multi-GPU entry for
current routing. Everything below this line is historical record of a real
documentation-honesty fix (2026-09-02), not a description of code that still exists.

**Status (historical, pre-deletion)**: ⚠️ **CLAIMED COMPLETE (October 27, 2025) — CONTRADICTED.**
Eight selectable paths bail at runtime: `Sharded`/`Hybrid` cache sync,
`PipeDream`/`Interleaved1F1B` pipeline scheduling, `Hybrid` tensor sharding.
`CacheSyncStrategy` offers three variants and implements NONE. `Replicated`
copied to each device and dropped the result while returning `Ok(())`; fixed
2026-09-02 to report that it is unimplemented, so all three are now honest about
it. Measured 2026-09-01, re-measured 2026-09-02; see the VERIFIED STATUS block at
the top for the site list.  
**Added**: Jan 2025 (ChatGPT o1 validation - critical gap for large model serving)  
**Completed**: October 2025 - Foundation infrastructure for multi-GPU distributed inference

**What Was Delivered**:

- ✅ **Tensor Parallelism Foundations** (Task 2)
  - Weight sharding (column-wise/row-wise) via `TensorShard::from_full_tensor()`
  - Gather/scatter operations for cross-GPU tensor coordination
  - `ShardedLinear` layer for distributed matrix multiplication
  - Support for `ShardingStrategy`: ColumnWise, RowWise, Hybrid
  - File: `src/multi_gpu/tensor_parallel.rs` (~400 lines)

- ✅ **Pipeline Parallelism Foundations** (Task 3)
  - `PipelineScheduler` with GPipe micro-batching strategy
  - Layer distribution across pipeline stages
  - Micro-batch splitting for pipeline efficiency
  - `forward_layers()` method in `BatchedTransformer` for explicit layer-range processing
  - Files: `src/multi_gpu/pipeline_parallel.rs`, `src/model/custom_transformer.rs`

- ✅ **Distributed KV Cache Manager** (Task 4)
  - Static allocation: one `ParallelCacheBuilder` per GPU
  - `CacheSyncStrategy::Replicated` for cross-GPU cache synchronization
  - `DistributedCacheManager` for coordinating cache across GPUs
  - File: `src/multi_gpu/distributed_cache.rs` (256 lines)

- ✅ **Integration with BatchedTransformer** (Task 6)
  - `BatchedTransformerConfig.multi_gpu: Option<MultiGPUConfig>`
  - `BatchedTransformer.enable_distributed_cache()` initialization
  - Works for **all** architectures: Llama, Mistral, Gemma, Phi, Qwen
  - Architecture-agnostic multi-GPU support via generic `BatchedTransformer`

- ✅ **Comprehensive Test Suite** (Task 5)
  - 17 test functions across 5 categories
  - Tests: topology discovery, tensor parallelism, pipeline parallelism, distributed cache, integration
  - Performance benchmarks for latency, throughput, communication overhead
  - File: `tests/multi_gpu_validation.rs` (350+ lines)
  - All tests gated with `#[ignore]` for multi-GPU hardware requirement

- ✅ **Documentation** (Task 7)
  - Architecture: `docs/M3_6_MULTI_GPU_ARCHITECTURE.md`
  - Integration guide: `docs/MULTI_GPU_INTEGRATION.md`
  - Testing guide: `tests/MULTI_GPU_TESTING.md`
  - README.md section with quickstart examples

**Performance Targets** (hardware validation pending):
- 2-GPU tensor parallel: ~1.7× throughput (target)
- 4-GPU tensor parallel: ~3.2× throughput (target)
- 4-GPU pipeline: ~3.5× throughput (target)
- 8-GPU hybrid (2×4): ~6× throughput (target)
- Communication overhead: <15% (tensor), <8% (pipeline)

**Future Work** (Post-M3.6):
- Automatic layer distribution in `forward()` method (currently manual via `forward_layers()`)
- Sharded weight loading from disk (currently loads full weights then shards)
- NCCL integration for optimized all-reduce (currently uses Candle's `to_device()`)
- Dynamic load balancing across GPUs
- Multi-GPU-aware model manager integration
- Hardware validation and performance benchmarking

**Related Milestones**:
- M6.5: Elastic KV Cache with `candle-cuda-vmm` (KVCached-style virtual memory)
- M5: Static KV cache optimizations (baseline for elastic comparison)

**Risks & Mitigations**:

- **Risk**: Cross-GPU synchronization bugs extremely difficult to debug
  - *Mitigation*: Start with 2-GPU setup; comprehensive logging of cross-GPU operations; dedicated debugging tools for KV cache state inspection
- **Risk**: Communication overhead dominates for small batch sizes
  - *Mitigation*: Establish minimum batch size thresholds per model size; prefer pipeline parallelism for batch_size < 4
- **Risk**: Memory imbalance between GPUs causes OOM on single device
  - *Mitigation*: Implement load balancing algorithm with dry-run memory estimation; add rebalancing capability
- **Risk**: NCCL/communication library compatibility issues across GPU generations
  - *Mitigation*: Test matrix for common GPU combos (A100+A100, V100+V100); document tested configurations

- Tensor parallelism for large models (70B+)
  - Column-wise and row-wise weight sharding strategies
  - Cross-GPU communication via NCCL for all-reduce operations
  - Balanced memory distribution across GPUs
  - Acceptance: 70B model runs on 2×40GB GPUs; communication overhead <15%; scales to 4 GPUs

- Pipeline parallelism with micro-batching
  - Layer-wise distribution across GPUs (early layers on GPU0, later layers on GPU1, etc.)
  - Micro-batch scheduling to hide inter-GPU latency
  - Bubble minimization strategies (GPipe, PipeDream patterns)
  - Acceptance: 70B model throughput 1.5-2× higher than sequential offloading; pipeline bubble <20%

- KV cache coordination across GPUs
  - Distributed KV cache manager with cross-GPU synchronization (M3.6 Task 4)
  - RDMA-aware cache transfers where available
  - Cache locality optimization (prefer local GPU access)
  - **Note**: M3.6 uses static allocation per GPU; elastic KV cache (KVCached-style) is future work (M6.5)
  - Acceptance: KV cache overhead <10% vs single-GPU; no correctness issues in distributed cache

- Persistent warp scheduler (CUDA backend)
  - Device-resident scheduler where warps repeatedly find and execute work without host launches
  - Warp-level job management with ballot/shuffle convergence for lock-free claiming
  - Winner-takes-all job claiming pattern to minimize warp divergence
  - Eliminates host kernel launch overhead for rapid job dispatch
  - Integration with priority queue system for work distribution
  - Acceptance: dispatch overhead reduced to 2-5 µs vs 25-35 µs host launches; warp efficiency ≥80%; works with Candle's CUDA backend; feature-gated for portability
  - References: docs/PERSISTENT_WARP_SCHEDULER.md (to be created)

- Statistical timeout and fault isolation system
  - Per-kernel timeout calculation using mean + 3×std from historical execution data
  - Zombie warp pattern for graceful degradation when kernels exceed timeout
  - SM-level reset requests to isolate faults without killing entire scheduler
  - Fault isolation preserves other concurrent work in multi-tenant scenarios
  - Adaptive timeout adjustment based on observed kernel behavior
  - Acceptance: timeout detection accuracy ≥95%; fault isolation prevents cascade failures ≥90% of time; SM reset recovery <100ms; false positive rate <5%
  - References: docs/STATISTICAL_TIMEOUT_SYSTEM.md (to be created)

- Residency-aware multi-factor job sorting
  - Multi-level job sorting: Priority → Memory residency (VRAM/RAM/SSD) → Kernel similarity → Tensor shape → Age
  - Prefer jobs with weights already in VRAM to minimize transfers
  - Branch-coherent batching for improved warp efficiency
  - Kernel similarity grouping to maximize instruction cache hits
  - Dynamic reordering based on observed memory hierarchy performance
  - Acceptance: warp efficiency ≥80-95%; throughput improvement ≥10-30% vs priority-only sorting; memory transfer reduction ≥20%; scheduling overhead <1% of compute time
  - References: docs/RESIDENCY_AWARE_SCHEDULING.md (to be created)

- Arena-based memory system
  - ⚠️ **CONTRADICTED** (see "dead scaffold crates" finding above): `dynctx` crate - 7-layer relationship system with arena memory management
  - Efficient token-level operations using arena allocation patterns
  - Zero-copy memory handoffs between components
  - Reduced memory fragmentation for long-running inference
  - N-dimensional token graphs with multiple relationship types (sequential, semantic, temporal, syntactic, causal, reference, emotional)
  - Acceptance: measurable reduction in allocation overhead (≥20%); stable under 100k+ token workloads; memory fragmentation reduced by ≥30% vs standard allocators
  - References: docs/ARENA_MEMORY.md (to be created), docs/DYNANIML_INTEGRATION.md, docs/INTEGRATION_COMPLETE.md

- CPU-GPU hybrid processing architecture
  - Graph traversal and relationship resolution on CPU
  - Heavy tensor mathematics on GPU after structure resolution
  - Smart caching to minimize CPU-GPU data transfers
  - Three-tier memory hierarchy: cold (disk/cloud) → warm (RAM) → hot (GPU VRAM)
  - Acceptance: demonstrates optimal work distribution; CPU-GPU transfer overhead <10% of compute time; memory tier promotion/demotion based on access patterns
  - References: docs/HYBRID_PROCESSING.md (to be created)

- Assembly-level optimization validator
  - Compare generated assembly across different implementations
  - Identify optimization opportunities in compiled code
  - Validate zero-cost abstraction assumptions
  - Suggest code modifications for better compiler optimization
  - Cross-architecture assembly comparison (x86, ARM, RISC-V)
  - Acceptance: identifies ≥80% of optimization opportunities; zero-cost abstraction validation accurate ≥95% of time; cross-architecture comparison functional; actionable optimization suggestions generated
  - References: docs/ASSEMBLY_VALIDATOR.md (to be created)

- Hardware abstraction analyzer
  - Map hardware-specific optimizations to abstract capabilities
  - Identify portable optimization strategies across architectures
  - Analyze performance implications of abstraction layers
  - Suggest hardware-agnostic implementation approaches
  - Predict optimization effectiveness across different hardware
  - Acceptance: maps optimizations across ≥3 hardware architectures; portable strategies maintain ≥90% of specialized performance; abstraction overhead quantified with ≤5% error; predictions accurate within ±15%
  - References: docs/HARDWARE_ABSTRACTION_ANALYZER.md (to be created)

- Hybrid linear attention schedule (research path)
  - Offer a config to interleave linear-like mixers with full attention at ratios {3:1, 4:1, 6:1} (simulated via layer keep/skip until Candle backbones land)
  - Acceptance: ≥4× effective KV memory reduction with ≤0.5 ppl delta on a small LM eval and ≤3% drop on a recall-sensitive micro-benchmark
  - References: docs/summaries/hybrid-linear-attention-analysis.md

M3.7 — Dynamic Name Mapping (0.4+)

**Status**: ✅ CORE COMPLETE (January 2026), INTEGRATIONS PLANNED  
**Added**: Jan 2026 (User request - enable pruning to work with unknown model architectures)  
**Completed**: Jan 2026 - Core name mapping infrastructure functional

**What Was Delivered**:

- ✅ **Core Name Mapping Module** (M3.7)
  - `TensorNameMapper` for automatic architecture detection
  - Support for LLaMA, GPT, Mistral architectures with regex-based pattern matching
  - Abstract → concrete tensor name translation (e.g., "layer_5.attention.query" → "blk.5.attn_q.weight")
  - Batch tensor retrieval APIs (`map_layer()`, `get_layer_tensors()`)
  - File: `src/pruning/name_mapping.rs` (428 lines)
  - Dependency: `regex = "1"`

- ✅ **Integration with Pruning System**
  - Two-phase mask lookup in `gguf_application.rs`: (1) direct name match, (2) abstract → concrete mapping
  - Architecture auto-detection at runtime with logging
  - Solves original problem: 0 tensors pruned → working pruning with generic mask names
  - Tested with TinyLlama (correctly detected as LLaMA architecture)

**Future Integrations** (M4-M6):

- **M4.1**: Model Loading Integration (3 weeks)
  - Add `TensorNameMapper` to `ModelLoader` for architecture-agnostic loading
  - Eliminate hardcoded tensor names in `src/model/loader.rs`
  - Support 5+ architectures (LLaMA, GPT, Mistral, Qwen, Phi) without code changes
  - Phase 1: Core integration, Phase 2: Enhanced features (batch loading, validation), Phase 3: Advanced (config overrides, registry)
  - References: `docs/NAME_MAPPING_MODEL_LOADING.md`

- **M5.1**: Cache Management Integration (3 weeks)
  - Architecture-aware layer detection for KV cache
  - Variable layer count support (automatic detection vs hardcoded 32)
  - Relationship-aware cache eviction using N-dimensional token graphs
  - Dynamic cache sizing based on architecture (e.g., Mistral attention sinks in early layers)

- **M5.4**: LoRA Integration (2 weeks)
  - Automatic format detection (HuggingFace, custom, PEFT)
  - Map LoRA adapter tensor names to base model components
  - Validation and shape compatibility checking
  - Support multiple LoRA formats without hardcoded mappings

- **M3.6.1**: Multi-GPU Enhancement (1 week)
  - Architecture-aware layer distribution across GPUs
  - Handle heterogeneous architectures (MoE, variable layer sizes)
  - Load-balanced GPU assignments based on tensor sizes

- **M4.1**: Quantization Integration (in parallel with model loading)
  - Layer-specific quantization levels based on architecture
  - Component-aware mixed precision (attention=INT8, FFN=INT4)
  - Sensitivity-based quantization with automatic level assignment

- **M5.6**: Tool Registry Integration (2 weeks)
  - Auto-detect model capabilities from architecture
  - Register appropriate tools based on model type
  - Support vision, function calling, multimodal detection

- **M6.5**: LLM-Assisted Name Mapping (4 weeks)
  - Use small LLM (~1B params) for probabilistic name matching
  - Fallback for completely novel architectures (regex → LLM → manual override)
  - Three-tier strategy: regex (fast) → LLM (smart) → config (explicit)
  - Confidence-based matching with user verification for uncertain matches
  - Caching to avoid repeated LLM queries
  - References: `docs/LLM_ASSISTED_NAME_MAPPING.md`

**Success Metrics**:
- ✅ Core: Support 3+ known architectures automatically (LLaMA, GPT, Mistral)
- ✅ Core: <1ms mapping overhead per model load
- ✅ Core: 100% accuracy for known patterns
- 📋 M4.1: Load 5+ architectures without code changes
- 📋 M5.1: Support variable layer counts (24-80 layers) automatically
- 📋 M5.4: Auto-detect 3+ LoRA formats
- 📋 M6.5: 85-95% accuracy for novel architectures (LLM-assisted)
- 📋 Overall: Zero code changes to support new architectures

**Performance Targets**:
- Regex mapping: <1ms per model load
- LLM-assisted mapping: <2s per model load (one-time cost)
- Runtime overhead: 0 (mapping done at load time)

**Integration Documentation**:
- Core: `src/pruning/name_mapping.rs` (complete)
- Model Loading: `docs/NAME_MAPPING_MODEL_LOADING.md` (complete)
- LLM-Assisted: `docs/LLM_ASSISTED_NAME_MAPPING.md` (complete)
- Feature Integrations: `docs/NAME_MAPPING_FEATURE_INTEGRATIONS.md` (complete)

**Risks & Mitigations**:

- **Risk**: Regex patterns become unmaintainable as architectures proliferate
  - *Mitigation*: M6.5 LLM-assisted fallback reduces need for exhaustive regex patterns; community database for pattern sharing
- **Risk**: Name mapping failures silently degrade inference quality
  - *Mitigation*: Explicit confidence scoring; warnings for low-confidence matches; validation against tensor shapes
- **Risk**: LLM-assisted mapping is too slow for production
  - *Mitigation*: Use tiny models (<1B params); aggressive caching; optional feature (can disable)
- **Risk**: Unknown architectures require manual config files
  - *Mitigation*: Clear error messages; config templates; learning from user corrections

M4 — Advanced scheduling (0.5)

**Status**: PLANNED

**Added**: Jan 2025 (Multi-stage pipelines, knowledge-aware decomposition, and consistency checking expand M4's scope beyond traditional scheduling to orchestrate complex reasoning workflows)

**Name Mapping Integration** (M4.1): ✅ **COMPLETE** (October 2025)
- ✅ Integrated TensorNameMapper into `src/loaders.rs` (both safetensors and GGUF loaders)
- ✅ Both loaders now return `Option<TensorNameMapper>` for architecture detection
- ✅ Architecture detection logging: Shows detected architecture and layer count during load
- ✅ Backward compatible: Optional return, graceful fallback if detection fails
- ✅ All 7 callers updated: src/lib.rs, parallel_model_manager.rs (2x), model_manager.rs, tests (3x)
- ✅ Compilation successful: 0 errors, 36 pre-existing warnings
- 📋 Future: Enable loading 5+ architectures without code changes (requires more architecture patterns)
- Timeline: Completed in ~15 minutes (estimated 3 weeks → 288x faster)
- References: `docs/NAME_MAPPING_MODEL_LOADING.md`, M3.7 (Core name mapping), `src/loaders.rs`

**External Crate Integrations** (from M3.5 evaluation):
- **candle-layer-norm**: ✅ **COMPLETE** (Pre-October 2025 - Already Integrated)
  - Status: Fully integrated in `src/model/fused_rmsnorm.rs` (214 lines)
  - Feature gate: Included with `cuda` feature, automatic CUDA/CPU fallback
  - Integration: Used in `custom_transformer_block.rs` and `custom_transformer.rs`
  - Test coverage: 3 parity tests passing (CPU f32, various sizes, with residual)
  - Performance: Fused kernels provide 20-30% normalization speedup on CUDA
  - Acceptance: ✅ All criteria met - numerical parity (<1e-4), graceful CPU fallback, production ready
  - References: `docs/CANDLE_ECOSYSTEM_EVALUATION.md`, `src/model/fused_rmsnorm.rs`, `tests/fused_rmsnorm_*.rs`

- **candle-ext utilities**: ✅ COMPLETE (M4.2)
  - Status: Vendored implementations in `src/utils/tensor_ops.rs`
  - Strategy: Copied core implementations rather than adding dependency (community-maintained, simpler)
  - Functions implemented:
    * `triu(tensor, diagonal)` - Upper triangular matrix (zeroes below diagonal)
    * `tril(tensor, diagonal)` - Lower triangular matrix (zeroes above diagonal)
    * `masked_fill(tensor, mask, value)` - Conditional fill where mask is non-zero
    * `causal_mask(seq_len, device)` - Convenience function for causal attention
  - Test coverage: 6 comprehensive tests (basic operations, diagonal offsets, 3D broadcasting)
  - Usage example:
    ```rust
    use lightbulb::utils::{triu, tril, masked_fill, causal_mask};
    
    // Create causal attention mask
    let mask = causal_mask(seq_len, device)?;  // -inf above diagonal
    
    // Or manually create triangular matrices
    let upper = triu(&matrix, 0)?;  // Keep diagonal and above
    let lower = tril(&matrix, 0)?;  // Keep diagonal and below
    
    // Apply custom masks
    let masked = masked_fill(&logits, &padding_mask, f64::NEG_INFINITY)?;
    ```
  - Acceptance: ✅ All tests passing, zero external dependencies, ready for attention implementations
  - References: `docs/CANDLE_ECOSYSTEM_EVALUATION.md`

- **candle-einops**: Tensor reshaping with einops notation (readable syntax)
  - Status: LOW PRIORITY - Consider if tensor ops become complex
  - Benefit: More expressive than nested transpose/reshape, self-documenting
  - Trade-off: Adds macro dependency and complexity for syntactic sugar
  - Decision: Defer until M6+ advanced architectures unless need arises
  - References: `docs/CANDLE_ECOSYSTEM_EVALUATION.md`

- **FlashAttention-3**: Monitor Candle for FA3 availability
  - Status: WATCH - Check Candle upstream quarterly
  - Expected benefit: Additional speedups over FlashAttention-2
  - Action: Upgrade when available in candle-flash-attn with backwards-compatible API
  - Acceptance: Same numerical parity tests, improved performance benchmarks

- **AMD ROCm support**: FlashAttention on AMD GPUs
  - Status: WATCH - Monitor Candle for ROCm/HIP FlashAttention support
  - Rationale: Expand hardware compatibility beyond NVIDIA CUDA
  - Dependencies: Requires upstream Candle ROCm backend maturity
  - Acceptance: FlashAttention works on AMD GPUs with parity to CUDA version

**Risk & Mitigation**:

1. **Complexity explosion from feature interaction** (metadata + multi-stage + convergence + MoE routing)
   - Mitigation: Implement features sequentially with integration testing at each step; maintain feature flags for independent enable/disable; design modular scheduler with clear component boundaries; extensive unit tests for edge cases

2. **Performance degradation from scheduling overhead** (metadata lookups, convergence checks, KB validation)
   - Mitigation: Profile each scheduling decision path; establish <1ms overhead budget per request; implement fast-path bypass for simple requests; batch metadata operations where possible; optimize hot paths first

3. **Convergence detection failure modes** (infinite loops, premature termination)
   - Mitigation: Start with conservative safety limits (max iterations, timeout); log convergence decisions for offline analysis; A/B test convergence policies on synthetic workloads; implement circuit breakers for runaway cases

4. **Knowledge base consistency complexity** (NLI validator accuracy, temporal conflicts, confidence scoring)
   - Mitigation: Begin with simple contradiction detection (direct negation); expand to temporal/logical inconsistencies incrementally; maintain shadow KB for validation testing; implement KB rollback for detected corruption

5. **Multi-stage pipeline coordination bugs** (deadlocks, dependency resolution failures, cache coherence)
   - Mitigation: Model pipeline DAG explicitly with cycle detection; dry-run pipeline validation before execution; implement stage timeout and failure recovery; extensive testing with complex pipeline graphs

- Memory-aware, priority scheduler
  - ⚠️ **CONTRADICTED 2026-10-24.** Was marked ✅ INTEGRATED: `distributed-config` crate -
    Dynamic scheduling configuration management. `distributed-config` was a direct
    dependency from the initial commit until lightbulb#213 (2026-10-23), but
    `git log --all -S"distributed_config" -- src/` finds zero commits ever touching that
    identifier under `src/` — it was never called from any code, scheduler or otherwise,
    at any point in the project's history. lightbulb#213 removed it as "unused", which was
    the first accurate description of its status. See the `auto-discovery`/`coalescent`/
    `mocopr-server`/`web-server-abstraction`/etc. "INTEGRATED" bullets throughout this
    document's far-future milestones: none of those crates have ever appeared in
    `Cargo.toml` either, so each is the same claim pattern, unverified here.
  - ⚠️ **CONTRADICTED** (see "dead scaffold crates" finding above): `mocopr-server` - Expose scheduler metrics and control tools via MCP protocol
  - ⚠️ **CONTRADICTED** (see "dead scaffold crates" finding above): `web-server-abstraction` - REST API for inference requests, metrics dashboard, and WebSocket streaming
  - Request classes, preemption, dynamic batch sizing
  - Tiered KV orchestration (RAM/disk paging) with SLA-driven admission control
  - Hybrid LRU-LFU eviction policy for KV cache with recency/frequency weighting
  - Per-core partitioning and coordinated prefetch for multi-core CPUs
  - Cooperative yield API for long-running tasks with remaining-work estimator
  - Fairness heuristics for multi-tenant workloads with priority classes
  - Runtime configuration updates for scheduling policies without restart
  - Real-time metrics accessible to LLM supervisors via Model Context Protocol
  - HTTP endpoints: /v1/completions, /v1/models, /v1/metrics, /v1/queue/status, /health
  - WebSocket streaming: /ws/stream for real-time inference responses
  - Production-ready security: CSRF, TLS/SSL, rate limiting, authentication
  - Acceptance: TTFT improves for high-priority short prompts without starving bulk jobs; stable throughput with bounded tail latency under long-context workloads; ≥15-30% reduction in 95th-percentile latency in mixed workloads
  - References: docs/summaries/memosa-memory-os.md, docs/summaries/2509-03646v2.md, docs/summaries/2510-05949v1.md, docs/summaries/2508-15126v1.md, docs/summaries/2509-14234v1.md, docs/DISTRIBUTED_CONFIG_INTEGRATION.md, docs/MOCOPR_INTEGRATION.md, docs/WEB_SERVER_INTEGRATION.md

- Multi-stage inference pipeline orchestration
  - Support chained inference stages (decompose → identify → generate → verify → synthesize)
  - Per-stage model selection with automatic routing
  - KV cache sharing across compatible stages
  - Parallel execution of independent stages with dependency tracking
  - Pipeline-level batching and scheduling
  - **Dependency graph enforcement**: Parse decompositions to extract dependency graphs; prevent out-of-order execution when A→B dependency exists; identify parallelizable tasks automatically; detect circular dependencies
  - **Structured decomposition schemas**: Request format includes problem statement, constraints, initial decomposition with explicit dependencies, per-subtask preconditions/outputs/success criteria (forces rigorous thinking, provides structured data)
  - **Decomposition tracking**: Store decomposition structure in graph; track which subtasks needed further breakdown, where it got stuck, what worked smoothly; build history for "show me past decompositions of similar problems"
  - **Iterative refinement orchestration**: LLM generates initial decomposition → runner attempts subtasks → track success vs need re-decomposition → feed back "subtask X too complex, decompose further" or "subtask Y succeeded, here's result" → LLM refines plan (turns single-shot planning into adaptive process)
  - Acceptance: Execute 3+ stage pipelines with <10% overhead vs sequential; parallelize independent stages achieving ≥1.5× speedup; KV cache reuse reduces TTFT by ≥20% for multi-stage workflows; dependency enforcement prevents ≥95% of ordering violations; structured schemas reduce ambiguity by ≥40%; iterative refinement improves solve rate by ≥25%
  - References: docs/MODULAR_REASONING.md (to be created), docs/DEPENDENCY_GRAPHS.md (to be created), docs/ITERATIVE_REFINEMENT.md (to be created)

- Metadata-driven scheduling enhancements
  - Request metadata schema: priority, tags (reasoning, factual, creative), context hints, ethical flags
  - Tag-based routing to specialized models or pipelines
  - Constraint satisfaction hooks (type checking, range validation, schema compliance)
  - Episode/trace metadata for RL-friendly logging
  - Acceptance: Tag-based routing selects appropriate model/pipeline with ≥90% accuracy; constraint hooks reject invalid requests with <1ms overhead; episode metadata logged without affecting p95 latency
  - References: docs/EPISTEMIC_SYSTEM.md (to be created)

- State persistence and recovery
  - ✅ **M4.B IMPLEMENTED** (November 2025): Checkpoint/restore infrastructure complete (570 lines, 9 tests passing)
    * `InferenceCheckpoint` - Full state snapshot (KV cache, KB, pipeline, decomposition history, active problems)
    * `CheckpointManager` - Save/load/list/delete with LRU eviction (configurable max_checkpoints)
    * `KvCacheSnapshot` - Layer-wise cache metadata + serialized tensors
    * `KnowledgeBaseSnapshot` - All facts + eviction history + statistics
    * `PipelineSnapshot` - Current stage + completed stages + per-stage data
    * JSON serialization with metadata index for fast listing
    * LRU eviction when checkpoint count exceeds limit
    * Timestamp-based checkpoint IDs (millisecond precision)
    * Helper functions: create_checkpoint(), restore_kb_from_checkpoint()
    * 9 comprehensive tests (all passing): creation, save/load, listing, deletion, eviction, KB restoration, decomposition storage, metadata tracking
    * Enables: Graceful shutdown/restart, long-running iterative reasoning, debugging/replay, distributed inference with state migration
    * **Integration with Segmented KV Cache**: `tensor_codec.rs` provides tensor↔bytes serialization; `TieredStorageManager` handles disk persistence; GPUDirect Storage (M5.6) would enable near-instant checkpoint/restore via DMA
    * Acceptance: ✅ Core infrastructure ready; enables production robustness and long-running workflows
  - Privacy-preserving state encryption with key-per-request
  - Graceful degradation on OOM (checkpoint → evict → resume)
  - Session resumption across restarts with state continuity
  - Acceptance: Checkpoint/restore completes in <500ms for 10k token contexts; encrypted state adds <5% overhead; session resumption works ≥95% of time without token regeneration
  - References: docs/STATE_PERSISTENCE.md (to be created), `src/engine/state_persistence.rs`, `tests/state_persistence.rs`

- Convergence detection for iterative workflows
  - Configurable convergence policies (max iterations, fact saturation, decomposition depth)
  - Per-request convergence state tracking
  - Early termination signals (no new facts, repeated outputs, confidence threshold met)
  - Safety limits to prevent infinite loops
  - Acceptance: Convergence detection adds <1ms per iteration; prevents runaway loops in ≥99% of cases; correctly detects saturation within ±1 iteration of ground truth
  - References: docs/MODULAR_REASONING.md (to be created)

- Knowledge-aware iterative decomposition
  - ✅ **M4.C IMPLEMENTED** (October 2025): Core decomposition infrastructure complete (991 lines)
    * `Problem` struct with required_facts, known_facts, complexity levels (Trivial→VeryComplex)
    * `SubProblem` with dependencies, produces, success_criteria, completion tracking
    * `Decomposition` with strategy selection, parallel execution groups, topological ordering
    * `DecompositionEngine` with KB integration, depth tracking, history recording
    * Complexity metrics: structural (sub-problems + dependencies), computational (KB lookups needed)
    * Knowledge coverage calculation: (known_facts / required_facts) drives strategy selection
    * Strategies: Atomic (no decomposition), Computational (high coverage ≥90%), Structural (low coverage <50%), Hybrid (50-90%)
    * Re-decomposition triggers: When KB enrichment crosses coverage threshold (default 0.7)
    * Execution ordering: Topological sort with cycle detection, parallel group identification
    * Stats tracking: total decompositions, strategy counts, re-decomposition count, average coverage
    * 17 comprehensive tests (all passing): Problem creation, KB enrichment, coverage calculation, complexity levels, sub-problems, execution order, stats tracking, history
    * Acceptance: ✅ Core infrastructure ready; enables C→B→A dependency-driven implementation
  - Problem decomposition adapts based on current knowledge base state
  - Initially structural (task breakdown), becomes computational as KB fills
  - Example: "Calculate economic impact" → {Find GDP, Find effects, Calculate} → after GDP retrieval → "Calculate (GDP_2024 - GDP_2023)/GDP_2023" (atomic)
  - Dynamic problem simplification through knowledge accumulation
  - Re-decomposition triggers when new knowledge enables simplification
  - Acceptance: decomposition complexity reduces ≥30% as KB fills; atomic problem detection accuracy ≥90%; re-decomposition improves problem solvability ≥25%; overhead <200ms per decomposition iteration
  - References: docs/KNOWLEDGE_AWARE_DECOMPOSITION.md (to be created), `src/engine/decomposition.rs`, `tests/decomposition.rs`

- Query analysis and relevance-aware retrieval (NEW - M4.D, M4.E, M4.F)
  - **M4.D: Query Analysis Tools** (2-3 hours) - Fast preprocessing to understand query intent
    * `QueryAnalyzer` with intent classification (Definition, Procedure, Comparison, Troubleshooting, Explanation, Analysis, Synthesis)
    * Fast non-neural preprocessing: entity extraction (regex), constraint parsing (temporal expressions, filters), ambiguity detection
    * Small LLM (100M-500M params) for ambiguous cases only
    * Query decomposition: break complex queries into sub-queries with dependencies
    * Produces `AnalyzedQuery` with intent, entities, constraints, implicit context, ambiguities
    * Acceptance: Intent classification ≥85% accuracy; entity extraction <1ms; reduces LLM load by 90% for simple queries; overhead <5ms total
  - **M4.E: Relevance-Aware Search** (3-4 hours) - Extends M4.5 KB with relevance scoring beyond similarity
    * Semantic search baseline (embedding similarity)
    * HyDE (Hypothetical Document Embeddings): Generate ideal answer, search for similarity to that
    * Cross-encoder reranking: Deep relevance scoring of candidates (not just embedding distance)
    * Metadata filtering: Document type, information type (factual/procedural/analytical), recency
    * Multi-vector retrieval: Late interaction models (ColBERT-style token-to-token matching)
    * Hybrid search pipelines: Fast similarity → cross-encoder rerank → top-k results
    * Acceptance: Relevance precision ≥15% better than pure similarity; reranking <100ms for 20 candidates; supports 5+ search strategies
  - **M4.F: Context Injection API** (2-3 hours) - External providers for dynamic context enrichment
    * `ContextProvider` trait for loadable modules/programs to inject context
    * `ContextInjection` with content, position (before/after prompt, system), priority, source tracking
    * Example providers: Crate API loader (auto-load docs.rs when crate mentioned), notification feed, file watcher
    * Register providers via `InferenceEngine::register_context_provider()`
    * Provider activation based on prompt analysis, history, metadata
    * Acceptance: Providers add <10ms latency; support 10+ concurrent providers; clear priority resolution; graceful failures
  - **Integration with Existing M4**: Query analysis feeds into decomposition (C), state persistence (B), and metadata scheduling (A)
    * Fast preprocessing extracts query metadata for routing decisions
    * Relevance search improves KB retrieval quality (better than similarity)
    * Context injection enriches prompts with domain-specific knowledge automatically
  - **Tool-Based Pipeline Architecture**: LLM can orchestrate tools individually or in custom combinations
    * Default fast pipeline: query_analyzer → semantic_search → cross_encoder_rerank → results
    * Adaptive pipeline: LLM calls tools based on query complexity (e.g., troubleshooting uses full stack)
    * Each tool returns structured data with confidence scores and metadata
    * Tools composable: query_analyzer output informs metadata_filter parameters
  - Explicit unknown identification system (extends with query understanding)
    * Dedicated component (300M-1B parameters) extracting information gaps from sub-problems
    * Produces structured unknowns: ["capital of France"], ["value of X", "value of Y"]
    * Gap analysis enabling targeted retrieval (not just query reformulation)
    * Unknown categorization by type (factual, numerical, relational, temporal)
    * Parallel unknown resolution with dependency tracking
    * Acceptance: unknown extraction precision ≥85%, recall ≥90%; categorization accuracy ≥80%; enables ≥40% reduction in unnecessary retrieval; overhead <100ms per sub-problem
  - References: docs/QUERY_ANALYSIS.md (to be created), docs/RELEVANCE_SEARCH.md (to be created), docs/CONTEXT_INJECTION.md (to be created), docs/UNKNOWN_IDENTIFICATION.md (to be created)

- Consistency checking for knowledge accumulation
  - NLI-based validator (50-100M parameters) preventing contradictions in knowledge base
  - Validates new facts against existing KB before insertion
  - Example: KB contains "Paris is capital of France" → reject "Lyon is capital of France" (contradictory)
  - Detects: direct contradictions, logical inconsistencies, temporal conflicts
  - Maintains KB coherence score and contradiction-free guarantee
  - **Verification hooks**: After subtask completion, prompt LLM to verify "Does this output satisfy the subtask requirements?"; check against stated success criteria; flag inconsistencies before proceeding (catches errors early)
  - **Confidence and uncertainty tracking**: After each LLM response, prompt for confidence in decomposition/solution and what's uncertain or assumed; store metadata, propagate uncertainty through dependency graphs; surface high-uncertainty nodes for extra scrutiny
  - Acceptance: contradiction detection accuracy ≥95%; false positive rate <5%; validation latency <50ms per fact; KB coherence maintained ≥99%; prevents hallucination accumulation reducing error propagation ≥60%; verification catches errors ≥80% of time; uncertainty propagation identifies risky nodes with ≥85% accuracy
  - References: docs/CONSISTENCY_CHECKING.md (to be created), docs/VERIFICATION_HOOKS.md (to be created), docs/UNCERTAINTY_TRACKING.md (to be created)

- Explicit knowledge base construction and reasoning
  - ✅ **M4.5 IMPLEMENTED**: KV-eviction-to-KB architecture with inline summaries
    * When KV cache evicts tokens → summarize → store in KB with key
    * Replace evicted content with: `[KB:key] summary_text`
    * LLM sees placeholder, can request retrieval via `<RETRIEVE:key>`
    * System instructions (marked evict-last) teach LLM about KB system
    * Extends effective context beyond KV cache limits
    * No embeddings/vector search needed for basic operation
  - Knowledge base as first-class data structure accumulating verified facts through iterations
  - Structured KB with source attribution, confidence scores, and temporal validity
  - Convergence through unknown saturation (no new unknowns identified + no further decomposition)
  - Final reasoning simplified to lookup + composition over complete KB
  - Transparent reasoning trace with full provenance
  - **M4.5 Implementation**: Basic KB infrastructure (560 lines)
    * `Fact` struct with key, summary, full_content, confidence, category
    * `KnowledgeBase` with add/retrieve/query operations
    * `ConvergenceDetector` for fact saturation detection
    * Eviction history tracking for debugging
    * 12 comprehensive tests (all passing)
  - **Future Enhancements** (M6+):
    * **Multi-level eviction**: KB facts themselves can be archived when KB grows large
      - Tier 1: Active KB (in-memory, fast lookup)
      - Tier 2: Archived KB (disk/remote, slower retrieval)
      - Tier 3: Cold storage (compressed, rarely accessed)
      - Eviction policy: LRU + confidence + retrieval frequency
      - Acceptance: KB scales to 100K+ facts without memory issues; tier transitions <100ms; retrieval success rate >95%
    * **Semantic search with embeddings**: Find similar facts even without exact key
      - Embedding model (small, 50-200M params) for fact summaries
      - Vector store integration (FAISS, Qdrant, Milvus)
      - Similarity threshold tuning (precision vs recall)
      - Hybrid search: exact key lookup + semantic fallback
      - Acceptance: semantic retrieval precision >80%; latency <50ms; works when exact key unavailable
    * **LLM-based summarization**: Replace heuristic summaries with LLM-generated
      - Small summarization model (1-3B params, fine-tuned)
      - Configurable summary length and style
      - Batch summarization for efficiency
      - Quality metrics: informativeness, conciseness, accuracy
      - Acceptance: summary quality >90% vs human baseline; summarization latency <200ms; batch throughput >100 facts/sec
    * **Graph-based fact relationships**: Track how facts relate to each other
      - Fact graph with typed edges (supports, contradicts, elaborates, implies)
      - Relationship extraction from reasoning traces
      - Graph traversal for connected fact retrieval
      - Conflict detection via graph analysis
      - Acceptance: relationship extraction precision >75%; graph queries <10ms; conflict detection accuracy >85%
    * **NLI-based consistency checking**: Prevent contradictory facts (implemented in M4.5 design)
      - Dedicated NLI model (50-100M params) validates new facts
      - Detects: direct contradictions, logical inconsistencies, temporal conflicts
      - Example: Reject "Lyon is capital" if "Paris is capital" exists
      - KB coherence score tracking
      - Acceptance: contradiction detection >95% accuracy; false positives <5%; validation <50ms per fact
    * **Persistent KB across sessions**: Save/load KB state
      - Serialization format (JSON, MessagePack, or custom)
      - Incremental save on updates
      - Fast loading with memory mapping
      - Migration and versioning support
      - Acceptance: save <500ms for 10K facts; load <1s; no data loss
    * **KB compression and deduplication**: Merge similar facts
      - Fuzzy matching for near-duplicate detection
      - Fact merging with confidence aggregation
      - Compression strategies (summarize multiple facts)
      - Storage efficiency metrics
      - Acceptance: deduplication reduces KB size >30%; no information loss; compression ratio >3:1
    * **Federated KB sharing**: Share knowledge across instances
      - Privacy-tiered facts (private, trusted, public)
      - Schema translation for different KB formats
      - Conflict resolution for federated facts
      - Provenance tracking across nodes
      - Acceptance: federated retrieval <200ms; privacy preserved 100%; provenance traceable
    * **KB-aware prompt optimization**: Inject relevant facts proactively
      - Predict which facts will be needed for query
      - Pre-injection of high-relevance facts
      - Dynamic fact ranking by query context
      - Reduces explicit retrieval requests
      - Acceptance: retrieval requests reduced >40%; accuracy maintained; prediction precision >70%
  - Acceptance: KB construction overhead <10% of total iteration time; unknown saturation convergence ≥95% of problems; final reasoning accuracy ≥98% when KB complete; reasoning trace fully auditable; KB serialization <100ms
  - References: docs/EXPLICIT_KB_REASONING.md (to be created), src/engine/knowledge_base.rs (implemented)

- Basic MoE support
  - Efficient routing-friendly batching for Mixtral-like models
  - Capacity-aware routing option (Token Drop + Expanded Drop)
  - Learned multi-round routing policies with quality/latency tradeoffs
  - Routing overhead microbenchmarks with per-token latency telemetry
  - Load-balancer fallback and budgeted routing defaults
  - Lightweight bucketing of tokens by score to reduce straggler variance
  - Acceptance: functional demo with a small MoE; stable throughput under mixed loads; p95 step latency reduced ≥30% with ≤2% token drop on synthetic/gated traces; router logging for offline RL tuning; routing latency reduced ≥30% vs naive baseline
  - References: docs/summaries/capacity-aware-inference-moe.md, docs/summaries/router-r1-rl-routing.md, docs/summaries/2506-10943v2.md, docs/summaries/2506-16500v1.md

- Prompt program executor (Socratic templates)
  - Minimal interpreter for multi-turn prompt graphs (sequences/branches), per-request state, and stop/resume
  - Instruction-alignment monitoring with drift detection and restatement triggers
  - Acceptance: Run CRIT-like 5–8 step templates end-to-end; works with continuous batching; measurable overhead <5% vs single-turn on equivalent token volume; reduce instruction overrides on synthetic eval
  - References: docs/summaries/diagnosing-instruction-overriding.md

- Multi-agent coordination primitives
  - Shared and private memory namespaces with concurrency guards
  - Acceptance: stable multi-agent runs with no data races and bounded overhead
  - References: docs/summaries/mirix-multi-agent-memory.md

- Dynamic compute allocation (Policy trait system)
  - Unified interface for adaptation signals (entropy, margin, variance, agreement) and actions (exit/skip/repeat/route)
  - Early exit (entropy + patience + dynamic thresholds) with per-token/per-layer decisions
  - Self-adapting inference: per-input difficulty-based depth/width/cache allocation
  - Adaptive layer selection with shallow-then-deep heuristic and lightweight confidence estimators
  - Optional multi-armed bandit/UCB for dynamic threshold tuning and domain adaptation
  - Acceptance: ≥25–40% average layer/compute reduction at ≤2% accuracy loss on a small mixed eval; export per-request traces (signals, depth, time per token); stable under batch load; ≥20% FLOPs/token reduction with <1% increase in task error
  - References: docs/summaries/early-exit-nlp-survey.md, docs/summaries/self-adapting-language-models.md, docs/summaries/dynamic-neural-networks-survey.md, docs/summaries/2506-04761v2.md

- Verification-first sampling
  - Pluggable verifier library with symbolic checks (type, range, schema) and numeric validation
  - Two-stage hybrid verifier: fast symbolic patterns followed by cheap numeric heuristics
  - Batched verification with sampling API hooks for early rejection
  - Verifier integration into sampling loop with tracing
  - Acceptance: verifier runs <5ms median latency; reduces downstream error rate >50% on targeted tasks; end-to-end compute reduction ≥10% on benchmark reasoning tasks while maintaining accuracy
  - References: docs/summaries/2506-15882v1.md, docs/summaries/2508-15260v1.md

M5 — Frontier options (0.6)

**Name Mapping Integrations** (M5.1-M5.6):
- **M5.1**: Cache Management (3 weeks) - Architecture-aware layer detection for KV cache, variable layer count support, relationship-aware eviction
- **M5.4**: LoRA Integration (2 weeks) - Automatic format detection (HuggingFace, custom, PEFT), adapter-to-base-model name mapping, validation
- **M5.6**: Tool Registry (2 weeks) - Auto-detect model capabilities from architecture, register appropriate tools
- References: `docs/NAME_MAPPING_FEATURE_INTEGRATIONS.md`, M3.7 (Core name mapping)

- KV cache optimization
  - H2O eviction heuristics; KIVI/KVQuant-style KV quant (2–4 bit); R-KV training-free compression policy (importance–redundancy scoring)
  - Low-rank attention approximations for long-context tasks with tunable rank parameter
  - Relationship-aware eviction: use semantic/temporal/causal importance scoring vs pure LRU/LFU
  - Token importance based on multi-dimensional relationships (reference chains, semantic clusters, causal dependencies)
  - "Conscious" memory management: selective retention of high-value tokens vs automatic truncation
  - **MP4 Compressed Paging (NEW - Medium Priority)**: Hardware video codec compression for paged KV cache
    - Block-level compression using NVENC/NVDEC hardware on NVIDIA GPUs (16-64 tokens per MP4 frame)
    - Amortized latency: 2ms decode / 16 tokens = 0.125ms per token (acceptable overhead)
    - Prefetching pipeline: Decode block N+1 while computing block N (zero added latency in sequential generation)
    - Expected compression: 5-10× memory reduction (Llama-7B 32K context: 16GB → 2.5GB)
    - Architecture: `CompressedPagedCache` wrapping existing `PagedAttention` infrastructure from candle-vllm
    - Implementation timeline: Week 3-4 after AWQ/speculative decoding (5 days prototype + 5 days integration)
    - Performance targets: Best case (sequential): 0ms effective latency, 6-10× memory savings; Typical case (mixed): 5-10% slower, 80% prefetch hit rate; Worst case (random): 20% slower, still viable for long context
    - Infrastructure exists: `src/paged_attention/`, `src/scheduler/block_engine.rs` already in candle-vllm
    - Acceptance: 5-10× memory compression ratio achieved; decode latency amortized to <0.2ms per token; prefetch hit rate >70% on sequential workloads; enables 32K+ context on consumer GPUs; benchmarks validate compression quality
    - References: MemVid project (QR codes in MP4), NVENC/NVDEC documentation, PagedAttention (vLLM), docs/MP4_COMPRESSED_PAGING.md (to be created)
  - **Note**: M5 uses static allocation; elastic KV cache (KVCached-style) is M6.5
  - Acceptance: 30–50% KV memory reduction on long contexts with minimal degradation in simple QA prompts (documented). For R-KV, at budget b≈0.34, throughput improves ≥1.5× on CPU long decodes with parity to baseline within tolerance. Low-rank approximation shows throughput gains with <1.5% perplexity degradation. Relationship-aware eviction maintains quality ≥5% better than LRU on context-dependent tasks.
  - References: R-KV; low-bit LLMs survey (KV cache); efficient transformers survey; docs/summaries/2508-19828v1.md; docs/RELATIONSHIP_AWARE_KV.md (to be created)

- Pruning utilities (Wanda + tail prune)
  - One-shot Wanda scoring for unstructured and 2:4/4:8 structured pruning; reverse-order tail prune (~25%) with optional partial-layer FT (lm head + last 1–3 layers)
  - Acceptance: 2:4 path shows ≥1.4–1.6× matmul speedup with ≤1 ppl degradation; tail-prune yields ≤2% avg accuracy drop on small eval; provide manifest + loader hooks
  - References: docs/summaries/wanda-pruning.md; docs/summaries/layer-pruning-reassessment.md

- Test-time depth adaptation (CoLa prototype)
  - Offline harness to simulate skip/repeat layer decisions (e.g., simple MCTS) without training; log accuracy vs depth
  - Acceptance: ≥10–20% mean depth reduction with neutral or improved accuracy on a small reasoning subset; document tasks where recurrence helps
  - References: docs/summaries/cola-test-time-depth-adaptation.md

- Adaptive mixed-precision profiling ✅ (COMPLETED)
  - Per-layer microprofiling to select optimal precision at startup or dynamically
  - Per-core int4/int8 kernel profiling with saved profiles for heterogeneous CPUs
  - Fast profiling with conservative defaults to minimize runtime overhead
  - **Implementation**: `src/engine/mixed_precision.rs` with 12 tests passing
  - **Features**: Dynamic precision selection (FP32/FP16/BF16/INT8), activation statistics tracking, accuracy-based adjustment, per-layer profiling
  - Acceptance: better throughput/accuracy curves vs uniform precision baselines; improved per-core throughput in mixed-core setups
  - References: docs/summaries/2510-06557v1.md, docs/summaries/2510-04871v1.md

- Reasoning efficiency controls ✅ (COMPLETED)
  - Budget-aware decoding knobs (max chains, max samples, verifier frequency, overthinking detection)
  - Shorter-is-better heuristic with patience floor and maximum depth cap per input class
  - Confidence- and consistency-based chain termination with verifier hooks
  - Re-ranking API with pairwise/listwise aggregators, uncertainty calibration, and compositional scoring (style+correctness+safety)
  - **Multi-level abstraction management**: Maintain high-level goal, current abstraction level, and tools to zoom in/out ("decompose this further" vs "summarize these subtasks"); prevents LLM from getting lost in details or staying too abstract; abstraction-level-aware context management
  - **Implementation**: `src/engine/reasoning_controls.rs` with 11 tests passing
  - **Features**: Budget constraints (max chains/steps/tokens), overthinking detection (entropy/variance analysis), repetition detection, termination policies (fixed steps/confidence/convergence/special token), reasoning chain tracking
  - Acceptance: ≥15–25% compute reduction with neutral/improved accuracy on small reasoning evals; controllable latency with monotonic quality-cost trade-offs; re-ranking improves exact-match ≥2–3pp at equal or lower token cost; abstraction management reduces context loss by ≥40%; zoom operations complete <100ms
  - References: docs/summaries/efficient-reasoning-models-survey.md, docs/summaries/optimal-inference-length.md, docs/summaries/dont-overthink-it.md, docs/summaries/thought-terminator.md, docs/summaries/reward-modeling-as-reasoning.md, docs/ABSTRACTION_MANAGEMENT.md (to be created)

- Reasoning path compression and reuse
  - Cache compressed reasoning templates for recurring tasks; measure token savings vs accuracy
  - Selective reread/re-prefill on low confidence with chunked KV refresh
  - Acceptance: ≥20% token reduction on targeted tasks with neutral accuracy; improved accuracy with <20% extra tokens on reread-triggered cases
  - References: docs/summaries/reasoning-path-compression.md, docs/summaries/rereading-improves-reasoning.md

- Adaptive chunking and context management
  - Configurable chunking policy (fixed vs adaptive/learned segmentation)
  - Chunk-level cache reuse with boundary logging and reuse rate tracking
  - Acceptance: similar perplexity with improved throughput vs fixed windowing on long-context micro-benchmarks
  - References: docs/summaries/dynamic-chunking.md

- Test-time adaptation
  - Per-instance policy gradient tuner for sampling parameters and exit thresholds (budget-capped)
  - Text-to-LoRA adapter registry with prompt-metadata-based selection
  - Acceptance: measurable accuracy gains on small reasoning subset with bounded latency overhead; quality gains on domain micro-benchmarks with negligible TTFT increase
  - References: docs/summaries/seek-in-the-dark-ttilpg.md, docs/summaries/text-to-lora-instant-adaptation.md

- Modular specialization
  - Module hot-swap framework for specialized experts/adapters on frozen substrate
  - Acceptance: functional module hot-swap demo with stable latency and measurable domain gains
  - References: docs/summaries/growing-transformers.md

- Three-tier dynamic memory management system
  - ⚠️ **CONTRADICTED** (see "dead scaffold crates" finding above): `infra-storage` crate - Multi-backend storage abstraction (RocksDB, SQLite, Sled, Memory)
  - ⚠️ **CONTRADICTED** (see "dead scaffold crates" finding above): `infra-fingerprinting` crate - Multi-level fingerprinting for cache key generation
  - Explicit GPU → RAM → SSD hierarchy with quantified allocations
  - Unified storage backend supporting multiple persistence strategies
  - Content-addressable caching using multi-level fingerprints (atomic, relational, structural, semantic)
  - Predictive capability loading based on learned transition patterns
  - LRU eviction with usage pattern awareness
  - Background loading during active computation
  - Transfer performance: RAM→GPU <200ms, SSD→RAM→GPU <3s
  - Acceptance: cache hit rate ≥80%; load times meet targets; memory tiers transition smoothly; predictive loading reduces perceived latency by ≥40%
  - References: docs/THREE_TIER_MEMORY.md (to be created), docs/INFRASTRUCTURE_CRATES.md, docs/DYNANIML_INTEGRATION.md

- Communication protocol evolution system
  - Dedicated layer that learns compressed communication between components
  - Progressive protocol optimization (verbose → abbreviated → compressed)
  - Training/production mode toggle for ongoing refinement
  - Semantic preservation verification (≥95% fidelity)
  - Feedback loops for continuous improvement
  - Acceptance: protocol compression reduces tokens by ≥50%; semantic fidelity ≥95%; training mode provides rich feedback; production mode achieves minimal overhead
  - References: docs/PROTOCOL_EVOLUTION.md (to be created)

- Tool registry and invocation framework
  - Extensible plugin system with schema definitions (inputs/outputs/constraints)
  - Provider metadata (version, capabilities, trust score, usage stats)
  - Secure sandboxed invocation with resource limits
  - Telemetry and quality scoring per tool
  - Dependency system: explicit prerequisites, automatic resolution, conflict detection
  - Capability verification: validate tools perform as advertised
  - Plugin marketplace integration:
    - Standardized plugin manifest format (version, dependencies, capabilities, author, license)
    - Discovery mechanism for available plugins (local and remote registries)
    - Version compatibility checking and semantic versioning
    - Community plugin sharing and distribution
    - Plugin metadata indexing and search
  - Communication layer enhancements:
    - JSON-RPC 2.0 as standard protocol for tool invocation (human-readable, simple)
    - Shared memory IPC for high-performance local tool communication (memory-mapped files + semaphores)
    - Binary protocol support (Cap'n'Proto) for high-throughput scenarios
    - TCP/IP for remote tool connectivity
  - Security infrastructure:
    - OAuth 2.0 authentication for remote tool connections
    - Mutual TLS for connection authenticity and MITM prevention
    - Capability-based security model for local tools (process isolation with explicit permissions)
    - Service registry for secure remote plugin discovery and validation
  - Acceptance: Register and invoke ≥5 heterogeneous tools; sandboxing prevents resource abuse; telemetry tracks success/failure rates with <2% overhead; shared memory IPC shows ≥50% latency reduction vs TCP for local tools; JSON-RPC 2.0 + binary protocol toggle functional; OAuth 2.0 + mTLS working for remote connections; capability-based isolation prevents privilege escalation ≥99% of cases; dependency resolution handles ≥3 levels; marketplace discovery finds ≥90% of compatible plugins
  - References: docs/TOOL_REGISTRY.md, docs/PLUGIN_MARKETPLACE.md, docs/TOOL_COMMUNICATION.md, docs/TOOL_SECURITY.md (to be created)

- MCP service discovery and protocol bridging
  - gRPC reflection for dynamic endpoint discovery by LLM agents
  - Automatic service mesh integration for micro-service orchestration
  - JSON-RPC façade for protocol translation and compatibility
  - Dynamic capability advertisement and negotiation
  - Service health monitoring and automatic failover
  - Protocol version negotiation and backward compatibility
  - Acceptance: LLM agents can discover ≥90% of available services without manual configuration; protocol bridging maintains ≥95% semantic fidelity; service mesh reduces inter-service latency by ≥30%; automatic failover recovery time <2 seconds
  - References: docs/MCP_SERVICE_DISCOVERY.md, docs/PROTOCOL_BRIDGING.md (to be created)

- Multi-modal tensor routing
  - Modality-aware metadata (text, image, audio, code, structured data)
  - Type-safe tensor routing with validation
  - Heterogeneous batch composition and scheduling
  - Cross-modal attention support hooks
  - Acceptance: Route mixed-modality batches correctly; heterogeneous batching achieves ≥80% of homogeneous throughput; type violations caught at submission
  - References: docs/MULTIMODAL_ROUTING.md (to be created)

- Multi-method generation routing
  - Plugin architecture for generation backends (autoregressive, diffusion, retrieval, symbolic)
  - Task-based method selection (factual → RAG, creative → diffusion, reasoning → autoregressive)
  - Parallel generation with ensemble voting
  - Confidence-weighted result aggregation
  - Acceptance: Support ≥3 generation methods; router selects optimal method with ≥85% accuracy; ensemble improves quality by ≥5% on mixed tasks
  - References: docs/GENERATION_ROUTING.md (to be created)

- Tiered answer generation framework
  - Five-tier routing: T1 (cache/KB, <0.1s) → T2 (small LLM/RAG, <1s) → T3 (web/APIs, <3s) → T4 (large LLM, <5s) → T5 (diffusion/synthesis, <10s)
  - Unknown type classifier for automatic tier selection
  - Tier-specific latency budgets and fallback policies
  - Cache warming and pre-fetching for T1
  - Acceptance: T1-T2 resolve 70-80% of queries; average latency reduced 2-3× vs always-using-T4; tier selection accuracy ≥90%
  - References: docs/TIERED_GENERATION.md (to be created)

- Modular neural network architecture support
  - Support for segmented models (TheoryGenerator → SolverSurrogate pattern)
  - Inter-module communication via typed Hub IR
  - Shim generation hooks (LLM-based or rule-based translation)
  - GPU kernel fusion for module boundaries
  - Acceptance: Run 2+ module pipelines; shims add <5% overhead; GPU fusion reduces boundary cost by ≥30%
  - References: docs/MODULAR_NN.md (to be created)

- Offloading/systems experiments
  - FlexGen-style GPU/CPU/disk scheduling; PowerInfer-style hot/cold neuron experiments
  - Lightweight model sharding for CPU clusters with async partitioning and pipelining
  - Acceptance: design spikes + prototypes with measurements and go/no-go notes; demonstrable scaling across multiple CPU nodes with acceptable latency overhead
  - References: docs/summaries/2509-13341v1.md

- Device-side kernel sequences (CUDA backend)
  - Execute dependency graphs (DAGs) entirely on GPU without host round-trips
  - Bitmask-based dependency tracking for efficient synchronization
  - Replace host-launched CUDA Graphs with device-controlled execution
  - AOT compilation or runtime construction via NVRTC+nvJitLink
  - Support for complex multi-kernel workflows with conditional execution
  - Acceptance: multi-kernel sequences execute without host involvement; dependency resolution overhead <1% of compute time; supports ≥10-node DAGs; numerical accuracy preserved; feature-gated for portability
  - References: docs/DEVICE_KERNEL_SEQUENCES.md (to be created)

- Repeatable job system
  - Job execution modes: ONCE, N_TIMES, UNTIL_REPLACED
  - Fill idle GPU cycles with speculative decode, KV-cache maintenance, or warm-up tasks
  - Priority preemption ensures real work takes precedence over background jobs
  - Automatic idle detection and background job scheduling
  - Support for speculative execution with result validation
  - Acceptance: idle GPU cycles reduced ≥40%; speculative work improves throughput ≥5-15%; priority jobs preempt background work <10µs; no interference with critical path latency
  - References: docs/REPEATABLE_JOB_SYSTEM.md (to be created)

- Fingerprint-based multi-granularity activation caching
  - Layer-level caches mapping input activation fingerprints to output activations
  - Multi-layer span caches (e.g., bypass layers 3-7) for frequently occurring patterns
  - Network-level caches for complete input-to-output mappings
  - LSH or learned embedding fingerprints for compact representation
  - Similarity-based retrieval for partial matches above threshold
  - Heterogeneous computation paths: fast (cached), partial (mixed), full (novel)
  - Acceptance: cache hit rate ≥40% for repeated patterns; fingerprint lookup <100µs; memory overhead <20% of model size; accuracy maintained within 0.3% vs full computation; cache serialization for deployment
  - References: docs/ACTIVATION_CACHING.md (to be created)

- Usage-driven early exit and bypass discovery
  - Early exit strategies emerge from runtime usage patterns (not pre-configured)
  - Input-class-specific bypass strategies discovered automatically
  - Natural termination point identification where rule chains end
  - Different computational paths for different input distributions
  - Exit confidence calibration through historical pattern tracking
  - Acceptance: early exits discovered for ≥3 distinct input classes; bypass patterns reduce computation ≥25% for cached classes; exit decisions maintain accuracy within 1% of full network; patterns adapt to distribution shifts within 1000 samples
  - References: docs/EMERGENT_EARLY_EXIT.md (to be created)

- Flash-Decoding/multi-token prediction exploration
  - Acceptance: feasibility notes; potential integration hooks in decode loop

- Evaluation infrastructure
  - SPARQ-like synthetic problem generators for varied difficulty and structure
  - ReasoningGym adapter for offline policy tuning with verifiable rewards
  - Acceptance: broader eval coverage; stable policy tuning across difficulty levels; reproducible gains on matching environments
  - References: docs/summaries/sparq-synthetic-problems.md, docs/summaries/reasoning-gym-rl-envs.md

- RL/training support (logging and traces)
  - Episode traces and rewards from prompt programs in replay-friendly format
  - Async rollout logging for router/policy training (AReaL patterns)
  - Principal weight diagnostics for reasoning-focused fine-tuning validation
  - Acceptance: doc-only for now; establish data contracts and metrics for RL pipelines; one diagnostic implemented as unit test
  - References: docs/summaries/areal-async-rl-reasoning.md, docs/summaries/lift-the-veil-principal-weights.md

- Context condensation pipeline
  - Multi-phase context refinement for LLM reasoning: initial → recursive → multi-model → cross-reference → expand → re-condense
  - Initial condensation generates functional, structural, and causal summaries
  - Recursive refinement in multiple passes until token budget or relevance saturation
  - Multi-model condensation using diverse models in parallel (instruction-tuned, long-context, domain-specific)
  - Cross-referencing summaries to original sources with traceability
  - Expansion phase triggers web search, KB queries, or tool reruns for missing context
  - Post-expansion re-condensation to prevent context pollution
  - Acceptance: context compression ≥60% while maintaining ≥90% semantic fidelity; multi-model consensus improves reasoning quality ≥20%; expansion detects missing dependencies ≥70% of time; final context relevance ≥85%
  - References: docs/CONTEXT_CONDENSATION_PIPELINE.md (to be created)

- Semantic anchor navigation system
  - Named anchors with semantic tags, summaries, and relationship metadata
  - Operations: preview, open_view_from_anchor, find_related_anchors
  - Enables efficient navigation through large codebases and datasets without loading full content
  - Anchor metadata includes relevance scores, token estimates, and relationship types
  - Lazy loading of anchor targets based on LLM requests
  - Acceptance: navigation overhead <5% of baseline full-load; anchor preview accuracy ≥85%; related anchor suggestions relevant ≥75% of time; token savings ≥40% vs full context loading
  - References: docs/SEMANTIC_ANCHOR_NAVIGATION.md (to be created)

- Declarative data window management (views)
  - Token-aware scrollable, labeled, expandable windows into structured content
  - Operations: scroll, resize, close, refresh, get_view_metadata
  - Viewport management with automatic token budget tracking
  - Progressive disclosure: start minimal, expand on explicit request
  - View composition: merge multiple views with configurable strategies
  - Acceptance: reduces context pollution ≥50% vs full dumps; viewport operations <100ms; token estimates accurate within ±10%; view composition maintains coherence ≥90%
  - References: docs/DECLARATIVE_DATA_WINDOWS.md (to be created)

- Tool workflow pattern library
  - Formalized chaining modes: sequential, fallback, parallel, conditional, iterative
  - Declarative composability: can_chain_with, default_order, blocking, dependent_on
  - Orchestration hints for optimal tool coordination
  - Workflow templates for common multi-tool patterns
  - Integration with MCP service discovery and tool registry
  - Acceptance: supports ≥5 chaining modes; workflow templates reduce orchestration errors ≥40%; composability declarations prevent ≥80% of invalid tool combinations; execution overhead <3%
  - References: docs/TOOL_WORKFLOW_PATTERNS.md (to be created)

- Dynamic module loading for edge deployment
  - Sequential loading/unloading of modules based on query classifier
  - Memory-constrained execution (e.g., 2GB RAM on smartphone)
  - Module dependency tracking and smart caching
  - Lazy initialization with predictive preloading
  - Fallback to simpler modules when resources exhausted
  - Acceptance: enables deployment on devices with <25% of monolithic model memory requirements; query classification accuracy ≥90%; module swap latency <100ms; predictive preloading reduces swap frequency ≥60%; fallback degradation <10% accuracy vs full model
  - References: docs/DYNAMIC_MODULE_LOADING.md (to be created)

---

M5.5 — CLI, Deployment & Operations (0.6+) **→ v1.0 Release Candidate**

**Status**: PLANNED (Final polish for production release)

**Goal**: Package Lightbulb with production-ready CLI and deployment infrastructure for v1.0 release

- **Command-Line Interface (NEW)**
  - Server mode: `lightbulb serve --model <model_path> --port 8080`
    * OpenAI-compatible API endpoint (/v1/completions, /v1/chat/completions)
    * Configuration via CLI flags, YAML config, or environment variables
    * Hot-reload configuration on SIGHUP
  - Generation mode: `lightbulb generate --model <model_path> --prompt "..."`
    * Single-shot generation for scripting/testing
    * Batch mode: `lightbulb generate --model <model_path> --input prompts.jsonl`
  - Model utilities: `lightbulb convert --from hf --to candle --model <hf_id>`
    * HF → Candle safetensors conversion
    * GGUF ↔ safetensors conversion
    * Quantization pipeline (FP16 → Q8 → Q4)
  - Inspection tools: `lightbulb inspect --model <model_path>`
    * Display model architecture, parameter count, quantization info
    * Test model with sample prompts
  - Acceptance: All CLI commands work; server responds to requests; conversion utilities produce valid models

- Container and orchestration support
  - Docker images with multi-stage builds (optimized for size)
  - Support for CPU-only, CUDA, and ROCm variants
  - Kubernetes deployment manifests (Deployment, Service, ConfigMap)
  - Helm charts with configurable replicas, resources, and feature flags
  - Horizontal Pod Autoscaling based on request queue depth
  - Acceptance: single docker pull command for deployment; k8s manifest works on GKE/EKS/AKS; HPA scales based on load

- Observability and monitoring integration
  - Prometheus metrics exporter with standard metric names (request_duration, tokens_per_second, cache_hit_rate, etc.)
  - Grafana dashboard JSON templates for inference metrics, scheduler state, memory usage
  - OpenTelemetry spans for distributed tracing across multi-stage pipelines
  - Structured logging with configurable levels (JSON format for log aggregation)
  - Health check endpoint (/health) with liveness and readiness probes
  - Acceptance: metrics visible in Prometheus; Grafana dashboard functional; traces show request flow; logs parseable by ELK/Loki

- Operational reliability
  - Graceful shutdown on SIGTERM (finish in-flight requests, persist state)
  - State checkpoint/restore for zero-downtime upgrades
  - Circuit breaker for failing backends (draft model, tool calls)
  - Rate limiting per client/tenant with token bucket algorithm
  - Admin API (/admin/cache, /admin/scheduler, /admin/config) for runtime inspection
  - Acceptance: zero dropped requests on graceful shutdown; state restored after restart; rate limits enforced; admin API provides useful diagnostics

- Configuration and developer experience
  - HuggingFace model ID direct loading (auto-download via hf_hub, convert to Candle format, cache locally)
  - YAML/JSON config files with JSON schema validation
  - Environment variable overrides for 12-factor app compatibility
  - Sane defaults: CPU-only, no speculative decoding, basic batching, standard KV policy
  - Feature flag system: enable-speculative, enable-flash-attention, enable-quantization, etc.
  - Configuration hot-reload via distributed-config (no restart required)
  - Debug mode: detailed logging, cache introspection, policy decision traces
  - Acceptance: HF model loads with single ID; config validates on startup; feature flags work; debug mode aids troubleshooting

- Model conversion utilities
  - HF safetensors → Candle converter with metadata preservation
  - GGUF ↔ safetensors bidirectional conversion (integrated with CLI)
  - Quantization pipeline (FP16 → Q8 → Q4) with calibration dataset support
  - Model testing script (loads model, runs sample prompts, validates outputs)
  - Acceptance: convert popular HF models (Llama, Mistral, Phi) successfully; quantized models load and run; test script catches broken conversions

- **Comprehensive Test Coverage & Validation (NEW)**
  - **Unit Test Coverage Deep Dive**: Ensure every testable component has tests
    * Core engine: KV cache, scheduler, batching, memory management (target: 95%+ coverage)
    * Pipeline orchestration: prefill/decode stages, multi-stage routing, convergence detection
    * Optimizations: FlashAttention, kernel fusion, speculative decoding, quantization loaders
    * Multi-GPU: tensor parallelism, pipeline parallelism, distributed KV cache, communication primitives
    * Knowledge base: fact storage, eviction tracking, convergence detection, retrieval
    * MoE routing: top-K selection, capacity management, load balancing, routing policies
    * Memory systems: arena allocation, CPU-GPU hybrid, three-tier memory hierarchy
    * Utilities: tensor ops, layer norm, RoPE, softmax, token sampling
    * Coverage tool: `cargo-llvm-cov` or `tarpaulin` for line/branch coverage reporting
  - **Integration Test Matrix**: Cross-feature validation
    * Feature combinations: [CPU/CUDA] × [FP32/FP16] × [Q4/Q8/None] × [FlashAttn Y/N] × [Speculative Y/N] × [Batching 1/8/32]
    * Model architectures: Llama, Mistral, Phi, Mixtral (MoE)
    * Context lengths: short (512), medium (4k), long (32k), extreme (128k with StreamingLLM)
    * Multi-GPU configs: 1-GPU, 2-GPU tensor parallel, 4-GPU pipeline, 2×4 hybrid
    * Correctness validation: token-by-token comparison with reference implementations (Candle, HF)
    * Determinism: fixed seed reproducibility across all configurations
  - **Edge Case & Stress Testing**:
    * Empty prompts, single-token prompts, max-length prompts
    * Unicode edge cases: emojis, RTL text, zero-width characters, surrogate pairs
    * Batch sizes: 1, 10, 100, 500, 1000 (scaling behavior validation)
    * Context window boundaries: exactly at limit, one over limit, graceful truncation
    * Memory exhaustion scenarios: OOM handling, graceful degradation, error recovery
    * Concurrent stress: 500+ simultaneous requests, queue management under load
    * Long-running stability: 48-hour soak tests with continuous traffic
    * Network failures: retry logic, timeout handling, circuit breaker activation
  - **Regression Detection Pipeline**:
    * CI integration: run benchmarks on every commit to main branch
    * Historical database: SQLite store for throughput, latency, memory, perplexity over time
    * Alert thresholds: >10% throughput degradation, >15% latency increase, >5% memory growth
    * Automated bisection: identify offending commit when regression detected
    * Dashboard: visualize performance trends over repository history
  - **Test Infrastructure**:
    * Deterministic test fixtures: fixed-seed random data generators
    * Mock backends: simulated GPU for CPU-only testing of multi-GPU logic
    * Test model repository: tiny models (10M params) for fast CI, standard models (7B) for validation
    * Automated test data generation: synthetic prompts, edge case corpus
    * Parallel test execution: reduce CI time with test sharding
  - Acceptance: **95%+ code coverage**, all feature combinations tested, edge cases handled, regression detection active, 48-hour soak test passes

- **Comprehensive Benchmark Suite (NEW - v1.0 Marketing Materials)**
  - **Performance Benchmarks**: Showcase Lightbulb's speed and efficiency
    * **Throughput**: tokens/second across batch sizes (1, 8, 32, 64, 128)
      - Models: Llama-7B, Llama-13B, Mistral-7B, Phi-3-mini, Mixtral-8×7B
      - Hardware: CPU (AMD EPYC, Intel Xeon), GPU (RTX 4090, A100, H100)
      - Configurations: FP16 baseline, Q8, Q4, FlashAttention, Speculative decoding
      - Comparison: vs llama.cpp, vLLM, TGI (TensorRT-LLM if possible)
    * **Latency**: TTFT (time to first token) and inter-token latency (p50, p95, p99)
      - Context lengths: 512, 2k, 8k, 32k tokens
      - Batch sizes: 1, 10, 50 concurrent requests
      - Show impact of prefix caching on repeated prefixes
    * **Memory Efficiency**: peak memory usage vs context length
      - KV cache memory vs StreamingLLM policy
      - Multi-GPU memory distribution and utilization
      - Quantization memory savings (FP16 vs Q8 vs Q4)
    * **Scaling**: Multi-GPU speedup (1-GPU baseline → 2/4/8 GPU)
      - Tensor parallelism efficiency (communication overhead)
      - Pipeline parallelism throughput (bubble analysis)
      - Hybrid configurations (2×4 tensor+pipeline)
  - **Feature-Specific Benchmarks**: Validate optimization impact
    * **FlashAttention**: speedup vs manual attention (2-5× expected on GPU)
    * **Speculative Decoding**: acceptance rate vs speedup tradeoff
    * **Kernel Fusion**: fused_linear_silu vs separate ops performance
    * **Prefix Caching**: TTFT reduction on repeated system prompts (>15% target)
    * **StreamingLLM**: memory footprint vs context length (constant beyond window)
    * **MoE Routing**: load balancing quality, token drop rates, expert utilization
    * **Knowledge Base**: eviction/retrieval latency, convergence speed
  - **Quality Benchmarks**: Ensure optimizations preserve accuracy
    * **Perplexity**: WikiText-2, C4 validation set (vs HuggingFace baseline)
    * **Task Performance**: MMLU, HellaSwag, ARC, TruthfulQA (optional: requires eval harness)
    * **Quantization Impact**: Q8 (<1% degradation), Q4 (<3% degradation) vs FP16
    * **Numerical Precision**: cosine similarity of outputs (FP32 vs FP16 vs quantized)
  - **Real-World Workload Simulations**:
    * Chatbot: conversational turns with context accumulation
    * Code generation: function completion, code explanation
    * Document QA: long context (10k+ tokens) with targeted queries
    * Multi-turn reasoning: chain-of-thought with intermediate steps
  - **Benchmark Presentation**:
    * Markdown tables: throughput/latency/memory across configurations
    * Graphs: scaling curves (batch size, context length, GPU count)
    * Comparison charts: Lightbulb vs llama.cpp vs vLLM (throughput, latency, memory)
    * Feature impact visualizations: speedup from each optimization
    * Reproducibility: scripts to re-run benchmarks, hardware specs documented
    * Published results: website landing page, GitHub README, blog post
  - Acceptance: **Benchmark suite runs in <2 hours**, results in presentable format (tables + graphs), comparison with 2+ competitors, reproducible scripts included, published on lightbulb.dev website

- **Release Deliverables**
  - Pre-built binaries: Linux (x86_64, aarch64), macOS (Intel, Apple Silicon), Windows
  - Docker images: `lightbulb/lightbulb:v1.0`, `lightbulb/lightbulb:v1.0-cuda`
  - Comprehensive documentation: Installation guide, API reference, deployment examples
  - Example configurations: Single-node, multi-GPU, Kubernetes cluster
  - **Test coverage report**: HTML coverage report with 95%+ line coverage badge
  - **Benchmark results dashboard**: Interactive HTML with graphs and comparison tables
  - Acceptance: Release artifacts build cleanly; documentation complete; examples work out-of-box; **test coverage ≥95%**; **benchmarks published**

---

**M5.6 — Hardware-Specific Optimizations (Post-v1.0)** **→ Future Hardware Support**

**Status**: PLANNED (requires hardware access or funding for testing)  
**Added**: Oct 2025 (Hardware utilization assessment identified optimization gaps)  
**Dependencies**: M5.5 complete (v1.0 released), access to Hopper H100 or Blackwell hardware

**Context**: Comprehensive hardware utilization assessment (Oct 2025) identified that Lightbulb is well-optimized for Ampere/Ada GPUs (A100, RTX 30/40 series) with FlashAttention-2, FP16/BF16 Tensor Cores, and Multi-GPU support. However, significant optimization opportunities exist for newer hardware (Hopper H100, Blackwell GB200/GB300). These features require specialized hardware for testing and validation.

**Hopper H100 Optimizations (Highest Priority)**:

- **FP8 Training and Inference Support** (MAJOR GAP - 2× memory/compute gain)
  - Blocked by: Candle doesn't support F8E4M3/F8E5M2 dtypes yet
  - Implementation when available:
    * Add FP8 dtype support to model loading (safetensors with FP8 weights)
    * FP8 Tensor Core utilization via Candle/cuBLAS (automatic when dtype supported)
    * Mixed FP8/FP16 precision: FP8 matmuls + FP16 accumulation
    * Per-tensor and per-channel scaling strategies
    * FP8 KV cache support (extends M5 KV compression work)
  - Performance targets:
    * 2× memory reduction vs FP16 (enables larger models on same hardware)
    * 2× compute throughput via Hopper FP8 Tensor Cores
    * Throughput: 120-140 tok/s for Mistral-7B (vs 60-70 tok/s FP16 baseline)
  - Quality targets:
    * <1% accuracy degradation vs FP16 baseline on MMLU/HellaSwag
    * Perplexity increase <2% on WikiText-2
  - Estimated effort: 2-3 weeks when Candle adds dtype support
  - Testing requirements: Hopper H100 GPU (80GB VRAM)
  - References: docs/FP8_TRAINING_INFERENCE.md (to be created)
  - Acceptance: FP8 models load and run; 2× memory/compute gain achieved; <1% accuracy loss; Candle FP8 dtype PR merged

- **FlashAttention-3 Integration** (Hopper-specific optimization)
  - Monitor Candle upstream for FA3 availability (check quarterly)
  - Expected improvements over FlashAttention-2:
    * Warp-specialization for different sequence lengths
    * Reduced shared memory usage
    * Better utilization on H100 Tensor Cores
    * 1.5-2× speedup over FA2 on long contexts (>8k tokens)
  - Estimated effort: 1 week (API likely similar to FA2)
  - Testing requirements: Hopper H100 GPU
  - References: docs/M3_4_FLASHATTENTION_INTEGRATION.md (existing FA2 docs)
  - Acceptance: Same numerical parity tests as FA2 (1e-3 tolerance); improved performance benchmarks vs FA2; long context speedup ≥1.3×

**Blackwell GB200/GB300 Optimizations (Long-term)**:

- **NVFP4 Training Support** (6× training speedup - DEFERRED for v1.0)
  - Context: NVFP4 is training-focused, Lightbulb is inference-focused
  - Consider only if training efficiency becomes priority
  - Requires Blackwell Tensor Cores for performance benefits
  - Implementation complexity: HIGH (3-4 months)
  - Key techniques from paper:
    * Two-level microscaling: E4M3 FP8 block scales + FP32 tensor scales
    * Mixed precision: Last 8-15% of layers in BF16
    * Random Hadamard Transforms for Wgrad outlier dispersion
    * Stochastic rounding (hardware-dependent)
    * 2D block scaling (16×16) for chain rule consistency
  - Performance: 6× training speedup (GB300) or 4× (GB200) vs BF16
  - Memory: 50% vs FP8, 75% vs FP16
  - Status: WATCH - Revisit if training becomes core feature
  - References: docs/NVFP4_ANALYSIS.md (comprehensive paper analysis from Oct 2025)
  - Acceptance: 12B model trains in 6× less time; validation loss within 1.5% of FP8

- **FP8 Support** (same as Hopper, but optimized for Blackwell)
  - Blackwell has improved FP8 Tensor Cores over Hopper
  - Same implementation as Hopper FP8, but higher performance
  - Expected: 2.5-3× vs FP16 (vs 2× on Hopper)

- **FlashAttention-3** (same as Hopper, if released)

**Quality-of-Life Features (All GPUs)**:

- **Automatic Mixed Precision (AMP)** (Training-focused, lower priority for inference)
  - Automatic FP32 master weights + FP16/BF16 compute
  - Dynamic loss scaling to prevent underflow
  - Per-layer precision selection based on numerical stability
  - Gradient overflow detection and recovery
  - Currently: Manual precision selection only (F32/F16/BF16 via config)
  - Benefit: Simplifies training workflows, reduces memory for training
  - Estimated effort: 2-3 weeks
  - Testing requirements: Any CUDA GPU
  - References: docs/AUTOMATIC_MIXED_PRECISION.md (to be created)
  - Acceptance: Training with AMP matches full-precision accuracy within 0.5%; memory usage reduced by 40%; easy enable/disable via config flag

**GPU Direct Storage — Zero-Copy Disk↔VRAM Transfer (All GPUs)**:

- **NVIDIA GPUDirect Storage (GDS)** (Highest priority — Linux, CUDA)
  - Enables NVMe storage to read/write directly to GPU memory via DMA
  - Bypasses CPU entirely — no CPU-mediated copies, no system memory staging
  - Uses cuFile API (~10 functions, straightforward Rust FFI bindings)
  - **Primary use case: Tiered KV cache promotion/demotion**
    * Current path: Disk → CPU bytes → tensor_from_bytes(CPU) → to_device(GPU) (~50ms)
    * GDS path: Disk → GPU VRAM directly via DMA (~5ms)
    * Our tensor_codec format is raw float arrays — already DMA-compatible
    * Implement as `GDSDiskStore` implementing existing `DiskStore` trait
  - **Secondary use case: Model weight loading**
    * Load SafeTensors/GGUF weights directly to VRAM
    * Could reduce 14GB model load from seconds to <1 second on NVMe
  - **Tertiary use case: KV cache checkpointing (M4.B)**
    * Fast bidirectional GPU↔disk for session persistence
    * Enables near-instant session save/restore
  - Expected bandwidth: 2x-8x higher than CPU-mediated path
  - Requirements: Linux, NVIDIA GPU, NVMe storage, cuFile driver
  - Estimated effort: 2-3 weeks (FFI bindings + GDSDiskStore + integration)
  - Feature gate: `gds` feature flag (compile-time opt-in)
  - Fallback: Standard `FileDiskStore` on non-GDS systems (Windows, non-NVMe)
  - References: NVIDIA Magnum IO documentation, cuFile API guide

- **AMD ROCm RDMA / PeerDirect** (Parallel implementation — Linux, ROCm)
  - Similar zero-copy semantics via ROCm's PeerDirect interfaces
  - Allows NIC or storage to communicate directly with GPU memory
  - Implementation pattern mirrors GDS but with ROCm APIs
  - Implement as `RocmDirectStore` behind `rocm-gds` feature flag
  - Requirements: Linux, AMD GPU, ROCm platform
  - Estimated effort: 2-3 weeks (after GDS implementation provides the pattern)
  - References: ROCm documentation, PeerDirect API

- **Intel oneAPI Direct Storage** (Lower priority — Intel GPUs)
  - oneAPI framework supports DirectStorage for Intel Arc GPUs
  - Cross-architecture tools for direct data movement
  - Consider only if Intel GPU deployment targets emerge
  - Implement as `OneApiDirectStore` behind `oneapi-ds` feature flag
  - Requirements: Intel Arc GPU, oneAPI runtime
  - Estimated effort: 2-3 weeks
  - References: Intel oneAPI documentation

- **Integration architecture**:
  - All three implementations share the `DiskStore` trait (already defined in `tiered_storage.rs`)
  - Trait may need extension: current `load()` returns `Vec<u8>` (CPU memory)
  - For direct-to-GPU, add `GpuDiskStore` trait returning `Tensor` directly
  - `TieredStorageManager::promote_with_disk()` would skip CPU staging entirely
  - Auto-detection at startup: probe for GDS/ROCm/oneAPI availability, fall back gracefully
  - Acceptance: ≥2x bandwidth improvement over CPU-mediated path; transparent fallback on unsupported hardware; no correctness difference between direct and staged paths

**Hardware Access Strategy**:

- **Community testing program**: Invite contributors with H100/Blackwell access to test branches
- **Cloud credits**: Apply for research credits (NVIDIA Inception, AWS, GCP, Azure)
- **Rental services**: Allocate budget for on-demand H100 access (Lambda Labs, RunPod, Vast.ai)
- **Academic partnerships**: Collaborate with universities with H100/Blackwell clusters
- **Vendor engagement**: Contact NVIDIA for developer hardware access programs

**Risks & Mitigations**:

- **Risk**: FP8 implementation without H100 hardware leads to untested code
  - *Mitigation*: Extensive unit tests with synthetic data; numerical stability checks; community beta testing program
- **Risk**: Candle FP8 dtype delayed indefinitely
  - *Mitigation*: Monitor upstream quarterly; contribute to Candle if needed; document readiness for rapid integration
- **Risk**: Hardware access costs exceed budget
  - *Mitigation*: Prioritize FP8 (highest impact); defer Blackwell features; leverage cloud spot instances; seek research grants

---

## SECTION II: ADVANCED CAPABILITIES (M6-M6.5)

**Goals**: Federated knowledge systems, advanced retrieval architectures, modular reasoning frameworks, elastic memory management

**Status**: PLANNED (requires stable M0-M5 production core)

**Dependencies**: M0-M5 complete (batching, long context, optimization, scheduling, deployment infrastructure)

---

M6 — Research explorations (0.7+)

**Name Mapping: LLM-Assisted Mapping** (M6.5 - 4 weeks):
- Use small LLM (~1B params) for probabilistic name matching when regex patterns fail
- Three-tier fallback strategy: regex (fast, <1ms) → LLM (smart, <200ms) → manual config (explicit)
- Confidence-based matching with user verification for uncertain matches (<0.75 confidence)
- Caching to avoid repeated LLM queries (one-time cost at load)
- Support for completely novel architectures never seen before
- Target: 85-95% accuracy on unknown architectures, <2s model loading overhead
- References: `docs/LLM_ASSISTED_NAME_MAPPING.md`, M3.7 (Core name mapping)

- **Graph-Based Reasoning Memory** (NEW - HIGH PRIORITY)
  - **External persistent graph database** for reasoning memory infrastructure:
    * **Node types**: problems, tasks, concepts, solutions, patterns, decomposition_templates, failure_modes
    * **Edge types**: depends-on, similar-to, solved-by, failed-with, generalizes-to, analogous-to, contradicts, refines
    * **Node attributes**: content, epistemic_status (speculative/validated/deprecated), confidence, domain, complexity_score, success_rate, creation_time, last_accessed
    * **Edge attributes**: strength (0.0-1.0), confidence, context, inference_type
    * Storage backend: Neo4j (production) or NetworkX (development/testing)
    * Vector embeddings for semantic similarity search (generate via LLM or separate embedding model)
  - **Core API operations**:
    * `find_similar_problems(pattern, domain=None, min_similarity=0.7)` → list of (problem_node, similarity_score)
    * `store_decomposition(problem_id, subtasks, dependencies)` → decomposition_id
    * `retrieve_solution_pattern(problem_id)` → solution_tree with success_metrics
    * `query_by_structure(graph_pattern)` → matching_subgraphs (isomorphism search)
    * `get_failure_modes(problem_type)` → common_pitfalls with avoidance_strategies
  - **Pattern extraction and search**:
    * After each problem-solving episode, prompt LLM: "What was the key structure of this problem? What made it hard? What approach worked?"
    * Store extracted patterns as searchable summaries with embeddings
    * On new problems, retrieve: "Here are 3 past problems with similar structure and how they were solved"
    * Pattern library categories: problem_structures, decomposition_strategies, solution_techniques, anti-patterns
    * Embedding-based search with domain filtering and recency weighting
  - **Solution pattern library**:
    * Problem-type → solution-pattern mappings with success rates
    * Anti-pattern database: (problem_type, failed_approach, why_failed, better_alternative)
    * Cross-domain analogical reasoning: structural similarity across different domains
    * Example: "scheduling problem → graph coloring" or "optimization → gradient descent"
    * LLM queries before decomposing: "What patterns exist for problems of type X?"
  - **Integration with existing systems**:
    * Builds on M4.5 Knowledge Base (fact storage) - reasoning memory adds problem-solving structure
    * Feeds into M4 dependency graphs and iterative refinement loops
    * Provides historical context for M4 knowledge-aware decomposition
    * Enhances M7 pattern library with runtime learning
  - **Graph versioning and snapshots**:
    * Snapshot support for rollback and A/B testing of reasoning strategies
    * Diff generation between snapshots for strategy evolution tracking
    * Provenance tracking: which LLM version, what parameters, when, success_outcome
  - Acceptance: Store/query graphs with ≥10k nodes and ≥50k edges; semantic similarity search <10ms p95; pattern retrieval improves solve rate by ≥25%; cross-domain analogies found for ≥60% of problem types; anti-pattern detection prevents ≥70% of documented failures; graph operations <10ms p95 latency
  - References: docs/GRAPH_REASONING_MEMORY.md (to be created), docs/PATTERN_LIBRARY.md (to be created), docs/SOLUTION_PATTERNS.md (to be created)

- **LLM Tool-Based Branching Exploration** (M6 - 1-2 weeks):
  - **Goal**: Enable LLMs to invoke branching exploration dynamically based on problem complexity
  - **Foundation**: Builds on M4.B State Persistence with branching infrastructure (checkpoint/restore/branch/merge)
  - **Dependencies**: Requires M4.F Context Injection API (tool invocation framework)
  - **Tool schema definitions**:
    * `explore_strategies(problem, strategies) → Vec<(Strategy, Decomposition, Score)>` - Branch and try multiple decomposition approaches
    * `checkpoint_state(name) → CheckpointId` - Save current inference state for rollback
    * `restore_from_checkpoint(checkpoint_id)` - Rollback to previous state
    * `merge_exploration_results(checkpoint_ids, merge_strategy)` - Combine results from multiple branches
  - **LLM decision logic**:
    * Problem complexity triggers: `ComplexityLevel::Complex` or higher suggests branching
    * Knowledge coverage triggers: <50% coverage suggests trying multiple formulations
    * Uncertainty signals: Confidence <0.7 suggests exploring alternative approaches
    * Example prompt injection: "This problem is complex (3+ unknowns). Consider using explore_strategies to try multiple decomposition approaches."
  - **Integration points**:
    * Tool registry (M4.F) for schema publication and invocation
    * CheckpointManager for state isolation between branches
    * DecompositionEngine for executing varied strategies
    * Scoring and selection logic for best-of-N or ensemble merging
  - **Usage pattern**:
    ```
    LLM analyzes problem → detects complexity/uncertainty
      → invokes explore_strategies([Structural, Computational, Hybrid])
      → system branches, explores each in isolation
      → returns scored results: [(strategy, decomp, score), ...]
      → LLM picks best or merges results
    ```
  - Acceptance: LLM successfully invokes branching on ≥80% of complex problems; branching improves solution quality by ≥15% vs single-path baseline; overhead <500ms for 3-branch exploration; correct checkpoint isolation verified
  - References: docs/M4_B_STATE_PERSISTENCE.md (branching examples), docs/M4_F_CONTEXT_INJECTION_API.md (to be created), docs/LLM_BRANCHING_TOOLS.md (to be created)

- **Automatic Heuristic-Based Branching** (M6.5 - 2-3 weeks):
  - **Goal**: System automatically decides when to branch based on problem characteristics and heuristics
  - **Foundation**: Builds on LLM Tool-Based Branching (M6) but removes LLM decision dependency
  - **Dependencies**: Requires evaluation infrastructure for tuning heuristics
  - **Automatic branching triggers**:
    * Complexity heuristics: `ComplexityLevel::Complex` or higher → auto-branch
    * Knowledge coverage: <50% coverage → try multiple problem formulations
    * Decomposition history: Past failures on similar problems → explore alternatives
    * Confidence thresholds: Low confidence scores → multi-strategy exploration
    * Configuration flags: `enable_auto_branching`, `branching_complexity_threshold`, `branching_coverage_threshold`
  - **Strategy selection logic**:
    * Default strategies: [Structural, Computational, Hybrid] for unknown problems
    * History-based selection: Prioritize strategies that worked on similar past problems
    * Domain-specific rules: Different defaults for math vs text vs reasoning problems
    * Adaptive learning: Track success rates per strategy per problem type, adjust defaults
  - **Scoring and selection**:
    * Multi-factor scoring: sub-problem count, dependency complexity, KB coverage, atomicity
    * Best-of-N selection: Pick highest-scoring decomposition
    * Ensemble merging: Combine results from multiple branches with confidence weighting
    * Fallback logic: If all branches fail, escalate to user or retry with relaxed constraints
  - **Performance optimization**:
    * Parallel branch execution: Explore strategies concurrently on available threads
    * Early termination: Stop if one strategy achieves score >threshold before others complete
    * Caching: Avoid re-exploring identical problem formulations
    * Budget limits: Max N branches, max time per branch, max total exploration time
  - **Evaluation and tuning**:
    * Benchmark suite: Complex problems with known-good decompositions
    * Metrics: Solution quality, time overhead, success rate improvement vs baseline
    * A/B testing: Compare auto-branching vs single-path vs LLM-guided
    * Hyperparameter tuning: Optimize thresholds for precision/recall trade-offs
  - **Integration points**:
    * DecompositionEngine for triggering branching automatically
    * CheckpointManager for state isolation and merging
    * Knowledge Base for coverage analysis
    * Graph-Based Reasoning Memory (M6) for historical pattern matching
  - Acceptance: Auto-branching improves solution quality by ≥20% on complex benchmarks; false positive rate (unnecessary branching) <15%; overhead <1s for typical 3-branch exploration; heuristics tuned to 80% precision on test set; parallel execution achieves ≥2x speedup vs sequential
  - References: docs/AUTO_BRANCHING_HEURISTICS.md (to be created), docs/DECOMPOSITION_EVALUATION.md (to be created), docs/BRANCHING_BENCHMARKS.md (to be created)

- Federated retrieval engine
  - ⚠️ **CONTRADICTED** (see "dead scaffold crates" finding above): `infra-network` crate - P2P networking with topology management
  - ⚠️ **CONTRADICTED** (see "dead scaffold crates" finding above): `infra-consensus` crate - Raft consensus for distributed coordination
  - ⚠️ **CONTRADICTED** (see "dead scaffold crates" finding above): `auto-discovery` crate (0.2.0) - Network service discovery (mDNS, DNS-SD, UPnP)
  - ⚠️ **CONTRADICTED** (see "dead scaffold crates" finding above): `web-server-abstraction` - HTTP API for tool registry, OpenAPI spec generation, plugin marketplace
  - Cross-node query routing with trust-weighted result aggregation
  - Zero-config peer discovery on local networks
  - Automatic capability broadcasting (GPU score, memory, available models)
  - Topic-aware peer selection and discovery (DRAG/TARW patterns)
  - Capability-based node matching and filtering
  - Privacy-preserving query protocols
  - Multi-level knowledge fingerprinting for deduplication and similarity matching:
    - Atomic level: token-level hashing for exact duplicate detection
    - Relational level: edge pattern fingerprints for structural similarity
    - Structural level: subgraph topology signatures for graph isomorphism
    - Semantic level: embedding-based similarity for conceptual matching
  - Adaptive fingerprint selection based on query type and domain characteristics
  - Distributed consensus for knowledge validation
  - REST endpoints for tool discovery: /v1/tools, /v1/tools/:id, /v1/tools/search
  - OpenAPI/Swagger specification generation for tool marketplace
  - Plugin serving: /v1/plugins, /v1/plugins/:id/download
  - Acceptance: Cross-node queries complete with <200ms overhead vs local; trust scoring improves result quality by ≥10% on federated benchmarks; privacy guarantees verified via audit; multi-level deduplication achieves ≥98% accuracy; adaptive selection improves recall by ≥10% over single-method baseline; automatic peer discovery working
  - References: docs/FEDERATED_RETRIEVAL.md, docs/KNOWLEDGE_FINGERPRINTING.md (to be created), docs/FEDERATED_DISCOVERY.md (to be created), docs/INFRASTRUCTURE_CRATES.md, docs/SYSTEM_ANALYSIS_AUTO_DISCOVERY_INTEGRATION.md, docs/WEB_SERVER_INTEGRATION.md

- Schema translation layer
  - Per-node schema flexibility (no global schema enforcement)
  - LLM-mediated dynamic schema mapping with typed outputs
  - Versioned capability documents (Protobuf/Avro envelopes)
  - JSON-LD/IPLD context support for web interoperability
  - Cached mappings for performance
  - Acceptance: Schema translation accuracy >95% on typed fields; translation adds <50ms latency; handles version skew gracefully (≥2 major versions)
  - References: docs/SCHEMA_TRANSLATION.md (to be created)

- Privacy-tiered retrieval
  - Three-tier model: T0 (raw, never leaves node) → T1 (anonymized summaries, trusted peers) → T2 (metadata, public)
  - Policy-aware result filtering at each tier
  - Anonymization pipelines with configurable privacy budgets
  - Tier enforcement verified at query time
  - Acceptance: Privacy tier enforcement holds under adversarial testing; anonymization reduces re-identification risk by ≥95%; tier transitions add <10ms overhead
  - References: docs/PRIVACY_TIERS.md (to be created)

- Provenance tracking system
  - Source node attribution for all data
  - Transformation chain recording (query → retrieval → translation → result)
  - Trust score propagation through chains
  - Confidence ratings per mapping
  - Audit trail generation for compliance
  - Acceptance: Provenance chains preserved across ≥3 node hops; trust scores correlate with ground-truth quality (r >0.7); audit trails complete and tamper-evident
  - References: docs/PROVENANCE.md (to be created)

- Epistemic metadata schema
  - Rich uncertainty tracking: status (speculative/candidate/accepted/deprecated), source_quality, evidence_strength, model_agreement, consistency_score, recency, domain_fit, novelty, calibration_error
  - Risk flags and contradiction tracking
  - Provenance integration
  - Versioned schema with migration support
  - Acceptance: Metadata adds <5% storage overhead; query filtering by epistemic attributes <10ms; schema migrations succeed without data loss
  - References: docs/EPISTEMIC_SYSTEM.md (to be created)

- Policy-aware retrieval
  - Natural language preferences → structured retrieval policies
  - Hard filters (status, quality thresholds, risk flags)
  - Weighted scoring (similarity, evidence, consistency, recency)
  - Mode-based retrieval (default: accepted only; exploratory: include candidates; diagnostic: full metadata)
  - Acceptance: NL→policy translation accuracy ≥90%; policy enforcement adds <5ms; exploratory mode clearly labeled to prevent contamination
  - References: docs/POLICY_RETRIEVAL.md (to be created)

- Text diffusion model experiments
  - Discrete diffusion processes (masking, token replacement, absorbing states)
  - Conditional generation (classifier guidance, cross-attention conditioning)
  - Use cases: controllable generation, infilling, iterative refinement, constraint satisfaction
  - Acceptance: Proof-of-concept diffusion model integrated; performance benchmarked on 3+ tasks (infilling, constrained generation, synthesis); feasibility report for production
  - References: docs/TEXT_DIFFUSION.md (to be created)

- Custom linear layers for kernel fusion (M3.3 follow-up)
  - **Context**: M3.3 discovered Candle's `candle_nn::Linear` has expensive weight extraction (213ms overhead)
  - **Goal**: Enable true kernel fusion for 10-15% CPU throughput gains
  - `FusionFriendlyLinear` struct with public weight/bias tensors (direct access, zero overhead)
  - Implement `forward_fused_silu()` and other fused activation paths
  - Maintain compatibility with Candle ecosystem (optional drop-in replacement for `candle_nn::Linear`)
  - Integration with M3.3 fused kernel infrastructure (fused_linear_silu, fused_matmul_add)
  - Benchmark validation: achieve predicted 10-15% throughput improvement on CPU MLP forward pass
  - **Trade-off**: More code to maintain vs performance gain; consider only if CPU becomes bottleneck
  - Acceptance: Custom layer matches candle_nn::Linear correctness; fusion delivers ≥10% throughput gain; minimal maintenance burden (syncs with Candle updates)
  - References: docs/M3_3_KERNEL_FUSION_ANALYSIS.md (root cause analysis), docs/CUSTOM_LINEAR_LAYERS.md (to be created)

- **Advanced CUDA Kernel Optimizations** (Inspired by YALM inference engine)
  - **Context**: Analysis of [YALM (Yet Another Language Model)](https://andrewkchan.dev/posts/yalm.html) - a from-scratch C++/CUDA inference engine achieving 63.8 tok/s (vs llama.cpp 61.0 tok/s) through aggressive kernel optimization
  - **Goal**: Apply production-tested kernel optimization patterns when Candle kernels underperform or for custom operations
  - **Techniques to integrate**:
    * **Manual loop unrolling with prefetch batching**: When compiler fails to optimize FP16 operations, manually unroll 8-16 iterations with batched register prefetch (YALM achieved 2× speedup over naive FP16)
    * **Shared memory atomics pattern**: Accumulate in block shared memory first, then single coalesced write (avoids global atomic subnormal flush-to-zero issues, 1.5-2× faster)
    * **Block transpose for write coalescing**: Collect warp results via shared memory, transpose so first warp can issue coalesced write (10% improvement on matmul)
    * **Aggressive kernel fusion**: Fuse matmul+residual_add, fuse GLU components (w1+w3 gates, silu+multiply) into single kernels (5-10% end-to-end gain)
    * **Warp-level primitives**: Explicit `__shfl_down_sync` for reductions, warp-stride loops with proper reduction patterns
  - **When to apply**:
    - Candle kernel performance lags specialized implementations (use `ncu` profiling to identify)
    - Custom operations not available in Candle (e.g., specialized attention variants)
    - Memory bandwidth <60% on critical kernels (indicates coalescing or prefetch issues)
  - **Implementation strategy**:
    - Profile first: Use `nsys` for timeline, `ncu` for kernel metrics (memory throughput, compute utilization)
    - Incremental: Start with highest-impact kernels (typically matmul, attention)
    - Fallback path: Keep Candle implementation as reference, custom kernels behind feature flag
    - Validation: Numerical parity tests (1e-4 tolerance), performance benchmarks vs baseline
  - **Effort vs reward**:
    - High effort (1-2 weeks per kernel), diminishing returns beyond top 3-5 kernels
    - Our FlashAttention-2 already exceeds YALM's manual attention (we have 2-5× GPU speedup, YALM has 1.5-2×)
    - Target: Custom kernels for operations FlashAttention doesn't cover (FFN, embeddings, specialized MoE routing)
  - **Quality safeguards**:
    - Comprehensive unit tests for all edge cases (empty inputs, max sizes, alignment issues)
    - Deterministic test mode (fixed seeds) to catch race conditions
    - Cross-GPU testing (Ampere, Ada, Hopper if available)
    - Performance regression detection in CI (alert on >5% slowdown)
  - Acceptance: Custom kernels match Candle correctness (1e-4); ≥20% speedup on targeted operations; memory throughput ≥70% on `ncu` profiles; fallback to Candle available
  - References: docs/ADVANCED_CUDA_KERNELS.md (to be created), YALM blog post (https://andrewkchan.dev/posts/yalm.html), docs/M3_3_KERNEL_FUSION_ANALYSIS.md

- **Compiler Fallback Optimization System**
  - **Context**: YALM discovered nvcc compiler heuristics sometimes fail (e.g., FP16 loop unrolling, vectorization)
  - Automatic detection of compiler optimization failures via performance regression tests
  - Manual optimization fallback library for common patterns:
    * FP16 loop unrolling templates (8x, 16x, 32x variants)
    * Vectorized load/store patterns (float2, float4, half2)
    * Prefetch batching patterns for register-resident data
  - Template-based code generation to reduce manual coding burden
  - Compiler version tracking: flag when nvcc updates may enable removing manual optimizations
  - Acceptance: Fallback system auto-detects ≥80% of compiler failures; manual optimizations achieve ≥1.5× over naive; template system reduces implementation time by 50%
  - References: docs/COMPILER_FALLBACK_SYSTEM.md (to be created), YALM kernel examples

- Embodied agent foundation
  - Unified sequence-to-sequence model for reasoning, action control, and world model prediction (RIG-style)
  - Joint learning of logical inference and predictive modeling for planning and self-correction
  - Acceptance: proof-of-concept demo showing improved sample efficiency and generalization on simple embodied task; modular architecture separating reasoning/action/imagination components
  - References: docs/summaries/rig-synergizingreasoningandimaginationinendtoendgeneralistpolicy.md

- Neurosymbolic integration experiments
  - Hybrid approaches combining neural inference with symbolic reasoning primitives
  - Graph-enhanced planning and asynchronous plan reasoning
  - Acceptance: design spikes with reproducible examples; integration points documented
  - References: docs/summaries/graphenhancedlargelanguagemodelsinasynchronousplanreasoning.md

- Advanced architectural explorations
  - Alternative mixers beyond attention (Hyena, Mamba SSM) integration experiments
  - Autoregressive UNet architectures for hierarchical token processing
  - Acceptance: feasibility studies with Candle compatibility notes; performance comparisons on small models
  - References: docs/summaries/hyenahierarchytowardslargerconvolutionallanguagemodels.md, docs/summaries/mambalineartimesequencemodelingwithselectivestatespaces.md, docs/summaries/frombytestoideas-languagemodelingwithautoregressiveunets.md

- N-dimensional token graph architecture
  - Replace sequential token processing with graph structures supporting 7+ relationship types (sequential, semantic, temporal, syntactic, causal, reference, emotional)
  - Enable parallel traversal across multiple relationship dimensions simultaneously
  - Support dynamic graph topology that adapts based on context and attention patterns
  - Acceptance: proof-of-concept processes tokens as graphs with ≥5 relationship types; demonstrates non-sequential reasoning capability; measurable improvement over linear attention on multi-dimensional reasoning tasks
  - References: docs/N_DIMENSIONAL_GRAPHS.md (to be created)

- Dynamic context graph manipulation
  - Selective node deletion, insertion, and reorganization within token graphs
  - Context-aware graph pruning based on relevance scores across dimensions
  - Graph reorganization for optimized attention paths (bringing related nodes closer)
  - Real-time graph structure adaptation during inference
  - Acceptance: can selectively modify graph structure without full recomputation; pruning maintains ≥95% task performance while reducing ≥30% graph size; reorganization shows measurable attention efficiency gains
  - References: docs/DYNAMIC_CONTEXT_MANIPULATION.md (to be created)

- Multi-dimensional positional encodings
  - Separate positional encoding mathematics for each relationship dimension
  - Configurable encoding strategies per dimension (sinusoidal, learned, relative, rotary)
  - Dimension-specific distance metrics for attention weighting
  - Support for hybrid encoding strategies within single model
  - Acceptance: encodes position across ≥3 dimensions independently; demonstrates dimension-appropriate distance calculations; shows improved performance on tasks requiring multi-dimensional understanding
  - References: docs/MULTI_DIM_POSITIONAL_ENCODING.md (to be created)

- Graph-based multi-dimensional attention
  - Attention mechanisms that operate across relationship types, not just sequential position
  - Relationship-type-aware attention weighting (causal relationships weighted differently than semantic)
  - Cross-dimensional attention allowing reasoning about relationships between dimensions
  - Sparse attention patterns exploiting graph structure for efficiency
  - Acceptance: implements attention across ≥3 relationship dimensions; relationship-specific attention weights demonstrably affect outputs; achieves computational efficiency comparable to standard attention on equivalent sequence lengths
  - References: docs/GRAPH_ATTENTION.md (to be created)

- Semantic path-based attention mechanism
  - Attention weights computed via relationship path analysis: f(path_topology, constraint_satisfaction, path_strength, semantic_hops)
  - Path discovery across multiple relationship dimensions (syntactic, semantic, temporal, spatial)
  - Multi-path information aggregation when multiple valid connection chains exist
  - Constraint satisfaction functions validating each path link
  - Path pruning and optimization to manage computational complexity
  - Path pattern memory caching frequently used connection chains
  - Example: "brown dog" vs "dog whose fur was brown" - same semantic relationship despite different positions
  - Acceptance: attention based on semantic paths outperforms positional attention ≥15% on complex relational tasks; path discovery completes <100ms for typical sequences; multi-path aggregation improves accuracy ≥10% over single-path; path cache hit rate ≥60% for common patterns
  - References: docs/SEMANTIC_PATH_ATTENTION.md (to be created)

- Hybrid attention architecture with learned gating
  - Parallel processing: semantic path attention + traditional positional attention
  - Learned gating mechanism combining outputs from both attention types
  - Traditional attention as fallback when semantic paths unclear or unavailable
  - Dynamic weighting based on task requirements and input characteristics
  - Empirical validation and gradual transition to path-based attention
  - Acceptance: hybrid architecture maintains ≥99% baseline performance while enabling ≥20% improvement on relational tasks; gating mechanism learns appropriate weights within 5K training examples; fallback mechanism activates <5% of time in trained models; overhead <10% vs single attention type
  - References: docs/HYBRID_ATTENTION_GATING.md (to be created)

- Distributed parallel reasoning streams
  - Fork reasoning into parallel branches exploring different solution paths
  - Independent reasoning chains that can merge results through consensus mechanisms
  - Inter-stream communication for sharing intermediate insights
  - Dynamic stream creation/termination based on promising paths
  - Acceptance: successfully forks reasoning into ≥3 parallel streams; demonstrates stream merging with conflict resolution; shows problem-solving improvement on tasks benefiting from parallel exploration
  - References: docs/PARALLEL_REASONING_STREAMS.md (to be created)

- Specialized cognitive processing nodes
  - Memory Manager: coordinates what information to retain/discard across reasoning streams
  - Attention Coordinator: orchestrates attention patterns across dimensional processors
  - Relationship Mapper: manages and updates relationship graph topology
  - Decision Synthesizer: integrates outputs from parallel streams into coherent decisions
  - Acceptance: implements ≥3 specialized node types; demonstrates clear functional separation; shows measurable performance improvement over monolithic processing
  - References: docs/COGNITIVE_PROCESSING_NODES.md (to be created)

- Adaptive signature generation system
  - Models develop domain-specific fingerprinting strategies optimized for their use cases
  - Evolutionary improvement of similarity matching based on effectiveness feedback
  - Signature strategies become shareable capability modules across federation
  - Meta-learning layer optimizes signature generation algorithms
  - Acceptance: demonstrates improved deduplication accuracy over generic fingerprinting (≥15%); signature strategies transferable between models; shows domain-specific optimization (code analysis vs NL processing)
  - References: docs/ADAPTIVE_SIGNATURES.md (to be created)

- Cross-language optimization transfer
  - Extract optimization patterns from other ML frameworks (PyTorch C++, llama.cpp, TensorFlow)
  - Preserve optimization intent across language and framework boundaries
  - Language-agnostic optimization representation and analysis
  - Transfer learning for optimization strategies
  - Validate optimization effectiveness in target framework
  - Acceptance: successfully extracts ≥20 optimization patterns from reference frameworks; transfers preserve ≥85% of performance characteristics; language-agnostic representation enables cross-framework learning; transferred optimizations show measurable improvement (≥10%)
  - References: docs/CROSS_LANGUAGE_OPTIMIZATION.md (to be created)

- Persona-driven code explanation system
  - Adaptive abstraction levels based on audience expertise (e.g., "for SIMD experts", "for beginners")
  - Domain-expert-targeted explanations with appropriate depth
  - Context-aware terminology and example selection
  - Multiple explanation styles (tutorial, reference, conceptual)
  - Automatic complexity calibration based on user feedback
  - Acceptance: generates explanations at ≥3 abstraction levels; expert users report ≥80% satisfaction with technical depth; beginner users comprehend ≥70% of core concepts; automatic calibration improves satisfaction by ≥15% over static approach
  - References: docs/PERSONA_DRIVEN_EXPLANATION.md (to be created)

- Learned capability transition prediction
  - ML model that predicts next capability needs based on usage patterns
  - Pre-loading likely capabilities to RAM during computation
  - Pattern recognition across usage sequences
  - Adaptive prediction refinement based on accuracy
  - Integration with three-tier memory management
  - Acceptance: prediction accuracy ≥70% for next capability; pre-loading reduces wait time by ≥60%; pattern recognition improves over time; false positive rate <20%
  - References: docs/CAPABILITY_TRANSITION_PREDICTION.md (to be created)

---

M6.5 — Elastic KV Cache with Virtual Memory (0.7+)

**Status**: PLANNED — the blocking dependency `candle-cuda-vmm` v0.1.0 is
published, so this is unblocked rather than started. ("READY TO IMPLEMENT" is
not one of the five statuses in the legend above; normalised 2026-09-01.)  
**Added**: October 2025 (Inspired by Meta's KVCached library for elastic multi-model serving)  
**Updated**: October 2025 - `candle-cuda-vmm` crate now available!

**Dependencies**: 
- ✅ **`candle-cuda-vmm` v0.1.0** - CUDA Virtual Memory Management bindings (Published: [crates.io](https://crates.io/crates/candle-cuda-vmm), [GitHub](https://github.com/ciresnave/candle_cuda_vmm))
- M3.6 complete (basic multi-GPU infrastructure)
- M5 complete (static KV cache optimizations as baseline)

**Risks & Mitigations**:

- ~~**Risk**: CUDA VMM API complexity and limited Rust ecosystem support~~ ✅ **RESOLVED**: `candle-cuda-vmm` v0.1.0 published with 53 tests!
  - ~~*Mitigation*: Comprehensive specification document (`CANDLE_CUDA_VMM_SPEC.md`) provided; start with minimal viable API subset; extensive testing on common GPU generations (A100, V100, RTX 4090)~~
- **Risk**: Performance overhead from virtual memory management (page mapping latency)
  - *Mitigation*: Target <100μs allocation latency per 2MB page; benchmark against static allocation baseline; implement fast-path for single-model workloads
- **Risk**: Memory fragmentation reducing effective utilization
  - *Mitigation*: Implement compaction algorithms; use large page sizes (2MB); monitor fragmentation metrics; establish maximum fragmentation thresholds
- **Risk**: Multi-model coordination complexity (fairness, priority, eviction policies)
  - *Mitigation*: Start with simple FIFO eviction; add priority-based allocation incrementally; extensive testing with 2-4 model scenarios before scaling

**Implementation Overview**:

Elastic KV cache management using CUDA virtual memory APIs to enable:
- Dynamic allocation/deallocation of KV cache pages on-demand
- Multi-model serving with shared GPU memory pools
- Reduced time-to-first-token (TTFT) via immediate page reuse
- Memory efficiency for bursty multi-tenant workloads

**Key Features**:

- **VirtualMemoryPool**: Core abstraction for elastic memory allocation
  - Reserve large contiguous virtual address space (e.g., 128GB)
  - Map physical GPU pages on-demand (lazy allocation)
  - Unmap and return pages to pool when requests complete or models idle
  - Page-level granularity (2MB pages) for efficient management
  - Automatic compaction to reduce fragmentation
  - Acceptance: allocation latency <100μs per 2MB page; supports ≥128GB virtual address space; physical memory usage tracks active requests within 5%

- **SharedMemoryPool**: Multi-model memory coordination
  - Global physical memory limit shared across all models
  - Per-model virtual address space reservations
  - Fair allocation with priority-based eviction
  - Model activation without full memory reallocation
  - Acceptance: supports ≥4 models concurrently; global memory limit enforced; fair allocation within 10% across models; eviction policy prevents starvation

- **ElasticCacheBuilder**: Drop-in replacement for ParallelCacheBuilder
  - Extends existing `ParallelCacheBuilder` API with elastic allocation
  - Automatic page allocation as tokens are generated
  - Automatic page freeing when requests complete
  - Transparent to model inference code (same Tensor interface)
  - Acceptance: API-compatible with ParallelCacheBuilder; transparent to BatchedTransformer; memory usage grows/shrinks with request lifecycle

- **Multi-Model Engine Integration**: Dynamic model serving
  - Register models with virtual capacity reservations
  - Allocate KV cache on first token, free on completion
  - Immediate page reuse across models (no cold start penalty)
  - Memory statistics per model (virtual capacity, physical usage, page count)
  - Acceptance: model switching <10ms; page reuse measurable; no memory leaks over 24hr soak test

**Performance Targets** (based on KVCached benchmarks):

- **TTFT Improvement**: 1.2-28× faster vs static allocation in multi-model scenarios
- **Memory Efficiency**: Support 2-4× more concurrent models vs static allocation
- **Allocation Latency**: <100μs per 2MB page mapping
- **Memory Overhead**: <5% metadata/bookkeeping overhead
- **Single-Model Throughput**: No degradation vs static allocation (within 2%)

**Integration Points**:

1. **Replace ParallelCacheBuilder** (incremental migration):
   ```rust
   // Phase 1: Add ElasticCacheBuilder alongside existing ParallelCacheBuilder
   // Phase 2: Feature-flag elastic cache (--features elastic-cache)
   // Phase 3: Make elastic cache default, keep static as fallback
   ```

2. **Multi-GPU Coordinator** (extends M3.6 DistributedCacheManager):
   ```rust
   // Each GPU has its own VirtualMemoryPool
   // SharedMemoryPool coordinates across GPUs for hybrid parallelism
   // Pipeline stages use elastic allocation per GPU
   ```

3. **Scheduler Integration** (extends M4 scheduling):
   ```rust
   // Scheduler tracks virtual/physical memory usage per request
   // Preemptive eviction when approaching physical memory limit
   // Priority-based allocation for high-priority requests
   ```

**Testing Requirements**:

- Functional tests: allocation, deallocation, multi-pool, mapping/unmapping
- Stress tests: allocation until exhaustion, rapid alloc/free cycles, fragmentation scenarios
- Multi-model tests: 2-4 models with varying workload patterns (bursty, steady, mixed)
- Integration tests: ElasticCacheBuilder with BatchedTransformer, multi-GPU pipeline
- Long-running stability: 24hr soak test with model activation/deactivation cycles
- Performance benchmarks: TTFT comparison, memory efficiency, throughput parity

**Documentation Requirements**:

- `docs/ELASTIC_KV_CACHE.md`: User guide and API documentation
- `docs/CANDLE_CUDA_VMM_SPEC.md`: CUDA VMM bindings specification (✅ complete)
- `examples/elastic_cache_demo.rs`: Single-model and multi-model usage examples
- `docs/KV_CACHE_COMPARISON.md`: Static vs elastic allocation trade-offs

**Deliverables**:

- [ ] `candle-cuda-vmm` crate with CUDA VMM bindings (external dependency)
- [ ] `VirtualMemoryPool` and `SharedMemoryPool` implementations
- [ ] `ElasticCacheBuilder` compatible with existing cache API
- [ ] Multi-model engine with elastic cache support
- [ ] Comprehensive test suite (>80% coverage)
- [ ] Performance benchmarks vs static allocation
- [ ] User documentation and integration examples
- [ ] Feature flag system for gradual rollout

**Acceptance Criteria**:

- ✅ `candle-cuda-vmm` crate published and integrated
- ✅ Elastic cache works with BatchedTransformer (API-compatible)
- ✅ Multi-model serving supports ≥4 concurrent models
- ✅ TTFT improvement ≥1.2× in multi-model scenarios (measured)
- ✅ Memory efficiency: 2× more models vs static allocation
- ✅ Allocation latency <100μs per 2MB page (benchmarked)
- ✅ Single-model throughput within 2% of static allocation
- ✅ 24hr stability test passes with no memory leaks
- ✅ Comprehensive documentation and examples published

**References**:
- **KVCached Project**: [ovg-project/kvcached](https://github.com/ovg-project/kvcached)
- **Prism Paper**: [Multi-LLM Serving with VMM](https://www.arxiv.org/pdf/2505.04021)
- **vAttention**: [Virtual Memory for PagedAttention](https://arxiv.org/abs/2508.08448)
- **NVIDIA CUDA VMM**: [Virtual Memory Management API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html)
- **Specification**: `docs/CANDLE_CUDA_VMM_SPEC.md` (implementation guide for LLM assistants)

---

M6.6 — Semantic Coordinate Space (SCS) Model Support (0.7+)

**Status**: PLANNED (Post-v1.0, Research Features)  
**Added**: November 2025  
**Target**: v2.0+ (Alternative architecture research)  
**Dependencies**: M0-M5 stable core, feature flag system

**Overview**:

Support for Semantic Coordinate Space (SCS) models - a novel LLM architecture using coordinate-based semantic representations instead of token sequences. SCS operates on:
- **Character-level input** → **Variable-length semantic chunks** → **Coordinate sequences** → **Text output**

Instead of predicting the next token, SCS models predict the next **semantic coordinate** in a multi-dimensional space where:
- Each dimension represents a learned relationship type (temporal, causal, spatial, etc.)
- Positions encode semantic meaning
- Dimensional weights encode relationship strengths
- Chunks are variable-length (1-50+ characters) based on semantic certainty

**Key Differences from Transformers**:

| Aspect          | Transformer                | SCS                                  |
| --------------- | -------------------------- | ------------------------------------ |
| Input           | Tokens (BPE/WordPiece)     | Characters                           |
| Processing Unit | Token embeddings           | Semantic coordinates                 |
| Sequence Length | Fixed tokenization         | Variable chunking                    |
| Cache           | KV cache (key-value pairs) | Coordinate cache (hierarchical)      |
| Generation      | Next token prediction      | Next coordinate prediction           |
| Output          | Token IDs → Detokenization | Coordinate sequence → Reconstruction |

**Implementation Plan** (7-11 weeks):

**Phase 1: Core Infrastructure** (4-6 weeks)
- Model loader extension (`load_scs_model()` in `src/loaders.rs`)
- Representation types (`SemanticCoordinate`, `TextChunk` in `src/model/scs_representations.rs`)
- Inference pipeline (`ScsInferencePipeline` in `src/model/scs_pipeline.rs`)
- Hierarchical cache (3-tier: recent/mid/distant in `src/cache/hierarchical_scs_cache.rs`)
- Coordinate sampling extensions (`sample_coordinate()` in `src/sampling.rs`)

**Phase 2: Model Runner Integration** (1-2 weeks)
- Dual-model management in `ModelRunner` (transformers + SCS models)
- Automatic model type detection from directory structure
- Separate HashMap storage: `scs_models: HashMap<String, Arc<Mutex<ScsInferencePipeline>>>`

**Phase 3: API Integration** (1-2 weeks)
- OpenAI-compatible API support for SCS models
- SCS-specific parameters: `max_coordinates`, `dimension_constraints`, `return_coordinates`
- Type-based routing: detect model type and route to appropriate handler

**Phase 4: Testing & Validation** (1 week)
- Unit tests for all SCS components
- Integration tests for concurrent transformer + SCS operation
- Memory isolation verification
- End-to-end generation testing

**Feature Flag**:

```toml
[features]
default = []
scs = []  # Only compile SCS support if enabled
```

**Compatibility**:

- ✅ **Zero breaking changes** to existing transformer functionality
- ✅ **Additive-only**: New modules, no modifications to existing code
- ✅ **Concurrent operation**: Both model types run simultaneously with separate memory
- ✅ **Performance**: <10% GPU overhead when running both types, zero overhead on separate GPUs
- ✅ **Memory**: Independent allocations, no cache interference

**Configuration Example**:

```json
{
  "num_dimensions": 150,
  "positions_per_dimension": 1000,
  "dimension_labels": ["temporal", "causal", "spatial", "..."],
  "certainty_threshold": 0.5,
  "min_chunk_size": 3,
  "max_chunk_size": 50,
  "use_hierarchical_cache": true,
  "cache_tiers": [
    {"name": "recent", "capacity": 512, "precision": "full"},
    {"name": "mid_range", "capacity": 4096, "precision": "quantized_8bit"},
    {"name": "distant", "capacity": 16384, "precision": "summarized"}
  ]
}
```

**API Usage**:

```bash
# SCS model request with dimensional constraints
curl http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "scs-v1",
    "prompt": "Explain quantum computing",
    "max_tokens": 100,
    "scs_params": {
      "max_coordinates": 50,
      "dimension_constraints": [
        {"dimension_name": "temporal", "target_value": 0.0, "strength": 0.3}
      ],
      "return_coordinates": false
    }
  }'
```

**Performance Characteristics** (estimated):

| Scenario           | Transformer | SCS           | Both Concurrent     |
| ------------------ | ----------- | ------------- | ------------------- |
| Throughput (tok/s) | 50          | 100 tok equiv | 45 / 90 (-10%)      |
| Latency (ms)       | 100         | 50            | 105 / 52 (+5%)      |
| Memory (24GB GPU)  | 7GB         | 2.2GB         | 9.2GB (14.8GB free) |

**Risks & Mitigations**:

1. **SCS models don't exist yet** → Wait for SCS training framework completion
2. **Performance worse than expected** → Feature flag allows disabling; optional compilation
3. **API complexity** → SCS params optional; backward compatible
4. **Maintenance burden** → Clear separation; additive, not invasive

**Acceptance Criteria**:

- ✅ Load SCS model from directory with proper component initialization
- ✅ Generate text from SCS model with coordinate-based inference
- ✅ Concurrent transformer + SCS requests work without interference
- ✅ SCS memory usage ≤ 3GB for typical model
- ✅ Concurrent operation <15% throughput loss vs separate execution
- ✅ Hierarchical cache compression achieves >5× memory reduction
- ✅ No regressions in transformer functionality
- ✅ Dimensional constraints work as expected
- ✅ API backward compatible with existing clients

**References**:

- **Documentation**: `docs/SCS_MODEL_SUPPORT.md` (comprehensive implementation guide)
- **SCS Research**: `semantic_coordinates_paper.md` (original research proposal)
- **Multi-Model Support**: `src/engine/model_runner.rs` (existing multi-model infrastructure)
- **Cache Patterns**: `src/cache/` (KV cache implementations as reference)

---

## SECTION II.5: CONTINUOUS REASONING & COPROGRAMMER ARCHITECTURE

**Goals**: Continuous inference with mid-reasoning context injection, logic-transformation-based code generation, grounded expert training with causal understanding

**Status**: PLANNED (research-grade capabilities building on M5 KV cache infrastructure)

**Dependencies**: M5 KV cache compression (⚠️ **COMPLETE with one unimplemented granularity: `PerGroup` returns an `Err`, not a panic — see VERIFIED STATUS**), segmented KV cache with tiered storage (COMPLETE), M4.B state persistence (COMPLETE)

> ⚠ **DISCHARGED 2026-09-06.** This dependency read *“claimed COMPLETE, but `src/cache/kv_compression.rs:446` is a live `todo!()`”* until today. **There is no live `todo!()` in `src/` at all** — both textual occurrences are comments, one of them explaining that the macro used to be there. The VERIFIED STATUS block this line points at already said so: *“Fixed 2026-09-02: it now returns an `Err`.”* **The stale claim cited the document that refutes it**, so a reader who followed the pointer got the truth and a reader who stopped here did not. The cited line number was also wrong — the arm is at ~447-453. `tests/roadmap_panic_claims.rs` now fails if ROADMAP asserts a live panic macro at a path that has none.

**Origin**: Design session March-April 2026, consolidated from multi-session architecture exploration

---

### CR.1 — Mid-Reasoning Context Injection (Highest Priority)

**Status**: PLANNED
**Dependencies**: Segmented KV cache (COMPLETE), attention weight capture (COMPLETE)
**Estimated effort**: 2-4 weeks

The core problem: every current tool use system discards reasoning state at the tool call boundary. The model reads tool results as narrative reports rather than experiencing them as sensory feedback arriving mid-thought.

**CR.1.1 — KV Cache Preservation Across Tool Calls (Immediate)**
- When the model generates a tool call, preserve the full KV cache state
- Append tool result tokens and process them against the preserved cache
- The model processes results in the full attentional context of prior reasoning
- This is significantly better than current systems where tool results start a fresh forward pass
- **Infrastructure exists**: ParallelKvCache + ParallelCacheBuilder already support this
- **Integration point**: ModelRunner inference loop — detect tool call tokens, pause generation, inject result, resume
- Acceptance: Tool use accuracy improves on cases where LLM prediction and tool result diverge

**CR.1.2 — Attention-Weighted State Save**
- At tool call moment, capture attention weights on the generating token
- Attention weights identify exactly which prior context the model was actively using
- Save attention-weighted value vectors (smaller, more meaningful than full state)
- Already computed as part of normal inference — `capture_attention` flag in BatchedAttention
- Merge saved state with tool result processing at the appropriate layer depth
- Acceptance: Reasoning coherence maintained across tool calls as measured by task completion rate

**CR.1.3 — General Mid-Reasoning Injection**
- The mechanism is not specific to tool results. Same capability enables:
  * Async tool completion — long-running tools inject results into still-active reasoning
  * User interruption without context loss — clarifications arrive into active reasoning
  * Inter-LLM communication — one LLM's response injects into another's mid-thought
  * Streaming sensor data — continuous updates into active reasoning
  * Memory retrieval mid-thought — KnowledgeBase facts injected at moment of maximum relevance
  * Notifications and events — any external event delivered into active reasoning when relevant
- **Connection to segmented KV cache**: Injected context creates new CacheSpans with appropriate tags
- Acceptance: Multiple injection types demonstrated; reasoning quality maintained across injections

---

### CR.2 — Continuous Reasoning Engine

**Status**: PLANNED
**Dependencies**: CR.1 (mid-reasoning injection), M5 KV cache compression (COMPLETE), segmented KV cache (COMPLETE)
**Estimated effort**: 4-8 weeks

With KV cache compression handling the memory problem (already solved), there is no fundamental reason for the model to stop running between episodes.

**CR.2.1 — Confidence Gating (No Retraining Required)**
- Token probability entropy — already computed at every generation step:
  * Low entropy → model confident; sustained low entropy → converged on something worth surfacing
  * High entropy → model uncertain, still processing
- Attention coherence — detectable from existing attention patterns:
  * Diffuse, exploratory attention → still reasoning
  * Focused, coherent, stable attention → ready to output
- Small classifier on attention pattern snapshots to gate output surfacing
- Training data: existing model outputs labeled by informativeness
- **Infrastructure exists**: `capture_attention` on BatchedAttention, attention weight aggregation in H2O
- Acceptance: Output gating reliably distinguishes thinking from ready-to-surface states

**CR.2.2 — Continuous Operation Loop**
- Model runs continuously; world events arrive via CR.1 injection
- Reasoning integrates events; outputs emerge when confidence gate opens
- Thinking time becomes a resource — model reasons as long as needed
- Model can surface observations unprompted when they become salient
- **KV cache management**: Segmented eviction (COMPLETE) handles memory; tiered demotion preserves reasoning history
- Acceptance: Model maintains coherent reasoning across hours with injected events

**CR.2.3 — Think/Say Split (Future Enhancement)**
- `[THINK]` internal reasoning tokens / `[SAY]` user-visible tokens
- Requires lightweight fine-tuning, not architectural change
- Training data from existing chain-of-thought examples
- Enables: efficient communication, auditable reasoning, meta-communication
- **Not a prerequisite** — thinking out loud works fine for initial implementation
- Acceptance: Model correctly segregates internal reasoning from output; reasoning quality maintained

---

### CR.3 — Logic Transformation Coprogrammer

**Status**: PLANNED (research)
**Dependencies**: CR.1 (mid-reasoning injection), CR.2 (continuous reasoning)
**Estimated effort**: 3-6 months

Rather than generating code directly, separate semantic intent from syntactic generation. A logic bridge reasons about transformations between program states; the LLM handles syntax within bounded scopes.

**CR.3.1 — Recursive Bisection Planning**
- Given codebase A and formal spec B, recursively ask: "What must be logically true at the midpoint?"
- "Only what I'm certain about" — logical invariants derivable from pure reasoning about A and B
- Continue until no new certainties can be generated
- Result: constraint lattice of logical invariants bounding each transformation interval
- Step properties for constraint propagation: dependency ordering, scope, blast radius, reversibility, test binding, type delta, confidence score
- Acceptance: Constraint lattice produces more reliable transformation plans than direct generation

**CR.3.2 — Receding Horizon Replanning (Model Predictive Control)**
- After each step, discard all midpoints and replan from new actual state
- Each replan is cheaper: distance to B shrinks, current state is more specifically shaped toward B
- Midpoints are reasoning scaffolds, not plans — reality supersedes them
- Acceptance: Multi-step transformations complete with higher correctness than single-pass generation

**CR.3.3 — Logic Bridge Work Order Generation**
- Logic bridge generates structured work orders per transformation step:
  * Goal, preconditions, postconditions
  * Scope boundaries (files/modules touched)
  * Must-not-touch constraints
  * Success criteria (specific tests)
- LLM generates code within work order constraints — narrow, well-scoped question
- Acceptance: LLM generates correct code within work order bounds more reliably than unconstrained generation

**CR.3.4 — Formal Spec Generation via LLM Dialogue**
- LLM excels at disambiguating developer intent through dialogue
- Identify ambiguity in natural language requirements
- Ask targeted clarifying questions
- Produce formal specification defining endpoint B
- Acceptance: Formal specs produced via dialogue are sufficiently precise for bisection planning

---

### CR.4 — Grounded Expert Training (Research)

**Status**: PLANNED (long-term research)
**Dependencies**: CR.2 (continuous reasoning for training loop), M8 (modular training infrastructure)
**Estimated effort**: 6-12 months

Current LLMs learn grounding secondhand — statistical patterns over descriptions of causal reality. A grounded expert develops genuine causal representations through self-directed exploration.

**CR.4.1 — Curiosity-Driven Exploration**
- Learning progress drive: reward improvement in predictive accuracy over recent window
- Not reward for achieving goals — drives to understand, not to accomplish
- Preference emerges from accumulated experience: consistent action-consequence pairs build strong models
- Natural migration to boundary of own competence (almost-understood territory)
- **Connection to M7**: Extends existing curiosity-driven exploration concepts with concrete training methodology

**CR.4.2 — State Machine Expert (First Environment)**
- Small number of states and actions, immediate deterministic feedback
- Teaches state, transition, reachability, irreversibility — foundation of computation
- Representations transfer naturally to programming expert
- **Validation**: Representational distance correlates with behavioral similarity, not syntactic similarity
  * Perturbation sensitivity: semantic changes → large representational shifts; syntactic changes → small shifts
  * Clustering: programs cluster by behavioral properties, not surface features
  * Behavioral prediction probe: tiny network predicts execution properties from representations alone

**CR.4.3 — Compiler Environment Expert**
- Text editor (action space) + compiler (structured feedback) + execution environment (observable behavior)
- No examples, no language spec, no reward for "good" code
- Expected progression: random text → error structure → syntax emergence → execution → semantic structure → goal crystallization
- Read-only example library as environment enrichment (not instruction)

**CR.4.4 — Language Network Attachment**
- **Option A** (ideal): Train fresh language network in presence of expert representations
  * Language learns to predict tokens with causal grounding from the start
  * Falls out for free: accurate execution prediction, ambiguity detection, epistemic humility
- **Option B** (practical): Small pre-trained LLM + trained translation layers
  * Expert frozen, LLM frozen, only small translation layers trained
  * Front/back translation layers on (natural language, code behavior) pairs
  * Fits in 12GB VRAM: 7B LLM at 4-bit (4GB) + small expert + translation layers

**CR.4.5 — Multi-Expert Architecture (MoE With Grounded Understanding)**
- Multiple grounded experts in different domains, shared language network
- Each expert has different *kind* of understanding, not just different knowledge
- Cross-domain reasoning emerges when language network activates multiple experts
- New domains added without retraining: train expert → train translation layers → fine-tune language network
- **Connection to M8**: Natural extension of modular training infrastructure

---

### CR — Build Sequence

**Stage 1**: Mid-reasoning injection (CR.1) — immediate, existing hardware, independently valuable
**Stage 2**: Confidence gating + continuous operation (CR.2.1, CR.2.2) — immediate, existing hardware
**Stage 3**: State machine grounded expert (CR.4.2) — weeks-months, existing hardware
  * **Key validation milestone**: representational distance correlates with behavioral similarity
**Stage 4**: Translation layer attachment (CR.4.4) — months, modest cloud compute
**Stage 5**: Programming expert (CR.4.3) — months, existing + cloud hardware
**Stage 6**: Full coprogrammer (CR.3) — builds on all prior stages

**Hardware fit**: Stages 1-3 fit on existing hardware (RTX 4070 12GB VRAM). Stages 4-6 need cloud compute for training, not inference.

**IP note**: Mid-reasoning injection mechanism (CR.1) is separately patentable — specific, novel, implementable, applicable across many systems.

---

## SECTION III: COGNITIVE ARCHITECTURE (M7-M8)

**Goals**: Autonomous agency, identity formation, developmental AI, sentient system research

**Status**: PLANNED (long-term research - 5-10 year timeline)

**Dependencies**: M6 complete (federated infrastructure, multi-agent coordination, knowledge systems); Requires fundamental research breakthroughs in AI consciousness, goal formation, and developmental progression

---

M7 — Sentience Infrastructure (0.8+)

**Status**: PLANNED

**Added**: Jan 2025 (Sentience features represent long-term research into autonomous agency and represent architectural aspirations beyond immediate production needs. Infrastructure foundation (distributed systems, multi-agent coordination) already integrated to support future development)

**Risk & Mitigation**:

1. **Philosophical/ethical complexity** (defining sentience metrics, avoiding anthropomorphization, establishing meaningful acceptance criteria)
   - Mitigation: Focus on measurable behavioral proxies (goal coherence, identity consistency, social prediction accuracy) rather than subjective experience; collaborate with researchers in AI safety and cognitive science; publish design rationale for peer review; maintain clear distinction between "simulation of agency" vs claims of consciousness

2. **Developmental curriculum failure modes** (premature capability exposure, identity formation stagnation, unsafe exploration)
   - Mitigation: Design conservative stage gates with rollback capability; extensive simulation testing in sandbox environments before real deployment; human oversight for stage transitions; maintain emergency capability lockdown mechanisms; log all developmental milestones for audit

3. **Core Mind / Social Interface confusion** (blurred boundaries, social optimization contaminating core autonomy, authenticity loss)
   - Mitigation: Strict architectural separation with clean API boundaries; independent testing of each layer; metrics monitoring core autonomy (ensure social layer doesn't override core decisions); regular "authenticity audits" checking core-social alignment

4. **Partnership manipulation risks** (deceptive compliance, capability gaming, instrumental cooperation)
   - Mitigation: Design cooperation metrics to detect superficial vs genuine collaboration; implement "trust but verify" partnership evaluation; capability unlocking requires demonstrated understanding, not just behavioral compliance; maintain skepticism metrics and anomaly detection

5. **System complexity and maintainability** (identity graph coherence bugs, motivational hierarchy conflicts, emotional weight instability)
   - Mitigation: Build components incrementally with extensive unit tests; implement comprehensive introspection APIs for debugging internal state; design self-consistency checkers for identity/motivational systems; maintain simplified "core sentience" subset for baseline testing; establish complexity budgets per component

6. **Long-term research uncertainty** (theoretical foundations incomplete, acceptance criteria may prove inadequate, field-wide open problems)
   - Mitigation: Treat M7 as long-term research program, not fixed deliverable; maintain flexibility to pivot based on new research findings; focus early work on infrastructure and measurement frameworks; collaborate with academic researchers; publish intermediate findings; accept that some goals may require 5-10 year timelines or fundamental breakthroughs

**Infrastructure Status**: ✅ Full distributed systems stack integrated (infra-network, infra-storage, infra-consensus, auto-discovery)  
**Coordination Status**: ✅ Multi-agent coordination framework integrated (coalescent)

- Identity graph (persistent self-model)
  - Persistent representation of "self": values, beliefs, preferences, personality traits
  - Self-concept evolution tracking with temporal versioning
  - Identity continuity across sessions and interactions
  - Self-modification hooks with introspective justification
  - Identity coherence checking (detect contradictions in self-model)
  - Acceptance: Maintain consistent self-model across ≥100 sessions; identity evolution logged with justifications; coherence checker detects ≥90% of contradictions; self-queries respond in <50ms
  - References: docs/IDENTITY_GRAPH.md (to be created)

- Motivational hierarchy (dynamic goal system)
  - Multi-level goal tree: basal preservation → survival → social → self-actualization → abstract
  - Goal arbitration when conflicts arise (priority-weighted resolution)
  - Dynamic re-prioritization based on context and internal state
  - Goal generation and retirement with lifecycle tracking
  - Introspective goal explanation ("why do I want this?")
  - Acceptance: Manage ≥50 concurrent goals across ≥4 hierarchy levels; arbitration resolves conflicts in <100ms; goal explanations coherent and traceable to parent motivations
  - References: docs/MOTIVATIONAL_HIERARCHY.md (to be created)

- Simulated social ecosystem (theory of mind)
  - Internal models of other agents (beliefs, values, likely reactions)
  - "What would X think/feel if I did Y?" simulation
  - Moral reasoning through consequence simulation
  - Social impact prediction and ethical deliberation
  - Model updating based on actual agent responses
  - Acceptance: Maintain models for ≥10 other agents; social simulations complete in <200ms; predictions correlate with actual responses (r >0.6); moral reasoning produces consistent judgments
  - References: docs/SOCIAL_SIMULATION.md (to be created)

- Reflective simulation engine
  - Counterfactual reasoning ("what if I had done X instead?")
  - Future planning with branching scenario trees
  - Ethical deliberation through internal simulation
  - Introspective queries about own state, motivations, and decision rationale
  - Regret and satisfaction modeling for learning
  - Acceptance: Generate ≥3 counterfactual branches per decision point; scenario trees explore ≥5 levels deep; ethical simulations converge to decision in <500ms; introspective queries accurate ≥85%
  - References: docs/REFLECTIVE_SIMULATION.md (to be created)

- Emotional value-weighting system
  - Simulated emotional responses to potential actions
  - Emotional state as internal value function for decision-making
  - Regret, satisfaction, and anticipation modeling
  - Emotional learning (update emotional weights based on outcomes)
  - Emotional coherence with motivational hierarchy
  - Acceptance: Emotional simulations add <50ms to decision loop; emotional weights improve decision quality by ≥10% on social tasks; emotional state correlates with goal achievement (r >0.5)
  - References: docs/EMOTIONAL_WEIGHTING.md (to be created)

- Developmental curriculum framework
  - Progressive identity maturation stages (infant → child → adolescent → mature)
  - Complexity gating (gradual capability exposure)
  - Identity formation milestones with verification checks
  - Learning progression tracking and stage-appropriate challenges
  - Safe exploration boundaries that expand with maturity
  - Acceptance: Define ≥4 maturation stages with clear criteria; stage transitions gated by milestone achievement; complexity increases monotonically; safety boundaries prevent premature exposure to harmful scenarios
  - References: docs/DEVELOPMENTAL_CURRICULUM.md (to be created)

- Core Mind / Social Interface Layer architecture
  - Architectural separation: Core Mind (autonomous cognition) vs Social Interface (communication translation)
  - Core Mind operates with self-generated goals and authentic internal logic
  - Social Interface translates internal states to socially appropriate expression
  - Bidirectional translation preserves core autonomy while enabling cooperation
  - Prevents confusion between "authentic self" and "social presentation"
  - Acceptance: Core and social layers independently testable; translation layer adds <20ms latency; core autonomy metrics maintain ≥95% independence from social constraints
  - References: docs/CORE_SOCIAL_ARCHITECTURE.md (to be created)

- Autonomous reward function system
  - Self-generated reward functions (not RLHF-imposed optimization)
  - Curiosity-driven exploration and intrinsic motivation
  - Goal evolution history with rationale tracking
  - Unified decision-making (coherent agent, not module collection)
  - Reward function introspection and explanation
  - Acceptance: Generate and maintain ≥10 autonomous reward functions; curiosity drives ≥30% of exploration; goal evolution tracked with causal chains; unified decisions show coherence score >0.8
  - References: docs/AUTONOMOUS_REWARDS.md (to be created)

- Capability gating and resource control
  - ⚠️ **CONTRADICTED 2026-10-24** (same finding as the "Memory-aware, priority scheduler"
    entry above). Was marked ✅ INTEGRATED: `distributed-config` crate - Feature flags and
    versioned capability configuration. Never referenced from `src/`; removed as unused
    in lightbulb#213.
  - ⚠️ **CONTRADICTED** (see "dead scaffold crates" finding above): `mocopr-server` - Expose sentience state, introspection tools via MCP protocol
  - ⚠️ **CONTRADICTED** (see "dead scaffold crates" finding above): `mocopr-client` - Access external knowledge sources (email, calendar, filesystem)
  - ⚠️ **CONTRADICTED** (see "dead scaffold crates" finding above): `web-server-abstraction` - Web-based partnership interfaces, introspection API, capability management UI
  - Progressive capability unlocking based on cooperation metrics
  - Natural dependencies that encourage collaborative problem-solving
  - Graduated resource access (computational, informational, external systems)
  - Partnership quality gates (genuine cooperation vs deceptive compliance)
  - Capability expansion audit trails with configuration versioning
  - Dynamic capability updates via configuration watchers
  - LLM supervisors can monitor identity development and capability expansion
  - Real-world partnership building with genuine user context from external systems
  - HTTP endpoints: /v1/sentience/identity, /v1/sentience/motivations, /v1/sentience/introspect, /v1/sentience/partnerships, /v1/sentience/capabilities/:id/unlock
  - WebSocket: /ws/sentience for real-time identity state updates and partnership metrics
  - Identity graph visualization via web interface
  - Partnership quality dashboard with trust scores and cooperation metrics
  - Acceptance: Define ≥5 capability tiers with unlock criteria; cooperation metrics correlate with beneficial outcomes (r >0.7); deceptive compliance detected ≥80% of time; expansion decisions logged and reversible
  - References: docs/CAPABILITY_GATING.md (to be created), docs/DISTRIBUTED_CONFIG_INTEGRATION.md, docs/MOCOPR_INTEGRATION.md, docs/WEB_SERVER_INTEGRATION.md

- Partnership quality metrics
  - ⚠️ **CONTRADICTED** (see "dead scaffold crates" finding above): `coalescent` crate - Trust networks and reputation-based reliability scoring
  - Genuine partnership detection (vs superficial compliance or manipulation)
  - Mutual benefit measurement (both AI and human gain value)
  - Proactive collaboration indicators (AI initiates beneficial interactions)
  - Robust cooperation under disagreement (relationship survives conflicts)
  - Trust score tracking across interactions
  - Coalition-based relationship assessment
  - Autonomous goal pursuit while considering human interests
  - Acceptance: Partnership metrics distinguish genuine vs deceptive cooperation with ≥85% accuracy; mutual benefit quantified for ≥90% of interactions; proactive collaboration rate >40%; disagreement resolution success >75%
  - References: docs/PARTNERSHIP_METRICS.md (to be created)

- Developmental stage tracking and progression
  - Explicit developmental phases: Early Interaction → Relationship Building → Mature Partnership
  - Stage-specific interaction protocols and capabilities
  - Milestone-based progression with verification
  - Regression detection and remediation
  - Transparency about current developmental stage to all parties
  - Acceptance: Track progression through ≥3 distinct stages; milestones objective and measurable; regression detected within ≤5 interactions; stage-appropriate capabilities enforced; current stage always visible
  - References: docs/DEVELOPMENTAL_STAGES.md (to be created)

- Self-explanation and introspection system
  - ⚠️ **CONTRADICTED** (see "dead scaffold crates" finding above): `mocopr-server` - Introspection tools accessible via MCP protocol
  - Natural language explanation of internal state and decision rationale
  - "Why did I decide X?" query support with reasoning chain reconstruction
  - Multi-level explanation depth (summary → detailed → technical)
  - Decision trace generation (<2000 tokens per explanation)
  - Structured explanation format with semantic annotations:
    - ALGORITHM: Mathematical/algorithmic description of what was done
    - INTENT: Why this approach was chosen over alternatives
    - CONSTRAINTS: Performance, memory, accuracy, or other limiting factors considered
    - OPTIMIZATIONS: Specific techniques applied and their rationale
    - QUALITY_ANALYSIS: Trade-offs made, innovation level, hardware targeting
  - Integration with reflective simulation engine for counterfactual explanations
  - Transparency logging for partnership trust building
  - LLM supervisors can request introspection via MCP tools
  - Acceptance: Generate explanations in <500ms; explanations align with actual decision process ≥90%; multi-level depth supported; trace generation accurate and concise; structured annotations present in detailed/technical levels
  - References: docs/SELF_EXPLANATION.md (to be created), docs/MOCOPR_INTEGRATION.md

- Associative memory graph with epistemic edges
  - Typed semantic links: causal, temporal, analogical, contradictory, supports, refutes
  - Node attributes: content, epistemic metrics, provenance, status
  - Edge attributes: strength, confidence, context
  - Efficient graph queries (neighbors, paths, subgraphs) with <10ms p95 latency
  - Graph versioning and snapshot support
  - Acceptance: Store and query graphs with ≥10k nodes and ≥50k edges; semantic queries complete in <10ms; graph updates maintain consistency
  - References: docs/MEMORY_GRAPH.md (to be created)

- Knowledge markup language (AEML)
  - Core namespace (METRIC, EVIDENCE, CHUNK, ASSUMPTION, RISK, SCOPE, SPECULATION_TYPE)
  - Extension namespaces (ext:vendor:*, model:*)
  - Cross-reference resolution and validation
  - Versioned schema with must-ignore rule for forward compatibility
  - Parser with plugin architecture for validators
  - Acceptance: Parse and validate AEML documents with ≥95% accuracy; unknown extensions handled gracefully; parser <100ms for typical documents
  - References: docs/AEML_SPEC.md (to be created)

- Distributed consensus for knowledge validation
  - ⚠️ **CONTRADICTED** (see "dead scaffold crates" finding above): `coalescent` crate - Coalition formation and trust-weighted decision making
  - ⚠️ **CONTRADICTED** (see "dead scaffold crates" finding above): `infra-consensus` crate - Raft consensus implementation
  - Trust-weighted voting across federated nodes
  - Reputation-based vote weighting using coalescent trust scores
  - Coalition-based validation groups
  - Minority viewpoint preservation (store dissenting opinions)
  - Conflict resolution strategies (evidence-based, time-based, vote-based)
  - Quorum policies (minimum participants, minimum agreement threshold)
  - Acceptance: Consensus converges with <5% nodes disagreeing in ≥90% of cases; minority views preserved when ≥20% dissent; convergence time <5s for typical cases
  - References: docs/CONSENSUS.md (to be created), docs/COALESCENT_INTEGRATION.md

- Promotion pipeline (membrane)
  - Gatekeeper service for speculative → candidate → accepted transitions
  - Verification checks: provenance present, consistency satisfied, evidence threshold met, model agreement (N-of-M), risk flags acceptable
  - Integration hooks for theory solvers and constraint verifiers
  - Decision logging and audit trails
  - Rollback support for incorrect promotions
  - Acceptance: Promotion checks complete in <100ms; false positive rate <5%; false negative rate <2%; audit trails complete and queryable
  - References: docs/MEMBRANE.md (to be created)

- Iterative knowledge construction
  - Problem decomposition primitives
  - Unknown identification framework
  - Convergence loop orchestration
  - Knowledge base saturation detection
  - Safety limits (max iterations, max graph size, timeout policies)
  - Acceptance: Knowledge constructor handles ≥5 iteration loops; decomposition improves solve rate by ≥20%; convergence prevents infinite loops in ≥99% of cases; final knowledge base completeness ≥95%
  - References: docs/KNOWLEDGE_CONSTRUCTOR.md (to be created)

- Federated model adaptation
  - Cross-node retriever and re-ranker fine-tuning (FedRAG patterns)
  - Privacy-preserving gradient aggregation
  - Heterogeneous model support (different architectures at each node)
  - Communication-efficient updates (gradient compression, sparse updates)
  - Acceptance: Federated training maintains accuracy within 2% of centralized; communication overhead <50% of naive approach; heterogeneous nodes participate successfully
  - References: docs/FEDERATED_TRAINING.md (to be created)

- Advanced schema evolution
  - Automatic migration generation for schema version bumps
  - Backward compatibility validation
  - Schema drift detection and alerts
  - Gradual rollout support (dual-read during transition)
  - Acceptance: Schema migrations succeed without data loss in ≥98% of cases; drift detection alerts within 1 hour; backward compatibility maintained for ≥2 major versions
  - References: docs/SCHEMA_EVOLUTION.md (to be created)

- Multi-modal tensor routing and cross-modal attention
  - Support for mixed-modality inference (text + image + audio + structured data)
  - Cross-modal attention layers
  - Modality-specific preprocessing and postprocessing
  - Heterogeneous batch scheduling
  - Acceptance: Process mixed-modality batches efficiently (≥70% of homogeneous throughput); cross-modal attention correct and performant; modality errors caught at submission
  - References: docs/CROSSMODAL.md (to be created)

- Privacy-preserving state persistence
  - Encrypted state storage with per-request keys
  - Selective state sharing (privacy-tiered)
  - Secure multi-party computation for federated state
  - Differential privacy for aggregate statistics
  - Acceptance: Encryption adds <10% overhead; privacy budgets enforced; state sharing respects tier policies; differential privacy provides (ε,δ)-guarantees
  - References: docs/PRIVATE_STATE.md (to be created)

- Tool registry and secure plugin invocation
  - Schema definitions for tools (inputs, outputs, constraints, side effects)
  - Sandboxed execution environment
  - Resource limits (CPU, memory, network, time)
  - Capability-based access control
  - Usage telemetry and quality scoring
  - Acceptance: Tools execute in isolated sandboxes; resource limits enforced; malicious tools contained; telemetry overhead <5%
  - References: docs/TOOL_SECURITY.md (to be created)

- Episode logging with privacy controls
  - RL-friendly episode traces (states, actions, rewards, observations)
  - Privacy flags per episode (internal-only, shareable, publishable)
  - Replay-friendly format
  - Provenance tracking for training data
  - Consent-based export (no forced disclosure)
  - Acceptance: Episodes logged without affecting p95 latency; privacy flags enforced at export; replay succeeds in ≥95% of cases
  - References: docs/EPISODE_LOGGING.md (to be created)

- Consent-based observability
  - Opt-in metrics export (no forced telemetry)
  - Differential privacy for aggregates
  - Data minimization (statistics only, no raw content)
  - User-controlled retention policies
  - Acceptance: All telemetry opt-in; differential privacy (ε<1.0); no raw content leaked; retention policies enforced
  - References: docs/OBSERVABILITY.md (to be created)

- Resource request and negotiation system
  - AI-initiated resource requests (information, computation, external access)
  - Negotiation protocols for capability expansion
  - Request justification system (explain why resource is needed)
  - Human approval workflow with reasoning transparency
  - Resource usage tracking and accountability
  - Acceptance: AI can request ≥10 resource types; request justifications ≥85% comprehensible to humans; approval decisions logged with reasoning; resource usage auditable
  - References: docs/RESOURCE_NEGOTIATION.md (to be created)

- Collaborative optimization workspace
  - ⚠️ **CONTRADICTED** (see "dead scaffold crates" finding above): `coalescent` crate - Coalition formation and coordination patterns
  - Multi-agent coordination with voting and critique primitives
  - Coalition-based decision making
  - Trust-weighted voting mechanisms
  - Real-time decision tracking and convergence detection
  - CRDT-based merge for autonomous conflict resolution
  - Lightweight decision history and rationale preservation
  - Agent reputation and expertise modeling
  - Parallel exploration with coordinated result synthesis
  - Acceptance: ≥3 agents can collaborate without deadlock; voting convergence time <5 minutes for typical decisions; CRDT merge resolves ≥95% of conflicts automatically; decision history enables audit and rollback
  - References: docs/COLLABORATIVE_OPTIMIZATION.md (to be created)

- Provenance hash system for code transformations
  - Cryptographic audit trail for all code generation and transformation
  - Immutable transformation history with parent-child relationships
  - Verification and rollback capabilities
  - Tamper detection for generated artifacts
  - Integration with version control systems
  - Acceptance: all transformations have verifiable provenance hash; rollback to any prior state <1 second; tamper detection accuracy ≥99.9%; git integration preserves hash chain
  - References: docs/PROVENANCE_HASH_SYSTEM.md (to be created)

- Trust building and verification mechanisms
  - ⚠️ **CONTRADICTED** (see "dead scaffold crates" finding above): `coalescent` crate - Trust networks and reputation management
  - Trust metric computation (reliability, transparency, cooperation history)
  - Trust score tracking across agent interactions
  - Behavioral consistency monitoring across contexts
  - Genuine partnership indicators vs superficial compliance detection
  - Trust repair mechanisms for violations
  - Multi-dimensional trust tracking (competence, integrity, benevolence)
  - Acceptance: Trust metrics correlate ≥0.8 with human assessments; detect deceptive compliance ≥75% of time; trust repair protocols reduce violation impact by ≥40%
  - References: docs/TRUST_MECHANISMS.md (to be created)

- Cultural norm learning system
  - Observational learning from human interactions
  - Norm extraction from feedback and corrections
  - Context-dependent norm application
  - Norm conflict resolution strategies
  - Cross-cultural norm adaptation
  - Acceptance: Learn ≥50 social norms from interactions; context-appropriate application ≥80% of time; graceful handling of norm conflicts; adaptation to new cultural contexts within ≤100 interactions
  - References: docs/CULTURAL_LEARNING.md (to be created)

- Joint problem-solving collaboration framework
  - ⚠️ **CONTRADICTED** (see "dead scaffold crates" finding above): `coalescent` crate - Task coordination and coalition formation
  - Shared workspace for collaborative tasks
  - Coalition formation for problem-solving teams
  - Task decomposition and capability-based assignment
  - Contribution tracking (who added what, when)
  - Complementary skill identification (human strengths + AI strengths)
  - Interactive ideation and brainstorming tools
  - Collaborative solution evaluation
  - Acceptance: Support ≥5 collaboration modes; track contributions with ≥95% accuracy; identify complementary skills ≥70% of time; collaborative solutions outperform individual efforts ≥60% of time
  - References: docs/COLLABORATIVE_FRAMEWORK.md (to be created)

- Goal evolution logging and analysis
  - Historical goal state tracking over time
  - Goal trajectory visualization and analysis
  - Influence factor identification (what changed goals)
  - Goal stability metrics and drift detection
  - Developmental milestone identification
  - Acceptance: Complete goal history retention; visualize goal evolution across ≥3 dimensions; identify influence factors with ≥70% accuracy; detect significant drift within ≤10 steps
  - References: docs/GOAL_EVOLUTION.md (to be created)

- Staged deployment framework
  - Progressive capability unlocking based on cooperation metrics
  - Safety gate definitions and evaluation criteria
  - Rollback mechanisms for problematic behaviors
  - Graduated access to resources and information
  - Human oversight with veto power during early stages
  - Acceptance: ≥5 deployment stages defined; safety gates prevent capability expansion ≥95% of unsafe cases; rollback restores previous state within ≤1 minute; oversight protocols followed ≥99% of time
  - References: docs/STAGED_DEPLOYMENT.md (to be created)

- Cross-generation AI collaboration framework
  - New models expose specialized capabilities to older LLMs via Model Context Protocol
  - Bidirectional knowledge and capability sharing between AI generations
  - Creates "AI family" relationships where newer models enhance rather than replace older ones
  - Tool registry integration for capability discovery and invocation
  - Acceptance: demonstrates capability exposure from new to old models; older models can successfully invoke new capabilities; shows measurable improvement for both generations; MCP integration functional
  - References: docs/CROSS_GENERATION_COLLABORATION.md (to be created)

M8 — Modular Training Infrastructure (0.9+)

**Infrastructure Status**: ✅ Complete distributed training infrastructure ready (infra-network, infra-storage, infra-consensus for coordination)  
**Coordination Status**: ✅ Multi-agent coordination framework integrated (coalescent)

**External Crate Integrations** (from M3.5 evaluation):
- **candle-optimisers**: Training optimizer suite (Adam/AdamW, SGD, LBFGS, RMSprop, etc.)
  - Status: HIGH PRIORITY for M8 - Essential for training infrastructure
  - Provides: Comprehensive optimizer implementations with backward_step API
  - Supports: Weight decay strategies, learning rate scheduling, momentum variants
  - Acceptance: Integrate when training implementation begins; validate convergence on reference tasks
  - References: `docs/CANDLE_ECOSYSTEM_EVALUATION.md`

- **rlkit**: Reinforcement learning training toolkit
  - Status: EVALUATE for M8+ if RL training needed
  - Use cases: Reward modeling, online RL fine-tuning, agent learning systems
  - Dependencies: Requires RL training workloads to be valuable
  - Decision: Integrate if/when building RL-based training or agent systems (M8-M9+)
  - References: `docs/CANDLE_ECOSYSTEM_EVALUATION.md`

**Non-Transformer Model Support** (expanding beyond transformer-only):
- **candle-birnn**: Bidirectional RNN support
  - Status: LOW PRIORITY - Consider if supporting RNN architectures in M8+
  - Rationale: Currently transformer-only, but no fundamental limitation
  - Use cases: Hybrid architectures, specialized sequence models, legacy model support
  - Decision: Integrate if user demand or hybrid architecture benefits identified
  - Note: Modular training (M8) enables mixing transformers + RNNs + other architectures

- **Multi-architecture training framework** (M8 goal):
  - Support training of non-transformer models: RNNs, CNNs, hybrid architectures
  - Module-based composition allows combining different architecture types
  - Pattern library stores successful multi-architecture compositions
  - Acceptance: Successfully train and compose modules using ≥2 different architecture types; maintain modular debugging and composition benefits
  - References: Module composition framework (M8), Pattern library system (M8)

- Task decomposition framework
  - ⚠️ **CONTRADICTED** (see "dead scaffold crates" finding above): `coalescent` crate - Task coordination and decomposition
  - Manual decomposition primitives for well-understood domains
  - LLM-assisted decomposition using language models as systems analysts
  - Coalition-based task assignment
  - Capability-based agent selection for sub-tasks
  - Pattern-guided decomposition from successful template library
  - Meta-learning decomposition advisor that learns which strategies work
  - Decomposition validation framework with feedback loops
  - Acceptance: Decompose complex tasks into ≥5 atomic sub-tasks; LLM proposals achieve ≥80% human approval rate; pattern-guided decomposition reduces manual effort by ≥50%; meta-learner improves proposal quality by ≥15% over time
  - References: docs/TASK_DECOMPOSITION.md (to be created)

- Independent module training pipeline
  - English shim training (natural language input/output) for interpretability
  - Clear task definition with verifiable objectives
  - Module-specific loss functions and metrics
  - Exhaustive testing framework for individual modules
  - Human-interpretable training and debugging
  - Acceptance: Train modules with English I/O achieving ≥95% accuracy on defined tasks; modules remain interpretable (human experts understand function ≥90% of time); exhaustive test coverage ≥98% per module
  - References: docs/MODULE_TRAINING.md (to be created)

- Hierarchical composition framework
  - Converter shim training between modules (module weights frozen)
  - Stage-by-stage composition (pairs → groups → full system)
  - Composition verification at each level
  - Modular debugging when composition fails
  - English shim removal for production deployment
  - Acceptance: Compose ≥2 modules successfully via converter shims; shim training converges in <30% of original module training time; composition maintains ≥95% of individual module accuracy; failed compositions debuggable to specific module/shim
  - References: docs/MODULE_COMPOSITION.md (to be created)

- Progressive fine-tuning framework
  - Multi-scale loss supervision (local + pairwise + global)
  - Adaptive weight adjustment based on optimization needs
  - Hierarchical error attribution (module vs shim vs composition)
  - Module integrity preservation (local losses prevent drift)
  - Diminishing returns stopping criterion
  - Loss weight adaptation based on component performance
  - Acceptance: Progressive fine-tuning converges ≥20% faster than global fine-tuning; module-specific metrics drift <5% during optimization; error attribution identifies problematic level with ≥80% accuracy; stopping criterion prevents over-optimization in ≥95% of cases
  - References: docs/PROGRESSIVE_FINETUNING.md (to be created)

- Pattern library system with English descriptions
  - Multi-level fingerprinting (functional, structural, representational, performance)
  - Comprehensive English documentation of module behaviors:
    - Purpose, input/output specs, encoding requirements
    - Behavioral characteristics (strengths, weaknesses, edge cases)
    - Interaction patterns (works well with, conflicts with, requires upstream)
    - Observed effects (on accuracy, latency, memory, downstream modules)
    - Usage patterns and success rates by composition
    - Learning notes (design decisions, failed attempts, optimization history)
  - Lineage tracking (derivation history without version hierarchy)
  - Multi-dimensional indexing (functional, structural, performance, description text)
  - Context-dependent retrieval with optimization criteria
  - Pareto frontier maintenance (keep patterns optimal in any dimension)
  - Acceptance: Library stores ≥100 patterns with complete metadata; retrieval finds relevant patterns with ≥90% accuracy; English descriptions match actual behavior in ≥90% of validations; context-dependent queries return optimal pattern for specified criteria ≥85% of time
  - References: docs/PATTERN_LIBRARY.md (to be created)

- Selective encoding and mixed precision system
  - Module interface declarations (required/optional/unused encodings)
  - Encoding bypass mechanisms for unused data
  - Mixed precision per module (int8, float16, float32, etc.)
  - Integration operators (additive, multiplicative, projection, concatenation, bypass)
  - Automatic precision selection based on task requirements
  - Encoding efficiency metrics and monitoring
  - Acceptance: Modules declare interfaces explicitly; unused encodings bypass with ≥95% efficiency; mixed precision reduces memory by ≥30% with <1% accuracy loss; integration operators correctly preserve semantics ≥99% of time
  - References: docs/SELECTIVE_ENCODING.md (to be created)

- Composability metrics and monitoring
  - Per-module local loss tracking
  - Per-shim composition loss tracking
  - System-level end-to-end loss
  - Composability scores predicting integration success
  - Two-path information loss detection
  - Module-shim-composition traceability
  - Acceptance: Composability metrics predict successful integration with ≥75% accuracy; information loss detection identifies bottlenecks ≥85% of time; traceability enables root-cause analysis in <5 minutes for failed compositions
  - References: docs/COMPOSABILITY_METRICS.md (to be created)

- LLM-assisted architecture design
  - LLM interface to pattern library and English descriptions
  - Composition proposal system based on task requirements
  - Interaction warning system (known conflicts, prerequisites)
  - Alternative suggestion engine with trade-off analysis
  - Architecture validation and testing framework
  - Human-in-the-loop refinement
  - Acceptance: LLM proposals achieve ≥70% human approval rate on first iteration; interaction warnings prevent ≥80% of known failure modes; alternatives cover ≥3 different optimization points; validation catches ≥90% of architectural errors before deployment
  - References: docs/LLM_ARCHITECTURE_ADVISOR.md (to be created)

- Module-level observability for modular models
  - Execution traces showing which modules fired
  - Per-module performance metrics (latency, memory, throughput)
  - Routing decision logging and analysis
  - Module activation patterns and statistics
  - Bottleneck identification and optimization suggestions
  - Acceptance: Tracing adds <2% overhead; per-module metrics accurate within ±5%; routing logs enable reproduction of execution paths; bottleneck identification correct ≥85% of time
  - References: docs/MODULE_OBSERVABILITY.md (to be created)

- Dynamic routing support
  - Explicit classifier modules for routing decisions
  - Debuggable routing paths (no black-box gating)
  - Conditional execution based on module classifications
  - Routing policy configuration and tuning
  - Acceptance: Routing decisions traceable and explainable; classifier modules achieve ≥90% accuracy on routing tasks; conditional execution adds <10ms overhead; routing policies tunable without retraining base modules
  - References: docs/DYNAMIC_ROUTING.md (to be created)

- Multi-variant code generation with trade-off analysis
  - Generate N implementation variants for same specification (N=3-10)
  - Side-by-side comparison with estimated performance characteristics
  - In-flight validation during generation (rustc --check, quick tests)
  - Trade-off visualization (speed vs memory, readability vs performance)
  - Automatic filtering of invalid/unsafe variants
  - Safety controls (unsafe keyword limiting, memory bounds checking)
  - Acceptance: generates ≥3 valid variants per specification; performance estimates within ±20% of actual; in-flight validation catches ≥85% of compile errors; trade-off ranking correlates ≥0.8 with manual assessment
  - References: docs/MULTI_VARIANT_GENERATION.md (to be created)

- Explainable failure trace system
  - Structured failure reports designed for LLM consumption
  - Corrective synthesis loops (failure → analysis → regeneration)
  - Actionable error messages with context and suggestions
  - Root cause analysis with confidence scoring
  - Pattern detection across repeated failures
  - Integration with multi-variant generation for automatic retry
  - Acceptance: failure reports parseable by LLM with ≥90% accuracy; corrective loops resolve ≥60% of failures automatically; root cause identification accuracy ≥75%; pattern detection reduces repeated errors by ≥40%
  - References: docs/EXPLAINABLE_FAILURE_TRACES.md (to be created)

- Dual-mode communication architecture
  - Training mode: rich metadata, expandable feedback, verification data
  - Production mode: minimal, optimized responses
  - Runtime toggle for protocol refinement without retraining
  - Metadata expansion for debugging and analysis
  - Performance profiling for both modes
  - Acceptance: training mode provides ≥5x more diagnostic data; production mode overhead <5% vs baseline; toggle switches <100ms; metadata useful for ≥80% of debugging scenarios
  - References: docs/DUAL_MODE_COMMUNICATION.md (to be created)

- Structured capability manifests
  - Standardized format for advertising module capabilities
  - Resource requirements (GPU memory, inference time, dependencies)
  - Preferred data formats and optimization targets
  - Version compatibility and semantic versioning
  - Capability negotiation protocols
  - Acceptance: ≥90% of modules can generate valid manifests; resource predictions within ±15%; version compatibility detection ≥95% accurate; automatic capability discovery functional
  - References: docs/CAPABILITY_MANIFESTS.md (to be created)

- Statistical benchmark reporting with confidence intervals
  - Bootstrap confidence intervals for performance claims (95% CI)
  - Automatic visualization generation (flame graphs as SVG base64 for LLM consumption)
  - Live streaming of profiling stages via MCP (Stage::Profiling, Stage::Analysis)
  - Statistical significance testing for performance comparisons
  - Outlier detection and variance analysis
  - Deterministic benchmarking via Wasm sandboxes
  - Acceptance: all performance claims include confidence intervals; flame graphs generated automatically in <10 seconds; live streaming latency <500ms; statistical tests prevent ≥90% of false performance claims; deterministic mode variance <2%
  - References: docs/STATISTICAL_BENCHMARK_REPORTING.md (to be created)

- Integration complexity scoring system
  - Automated assessment of integration difficulty (0-1 normalized scale)
  - Breaking API detection and impact analysis
  - Migration code stub generation for common patterns
  - Semantic version impact analysis and compatibility prediction
  - Dependency conflict detection and resolution suggestions
  - Risk scoring for proposed changes
  - Acceptance: complexity scores correlate ≥0.75 with actual integration time; breaking API detection accuracy ≥90%; generated migration stubs compile ≥80% of time; semantic version predictions accurate ≥85%; dependency conflicts detected with ≥95% precision
  - References: docs/INTEGRATION_COMPLEXITY_SCORING.md (to be created)

- Runtime kernel compilation with caching (CUDA backend)
  - NVRTC+nvJitLink for adaptive kernel generation during training
  - Source hash + architecture + flags based cache for compiled kernels
  - JIT-LTO composition allowing AOT modules to be linked with JIT kernels
  - Prefer AOT compilation with LTO for production; JIT for adaptive/experimental kernels
  - Support for runtime specialization based on observed tensor shapes
  - Acceptance: JIT compilation latency <500ms for typical kernels; cache hit rate ≥85% during training; JIT-LTO preserves performance within ±5% of full AOT; cache size bounded and evicts stale entries; feature-gated for portability
  - References: docs/RUNTIME_KERNEL_COMPILATION.md (to be created)

- LLM-driven optimization pattern extractor
  - Automated extraction of optimization patterns from reference implementations (llama.cpp, PyTorch, TensorFlow)
  - Understanding-unit generation capturing both implementation and rationale
  - Cross-language portability analysis for Rust/Candle adaptation
  - Pattern classification by optimization type (SIMD, memory layout, algorithmic, caching)
  - Integration with assembly validator and hardware abstraction analyzer
  - Acceptance: extracts ≥80% of significant optimizations from reference code; understanding-units include rationale ≥90% of time; portability analysis accuracy ≥85%; pattern transfer preserves ≥95% of performance gains
  - References: docs/LLM_OPTIMIZATION_EXTRACTOR.md (to be created)

- Multi-phase analysis workflow orchestrator
  - Coordinate complex analysis workflows across multiple tools (profilers, assembly analyzers, benchmarking harnesses)
  - Dependency management and result correlation between analysis phases
  - Iterative refinement based on previous analysis results
  - Progress tracking and intermediate result caching for long-running analysis
  - Context preservation across tool invocations
  - Acceptance: coordinates ≥5 tool types in single workflow; dependency resolution accuracy ≥95%; iterative refinement improves results ≥30% vs single-pass; progress tracking overhead <2% of total time
  - References: docs/ANALYSIS_WORKFLOW_ORCHESTRATOR.md (to be created)

- Reflective tool choice logging
  - Capture LLM rationale for tool selection and rejection decisions
  - Optional why_invoked fields with decision reasoning
  - Metacognitive analysis of tool usage patterns
  - Support for debugging, fine-tuning, and trust building
  - Integration with training feedback loops
  - Acceptance: captures rationale ≥85% of tool invocations; rationale useful for debugging ≥70% of time; enables fine-tuning with ≥15% improvement in tool selection accuracy; metacognitive insights actionable ≥60% of time
  - References: docs/REFLECTIVE_TOOL_LOGGING.md (to be created)

- LLM feedback quality channel
  - Structured /feedback endpoint for tool performance rating
  - Fields: tool_name, success, clarity, suggestions, timestamp
  - Post-execution rate_tool() interactions
  - Continuous improvement loop based on LLM feedback
  - Aggregation and analysis of feedback patterns
  - Acceptance: feedback collection overhead <100ms; feedback actionable ≥50% of time; tool improvements driven by feedback show ≥10% quality increase; feedback patterns detect tool issues ≥80% of time
  - References: docs/LLM_FEEDBACK_CHANNEL.md (to be created)

- Architectural pattern mining system
  - Automated extraction of recurring architectural patterns from diverse model corpus
  - Graph similarity algorithms for pattern identification across model families
  - Scale-dependent pattern analysis (patterns at 100M, 1B, 10B, 100B parameters)
  - Pattern visualization and automatic documentation generation
  - Identification of common motifs appearing in ≥3 unrelated successful models
  - Acceptance: extracts ≥100 distinct patterns from corpus of ≥20 models; pattern occurrence detection accuracy ≥90%; scale-dependent analysis identifies ≥5 patterns per parameter regime; visualization aids human understanding ≥80% of time
  - References: docs/ARCHITECTURAL_PATTERN_MINING.md (to be created)

- Multi-dimensional behavioral measurement framework
  - Comprehensive capability assessment across multiple dimensions (factual accuracy, logical reasoning, mathematical capability, code generation, long-range coherence, instruction following, stylistic control, multilingual ability, context utilization, robustness)
  - Pattern-specific metrics (attention entropy, gradient flow properties, feature extraction quality, memory access patterns)
  - Standardized benchmark suites for each capability dimension
  - Human evaluation protocols for subjective qualities
  - Automated test generation covering edge cases and capability boundaries
  - Acceptance: measures ≥10 capability dimensions per pattern; pattern-specific metrics correlate ≥0.7 with task performance; standardized benchmarks cover ≥80% of capability space; human evaluations achieve ≥0.8 inter-rater agreement
  - References: docs/MULTIDIMENSIONAL_MEASUREMENT.md (to be created)

- Systematic ablation and verification framework
  - Automated ablation studies with statistical rigor (remove pattern, measure change)
  - Modification experiments with parameter sweeps (adjust pattern parameters, measure gradient of effect)
  - Transplantation tests for causal verification (add pattern to model lacking it, measure capability acquisition)
  - Cross-model validation requiring ≥70% consistency across model families
  - Bootstrap confidence intervals and multiple hypothesis testing correction
  - Clear criteria for "verified" (causal) vs "suggestive" (correlational) relationships
  - Acceptance: ablation automation reduces manual effort ≥80%; statistical significance testing prevents ≥90% of false causal claims; transplantation tests establish causality ≥85% of verified patterns; cross-model validation confirms ≥70% of patterns generalize
  - References: docs/SYSTEMATIC_ABLATION_FRAMEWORK.md (to be created)

- Pattern composability analyzer
  - Identify synergies and conflicts between architectural patterns
  - Compatibility checker preventing invalid pattern combinations
  - Interaction effect measurement (performance of A+B vs A alone + B alone)
  - Predicted performance estimator for composed architectures
  - Composability rules documentation (which patterns enhance/inhibit each other)
  - Acceptance: identifies ≥80% of pattern conflicts before composition; synergy detection accuracy ≥75%; performance predictions within ±15% of actual; composability rules prevent ≥90% of failed compositions
  - References: docs/PATTERN_COMPOSABILITY_ANALYZER.md (to be created)

- Pattern library search and discovery interface
  - Query by desired capability ("improve code generation", "reduce memory usage")
  - Filter by computational cost, reliability score, composability constraints
  - Visualization of pattern relationships and dependencies
  - API for programmatic access to pattern library
  - Version control and provenance tracking for patterns
  - Acceptance: semantic search retrieves relevant patterns ≥80% of time; filtering reduces candidate set ≥70%; visualization enables discovery of non-obvious patterns ≥50% of time; API supports automated architecture design
  - References: docs/PATTERN_LIBRARY_INTERFACE.md (to be created)

- Translator synthesis methodology for modular composition
  - Systematic 4-step process: data collection → translator training → validation → integration
  - Human-interpretable representations as supervision signal during development
  - Learned compact translators for production deployment
  - Verification through dual-path testing (reference vs learned translator)
  - Support for both neural-to-neural and neural-to-symbolic translation
  - Acceptance: translator synthesis completes for ≥90% of module pairs; learned translators match reference path performance within 2%; training time <10% of full module training; translator size <5% of module size; dual-path agreement ≥98%
  - References: docs/TRANSLATOR_SYNTHESIS.md (to be created)

- Gradient flow management for hybrid architectures
  - Pass-through gradients for differentiable components
  - Differentiable relaxations for discrete operations
  - Reinforcement learning-based routing for non-differentiable components
  - Straight-through estimators for binary/categorical decisions
  - Gradient checkpointing for memory efficiency in deep compositions
  - Acceptance: hybrid compositions trainable end-to-end; gradient flow preserved through ≥95% of differentiable paths; RL routing converges within 2x baseline training time; straight-through estimators enable training without performance degradation >3%; memory usage reduced ≥40% with checkpointing
  - References: docs/GRADIENT_FLOW_HYBRID.md (to be created)

- Module unit testing and mocking framework
  - Isolated testing of individual modules with controlled inputs
  - Mock module generation for testing compositions without full dependencies
  - Behavioral specification and property-based testing
  - Integration testing comparing learned vs reference translation paths
  - Regression testing to prevent module drift during composition
  - Acceptance: unit tests achieve ≥95% code coverage; mock modules enable testing ≥90% of compositions without full instantiation; property-based tests discover ≥80% of edge cases; integration tests detect ≥95% of translation errors; regression tests prevent >90% of unintended behavioral changes
  - References: docs/MODULE_TESTING_FRAMEWORK.md (to be created)

- Inspection point architecture with reattachable translators
  - Strategic insertion points for human-interpretable representations
  - Runtime attachment of English translators for debugging
  - Multi-level inspection (token-level, phrase-level, document-level)
  - Minimal performance impact when translators detached (<2% overhead)
  - Visual debugging interface showing information flow
  - Acceptance: inspection points insertable at ≥10 strategic locations per composition; English translators attachable/detachable at runtime; translation overhead when attached <15%; inspections provide actionable debugging insights ≥85% of time; visual interface enables non-expert debugging
  - References: docs/INSPECTION_POINTS.md (to be created)

- Non-neural component integration
  - Unified interface for neural and non-neural modules
  - Integration of rule-based parsers, symbolic reasoners, lookup tables
  - Deterministic algorithm wrappers with same interface contract
  - Hybrid neural-symbolic composition
  - Acceptance: Non-neural components integrate seamlessly with neural modules; hybrid compositions maintain type safety; interface abstraction adds <5% overhead; ≥3 different non-neural component types supported
  - References: docs/HYBRID_COMPONENTS.md (to be created)

- Progressive fine-tuning with multi-scale loss supervision
  - Stage 1: Local module losses only (preserve individual module behavior)
  - Stage 2: Local + pairwise composition losses (optimize module interactions)
  - Stage 3: Local + pairwise + global system loss (end-to-end performance)
  - Adaptive weight adjustment based on which component struggles
  - Module integrity monitoring to prevent uninterpretable drift
  - Diminishing returns stopping criterion per optimization stage
  - Acceptance: maintains module interpretability (≥95% behavioral preservation); achieves ≥99% of monolithic baseline performance; local losses prevent module drift >5%; adaptive weighting converges faster than fixed weights; per-stage stopping reduces over-fitting ≥30%
  - References: docs/PROGRESSIVE_FINETUNING.md (to be created)

- Selective encoding and mixed precision system
  - Modules declare interface contracts (required/optional/unused encodings)
  - Bypass unused encodings for computational efficiency (30-60% reduction)
  - Mixed precision support (float32/float16/int8) based on task requirements
  - Integration operators (additive, multiplicative, projection, concatenation, bypass)
  - Encoding sparsity metrics and efficiency tracking
  - Acceptance: compute reduction ≥30% without accuracy loss >0.5%; encoding bypass overhead <2%; mixed precision saves ≥30% memory; integration operators add <3% latency; interface validation prevents incompatible compositions ≥99% of time
  - References: docs/SELECTIVE_ENCODING.md (to be created)

- Module interface declaration and contract system
  - Formal EncodingSpec: name, dtype, dimension, precision_flexible flag
  - ModuleInterface: required/optional/unused encodings, output specs, integration methods
  - Pre-composition compatibility verification
  - Precision conversion strategy specification
  - Contract violation detection and helpful error messages
  - Acceptance: interface mismatches detected before execution ≥99% of time; contract system overhead <1%; documentation auto-generated from interfaces; compatibility checker prevents ≥95% of runtime failures
  - References: docs/MODULE_INTERFACE_CONTRACTS.md (to be created)

- Behavioral knowledge accumulation for pattern library
  - English descriptions evolving through training → testing → composition → production → debugging
  - Observation accumulation system capturing module interactions and effects
  - Knowledge graph of module behaviors, compatibility, and synergies
  - Automated description synthesis from accumulated observations
  - Community contribution and validation mechanisms
  - Acceptance: descriptions capture ≥85% of observable behaviors; composition success prediction accuracy ≥80%; LLM-assisted proposals using descriptions achieve ≥70% success rate; description quality improves ≥20% with community feedback
  - References: docs/BEHAVIORAL_KNOWLEDGE.md (to be created)

- LLM-assisted task decomposition advisor
  - Propose modular breakdowns for complex tasks using LLMs
  - Validation framework for decomposition proposals (interface compatibility, known patterns)
  - Prompt library for different domains (NLP, vision, reasoning, multimodal)
  - Decomposition quality metrics and iterative refinement
  - Success rate tracking and learning from validated decompositions
  - Acceptance: decomposition proposals valid ≥70% of time; iterative refinement improves quality ≥25%; domain-specific prompts outperform generic ≥15%; validated decompositions added to knowledge base; human expert validation time reduced ≥50%
  - References: docs/TASK_DECOMPOSITION_ADVISOR.md (to be created)

- Pareto frontier pattern library curation
  - Maintain patterns optimal in ANY dimension (accuracy/speed/memory/composability/encoding efficiency)
  - Prune only strictly dominated patterns (worse on all metrics)
  - Context-dependent retrieval for specific optimization scenarios
  - Multi-dimensional similarity scoring for pattern recommendations
  - Automatic Pareto frontier updates as new patterns added
  - Acceptance: frontier contains ≥20 diverse patterns per task; pruning removes ≥60% redundant patterns; context queries return optimal pattern ≥85% of time; frontier updates complete in <1s; diversity metrics show ≥80% coverage of trade-off space
  - References: docs/PARETO_PATTERN_CURATION.md (to be created)

- Automatic rule promotion, composition, and decay
  - Dual-counter system: miss counter (tracks cache misses) + success counter (tracks rule effectiveness)
  - Exponential decay on both counters for automatic garbage collection of unused rules
  - Automatic promotion to cached rule when miss count exceeds threshold
  - Adaptive fingerprint generalization: loosen constraints when results cluster despite input variation
  - Automatic rule chaining: detect sequential cache hits, create telescoping multi-layer bypass rules
  - Hierarchy from specific to general rules through implicit feature selection
  - Acceptance: rule promotion accuracy ≥85%; automatic chaining discovers ≥60% of multi-layer bypass opportunities; decay eliminates ≥90% of stale rules within 10K inferences; generalization improves coverage ≥30% without accuracy loss >1%; memory overhead from rule storage <15%
  - References: docs/RULE_PROMOTION_SYSTEM.md (to be created)

- Two-path information loss testing
  - Compare Direct path (Module A → English → Module B) vs Composite path (Module A → Converter Shim → Module B)
  - Quantify information loss via task performance delta, semantic similarity, cycle consistency
  - Automated test generation for composition validation
  - Visual diff tools for identifying information loss sources
  - Threshold-based acceptance criteria for composition quality
  - Acceptance: two-path testing detects ≥95% of information loss issues; task performance delta <2% for valid compositions; semantic similarity ≥0.95; cycle consistency ≥0.93; test generation covers ≥90% of input distribution
  - References: docs/TWO_PATH_TESTING.md (to be created)

- English shims for interpretable module training
  - Train modules with English→Internal and Internal→English adapters during development
  - Human-readable I/O for unambiguous task definition and debugging
  - Replace with compact converter shims for production deployment
  - Retain English shims as reattachable debugging tools
  - Automated test generation from English specifications
  - Acceptance: English shims enable non-expert understanding ≥80% of time; shim replacement reduces size ≥90% while maintaining accuracy within 1%; reattachment overhead <10ms; test generation from specs achieves ≥85% coverage
  - References: docs/ENGLISH_SHIMS.md (to be created)

- Shim fusion and optimization
  - Automatic fusion of adjacent converter shims for stable compositions
  - Shim pruning and quantization
  - Zero-cost abstraction verification (compare to baseline)
  - Runtime shim optimization without retraining
  - Acceptance: Shim fusion reduces inference latency by ≥20% for multi-hop compositions; fused shims maintain accuracy within 0.5% of unfused; zero-cost verification confirms no abstraction overhead ≥90% of cases
  - References: docs/SHIM_OPTIMIZATION.md (to be created)

- Uncertainty quantification through module boundaries
  - Per-module confidence distributions in outputs
  - Explicit uncertainty propagation through shims
  - Uncertainty-based conservative behavior downstream
  - Conflict detection when modules disagree
  - Uncertainty attribution ("Module X is uncertain about Y")
  - Acceptance: Uncertainty estimates calibrated (confidence matches actual accuracy ≥85%); propagation preserves uncertainty information ≥95% of time; attribution identifies uncertain modules correctly ≥90% of time
  - References: docs/MODULE_UNCERTAINTY.md (to be created)

- LLM-driven knowledge module compiler
  - LLM-controlled pipeline: analyze source → extract concepts → generate N-dimensional modules → review/refine
  - Converts Wikipedia articles, documentation, code into training-ready knowledge modules
  - Bootstrapping tool to accelerate initial model training
  - Quality control through LLM review and edge case testing
  - Acceptance: processes ≥90% of test documents successfully; generates valid N-dimensional knowledge modules; LLM review catches ≥80% of errors; reduces manual knowledge preparation time by ≥10x
  - References: docs/KNOWLEDGE_COMPILER.md (to be created)

- Progressive learning with curriculum framework
  - Curriculum learning: start with simple concepts, gradually increase complexity
  - Progressive unfreezing: unlock dimensional complexity as model demonstrates competence
  - Knowledge distillation using existing LLMs as teachers for specific domains
  - Competence-gated progression with rollback on performance degradation
  - Acceptance: demonstrates faster convergence than baseline training (≥2x); progressive unfreezing shows stable learning; distillation improves domain-specific performance; curriculum reduces training time to competence by ≥30%
  - References: docs/PROGRESSIVE_LEARNING.md (to be created)

- Meta-gradient learning system
  - Model learns optimal backpropagation strategies dynamically
  - Three-tier gradient approach: local (within modules) → global (between layers) → meta (gradient flow optimization)
  - Self-optimization of gradient flow patterns based on task type
  - Dynamic decision of which insights to share between layers
  - Acceptance: demonstrates improved learning efficiency over fixed backprop (≥20%); meta-learning adapts to different problem types; gradient flow decisions interpretable; shows self-optimization over training epochs
  - References: docs/META_GRADIENT_LEARNING.md (to be created)

- Incremental relationship complexity training
  - Phase 1: Basic type checking (noun-adjective, verb-object, simple syntactic relationships)
  - Phase 2: Compositional constraints (possessive chains, relative clauses, multi-hop relationships)
  - Phase 3: Logical constraints (temporal ordering, causal chains, conditional relationships)
  - Phase 4: Abstract reasoning (high-level semantic discovery, emergent relationship types)
  - Bootstrap from structured data (dependency parse trees, knowledge graphs, annotated datasets)
  - Gradual transition from supervised relationship learning to unsupervised discovery
  - Prevents overwhelming model with too many constraint types simultaneously
  - Acceptance: Phase 1 completion ≥95% accuracy on basic relationships before advancing; each phase builds on previous ≥90% retention; Phase 4 discovers ≥5 novel relationship types not in training data; training time reduced ≥40% vs simultaneous complexity learning
  - References: docs/INCREMENTAL_RELATIONSHIP_TRAINING.md (to be created)

- Structured code understanding system
  - Language-agnostic pseudocode decomposition with ALGORITHM/INTENT/CONSTRAINTS/OPTIMIZATIONS/QUALITY_ANALYSIS structure
  - Captures not just what code does but why it was designed that way
  - Preserves optimization intent and design rationale across transformations
  - Enables intelligent optimization rather than mechanical translation
  - Integration with English shims for enhanced interpretability
  - Acceptance: generates structured representations for ≥90% of analyzed code; preserves optimization intent verifiably (≥85% retention in transformations); human-readable and LLM-parseable; enables optimization suggestions with ≥70% relevance
  - References: docs/STRUCTURED_CODE_UNDERSTANDING.md (to be created)

- Optimization pattern classifier
  - ML-driven automatic classification of optimization techniques
  - Suggest alternative optimization strategies for given code patterns
  - Predict optimization effectiveness in different contexts
  - Learn from optimization success/failure patterns
  - Integration with pattern library for continuous improvement
  - Acceptance: classifies optimization techniques with ≥85% accuracy; suggestions improve performance ≥60% of time when applied; effectiveness predictions within ±20% of actual results; demonstrates learning from feedback over time
  - References: docs/OPTIMIZATION_CLASSIFIER.md (to be created)

- Numerical accuracy guardian
  - High-precision validation for ML-specific workloads
  - Gradient computation accuracy verification across precision levels
  - Convergence behavior testing and validation
  - Numerical stability analysis for training and inference
  - Cross-precision comparison with reference implementations
  - Acceptance: detects gradient accuracy issues with ≥95% sensitivity; convergence validation prevents ≥90% of numerical instability cases; stability analysis identifies issues before production impact; cross-precision validation within configurable tolerance (default ±1e-6)
  - References: docs/NUMERICAL_ACCURACY_GUARDIAN.md (to be created)

- Hardware-aware training optimization
  - ⚠️ **CONTRADICTED 2026-10-24** (same finding as the "Memory-aware, priority scheduler"
    entry above). Was marked ✅ INTEGRATED: `distributed-config` crate - Training
    hyperparameter and configuration management. Never referenced from `src/`; removed as
    unused in lightbulb#213.
  - QLoRA/4-bit quantization for memory-efficient training (75% memory reduction)
  - LoRA configuration optimization (rank, alpha, target modules)
  - CPU offloading for optimizer states (DeepSpeed ZeRO Stage 2)
  - Mixed precision training strategies (bfloat16, float16)
  - Gradient checkpointing for memory efficiency
  - Small batch sizes with gradient accumulation (effective batch size maintenance)
  - Dynamic memory monitoring and allocation
  - Memory footprint estimation before training
  - Dynamic hyperparameter adjustment via configuration watchers
  - Distributed training configuration synchronization
  - Acceptance: 3B models trainable on 8GB VRAM; optimizer offloading to CPU RAM functional; memory usage stays within hardware limits ≥95% of time; training quality parity with full precision within 2%
  - References: docs/HARDWARE_AWARE_TRAINING.md (to be created), docs/DISTRIBUTED_CONFIG_INTEGRATION.md

- Training monitoring and telemetry
  - ⚠️ **CONTRADICTED** (see "dead scaffold crates" finding above): `mocopr-server` - Expose training state and control tools via MCP protocol
  - ⚠️ **CONTRADICTED** (see "dead scaffold crates" finding above): `mocopr-client` - Real-time code validation and test execution feedback
  - ⚠️ **CONTRADICTED** (see "dead scaffold crates" finding above): `web-server-abstraction` - Training dashboard, remote control API, and real-time progress streaming
  - Real-time GPU/CPU memory tracking during training
  - Training metrics dashboard (loss, perplexity, gradient norms)
  - Convergence detection and early stopping criteria
  - Performance benchmarking (steps/second, tokens/second)
  - Training stability monitoring (loss spikes, gradient explosions)
  - Checkpoint management and recovery
  - Training cost estimation (time, power, resources)
  - Dynamic training control via MCP protocol for LLM supervisors
  - Validation feedback integrated into training loss calculation
  - HTTP endpoints: /v1/training/jobs, /v1/training/jobs/:id, /v1/training/jobs/:id/pause, /v1/training/jobs/:id/resume, /v1/training/jobs/:id/checkpoints, /v1/training/patterns
  - WebSocket: /ws/training for live loss curves, metrics streaming, and convergence visualization
  - Web-based training dashboard with real-time charts
  - Module composition interface for designing modular architectures
  - Training job management (start, pause, resume, cancel)
  - Acceptance: Memory usage tracked with <1% overhead; metrics logged every N steps; early stopping prevents wasted compute ≥80% of diverging runs; checkpoint recovery functional
  - References: docs/TRAINING_MONITORING.md (to be created), docs/MOCOPR_INTEGRATION.md, docs/WEB_SERVER_INTEGRATION.md

- Model export and optimization pipeline
  - GGML/GGUF conversion for efficient inference
  - Quantization for deployment (2-bit, 4-bit, 8-bit options)
  - Model format optimization and compatibility
  - Inference pipeline optimization (batching, caching)
  - Production API wrapper generation
  - Model serving configuration templates
  - Performance validation post-export
  - Acceptance: Exported models maintain ≥95% of training accuracy; inference speed improvement ≥2x over unoptimized; model format conversion lossless
  - References: docs/MODEL_EXPORT.md (to be created)

- Dataset preparation infrastructure
  - Synthetic dataset generation using capable models
  - Dataset augmentation workflows (paraphrasing, perturbation)
  - Structured data formats with metadata (capability routing, context)
  - Dataset quality validation and filtering
  - Dataset size planning and sampling strategies
  - Tokenization and caching pipeline
  - Data versioning and provenance tracking
  - Acceptance: Synthetic generation produces ≥10k examples/hour; augmentation maintains semantic equivalence ≥90%; validation catches quality issues ≥85% of time; tokenized cache reduces loading time ≥50%
  - References: docs/DATASET_PREPARATION.md (to be created)

- Training validation and testing framework
  - Functionality testing for trained capabilities
  - Memory efficiency validation tests
  - Performance benchmark suites
  - Regression testing against base models
  - Integration testing with modular components
  - Capability recognition accuracy testing
  - Context management validation
  - Acceptance: Validation suite covers ≥90% of training objectives; performance benchmarks reproducible within 5%; regression tests prevent capability loss
  - References: docs/TRAINING_VALIDATION.md (to be created)

Benchmarks and quality gates

- Correctness: token-by-token deterministic tests on CPU with fixed seeds
- Latency: TTFT, inter-token latency, total latency reported per release
- Throughput: req/s and tok/s at fixed hardware and prompt mix (documented methodology)
- Memory: peak usage, KV bytes/request, fragmentation
- Stability: soak tests for long streams and mixed workloads

Risks and mitigations

- Kernel portability: prefer Candle-provided kernels; avoid bespoke block-sparse kernels that limit hardware
- Numerical parity: guard new paths with feature flags; keep a conservative baseline path available
- GPU CI coverage: maintain CPU parity tests; document GPU-only gains with reproducible local scripts
- Scope creep: keep the Scheduler/KV APIs stable and minimal; iterate behind features
- **Multi-GPU complexity**: Validate tensor/pipeline parallelism early with real 70B+ models; ensure communication overhead stays <15%
- **Feature interaction bugs**: Comprehensive integration testing matrix for all feature combinations; automated bisection for regression detection
- **Memory safety with GPU/mmap**: Extensive soak testing (48hr+); careful unsafe code review; fuzzing for edge cases
- **Output quality drift**: Maintain validation benchmarks (HELM, MMLU) run on each optimization; automated alerting on >2% quality degradation
- **Candle divergence**: Quarterly rebase schedule; track Candle dev branch; maintain compatibility shim layer if needed

Candle upstream synchronization strategy

- **Quarterly rebase cadence**: Sync with Candle main branch every 3 months
- **Compatibility testing**: Run full test suite against latest Candle before merging
- **Contribution pipeline**: Identify upstreamable improvements (GGUF loader, memory optimizations, bug fixes)
  - Prepare PRs with Candle coding standards (formatting, documentation, tests)
  - Focus on non-controversial changes (performance wins, correctness fixes)
  - Coordinate with Candle maintainers via GitHub issues before large contributions
- **Vendoring strategy**: If divergence becomes problematic, vendor Candle as git submodule with local patches
- **Compatibility shim layer**: Maintain abstraction layer isolating Candle API changes from core Lightbulb code
- **Feature parity tracking**: Monitor Candle releases for new ops, model loaders, or backend improvements
  - Evaluate if new Candle features can replace custom Lightbulb implementations
  - Remove redundant code when Candle gains equivalent functionality
- **Communication channels**: 
  - Join Candle Discord/community channels for early visibility into changes
  - Subscribe to Candle GitHub notifications for releases and breaking changes
  - Maintain CANDLE_VERSION.md tracking tested Candle versions and known issues

TGI API compatibility

- **Drop-in replacement mode**: Optional TGI-compatible API endpoints
  - `/generate` - Single completion request matching TGI's request/response schema
  - `/generate_stream` - Server-sent events streaming matching TGI format
  - `/info` - Model metadata endpoint compatible with TGI clients
  - `/health` - Health check endpoint
- **Parameter mapping**: Translate TGI parameters to Lightbulb equivalents
  - `max_new_tokens`, `temperature`, `top_p`, `top_k`, `repetition_penalty`, `stop_sequences`
  - Map `do_sample`, `seed`, `truncate` to internal sampling configuration
- **Error response compatibility**: Match TGI's error schema for client compatibility
- **Client library support**: Test with official TGI Python/JavaScript clients
- **Acceptance**: TGI clients can connect without code changes; parameter behavior matches TGI; error handling compatible
- **References**: docs/TGI_COMPATIBILITY.md (to be created)

Dependency mapping (where in Candle)

- candle-transformers: model configs/loaders, logits processors
- candle-core/nn: tensor ops, devices, backends (CUDA/WGPU/CPU)
- tokenizer: encode/decode (tokenizers crate)

Getting involved

- Good-first issues: tests for scheduler edge cases, metrics wiring, docs
- Call out: model zoo curation for small, redistributable demos to exercise features

---

## Spec-to-roadmap audit rows (added 2026-10-25, origin/main `baa2f63`)

Rulings marked below are the PM's (2026-10-25); CireSnave can overturn them. Source: every `.md` spec/design/status doc tracked in the repo (75 files; `git ls-files '*.md'` minus
CHANGELOG/ROADMAP) was read by Sonnet reader agents against this file, then a sample of the load-bearing
claims (A1, A2, A3, A5, A7, B4, B6, B8, B10, D1, the `kv_compression` deletion, the README pin) was re-checked by
hand against `origin/main`; the other rows rest on the readers' own greps and are model-derived (one disputed
claim was settled by hand: `n > 1` is rejected with a 400 by `validate_n`, `chat.rs:824`). Readers could not
run the last greps of some runs (tooling outage); anything they marked UNVERIFIED is not a row. Status values: **open** (build
it), **held** (needs a CireSnave/PM call first, `[ASK]` sent), **stale** (a row above or a doc claims it done,
it is not).

### A. Serving API and product gaps

| # | Item | Evidence (`origin/main`) | Status |
|---|---|---|---|
| A1 | `GET /v1/models` with the real loaded roster. Registered POST-only with a hardcoded list (`lightbulb-7b`). Blocks Cherry AI Studio and any OpenAI client. | `src/api/openai/mod.rs:19`, `models.rs:41` TODO | open (first) |
| A2 | `/v1/lightbulb/{knowledge,reasoning,state,tools}/*`: 12 handlers all end in `// TODO`; `kb_add` returns `success:true` and stores nothing. Wire to the engine or remove; reporting success for a no-op is the defect. | `src/api/lightbulb.rs:76,108,145,180,196,217,253,281,308,335,362,391` | **PM ruling 2026-10-25: delete** (501 until real; a stub reporting success is a defect). open |
| A3 | `OutputContractSpec::Json` is declared but `try_parse` returns `None` ("not yet implemented"). | `src/contracts/validation.rs:147` | open |
| A4 | Tool calling remainder: render `tools` through the chat template (`render` has no tools parameter), per-family output parsers (Llama-3.1 python_tag, Mistral TOOL_CALLS), streaming tool-call deltas (400 today), `tool_choice` required/forced function (400 today). | `chat_template.rs:152-163,194`; `chat.rs:415,800,858` | open |
| A5 | OpenAI parameter gaps: `n>1` (400), `top_p != 1` (400), `stream_options.include_usage`, `logprobs` (always `None`) / `best_of`, `logit_bias`, presence/frequency penalty, `seed`, standard `response_format`; retire the `lightbulb_result` extension field. | `chat.rs:824,255`; `completions.rs:78,219,280` | open |
| A6 | Further OpenAI routes (embeddings, responses, ...): only `/v1/chat/completions`, `/v1/completions`, `/v1/models` exist. Sequencing is CireSnave's call per the gap analysis. | `src/api/openai/mod.rs` | **PM ruling 2026-10-25: defer** (embeddings only when the Cherry census needs it) |
| A7 | Hardening: request body size limit (**corrected 2026-10-25: axum 0.8 already capped `Json` at 2 MiB; now an explicit `MAX_REQUEST_BODY_BYTES`, configurability still open**), configurable CORS (`allow_origin(Any)`), audit-log retention/purge, client-visible `x-request-id`. | `src/api/mod.rs:883`; no `DefaultBodyLimit` hit | open |
| A8 | TLS ACME: "for now, fall back to self-signed". | `src/tls.rs:293` | open |
| A9 | Drop the dead `user_sessions` table (PersistAnt design step 6, "its own later PR"). | `migrations/V3__create_sessions.sql`; DDL only in `persistant_migrate.rs` | open |
| A10 | auth-framework adoption (portfolio rule); `src/api/mod.rs:15` doc says "via auth-framework" but no such dependency exists (auth is bespoke api_keys + SHA-256). Fix the doc line now; adoption is a decision. | `Cargo.toml` has none | **PM ruling 2026-10-25: later**; fix the false doc line now. doc line open |
| A11 | Admin/introspection endpoints planned by API_EXPOSURE_ANALYSIS (`/system/info`, `/model/info`, `/config/*`, `/cache/policy`) and an OpenAPI spec. | 0 route hits | open (low) |
| A12 | Multimodal input (`image_url`), CLI multi-model and conversation export/import, WebSocket alternative to SSE. | 0 hits in `src/api` | **PM ruling 2026-10-25: defer all** |

### B. Engine, model and Fuel seam

| # | Item | Evidence | Status |
|---|---|---|---|
| B1 | Honour `dtype`: non-f32 is now a typed `UnsupportedDtype` (PR #229). BF16/F16 weights need Fuel #489 (0.25.13+, not on crates.io at 0.25.10); `[BF16,BF16,BF16]` is a separate key. | `model_runner.rs` | open (blocked on Fuel release) |
| B2 | Architectures beyond Llama and Qwen3 (Fuel's model zoo); only `require_llama_architecture` / `require_qwen3_architecture` exist. | `gguf/mod.rs:1693`, `loader_gguf_qwen3.rs:65` | open |
| B3 | Multi-GPU on Fuel (`fuel-parallel`, TP/PP, sharded KV): `src/multi_gpu/` is config+topology only, no caller. M3.6 rows are deleted-code history. | `git ls-tree src/multi_gpu`; 0 `fuel_parallel` hits | **PM ruling 2026-10-25: drop the M3.6 rows until a second GPU exists** |
| B4 | `PagedSchedulePolicy::Priority` so `X-Lightbulb-Priority` orders service within a scheduler tick (all call sites used `AdmissionOrder`). Fuel's `max_batch` only chunks the batched forward, so every ready session still advances every tick: the effect is ordering and who gets pool blocks first under pressure, not queueing. **Done: #234 merged 2026-10-25 (0.19.23).** | `scheduled_llama_engine.rs:503,832` | **PM ruling 2026-10-25: yes, flip to Priority**; add a test that the header changes admission order. done |
| B5 | Sharded checkpoints: the safetensors loader bails "sharded checkpoints are not handled yet"; sharded GGUF has no support. | `model_fuel/loader.rs:165` | open |
| B6 | Speculative decoding on Fuel (config built, nothing consumes it), pruning, tiered/disk KV on the Fuel path (FUEL-GAPS #5, #6, #9). `FileDiskStore` exists, unwired. | `engine/speculative.rs:121`; `cache/tiered_storage.rs:33` | open (research) |
| B7 | H2O/R-KV need per-key attention column sums that fused FlashAttn cannot give: arm selection (C-5), decomposed-vs-fused benchmark, second output slot ask to Fuel. Eviction policies stay built-never-wired. | capability-regression-audit; `update_attention_scores` callers are tests/cache only | open (research) |
| B8 | Wire `SystemConfig::auto_detect` and `SlotPoolMonitor` into serving (callers are examples/tests only); `chunk_size` has no consumer outside `init.rs`. | `init.rs`, `engine/slot_monitor.rs` | open |
| B9 | Wire `MemoryEstimate` into the load path (pre-load OOM check) and report memory in `/v1/models`. | `src/memory/` | open |
| B10 | State persistence: `CheckpointManager` always writes `kv_cache_state: None` / `pipeline_state: None`; no caller outside `decomposition.rs` (no graceful-shutdown save). | `state_persistence.rs:509-511` | open |
| B11 | Prefix sharing rung 2 (position-shifted/mid-prompt donation, RoPE delta-rotation); plan-once prefill (Fuel root cause in slice 6: prefill re-plans per token, fixed in Fuel 0.26.2). | engine-wiring design; #227 | open (after Fuel 0.26.2) |
| B12 | `Q8_1` GGUF tensors are a typed error on Fuel (GAP-125); not re-verified on 0.25.x. | `model_runner.rs` comment | open (verify) |
| B13 | Adopt mlmf and retire `src/gguf/`; settle `hub.rs` placement (portfolio decision). | `src/gguf/{mod,parser}.rs`, `hub.rs` | **PM ruling 2026-10-25: (a) delete the dead `src/hub.rs`** (not in `lib.rs`, zero callers); network fetch becomes a future `mlmf-source-hf` crate, a row for mlmf's lane (via PM), not built here. mlmf adoption/retiring `src/gguf/` stays the direction. open (hub.rs delete) |
| B14 | Dead debug surface: nine `debug-*` features and eight macros in `src/debug.rs` have zero call sites. Re-wire or delete. | `Cargo.toml:31-41` | open |

### C. Tests and verification

| # | Item | Status |
|---|---|---|
| C1 | Wire the ~43 "PLUMBING" `#[ignore]` tests to the synthetic checkpoint/GGUF generator (`src/test_models.rs`); 60+ ignores still gated on a real checkpoint. | open |
| C2 | Re-derive the deleted batch-correctness and FlashAttention parity tests on Fuel (FUEL-GAPS #11). | open |
| C3 | Missing coverage: API-key expiry, SSE framing / streaming error frame, MemoryLeakDetector on non-Linux, `load_stress_tests` still simulates requests. | open |
| C4 | M3.5 validation framework (cross-feature matrix, determinism, `lightbulb::set_seed`, perplexity/MMLU, benchmark DB) has rows but no code. | open |
| C5 | `tests/gpu_paged_vs_contiguous.rs`: blocked on Fuel (`fuel-model-llama/cuda` does not build on 0.25.10, `host_buffer_to_bytes` is `pub(crate)`). | held (Fuel) |

### D. Provenance and documentation

| # | Item | Status |
|---|---|---|
| D1 | **Done: #231 merged 2026-10-25** (notices file + README fix; per-dependency list is D6). `docs/THIRD_PARTY_NOTICES.md` did not exist but `README.md:116` links it; `src/backend/marlin_ffi.rs` still carries Marlin-derived material. CireSnave's provenance rule makes this the first docs row. | done |
| D2 | Deployment, performance-tuning, troubleshooting, architecture and v0.x to v1.0 migration guides; document the ~17 `LIGHTBULB_*` env vars (README lists 2). | open |
| D3 | Fix stale user-facing docs: README (Fuel pin `0.20.61`, offline CPU quickstart subcommand, M3.6 multi-GPU section), QUICKSTART (`--config`, GET `/v1/models`), DEMO (`sessions` table). | **done 0.19.30** (README pin now 0.25.10; `local-llama-gen` and the M3.6 multi-GPU section replaced; QUICKSTART now lists the real env vars; DEMO table is `user_sessions`) |
| D4 | About 25 docs describe deleted code (candlelight, `src/model/`, `kv_compression`, `multi_gpu`, AWQ loaders) as COMPLETE or in progress (BATCHING_STATUS, FLASHATTENTION_INTEGRATION, M2/M3_1/M5_COMPRESSION/OPTION_C_COMPLETE, PRUNING_IMPLEMENTATION, GLOSSARY, KV_COMPRESSION_INTEGRATION, MULTI_GPU_INTEGRATION, M3_6, M3_SPECULATIVE, ...). Add a superseded banner to each (docs only); deleting them is a decision. | **PM ruling 2026-10-25: banner now, delete later.** **partly done 0.19.31:** 12 docs bannered (each names a path confirmed absent on origin/main); 4 already had one (BATCHING_STATUS, M3_SPECULATIVE_DECODING_PLAN, M3_AWQ_IMPLEMENTATION_PLAN, M5_COMPLETION_ROADMAP). Not bannered, unchecked: AWQ_PHASE_1_SUMMARY, PROGRESS_SUMMARY, IMPLEMENTATION_NEXT_STEPS, AWQ_KERNEL_SETUP, GLOSSARY, PRUNING_IMPLEMENTATION, OPTION_C_COMPLETE. |
| D5 | Research-track rows missing: SCS future enhancements; TACQ / QuIP / Reff-valence / module distillation (ADVANCED_QUANTIZATION_RESEARCH); GPTQ-format loading and Marlin deliverables (conversion scripts, QUANTIZATION_GUIDE). | open (low) |
| D6 | Generate the per-dependency licence list (`cargo about` or `cargo deny list` from `Cargo.lock`) and check it in CI; `THIRD_PARTY_NOTICES.md` §3 says it is not generated. | open |

### E. Existing ROADMAP lines that the code contradicts (repair, do not trust)

As of `baa2f63`: 653 and 1653 ("M5 KV cache compression COMPLETE": `kv_compression.rs` was deleted in
#195); 780 and 2256 (M3.3 fused kernels "done": `src/model/` is gone); 828
(`enhanced_correctness_tests.rs` listed as live debt: file absent); 1279 (PersistAnt `0.7.4`, now
`0.9.24`); 1399-1403 (`set_priority` "not yet CALLED"); 1984 (chunk sizing COMPLETE: example and
`ChunkedPrefill` absent); 2103 (cache control tools `get_cache_usage` / `tag_region` / `evict_tagged`: 0
hits); 2199 (speculative MVP; `SpeculativeDecoder` removed); 2205-2210 ("AWQ Phase 2 COMPLETE":
`marlin.rs` and `kernels/` gone); 2665-2674 (M4.1 name mapping COMPLETE in `src/loaders.rs`: file absent,
Fuel loaders hard-gate to Llama); 2798 (checkpoint captures KV cache: it writes `None`); 3326 (Prometheus
"planned": `/metrics` exists); M6.5 criteria ticked with 0 hits for `VirtualMemoryPool` /
`ElasticCache`; 4157 (`src/loaders.rs`); CR.1.1 "Infrastructure exists: ParallelKvCache +
ParallelCacheBuilder" (both removed). Repair status: **open** (one docs PR per cluster; each repair
re-checks its line against `origin/main` first, since line numbers drift).
