<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
# Prefix reuse after the donor finishes (ROADMAP P0-1 open defect (a))

Status: DESIGN, nothing built. Task: pm-kernels `[TASK] lightbulb-go` item 3, 2026-11-27. Refs: lightbulb `origin/main`
at `1f346ca`+ (this branch), Fuel crates 0.59.0 (`fuel-inference-0.59.0/src/multi_session.rs`).

## 1. The defect, as the code has it

With `LIGHTBULB_PREFIX_SHARING` on, a new request whose prompt starts with a block-aligned prefix already prefilled by a
LIVE session splices that session's blocks instead of re-prefilling (measured 2026-10-25, CPU, n=1 per run, 2 runs:
sharer TTFT 60.4 s OFF -> 2.3 s ON). When the donor finishes, `ServingDriver::mirror_departure`
(`src/model_fuel/scheduled_llama_engine.rs:199-223`) does two things that end all reuse of its prompt:

1. `prefix_index.forget_donor(id)` (`policies.rs:637`) drops every index entry that names the donor;
2. it releases the donor's `registered_prefixes` entries (`release_prefix_inner`).

So the second identical request an hour later is cold, which is the common chat case (same system prompt, new user turn).

## 2. What Fuel gives us (quoted, so the design stands on text and not on memory)

`fuel-inference-0.59.0/src/multi_session.rs:1960-1967`:

> Register the first `prefix_blocks` FULLY-FILLED blocks of a live session as a shared [`PrefixId`] whose lifetime the pool's
> registry controls ... The donor may then be reaped/discarded; the owner keeps those blocks resident so a later
> [`add_session_sharing_prefix`] can splice them without racing the donor's teardown.

`:1990-1993`:

> Release a registered prefix: drop the owner handle so each shared block's refcount falls by one, freeing only the blocks no
> live sharer still references. Live sharers keep the prefix alive independently.

`:1973-1983` is the constraint that shapes everything below: `register_prefix` looks the donor up in `self.sessions`
and returns `Err("unknown session")` otherwise. **A prefix can only be registered while its donor is live.** By the time
`mirror_departure` runs, the session is reaped, so "just do not release it there" retains nothing: the registration has
to happen EARLIER. Today `ensure_prefix_registered` (`scheduled_llama_engine.rs:337-361`) registers lazily, on the first
sharer, which is why a donor that nobody shared from has nothing to keep.

So the defect is unblocked on Fuel (the API exists and says it is meant for this), but the cheap fix (skip the release)
does not work, and that is the finding that makes this a design and not a one-liner.

## 3. Design

### 3.1 Index keyed by what survives, not by who computed it

`BlockPrefixIndexFor<D>` is already generic over the donor type (`policies.rs:482`). Instantiate it with a source enum
instead of a bare `SessionId`:

```rust
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum PrefixSource { Live(SessionId), Retained(RetainedId) }   // RetainedId = a lightbulb counter, never reused
```

`forget_donor` then only ever drops `Live(id)` entries, and a `Retained` entry is removed only by eviction (3.3).

### 3.2 Register at the moment the blocks are full, while the donor is live

When a session's PROMPT prefill completes, its `n_full = prompt_len / block_size` leading blocks are fully filled and the
session is still live. At that point, `register_prefix(donor, d)` for the depths `d` we want to be able to hit, store
`RetainedPrefix { prefix: PrefixId, blocks: d, last_used }`, and re-point the chain-hash entries at
`Retained(rid)`. `registered_prefixes` is keyed by `(donor, blocks)` EXACTLY for a reason documented at
`scheduled_llama_engine.rs:157-160` (a `PrefixId` splices every block it covers, so a deeper registration would graft KV
of tokens the new prompt does not contain). That rule stands: retention registers a depth per hit we want, not one deep id.

Registering several depths of one prompt costs refcount bumps and bookkeeping, not block memory (the blocks are shared).
Depth set: every depth up to 64 blocks, then geometric (64, 96, 128, ...) plus the full depth. Cap and stride are
ASSUMPTIONS to be settled by the measurement in section 5, not facts.

UNVERIFIED: where in the serving step loop "prompt prefill finished" is observable. `try_admit` admits; prefill happens in
the scheduler's steps. Slice 2 starts by finding that hook and writing down the line.

### 3.3 Bounded, evictable, never starving a live request

Pinned blocks are memory a live request cannot use. So:

- Budget `LIGHTBULB_PREFIX_RETENTION_BLOCKS=N` (0 = off = today's behavior, the default). A bad value stops startup.
- Accounting is in distinct pool blocks pinned by retained prefixes (read back from the pool, not summed from our own
  tallies: the same discipline as the F2 unload measurement).
- LRU over whole chains: evict the least-recently-hit retained chain, deepest depth first (so the index stays prefix-closed
  and `lookup`'s stop-at-first-miss stays correct), `release_prefix` each `PrefixId`, drop the chain entries.
- **Pressure rule:** if an admission is refused for pool space, evict retained chains LRU-first and retry before refusing.
  A retained prefix must never be the reason a live request fails. This gets its own test (3.5).
- **Measured depth cap (2026-11-28):** the first implementation registered at most 64 blocks per prompt. On a ~1149-token prompt
  (71 blocks) that shared 1024 tokens and left ~125 to prefill: 23-40 s to first token against ~5 s with the whole prefix (CPU,
  loaded machine, n=1). The cap is 256 blocks (4096 tokens); registrations are refcount bumps, not block memory.
- **Implemented (slices 2-3), with two facts the code found:** (1) Fuel's paged admission is optimistic: pool
  exhaustion surfaces later as that session's `Failed`, never as a refusal, so there is nothing to retry on. The
  pressure rule is therefore an up-front check in `try_admit`: if free blocks < `ceil((prompt + max_new) / block)`,
  evict retained entries LRU-first, deepest depth first, until it fits or nothing is retained. It stops as soon as
  the request fits, so a chain can be left shortened (its shallow depths stay). (2) `register_prefix` needs the
  donor's blocks live, and a session that finishes IN its prefill step (`max_new` 1, or EOS on the first token)
  has released them before the post-step probe runs, so such a prompt is never retained. Chat requests with a
  real generation length are unaffected.
- `cache_clear` (the admin endpoint; its implementation follows `release_prefix_inner` in `scheduled_llama_engine.rs`) releases all retained prefixes.

### 3.4 Correctness limits that get worse once entries persist

- Key (c) in the ROADMAP row: the chain key is the token chain alone (seed `lightbulb.block-prefix.v1`). A live-donor hit
  could only cross requests that were concurrent; a retained hit crosses time. With adapters, a different model, or a
  different chat template, the same tokens give different KV. **Retention must not be enabled until the key carries the
  adapter and model identity.** Slice 1 adds the key dimension (as part of the chain seed) before any retention code lands.
- (b) the index records only the prompt, never the generated tokens, so a follow-up turn matches only up to the previous
  PROMPT. Out of scope here; noted so nobody expects multi-turn reuse from this.

### 3.5 Slices, each with the control that fails if the feature is faked

1. `PrefixSource` + a pure `RetainedStore` (insert, lookup-touch, LRU evict, budget) with unit tests. Control: a store that
   never evicts fails the budget test; one that evicts shallow-first fails the prefix-closed test. Also the key dimension.
2. Register at prefill completion; departure keeps `Retained` and drops only `Live`. Control: a driver test where the donor
   finishes and a second identical prompt still takes `prefix_shared_admissions += 1`; with retention 0 the same test must
   show a cold admit (so the test can tell the two arms apart).
3. Eviction under pressure + `cache_clear`. Control: fill the pool with retained chains, then admit a live request that
   needs the space and assert it admits and a retained chain was released; a variant with eviction disabled must refuse.
4. Config, metrics (`cache_stats.entries` / `evictions` already exist), and the `/v1/lightbulb` admin read of pinned blocks.
5. Measurements (section 4).

## 4. Measurement plan (none of this exists yet; the only number is CPU, n=1, 2 runs)

- CPU, SmolLM2-135M Q8_0, the same ~1149-token prefix as 2026-10-25: donor completes FIRST, then the sharer. TTFT with
  retention on vs off, n >= 5 per arm, report the spread, completions bit-identical to the off arm.
- GPU, TinyLlama (the harness `tests/gpu_paged_vs_contiguous.rs` already loads it) under `gpu-run`, compiled outside the lock:
  sharer TTFT on vs off, and `kv_bytes_resident` with 0 / N retained to price the pinned memory.
- Pressure: the slice 3 scenario on the GPU, to show eviction frees blocks the next request then uses.
- Default stays off until the GPU rows are in the ROADMAP row.

## 5. Open questions for pm-kernels

- Is a single global retention budget enough, or per-model (the manager can hold several chat models)?
- Is it acceptable that retained prefixes are lost on model unload (they live in the model's pool)? Recommended: yes, and
  `unload` reports it.
