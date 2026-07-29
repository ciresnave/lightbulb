# Fuel API surface — verified signatures for the port

**Tree**: `C:\Projects\fuel-lightbulb-port` (detached worktree) at **`13279179`**.
**Captured**: 2026-07-29.
**Why this exists**: every prior structural claim about Fuel in this project required
correction on contact. Later plans are written against *this document*, not against
inference. Re-verify against a newer commit before trusting it — `fuel-core` is mid-retirement
into `fuel-ir`/`fuel-hardware`/`fuel-memory` and paths move.

**Evidence key**: `[read]` = read from source at the stated `file:line`. `[ran]` = executed.

---

## 0. The rule that shapes everything: graph affinity

**`[ran]` — cost me a build cycle to discover.** Every `LazyTensor::from_*` constructor
(`from_f32`, `from_bf16`, …) and `zeros`/`full`, which delegate to them, **mints a brand-new
graph**. Two independently-constructed tensors live on different graphs and **cannot be
combined** — every binary op asserts they match (25 such assert sites).

```rust
// WRONG — panics at matmul
let a = LazyTensor::from_f32(a_data, [2, 3], &dev);
let w = LazyTensor::from_f32(w_data, [3, 2], &dev);   // a SECOND graph
let y = a.matmul(&w);

// RIGHT — weights built on the activation tensor's graph
let a = LazyTensor::from_f32(a_data, [2, 3], &dev);   // the root
let w = a.const_f32_like(w_data, [3, 2].into());       // same graph
let y = a.matmul(&w);
```

**In model terms: the activation tensor is the root; the weights are `const_*_like` off it.**

`const_f32_like` (`lazy.rs:189`), `const_bf16_like` (`:213`), `const_like_dtype` (`:234`).
`const_bf16_like` exists specifically for f32 activations with bf16 weight matrices sharing
one graph. `[read]`

---

## 1. Weight loading

- **`Llama2cModel::from_hub(repo_id: &str) -> Result<Self>`** — `lazy.rs:9119` / `:9712`.
  `[ran]` This is what the working `llama-lazy` binary uses. Downloads and loads in one call.
  Highest-level entry; start here.
- **`LazyVarMap`** — `lazy_nn_varmap.rs`. `new()`, `insert(LazyVar)`, `get(&str)`,
  `all_vars()`, **`load<P: AsRef<Path>>(&self, path) -> Result<()>`** (`:110`),
  `save(path)` (`:68`). `[read]`
- **`LazyVarBuilder`** — `lazy_nn_varbuilder.rs:39`. `from_varmap(LazyVarMap, DType, Device)`
  (`:51`), `pp(name)` for prefixing (`:66`), `get(shape, name) -> Result<LazyVar>` (`:84`),
  `get_with(...)` (`:94`), plus `dtype()`/`device()`/`prefix()`. `[read]`

**Gap, stated rather than invented**: `LazyVarBuilder` has **no** `from_mmaped_safetensors`
equivalent — it builds from a `LazyVarMap`, and the map's `load` takes a path. The
Candle-style "VarBuilder straight off mmapped safetensors" shape does not exist here. Resolve
the exact safetensors route before writing loader code.

**CPU constraint**: weights must be F32 on CPU. `[ran]` — a `[F32, BF16, F32]` matmul has no
CPU kernel; as of `13279179` the optimizer's dtype-reconciliation pass inserts a promoting
cast automatically, but that changes accumulation precision and doubles resident bytes, and
the promotion is reported only via `tracing::warn!`.

---

## 2. Model construction and forward

`LlamaModel` — `lazy.rs:6804`. `LlamaConfig` — `:6363`. `[read]`

Forward variants, in ascending order of relevance to this port:

| Method | `file:line` | Notes |
| --- | --- | --- |
| `forward(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor>` | `:6821` | Simplest. **No KV cache.** What `llama-lazy` exercises. |
| `forward_hidden(...)` | `:6957` | Hidden states rather than logits |
| `forward_embeds(...)` / `forward_hidden_embeds(...)` | `:6851` / `:6869` | Pre-embedded input |
| `forward_hidden_embeds_with_mask(...)` | `:6915` | Explicit mask |
| `forward_with_kv_context(...)` | `:7355` | KV-cached |
| `forward_with_kv_context_all_positions(...)` | `:7375` | |
| **`forward_with_kv_context_persistent(...)`** | **`:7631`** | The scheduler's serial arm + prefill |
| **`forward_with_kv_context_captured(...)`** | **`:8458`** | **The `CapturedRun` path — the port's target** |

`Llama3Model` (`lazy_llama_full.rs`) adds three-band Llama-3.1 long-context RoPE scaling and
`LlamaFullConfig::from_hf_json_str`. Produces bit-identical output to `LlamaModel::forward`
when `rope_scaling` is absent. `[read]`

**`Llama2cModel` is a documented thin wrapper over `LlamaModel`** (`lazy_llama2c.rs:10`) —
every forward constructs a `LlamaModel` and delegates. So `llama-lazy`'s verified run does
exercise the canonical decoder. `[read]`

---

## 3. Realize — the graph→values boundary

- `realize_f32(&self) -> Vec<f32>` — `lazy.rs:1525` `[ran]`
- `realize_u32(&self) -> Vec<u32>` — `:1007`
- `realize_f32_cuda(...)` — `:1616`

**These panic on failure** (`.expect("realize_f32 via PipelinedExecutor")`) rather than
returning `Result`. Wrap in `catch_unwind` when probing, or expect a panic to be the failure
mode. `[ran]` — this is how the missing-kernel error surfaced.

---

## 4. The consumer seam: `DecodeModel`

**`fuel-inference/src/multi_session.rs:55`** `[read]`. Moved out of `fuel-core` on
2026-07-29 (Q2), which resolved the layer-drift defect. **This is where Lightbulb plugs its
model in.**

```rust
pub trait DecodeModel {
    fn n_layers(&self) -> usize;
    fn n_kv_heads(&self) -> usize;
    fn head_dim(&self) -> usize;

    fn forward_with_kv_context_persistent(
        &self,
        tokens: &[u32],
        cache: &mut KvCache,
        ctx: &mut InferenceContext,
        session: &mut Option<DecodeSession>,
    ) -> fuel::Result<Vec<f32>>;

    // + build_batched_decode_logits — the batched arm over K sessions,
    //   all-or-nothing: errors before mutating any cache, never panics.
}
```

**Two things worth noting.** It returns **`Vec<f32>`** — *realized* logits, not a
`LazyTensor`. The realize boundary sits exactly at logits→sampling, which is what the port
assumed. And **`SamplingStrategy` is deliberately not in the trait** — it stayed a
`fuel-core` type because sampling is consumer policy. §15's mechanism/policy line holding in
the actual API.

---

## 5. KV allocator

### Pure core — `fuel-core/src/kv_block_pool.rs` `[read]`

`KvBlockPool`, `SessionHandle`, `PoolCapacity`, `EvictReport`, `Externalized`,
`Fidelity{Lossy,Exact}`, `KvAllocError` (incl. `BadBlockIndex`).

Verbs: `open` · `append(s, n_tokens)` · `capacity()` / `free_blocks()` ·
`blocks_required(cur_filled, add_tokens)` · `blocks_required_batch(&[(filled, add)])` ·
`kv_bytes_resident()` · `evict(s)` · **`evict_blocks(s, &[logical_index])`** ·
**`evict_range(s, from, to)`** · `restore(s, handle)` · `discard(s)` ·
`splice(src, dst, from, to)` · `cow_break(s, i)`.

`EvictReport { freed: Vec<usize>, still_shared: Vec<usize>, handle }` — per-block
attribution, not counts.

```rust
pub struct KvGeometry {
    pub n_layers: usize,   // vLLM shared block table: physical block p = slot p in ALL layers
    pub num_blocks: usize,
    pub block_size: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub elem_size: usize,
}
```

**Single geometry per pool.** Compressed-KV regimes each need their own pool; a session
changing regime migrates (evict → re-encode → restore), because a re-encode is not a pure
restore.

### Device layer — `fuel-core/src/kv_block_pool_device.rs` `[read]`

`DeviceKvPool::new(geom, dtype, device)` (`:151`). Real `n_layers × 2` pool buffers.
`k_pool(layer)` / `v_pool(layer)` (`:204`/`:208`), `write_block` (`:251`), `read_block`
(`:305`), **`materialize_block_table`** (`:354`) for `Op::PagedAttn`, `evict` (`:384`),
`evict_blocks` (`:398`), `restore` (`:436`). `DeviceEvicted` carries `covers()`,
`fidelity()`, `saved_block_count()`. `PageTableHost` gives `block_table_shape()` /
`context_lens_shape()`.

**f32-only and CPU-gated as of `13279179`** — which happens to match the port's F32 oracle
path. **Not yet wired to any consumer**; that wiring point is where Lightbulb's retained
cache policies sit.

---

## 6. Where the runnable shape lives

- **Decoder path**: `fuel-lazy-examples/src/bin/llama-lazy.rs` — `[ran]`, produces tokens.
- **Serving path**: **no end-to-end example exists.** `multi_session.rs` and
  `kv_block_pool_device.rs` are exercised only by their own tests, so **read the test bodies**
  for the runnable shape. Lightbulb writes the first end-to-end serving consumer.
