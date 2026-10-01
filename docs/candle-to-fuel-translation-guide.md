# Candle → Fuel Translation Guide — serving-layer symbols

**Scope**: candlelight usage that survives the move-manifest (i.e., stays in lightbulb as serving-layer code) but needs to do its tensor/device/dtype work the Fuel way. Excludes everything the manifest task is moving/deleting (`src/model/*`, `kernels.rs`, `backend/`, `multi_gpu/*`, `quantization/*`, `lora/*`, `pruning/*`, `loaders/awq.rs`, `gguf/*`, `hub.rs`, cache compression/codec kernels).

**Method**: `git show origin/main:<path>` only, both repos, never a working tree (portfolio CLAUDE.md §2). Lightbulb at `32e91fc3` (2026-09-30). Fuel at `5bd79337` (2026-09-30) — current post-dissolution layout, re-verified independent of the two 2026-09-24 audit docs, which predate the crate split.

**Verified with `git show origin/main:src/lib.rs | grep -n candlelight`, `git ls-tree -r --name-only origin/main -- src`, and per-file `grep -n candlelight`** against every file in lightbulb's tree. 49 files reference `candlelight`; of those, the ones below are what's left after subtracting the manifest's scope.

---

## 0. Files touched, by verdict

| File | Verdict | Why |
|---|---|---|
| `src/cache/parallel_cache_builder.rs` | **DELETE**, reference only | Candle-based KV cache span builder. Already superseded in-repo by `src/model_fuel/policies.rs` (fuel `kv_block_pool`-based), confirmed by the 2026-09-24 capability audit and unchanged since. Table below is offered only in case any span/eviction *logic* (not the tensor plumbing) needs manual porting rather than a straight delete. |
| `src/cache/prefix_cache.rs` | **DELETE, WIRE to fuel-inference** | See §1 — fuel ships this exact module already. |
| `src/engine/speculative.rs` | **DELETE, WIRE to fuel-inference** | See §2 — fuel ships this exact module already. |
| `src/hardware/mod.rs` (`to_device()`) | **TRANSLATE** | Real shape hazard, no drop-in replacement exists — see §3. |
| `src/memory/estimate.rs`, `src/memory/speculative.rs`, `src/memory/utils.rs` | **TRANSLATE, simplify** | See §4 — one-line swap, and fuel's version is *more correct* than lightbulb's hand-rolled match. |
| `src/tools/mod.rs` (test-only `DType`/`Device`/`Tensor` import) | **TRANSLATE, trivial** | Test-only; migrates for free when the module it's testing does. No production code. |
| `src/lib.rs` (`hello_generate`, `local_llama_generate`) | **OUT OF THIS TASK'S SCOPE — flag for the manifest doc** | Full Candle Llama reference-generation path (uses `candlelight::transformers::generation::{LogitsProcessor,Sampling}`, `candlelight::transformers::models::llama::LlamaEosToks`, and `crate::loaders::load_local_llama`). Entirely superseded by `src/model_fuel/generate.rs`. Wasn't in the manifest's named list (only `lib.rs`'s module declarations were expected) — **recommend the manifest doc add `src/lib.rs`'s `hello_generate`/`local_llama_generate` fns as DELETE**. |
| `src/loaders/mod.rs` | **OUT OF THIS TASK'S SCOPE — flag for the manifest doc** | Only `loaders/awq.rs` was named in the manifest's list, but `loaders/mod.rs` itself is a full Candle SafeTensors+GGUF Llama loader (`candlelight::nn::VarBuilder`, `candlelight::transformers::models::llama::{Llama,Cache,Config}`, `candlelight::core::quantized::gguf_file::{Content,Value}`, `candlelight::core::safetensors::load`) — the exact thing `src/model_fuel/loader.rs`/`loader_f32.rs`/`loader_gguf.rs` already replace on the Fuel path. **Recommend the manifest doc add the whole file as DELETE**, not just `awq.rs`. |
| `src/api/chat_template.rs`, `src/engine/model_runner.rs`, `src/model_fuel/{engine_model,loader_gguf,mod,policies}.rs` | **No action needed** | `candlelight` appears only in prose/doc-comments in these files (confirmed by per-line grep above) — no actual API call to translate. |

No true gap was found anywhere in this scope — consistent with CireSnave's framing that this is a translation exercise, not a gap hunt.

---

## 1. `src/cache/prefix_cache.rs` → `fuel_inference::prefix_cache`

Lightbulb's `PrefixKvEntry`/`PrefixKvCache` (hash-keyed LRU cache of per-layer `(Tensor, Tensor)` KV pairs) is duplicated, nearly line-for-line in *purpose*, by **`fuel-inference/src/prefix_cache.rs`** (read in full at `origin/main`): `PrefixCache::{new,lookup,insert}`, `LayerKvState = (Tensor, Tensor)`, same LRU-by-access-counter design, same stated 15–50% TTFT-reduction rationale (verbatim-similar doc comment — this is fuel's own port of the same idea, not a coincidence).

| Candle symbol | Used at (lightbulb) | Fuel way | Shape difference |
|---|---|---|---|
| `candlelight::core::{Device, Tensor}` (`prefix_cache.rs:33`) | Struct field `kv_by_layer: Vec<(Tensor, Tensor)>`, fn param `_device: &Device` (`prefix_cache.rs:232-233`, unused — `_device` is dead already) | `fuel::lazy::Tensor` — same pair shape, `fuel_inference::prefix_cache::LayerKvState` is literally `(Tensor, Tensor)` | Fuel's `Tensor` is a lazy-graph node, not eager storage — cloning a `Tensor` clones a graph reference, not the underlying buffer; inserting into a long-lived cache pins that subgraph alive rather than a materialized value. `fuel_inference::prefix_cache` already handles this (it stores the `Tensor` handles the same way, so the concern is already resolved in fuel's implementation, not something the porter has to solve fresh). |

**Recommendation**: don't translate this file symbol-by-symbol — delete it and depend on `fuel-inference`, calling `PrefixCache::{new,lookup,insert}` from wherever lightbulb currently owns a `PrefixKvCache`. Note lightbulb doesn't declare `fuel-inference` as a `src/` dependency today (only two integration tests do, per the 2026-09-24 status doc — reconfirm still true before wiring, since that doc is now stale on other points).

---

## 2. `src/engine/speculative.rs` → `fuel_inference::speculative`

Lightbulb's `SpeculativeModel` trait (`forward_logits(&mut self, tokens: &[u32], position: usize) -> Result<Tensor>`, `device(&self) -> &Device`) plus its own accept/reject verification logic is the same algorithm **`fuel-inference/src/speculative.rs`** already implements: `SpeculativeConfig`, `SpeculativeStats`, `verify_draft(draft_tokens, draft_logprobs, target_logprobs, config, stats)` — Leviathan et al. 2023 / Chen et al. 2023, cited identically in both files' doc comments.

| Candle symbol | Used at (lightbulb) | Fuel way | Shape difference |
|---|---|---|---|
| `candlelight::core::{Device, Tensor}` (`speculative.rs:26`) — trait methods `forward_logits`/`device` return these | `SpeculativeModel` trait (`speculative.rs:137,140`), closures `FnMut(&Tensor) -> Result<u32>` (`speculative.rs:216,301`) | **`fuel_inference::speculative::verify_draft` takes no `Tensor` at all** — its own doc comment states plainly: *"These are HOST values — verification is control flow over already-realized logits, so no tensor is involved."* Signature is `Vec<u32>` (draft tokens) + `Vec<Vec<f32>>` (draft/target logprobs, already realized) + `&SpeculativeConfig` + `&mut SpeculativeStats`. | This is the single biggest shape difference in this guide: Candle's version threads a live `Tensor`/`Device` through the verify loop; fuel's version pushes the tensor→host realization boundary *before* verification, so the accept/reject math is pure host arithmetic. A porter must call `.realize_f32()` (or the appropriate realize variant) on logits **before** calling `verify_draft`, not inside it. |

**Recommendation**: delete lightbulb's `SpeculativeModel` trait and its own verify loop; wire whatever calls it today to call the model's forward pass, realize the logits to `Vec<f32>` per row, and hand those to `fuel_inference::speculative::verify_draft`. This is a WIRE, not a type-for-type swap — the model boundary moves.

---

## 3. `src/hardware/mod.rs::to_device()` — real, unavoidable shape hazard

This is lightbulb's own `HardwareType` enum → device constructor. **No drop-in fuel module replaces this** (it's lightbulb's own hardware-selection policy, which HANDOFF explicitly says stays in lightbulb) — this genuinely needs hand-translation.

Current (candlelight, `hardware/mod.rs:239-248`):
```rust
pub fn to_device(&self) -> candlelight::core::Device {
    match self {
        Self::Cpu => candlelight::core::Device::Cpu,
        Self::Cuda => candlelight::core::Device::cuda_if_available(0).unwrap_or(candlelight::core::Device::Cpu),
        Self::Rocm => candlelight::core::Device::Cpu, // TODO: Add ROCm support to Candle
        Self::Metal => candlelight::core::Device::new_metal(0).unwrap_or(candlelight::core::Device::Cpu),
        Self::Vulkan => candlelight::core::Device::Cpu, // TODO: Add Vulkan support
    }
}
```

| Candle symbol | Fuel way (verified at fuel `origin/main` `5bd79337`) | Shape difference |
|---|---|---|
| `candlelight::core::Device::Cpu` (enum variant) | `fuel::Device::cpu()` (method call) | **The hazard lightbulb's own `src/model_fuel/mod.rs` doc comment already names explicitly**: *"Fuel's `Device` is a struct with `Device::cpu()`, not an enum with a `Device::Cpu` variant... the error is `E0599 no associated function or constant named 'Cpu'`."* `fuel-core/src/device.rs:33-36`: `pub struct Device { pub(crate) inner: Arc<dyn DynBackendDevice> }` — not an enum at all; CPU/CUDA/Metal/Vulkan are all the *same type*, distinguished by which trait object is inside, not by variant. |
| `candlelight::core::Device::cuda_if_available(0)` | `fuel::cuda_backend::device_if_available(0)` — returns `Result<Device>` (verified `fuel-core/src/cuda_backend/mod.rs:51`) | Same `.unwrap_or(Device::cpu())` fallback pattern works, but it's a free function in a feature-gated module (`cuda_backend`), not a method on `Device` itself. |
| `candlelight::core::Device::new_metal(0)` | `fuel::metal_backend::device_if_available(0)` — `Result<Device>` (verified `fuel-core/src/metal_backend/mod.rs:40`) | Same fallback pattern, same free-function-not-method shape. |
| Vulkan: `candlelight` has **no** Vulkan device — the `// TODO` stub already reflects that Candle can't do this | `fuel::vulkan_backend::new_device()` — `Result<Device>` (verified `fuel-core/src/vulkan_backend.rs:40`, real: `Ok(VulkanBackend::new()?.into())`) | **Real capability gain, not just a translation**: fuel has a working Vulkan backend where candlelight has none. No `_if_available` variant exists for Vulkan specifically — use `.unwrap_or_else(|_| fuel::Device::cpu())` the same way the CPU fallback works elsewhere. Confirmed via full-file read; no other constructor exists in that file. |
| ROCm: `candlelight` has no ROCm — `TODO` stub | **Also absent in fuel** — no `rocm_backend` module found anywhere in the fuel tree listing (`fuel-core/src/{cuda_backend,metal_backend,vulkan_backend}` exist; no `rocm_backend`) | Not a regression introduced by the port — carry the same `TODO`/CPU-fallback forward unchanged. Positive control that this null result isn't a broken search: the identical directory-name check found real `cuda_backend`/`metal_backend`/`vulkan_backend` modules, so the absence of a `rocm_backend` entry is a real absence, not a missed query. |

---

## 4. `src/memory/{estimate,speculative,utils}.rs` — `DType` size lookups (simplification, not just translation)

All three files do the same thing: hand-written `match dtype { F32 => 4, F16|BF16 => 2, _ => <default> }` to get a dtype's byte width, purely for capacity-planning arithmetic (confirmed — no tensor is ever touched, `DType` is used only as an enum key).

| Candle symbol | Used at | Fuel way | Shape difference |
|---|---|---|---|
| `candlelight::core::DType` (`estimate.rs:5`, `speculative.rs:7`, `utils.rs:79,82-83,133`) | Struct fields (`WeightMemory::Unquantized.dtype`, `ActivationMemory.dtype`, `KvCacheMemory.dtype`, etc.) and hand-rolled `match` arms computing bytes-per-element | `fuel::DType` (re-exported from `fuel_ir::dtype::DType` via `fuel-core/src/dtype.rs:13`) has a **native `size_in_bytes()` method** (`fuel-ir/src/dtype.rs:581-600`, read in full): `U8/I8/F8E4M3/F8E5M2/F8E8M0/F8E6M2/Bool→1, I16/BF16/F16→2, U32/I32/F32→4, I64/F64→8, F6E2M3/F6E3M2/F4→0` (sub-byte types) | This is a straight upgrade, not just a swap: lightbulb's hand-rolled matches fall through to a guessed default (`_ => 4` in `estimate.rs`, `_ => 2` elsewhere) for any dtype they didn't enumerate, so today `I8`/`I16`/`I32` silently get the *wrong* size. Fuel's `DType` also has more variants than candlelight's (adds `I8`, `I16`, `I32`, `F8E4M3`, `F8E5M2`, `F8E6M2`, `F8E8M0`, sub-byte float families, `Bool`) — a porter should call `dtype.size_in_bytes()` directly rather than re-deriving a match, which both fixes the silent-default bug and stays correct as fuel adds dtypes. |

**Recommendation**: replace every hand-rolled match in these three files with `dtype.size_in_bytes()`. This also means `Option<DType>` fields (`memory/speculative.rs:67`) need no change beyond the import swap — `.unwrap_or(DType::F16)` works identically on `fuel::DType`.

---

## 5. Trivial / no-op sites

- **`src/tools/mod.rs:316`**: `use candlelight::core::{DType, Device, Tensor}` is inside `#[cfg(test)]`. Confirmed no production code path. Swap the `use` line to `fuel::{DType, Device, lazy::Tensor}` whenever the surrounding test module's subject migrates — no independent action needed.
- **`src/api/chat_template.rs:347`, `src/engine/model_runner.rs:131`, `src/model_fuel/{engine_model.rs:111,183,282,290; loader_gguf.rs:116; mod.rs:4,91,108,145; policies.rs:62,905}`**: every `candlelight` hit in these files is inside a doc comment or `//` note, not an import or call (verified — see the per-file grep output this guide was built from). No code change needed; several of these comments (e.g. `model_fuel/mod.rs`'s note about `candlelight`-frozen `crate::model`) will need a wording update once the manifest's DELETE items actually land, but that's editorial, not translation.

---

## 6. General shape rules a porter needs, beyond the per-symbol table (from `fuel::lazy::Tensor`, read in full at `fuel-core/src/lazy.rs`)

- **Graph affinity** (already named in `src/model_fuel/mod.rs`'s own doc comment, re-verified against `fuel-core/src/lib.rs`'s crate-level example): every `Tensor::from_*`/`zeros`/etc. call mints a *new* graph; a second operand must be created `_on(existing.graph(), ...)` or built as `const_*_like`. Combining tensors from different graphs panics at runtime, not compile time.
- **Indexing**: candlelight's `IndexOp`/`.i(...)` trait (used in `parallel_cache_builder.rs:628,642,2672,2683` — a DELETE-scope file, but the pattern will recur) has no equivalent trait in fuel. Use explicit methods instead: `.i(idx)` on dim 0 → `Tensor::get(idx)` (`fuel-core/src/lazy.rs:7167`); `.i((.., idx))` selecting on an arbitrary dim → `Tensor::get_on_dim(dim, idx)` (`fuel-core/src/lazy.rs:7093-7101`, verified equivalent to `self.slice(dim, index, 1)?.squeeze(dim)`).
- **Realization is flat, not shape-aware**: candlelight's `.to_vec1/2/3::<T>()` returns nested `Vec`s matching tensor rank. Fuel's `Tensor::realize_f32()` (and the `u8`/`u32`/`f64`/`bf16`/`f16` variants, `fuel-core/src/lazy.rs:1332-2049`) returns a **flat** `Vec<T>` — the porter must read `.shape().dims()` separately and reshape by hand if nested structure is needed. `realize_many_f32(&[&t1, &t2])` batches multiple tensors in one graph execution — prefer it over looping single realizes when realizing several tensors from the same graph.
- **Error type is drop-in compatible**: `fuel::Error::Msg(String)` and the `fuel::bail!` macro (re-exported explicitly from `fuel-core`'s macro namespace since glob re-exports don't carry macros, per `fuel/src/lib.rs`'s own comment) have the same three call shapes as candlelight's (`bail!("msg")`, `bail!("msg {}", x)`, `bail!(existing_error)`) — verified against `fuel-core/src/error.rs:16-24`. The one difference: fuel's macro calls `.bt()` (attach backtrace) on construction; candlelight's does not carry this call. Harmless to carry over unchanged.
