# B8 / B9 / B10 wiring design (draft for PM review)

Status: design only, no code. Read at origin/main 9209042 (0.20.4). Fuel facts read from the
registry copy of fuel-backend-contract 0.25.10 (lightbulb pins 0.25.10).

## B9: pre-load memory check, memory in `/v1/models` (do first)

**Free-memory source (PM 2026-10-30: ask the backend).** `fuel_backend_contract::backend::BackendRuntime`
has `available_bytes() -> Option<u64>`, `total_bytes() -> Option<u64>` and
`would_fit(size) -> FitStatus` (`WontFit | Tight | Comfortable | Unknown`). It is implemented for
`CpuBackendDevice` and `CudaDevice` at 0.25.10 (its doc comment names Vulkan as a backend that may override `would_fit`; I did not check
that impl or Metal). Its honesty contract: `None`/`Unknown` means *no signal*, never "zero free".
Lightbulb does not use it today: `hardware::HardwareProfile::detect()` (sysinfo-style, `vram_size` of
`Option<MB>`) is the only source, and it is a different, coarser one.

**Open question 1 (must be answered before code):** how does lightbulb obtain a `&dyn BackendRuntime`
from the `fuel::Device` it already builds (`hardware/mod.rs:241-270`)? `fuel` 0.25.10 re-exports no
`BackendRuntime` that I found; the impls live on backend device types (`CpuBackendDevice`,
`CudaDevice`). Candidates: a `Device` accessor in fuel-tensor, or constructing the backend device
directly (we already depend on `fuel-cpu-backend` and, with `fuel-cuda`, `fuel-cuda-backend`).
If neither exists this is a Fuel ask (a `Device::runtime()`), not hand-rolled probing.

**Design.**
1. `memory::preflight(estimate: &MemoryEstimate, rt: Option<&dyn BackendRuntime>) -> Preflight`
   where `Preflight = Fits | Tight | WontFit{need,available} | Unknown`. Pure function over
   `rt.would_fit(estimate.total_bytes() as u64)`; `None` runtime => `Unknown`.
2. Inputs to `MemoryEstimate` come from what the loaders already know before allocating: GGUF tensor
   byte sizes (header only), safetensors file sizes / shard index total, plus `KvCacheMemory` from the
   config and the served context length.
3. Call site: once in `run_fuel_engine` (`scheduled_llama_engine.rs`), before any of the three loaders runs,
   using the model's on-disk weight size as a lower bound. `WontFit` follows the ruling below (GPU refuses via the
   existing `load_failed` path, CPU warns), the message naming need/available. `Tight` => `tracing::warn!`.
   `Unknown` => proceed (never refuse on no signal).
4. `/v1/models`: add the estimate breakdown (weights / KV / activations / total bytes) to the loaded
   model's object under a clearly non-OpenAI field (`lightbulb_memory`), absent when unknown. `models.rs`
   already lists exactly the loaded model.

**Tests / drills.** Mock `BackendRuntime` (as fuel's own tests do) for each `FitStatus`; mutation: make
`Unknown` refuse (must turn the "no signal proceeds" test red). Controls: the same estimate against a
runtime reporting plenty of memory must pass.

**Ruling (PM 2026-10-30).** GPU `WontFit` refuses (VRAM is a hard limit); CPU `WontFit` WARNS and proceeds
(free RAM is soft: page cache, swap, mmap'd weights), the message naming both figures. One explicit override,
`LIGHTBULB_ALLOW_OVERSIZE_MODEL` (`ApiConfig::model_allow_oversize` / `EngineOptions::allow_oversize_model`),
forces the load on a GPU too. `Tight` warns; `Unknown` never refuses. Implemented in 0.21.2.

**Risk.** CPU `available_bytes` is system-wide and noisy (contract doc says so); hence the CPU rule above.

## B8: `SystemConfig::auto_detect` and `SlotPoolMonitor`

Callers today: examples/tests only. The scheduler is built in `main.rs` from five env vars
(`scheduler_config_from_env`) with fixed defaults (8 GB, 100 MB/slot, 50 KB/token).

**Proposal (smaller than "wire everything").**
- `auto_detect` currently `println!`s and re-detects hardware. Replace the prints with `tracing` and take
  the B9 free-memory figure as an input instead of `HardwareProfile` RAM/VRAM where the backend reports
  one. Use it only to compute the *defaults* of `MemoryAwareConfig.max_memory_bytes` and slot pool size
  when the env vars are unset; an explicit env var always wins. This makes B8 depend on B9's runtime
  accessor.
- `SlotPoolMonitor`: wire only if the engine actually resizes the slot pool at runtime. I have not
  confirmed it does; `SlotPool` resize support must be read first. If there is none, B8's monitor half
  is dead code with nothing to adjust, and the honest options are delete or defer, not wire. **Needs a
  PM call after I read `slot_pool.rs`.**
- `chunk_size` has no consumer outside `init.rs`; wire into prefill chunking only if the Fuel prefill
  path chunks (not checked). Otherwise delete the field.

## B10: state persistence

`create_checkpoint` writes `kv_cache_state: None` / `pipeline_state: None` (`state_persistence.rs:509-511`)
and its only caller is `engine/decomposition.rs`. A graceful-shutdown save needs (a) a KV serialisation
format and (b) a shutdown hook. Both are large and (a) is entangled with paged KV and prefix sharing
(#227). Recommendation: **defer B10 behind #227 and Fuel paged-KV export**; do not hand-roll a KV
serialiser. Restoring KB/decomposition state at startup is separable and small, but nothing in the
product consumes it yet.

## Order and sizing
1. B9 preflight (+ `/v1/models` field), after open question 1 is answered. One PR.
2. B8 defaults-from-backend, one PR, after B9. Monitor/chunk_size: PM call.
3. B10: deferred.

Nothing here adds a dependency or hand-rolled config/secret/HTTP code.
