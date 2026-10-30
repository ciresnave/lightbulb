# L2 / L4 / L5: mixed-vendor device selection, hybrid KV budget, prefix sharing for hybrids (design, no code)

Status: draft for PM review, 2026-10-30. Read at `origin/main` dcda3f1 (0.23.13). Part of ROADMAP F10 (plan: `2026-10-30-qwen38-27b-plan.md`, rows L2, L4, L5). Nothing here has been built or measured; every "today" line was read in the source at that ref. L3 is already per-device and is only touched where L4 needs it.

## What exists today

- **One device, chosen in one place.** `model_fuel::device::select()` (`src/model_fuel/device.rs:23`) returns CUDA device `hardware::cuda_device_index()` (env `LIGHTBULB_CUDA_DEVICE_INDEX`, default 0, read at each model load) when built with `fuel-cuda` and the device opens, else CPU, with a warn on fallback. `BatchedServeConfig::new` calls it (`scheduled_llama_engine.rs:501`), and `run_fuel_engine` hands that same device to the loader and to `preflight::enforce` (`:615`), so the weights check runs against the device actually selected. That is L3's "per device" half, already true for any single device.
- **Preflight checks weights only.** `preflight::enforce` -> `enforce_with` -> `check_with(weights_bytes, runtime)` compares the B17 resident-weights estimate with `BackendRuntime::available_bytes()` (`preflight.rs:125-190`). KV and any fixed per-session state are not in the figure, so L3's "weights + KV + f32 state" is not done.
- **KV budget assumes every layer holds KV.** `KvBudget { block_size: KV_BLOCK_SIZE, num_blocks: max_batch_size * ceil(context_length / KV_BLOCK_SIZE) }` (`scheduled_llama_engine.rs:505-508`). It counts blocks, not bytes: bytes per block come from Fuel's pool, which sizes it for the model's layer count. Lightbulb has no per-layer notion at this seam.
- **Prefix sharing is a bool.** `EngineOptions::prefix_sharing` (default false, env `LIGHTBULB_PREFIX_SHARING`) sets `prefix_sharing_enabled`; when true, admission takes the sharing path (`scheduled_llama_engine.rs:404`). Nothing consults the model kind.
- **Vulkan has no selection path.** `select()` has no Vulkan arm; `Cargo.toml` features and Fuel's Vulkan backend are the F6 row, behind Fuel row 124.

## L2: mixed-vendor enumeration and selection

**Need.** CireSnave's rig (via PM): RTX 4060, Radeon VII, Arc A770, one P40, MI50 on order, dual boot. Target order from the plan: single P40 via CUDA, then MI50, then Vulkan. Today the only choice is a CUDA ordinal in an env var; on a box with a 4060 and a P40, "index 1" means whatever the driver enumerates second, which can change when a card is added. Selecting by name is the minimum useful step.

**Proposal.**
1. A `DeviceInfo { backend, index, name, total_bytes, free_bytes }` list from a `device::enumerate()` that asks each compiled-in backend for its devices. CUDA part: the same `CudaDevice::new(i)` loop `select()` already implies, plus `BackendRuntime::{total,available}_bytes` for the figures. Vulkan part: **blocked** on a Fuel enumeration API (open question 1).
2. `select()` becomes `select_from(&[DeviceInfo], &Selector) -> Device` where the selector is `LIGHTBULB_DEVICE` = `auto` (default) | `cpu` | `<backend>:<index>` | a case-insensitive substring of the device name (`p40`). Ambiguous substring is an error naming the candidates, never "first match".
3. `auto` = today's behaviour when exactly one accelerator exists. With several, `auto` picks the one with the most free bytes that fits the model's weights estimate, and logs the choice and the runners-up. This is the only new behaviour, and it only differs from today on multi-device boxes.
4. `LIGHTBULB_CUDA_DEVICE_INDEX` keeps working and maps to `cuda:<n>`; setting both is an error.
5. An explicit selector that names an absent device is an error, not a CPU fallback (today's fallback is right for "no GPU present", wrong for "you asked for the P40 and it is missing"). `auto` and unset keep the CPU fallback.

**Not in scope.** Multi-device groups (a model split over P40 + MI50): needs a placement plan (L3 "later") and Fuel support.

**Test plan.** `select_from` is pure over a `Vec<DeviceInfo>`, so selector parsing, ambiguity, the unknown-device error and the `auto` ordering are unit tests with fabricated lists; a mutation test flips the `auto` comparison (most-free vs first) and must redden the ordering test. Enumeration itself needs hardware; the single-device CUDA path stays covered by the existing `select_returns_a_usable_device` plus a run on this box's RTX 4060.

## L4: KV and fixed-state budget for a hybrid model

**Facts from the plan, not re-measured here.** Qwen3.5 27B: 16 of 64 layers hold KV, the rest are gated-delta-net layers holding fixed-size recurrent state per session. So KV grows with context for a quarter of the layers, and each session also costs a constant that does not grow.

**Proposal.**
1. Lightbulb must not guess layer counts. Ask the model for its memory shape through a small trait object supplied by the loader: `bytes_per_kv_block()` and `fixed_state_bytes_per_session()` (0 for a non-hybrid). Fuel's pool already knows the first; the second is Fuel's `PagedDecodeModel` hybrid impl (plan rows 117/118) to expose. Until Fuel gives both, lightbulb keeps today's block-count budget for non-hybrids and refuses hybrids with the existing qwen35 message, so nothing changes silently.
2. `max_batch_size` for a hybrid is bounded by `free_bytes - weights - fixed_state * batch` before the KV blocks are sized, instead of `max_batch_size * ceil(ctx/block)` unconditionally.
3. B17/L3 preflight gains the same two terms: needed = weights + `max_batch_size * fixed_state` + KV for `max_batch_size * context_length`. The refuse-on-GPU / warn-on-CPU rule and `LIGHTBULB_ALLOW_OVERSIZE_MODEL` are unchanged. For non-hybrids, adding KV to the check **is a behaviour change** (a model whose weights fit but whose KV does not would newly refuse); I recommend landing it separately with its own version and CHANGELOG line.
4. B8b (real slot-pool resize) stays waiting on a Fuel hook; the mirror scheduler is dead for serving (`policies.rs` note 3), so it is not updated.

**Test plan.** The arithmetic is a pure function of (free, weights, per-block, per-session, ctx, batch); tests include the 27B shape (16 KV layers of 64, one P40 at 24 GB, Q4_K_M weights 15.33 GiB) as a numeric case and a non-hybrid case that must equal today's number exactly (regression pin). Mutation: drop the fixed-state term and the 27B case must redden.

## L5: prefix sharing refused for hybrids

**Why.** Prefix sharing splices cached KV blocks so a second request skips prefill of a shared prefix. A recurrent layer's state after the prefix is not in any KV block; splicing blocks alone leaves those layers at state-zero, which yields wrong output with no error. A state snapshot at the prefix boundary would make it sound but is Fuel-side and costs fixed-state bytes per cached prefix.

**Proposal.** `prefix_sharing_enabled` stays an operator opt-in, but `run_fuel_engine` computes `effective = options.prefix_sharing && !model.is_hybrid()`; if the operator asked for sharing and the model is hybrid, log a warn naming the reason and run with sharing off. A refusal at startup would also be defensible; I recommend warn-and-disable because the flag is global and F2 will serve several models under one setting. `is_hybrid()` comes from the same loader-supplied memory shape as L4, so L4 and L5 share one new type.

**Test plan.** Pure function `effective_prefix_sharing(requested, hybrid)` with the four combinations; an engine-level test with a fake hybrid shape asserts the admission path taken is the non-sharing one (counter `prefix_shared_admissions` stays 0 after two requests with an identical prefix). Mutation: remove the `&& !hybrid` and that test reddens.

## Sequencing and what blocks what

| Step | Needs from Fuel | Can start |
|---|---|---|
| L2 selector, enumeration for CUDA, `auto` | nothing | now |
| L2 Vulkan devices | enumeration API for the Vulkan backend (F6) | after Fuel row 124 and F6 |
| L3 per-device figures into `DeviceInfo` | nothing (`BackendRuntime` already reports) | with L2 |
| L4 memory-shape trait, preflight terms | per-block bytes and fixed-state bytes from the hybrid `PagedDecodeModel` (rows 117/118) | the trait and the arithmetic now, wired when 117 lands |
| L5 | `is_hybrid` from the same shape | with L4 |

## Open questions for the PM

1. Does Fuel expose (or will it expose) a device enumeration for Vulkan and ROCm with name and free bytes, or must lightbulb probe? Lightbulb should not add OS or cgroup memory probes without an ack (standing rule), so the answer decides whether L2's Vulkan arm waits for Fuel.
2. Selector spelling: `LIGHTBULB_DEVICE` as above, or fold into a ConfigurAnt key when that migration lands (WAIT item)? I would add the env var now and let ConfigurAnt own it later.
3. L5: warn-and-disable (recommended) or refuse at startup?
4. Land KV-in-preflight for non-hybrids as its own behaviour-changing PR, or leave preflight weights-only until a hybrid exists?
