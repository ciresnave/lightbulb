# F2: several models per server, runtime switching, `run` / `list` (design, no code)

Status: draft for PM review, 2026-10-30. Read at `origin/main` f04cf26b (0.23.5). Part of ROADMAP F2 (census rows 3, 4, `docs/OLLAMA-CHERRY-CENSUS.md`). Nothing here has been built or measured except where a line says **measured**.

## What exists

- **One model per process, wired once.** `AppState` holds a single `inference_tx`, `cache_control` and `chat_template` (`src/api/mod.rs:437-458`). `try_start_model_runner` (`mod.rs:722-790`) fills them at startup from `LIGHTBULB_DEFAULT_MODEL` under `LIGHTBULB_MODELS_DIR`; nothing changes them afterwards.
- **Any other name is refused.** `chat.rs:312-322` and `completions.rs:108` return a 400 "model 'x' is not loaded on this server; the loaded model is 'y'". That check was added deliberately (2026-09-24) so a caller trying candidate models in sequence does not get one model's output under several names; any multi-model design must keep a *truthful* answer for an unknown name.
- **`/v1/models` lists exactly the loaded model** (`models.rs:55`), by design (the old hardcoded roster was the defect, A1).
- **A runner stops when its channel closes.** `scheduled_runner.rs:196-199` exits the loop on `RecvTimeoutError::Disconnected`, so *unloading* a model is "drop its `inference_tx` and `ctrl` senders". `ModelRunner::start_with_options` discards the thread handle (`model_runner.rs:398`), so today nothing can wait for the model's memory to actually be released.
- **Per-model state that is currently global in `AppState`:** `chat_template` (per model by nature), `eos_monitor` (its signal is "this model's template looks wrong", so per model), `cache_stats` (prefix-cache counters, per runner). `scheduler` (`MemoryAwareScheduler`) is a *mirror* of Fuel's admissions (B8b), one shared budget.
- Size of the refactor: matching *lines* of `git grep -c "inference_tx\|cache_control\|default_model" -- src` outside `src/api/mod.rs` at f04cf26b: `chat.rs` 19, `completions.rs` 8, `models.rs` 8, `admin.rs` 6, `lightbulb.rs` 2, `main.rs` 2, `chat_template/registry.rs` 1 (counts lines, not call sites; the tests inside those files are included).

## The constraint that shapes the design

**This box's GPU has 8188 MiB (measured: `nvidia-smi`, 2026-10-30), and the installed models are 2 to 9 GB** (`ollama list`). Several 5 to 9 GB models cannot be resident together. So "multi-model" here mostly means **load on demand and evict under memory pressure**, with co-residency whenever the B9 preflight says both fit. The B9 preflight already answers "does it fit now" (GPU refuses, CPU warns).

## Proposal

1. **`ModelRegistry`** replaces the three singleton fields: `name -> Entry`, where `Entry` is `Unloaded | Loading | Ready { handle, last_used, in_flight }`, `handle = { inference_tx, cache_control, chat_template, eos_monitor, cache_stats, join }`.
2. **Request path:** resolve `request.model` to an entry; unknown name -> 400 as today (truthful); known but `Unloaded` -> load it (single-flight per model; concurrent requests for it wait, bounded by the existing request timeout); `Ready` -> admit as now.
3. **Making room / co-residency:** before a load, run the B9 preflight. If B9 says the new model fits alongside what is resident, load it alongside (co-residency whenever both fit, PM default). On `WontFit`, unload the least-recently-used *idle* model(s) (`in_flight == 0`), **wait for their thread to exit** (needs the `JoinHandle` that `start_with_options` drops today), then retry the preflight. If nothing idle can be evicted, answer 503 with `Retry-After`, never block forever.
4. **No timer; unload under pressure only (PM default 2026-10-30).** A model stays loaded until memory pressure forces it out: at the next admission, if the B9 preflight says the model being loaded does not fit, evict the least-recently-used idle model(s) lazily, never one with an in-flight request. An Ollama-compatible `keep_alive` request field is honoured by being checked at the next request or admission, never by a background scheduler, so there is no scheduling concern under the ant-first rule. An explicit unload API exists (admin route; shape to be designed in P2).
5. **Discovery:** the available set is the files in `LIGHTBULB_MODELS_DIR` (`*.gguf`, and SafeTensors directories) whose header passes `detect_architecture` (`gguf/mod.rs:1732`). `/v1/models` lists those, with a non-OpenAI field `lightbulb_loaded: bool` beside the existing `lightbulb_memory`. Listing unloaded models is truthful only because a request will now load them; unsupported architectures (23 of CireSnave's 30, census row 15) are **omitted**, so the list never offers a model that will fail.
6. **CLI:** `lightbulb-cli list` (GET `/v1/models`) and `lightbulb-cli run <model>` (a chat REPL that sends `model: <name>`, the server loads on demand). `pull` stays F1 (blocked on mlmf). There is deliberately no `lightbulb run` that loads weights in the CLI process: that would be a second serving path.

## Phases (each its own PR)

| P | Change | Gate |
|---|---|---|
| P1 | Registry with one entry; behaviour identical; `start_with_options` also returns the thread's `JoinHandle`. | existing tests green; a test that dropping the senders ends the thread. |
| P2 | **Blocked by the measurement below.** As written (unload = drop the senders, thread exits) it does not free the dominant allocations in this Fuel build. P2 needs one of the three options under "Measured". | n/a until an option is chosen. |
| P3 | `/v1/models` lists available models with `lightbulb_loaded`; `lightbulb-cli list` / `run`. | tests for the omit-unsupported rule, using the corpus files. |
| P4 | (folded into P2: co-residency whenever B9 says both fit.) | - |

## Open questions and risks

- **Does unloading free VRAM?** Measured 2026-10-30: **no** (next section).
- **Load latency vs client timeouts.** A 9B Q4_K_M load will take seconds to tens of seconds (not measured; the only model I have timed is 135M). Cherry's request timeout is unknown to me (its docs do not say). The server must hold the request open rather than 503 in the common case.
- **Shared mirror scheduler.** One `MemoryAwareScheduler` mirrors all models; its per-token memory guess is already a guess (B8). With several models the mirror is less meaningful, not more. Acceptable only while it is a mirror.
- **Model-name semantics.** Ollama names carry tags (`qwen3:8b`). Whether ids are file names, or a name map, is a user-facing decision; the proposal uses the file name (as `default_model` does today).
- **Security:** a client choosing a model by name must not reach outside `LIGHTBULB_MODELS_DIR` (reject separators and `..`, same rule as the shard-name check in `loader.rs`); resource use by loading arbitrary listed models is bounded by the preflight, not by auth.

## Measured 2026-10-30: dropping a runner does not return its VRAM

Probe: `examples/vram_return_probe.rs` (this PR). It starts a runner, serves one tiny job so weights and KV pool really exist, reads device-wide used memory from `nvidia-smi`, drops **both** senders (the runner thread then sees `Disconnected` and exits), and polls up to 30 s for memory to fall back to the pre-load level. **Positive control** in every trial: the same load with a clone of the job sender deliberately retained must stay up; it did, in all trials, so the probe can tell "returned" from "never allocated". GPU: RTX 4070 Laptop, 8188 MiB. Under the machine GPU lock, `pwsh -File`. Build: lightbulb a8dcf11 plus the local Fuel patch (`lightbulb-wt-bench`), `fuel-cuda`, release. Model SmolLM2-135M-Instruct.

| Case | n | baseline | loaded | 30 s after drop | returned |
|---|---|---|---|---|---|
| f16, ctx 2048, no device held (server-like) | 3 | 0 MiB | +1302 | +1056 | ~246 MiB |
| f16, ctx 2048, a device held by the probe | 3 | 125 | +1302 | +1056 | ~246 MiB |
| Q8_0, ctx 2048, a device held | 3 | 125 | +1302..1312 | +1056 | ~250 MiB |
| f16, ctx 4096, a device held | 3 | 125 | +1654 | +1408 | ~246 MiB |

Reading it, and what it does **not** show:

- RELEASE returned memory in **0 of 12** trials (control valid 12 of 12). The result is the same whether or not the probe holds a device, so the retention is not the probe's own handle.
- Memory is **stable across cycles** (identical figures in trials 1-3): a cache that is reused, not a growing leak.
- The returned part is a constant ~246-256 MiB regardless of file size or context. The retained part **does not depend on the quantisation** (f16 and Q8_0 end identically, as if weights were dequantised to the same size) and **grows with context** (+352 MiB for ctx 2048 to 4096, about the KV pool at batch 4). That points at weights and the KV pool staying allocated after the runner thread exits.
- With no device held, 1056 MiB (absolute; the pre-load figure was 0) is still in use after the drop: the same +1056 as when the probe holds a device (1181 absolute, of which 125 is the held context). So the memory is not kept alive by the probe's own handle, and something in the process (Fuel's allocator or device cache, CUDA, or a library handle) holds it. **I did not find which.**
- Not measured: whether a **larger** model loaded after the drop reuses that retained memory (then swapping models of different sizes may still fit) or adds to it; whether this build is representative of a published Fuel release; larger real models (only 135M variants are in the corpus).

**Options for P2 (decision needed; the first two are not mine to build):**
1. *Fuel-side release.* Ask Fuel for an explicit way to release device memory held for a dropped model (pool trim / device cache clear), or confirmation that one exists. Cheapest if it exists.
2. *One worker process per model.* A supervisor spawns a `lightbulb` worker per model and unloads by process exit, which is the one release the OS guarantees. Large change (IPC or a reverse proxy in front); would also isolate CUDA faults.
3. *Swap by restart, no in-process unload.* Keep P1 (registry) and P3 (`list`/`run`, listing); a request for a non-resident model is answered "restart with X" or the server re-execs itself. Smallest, least Ollama-like.

Recommendation: ask Fuel (option 1) first, since it also unblocks B8b-style work; if there is no release API, option 2 is the honest route to Ollama-style switching.

## PM defaults 2026-10-30 (CireSnave can override)

1. No timer thread; evict LRU lazily at admission when B9 says the new model does not fit; `keep_alive` honoured at the next request/admission only; explicit unload API yes.
2. `/v1/models` lists unloaded models, with a state field (`lightbulb_loaded`).
3. Co-residency whenever B9 says both fit; otherwise evict LRU; never evict a model with an in-flight request.
4. P2 was gated on the VRAM-return measurement (n>=3, with the retained-handle positive control). Done: the answer is no (next section).

Recommendation: P1 now (no behaviour change; it also becomes useful for any option below). Choose among the options in "Measured" before P2.

## Update 2026-11-04 (origin/main 266be35, Fuel 0.36.4)

**Answer to "one runner per model, or swap, and B9 per model":** one runner per model in the registry, loaded on demand, co-resident whenever the B9 preflight says the new model fits next to what is resident, otherwise evict the least-recently-used idle model (the PM defaults above stand). Swap-by-restart is the fallback only if P2 below has no answer. B9 runs once per model load, at the moment of the load; its device figure must be the live free memory (the runtime verdict), not the file size alone, so resident models are already counted. **Not checked:** that B9's runtime verdict reads device free memory rather than a total minus its own bookkeeping; P2 must confirm that before relying on it.

**Status:** P1 is partly done: `ModelRunner::start_with_handle` returns the serving thread's `JoinHandle` (0.23.22). The `AppState` registry itself is not started (about 274 references to `inference_tx`, `cache_control` and `chat_template` across 11 files). P2 (unload) is still blocked on the measurement above.

**New lead on P2, from reading source only (not measured):** `fuel-cuda-backend 0.36.4`, `src/device.rs:533-534`, takes the driver's default memory pool for the device and sets its release threshold to `u64::MAX`, so memory freed back to that pool is **held, not returned to the OS**; the comment at `:531` lists a `trim_to` at realize-end as an unbuilt follow-up. `baracuda-driver 0.1.0-alpha.80` has `MemPool::trim_to(min_bytes_to_keep)` (`src/mempool.rs:171`). That fits the probe's reading (constant retention after the runner thread exits, stable across cycles) but the probe did not test it: allocations that bypass the pool (if any) would not be returned by a trim either.

**Next measurement (needs a GPU slot, PM "slot go"):** rerun `examples/vram_return_probe.rs` with one added step after the drop: `default_pool(..).trim_to(0)` on the same device, n >= 3, with the retained-sender positive control. If memory returns, the options are (1) ask Fuel for a public `trim`/release call on `CudaDevice` (preferred: no new dependency), or (2) Lightbulb calls `baracuda-driver` directly (a new crates.io dependency, and it must hold a context for that device). If it does not return, the three options above stand and the worker-process route is next.

## Measured 2026-11-05: `trim_to(0)` on the default pool returns the memory

Probe: `examples/vram_return_probe.rs` with `LB_TRIM=1`. Same protocol as above (RELEASE and CONTROL per trial, n = 3 per mode, `nvidia-smi` device-wide, under the machine GPU lock), with one added step: when memory has not returned 30 s after both senders were dropped, call `baracuda_driver::mempool::MemoryPool::trim_to(0)` on the device's default pool and read the pool's own `used_bytes` / `reserved_bytes` and `nvidia-smi` again. The pool handle (and its context) is created before the baseline, so it is part of the baseline. Build: lightbulb 1f8e244 plus Fuel 0.36.7 (the move in this PR), `fuel-cuda`, release; `baracuda-driver 0.1.0-alpha.80`. GPU RTX 4070 Laptop 8188 MiB, SmolLM2-135M-Instruct f16, ctx 2048.

| `LB_HOLD_DEVICE` | n | baseline | loaded | 30 s after drop | pool used / reserved before trim | after `trim_to(0)` | memory back to baseline |
|---|---|---|---|---|---|---|---|
| 1 | 3 | 238 MiB | 1485-1549 | 1230-1294 | 0 / 992-1056 MiB | 238 MiB in 0.1-0.4 s | 3 of 3 |
| 0 | 3 | 113 MiB | 1361-1427 | 1105-1169 | 0 / 992-1056 MiB | 113 MiB in 0.1 s | 3 of 3 |

- Without the trim, **0 of 6** trials returned memory within 30 s (as measured on 2026-10-30). With it, **6 of 6** returned to the baseline. The positive control (a retained sender; the memory must stay up) held in 6 of 6.
- The pool's `used_bytes` was 0 before the trim: the retained memory is freed blocks the pool keeps (`fuel-cuda-backend` sets the pool's release threshold to `u64::MAX`, `device.rs:533-534`), not live allocations. `trim_to` cannot release in-use memory, so calling it while another model is resident frees only that pool's unused reserve.
- **Consequence for P2:** unload is drop the senders, join the runner thread (`start_with_handle`, 0.23.22), then `trim_to(0)` on the device's pool. The worker-process route is not needed for this model. Fuel has no public call for it today; Lightbulb would call `baracuda-driver` directly (a new crates.io dependency, and a context for the device), or Fuel adds one.
- **Not shown:** a larger model (only 135M is in the corpus here); two models resident at once (what `trim_to` does to the survivor's reuse: it must re-reserve on its next allocation, a latency cost not measured); a Fuel-side API (none exists in 0.36.7: `rg trim fuel-cuda-backend` finds only a comment); the CPU, Vulkan and ROCm backends; a server process rather than the probe.
- The 2026-10-30 section above stays as measured: its trials had no trim, and its conclusion ("dropping the senders frees nothing") is still true; its option list is superseded by this section.

## Update 2026-11-05 (P1 landed, 0.23.33)

P1 is done as designed and **changes no behaviour**: `AppState.inference_tx`, `cache_control` and `chat_template` became `AppState.models: ModelRegistry` (`src/api/registry.rs`), a name -> `ModelEntry` map holding at most one entry, read through `AppState::model()`. The three pieces stay independently optional (a test server has a template and no runner; a control sender exists only with a runner). Two simplifications against the design above, on purpose: the entry has no `join` handle yet (P2 adds it with unload, when something joins), and `eos_monitor` / `cache_stats` stay in `AppState` (the runner takes `cache_stats` at start, so it moves with P2). The size note above overstated the work: of the "about 274 references", about 12 production sites and 14 `AppState` literals needed a change; most `chat_template` hits are the module path. Not built: resolving `request.model` to an entry (the handlers still compare against `config.default_model`), load on demand, eviction.
