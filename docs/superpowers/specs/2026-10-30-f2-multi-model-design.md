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

Probe: `examples/vram_return_probe.rs` (this branch's PR). It starts a runner, serves one tiny job so weights and KV pool really exist, reads device-wide used memory from `nvidia-smi`, drops **both** senders (the runner thread then sees `Disconnected` and exits), and polls up to 30 s for memory to fall back to the pre-load level. **Positive control** in every trial: the same load with a clone of the job sender deliberately retained must stay up; it did, in all trials, so the probe can tell "returned" from "never allocated". GPU: RTX 4070 Laptop, 8188 MiB. Under the machine GPU lock, `pwsh -File`. Build: lightbulb a8dcf11 plus the local Fuel patch (`lightbulb-wt-bench`), `fuel-cuda`, release. Model SmolLM2-135M-Instruct.

| Case | n | baseline | loaded | 30 s after drop | returned |
|---|---|---|---|---|---|
| f16, ctx 2048, no device held (server-like) | 3 | 0 MiB | +1302 | +1056 | ~246 MiB |
| f16, ctx 2048, a device held by the probe | 3 | 125 | +1302 | +1056 | ~246 MiB |
| Q8_0, ctx 2048, a device held | 3 | 125 | +1302..1312 | +1056 | ~250 MiB |
| f16, ctx 4096, a device held | 3 | 125 | +1654 | +1408 | ~246 MiB |

Reading it, and what it does **not** show:

- RELEASE returned memory in **0 of 12** trials (control valid 12 of 12). The result is the same whether or not the probe holds a device, so the retention is not the probe's own handle.
- Memory is **stable across cycles** (identical figures in trials 1-3): a cache that is reused, not a growing leak.
- The returned part is a constant ~246-250 MiB regardless of file size or context. The retained part **does not depend on the quantisation** (f16 and Q8_0 end identically, as if weights were dequantised to the same size) and **grows with context** (+352 MiB for ctx 2048 to 4096, about the KV pool at batch 4). That points at weights and the KV pool staying allocated after the runner thread exits.
- With no device held, ~1056 MiB stays and the pre-load figure was 0, so even the CUDA context outlives the runner: something process-wide in Fuel (or CUDA) keeps device state alive. **I did not find which**; it is Fuel's allocator/device cache until shown otherwise.
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
