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
| P2 | On-demand load + lazy LRU eviction under the B9 preflight; `keep_alive` field checked at admission; explicit unload API. | **GPU measurement first (PM): n>=3, load A, unload A, load B on the 8 GB card; confirm VRAM is returned when the runner thread exits** (**unverified** today). Positive control: a deliberately retained handle must show VRAM NOT returned, so the measurement can tell the two apart. |
| P3 | `/v1/models` lists available models with `lightbulb_loaded`; `lightbulb-cli list` / `run`. | tests for the omit-unsupported rule, using the corpus files. |
| P4 | (folded into P2: co-residency whenever B9 says both fit.) | - |

## Open questions and risks

- **Does unloading free VRAM?** Unmeasured (see P2). If Fuel keeps allocator pools alive past the model, P2 degrades to "restart the process to switch", which is a different, smaller feature.
- **Load latency vs client timeouts.** A 9B Q4_K_M load will take seconds to tens of seconds (not measured; the only model I have timed is 135M). Cherry's request timeout is unknown to me (its docs do not say). The server must hold the request open rather than 503 in the common case.
- **Shared mirror scheduler.** One `MemoryAwareScheduler` mirrors all models; its per-token memory guess is already a guess (B8). With several models the mirror is less meaningful, not more. Acceptable only while it is a mirror.
- **Model-name semantics.** Ollama names carry tags (`qwen3:8b`). Whether ids are file names, or a name map, is a user-facing decision; the proposal uses the file name (as `default_model` does today).
- **Security:** a client choosing a model by name must not reach outside `LIGHTBULB_MODELS_DIR` (reject separators and `..`, same rule as the shard-name check in `loader.rs`); resource use by loading arbitrary listed models is bounded by the preflight, not by auth.

## PM defaults 2026-10-30 (CireSnave can override)

1. No timer thread; evict LRU lazily at admission when B9 says the new model does not fit; `keep_alive` honoured at the next request/admission only; explicit unload API yes.
2. `/v1/models` lists unloaded models, with a state field (`lightbulb_loaded`).
3. Co-residency whenever B9 says both fit; otherwise evict LRU; never evict a model with an in-flight request.
4. P2 stays gated on the VRAM-return measurement (n>=3, with the retained-handle positive control).

Recommendation: P1 now (no behaviour change, de-risks the rest); do the VRAM-return measurement before P2.
