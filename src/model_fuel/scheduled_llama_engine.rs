//! Wires `FuelSchedulerDriver` (`super::scheduler_driver`) into
//! `ModelRunner::start`'s `fuel-engine` arm (board item 97's actual remaining
//! deliverable: everything before this file only unblocked it) — but NOT
//! unconditionally. See "Routing" below.
//!
//! `ModelRunner::start`'s `fuel-engine` arm can load either an eager
//! SafeTensors checkpoint (`LlamaModel`) or a GGUF-quantized one
//! (`QuantizedLlama3Model`) — two different concrete `DecodeModel` types, so
//! one `FuelSchedulerDriver<'m, M>` cannot hold either interchangeably.
//! `LlamaSchedulerDriver` (below, `pub(crate)` — not an intra-doc link, same
//! reason as `scheduler_driver.rs`'s own module doc) is the small
//! enum-dispatch wrapper that lets
//! `run_scheduled_jobs` stay generic over ONE `SchedulerDriver` regardless of
//! which checkpoint format was loaded.
//!
//! Qwen3 GGUF loading is a deliberately separate follow-up — `loader_gguf.rs`
//! dispatches on architecture for Llama only today; nothing here loads a
//! Qwen3 checkpoint yet.
//!
//! # Routing: multi-token EOS checkpoints stay on the serial path
//!
//! Some checkpoints declare `LlamaEosToks::Multiple` (more than one stop
//! token). Fuel's own `SessionState` tracks exactly one `eos_id: Option<u32>`
//! — not a set — so the batched path can only ever honor ONE of several
//! declared EOS tokens: a checkpoint with multiple EOS tokens would stop
//! correctly on whichever one was chosen but run PAST the others to the
//! budget ceiling, a real user-visible regression (not a nuance) against the
//! old serial `FuelEngineModel` path, which checks the full set every token
//! via `LlamaEosToks::is_eos`.
//!
//! So a `Multiple` checkpoint is routed to the EXISTING serial path instead —
//! loaded once, reused by `FuelEngineModel::from_loaded` (`pub(crate)`, not an
//! intra-doc link here either) rather than loaded a second time. `Single`/`None` checkpoints (the common
//! case) take the new batched path. Tracked with the fuel lane
//! (`SessionState::eos_id` becoming a set, i.e. `eos_ids`); when that lands,
//! the `Multiple` branch below can move to the batched path too.
#![allow(dead_code)]

use std::path::Path;
use std::sync::mpsc::Receiver;

use anyhow::Result;

use fuel_model_llama::LlamaModel;
use fuel_transformers::models::lazy_llama_full::LlamaEosToks;
use fuel_transformers::models::lazy_quantized_llama::QuantizedLlama3Model;

use fuel_inference::multi_session::{KvBudget, SchedulePolicy};

use super::engine_model::{FuelEngineModel, LoadedModel, effective_generation_budget};
use super::scheduler_driver::FuelSchedulerDriver;
use crate::engine::model_runner::{InferenceJob, drain_with_error, run_jobs};
use crate::engine::scheduled_runner::{
    AdmitError, FinishOutcome, SchedulerDriver, run_scheduled_jobs,
};

/// Which path a checkpoint takes, decided from its own EOS info — see this
/// module's "Routing" doc comment for why `Multiple` cannot take `Batched`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EngineRoute {
    Batched,
    Serial,
}

/// The single decision point both load branches (`.gguf` / SafeTensors
/// directory) call, so they cannot drift from each other on this choice.
fn route_for(eos: Option<&LlamaEosToks>) -> EngineRoute {
    match eos {
        Some(LlamaEosToks::Multiple(_)) => EngineRoute::Serial,
        None | Some(LlamaEosToks::Single(_)) => EngineRoute::Batched,
    }
}

/// The checkpoint's one EOS id. Callers must have already routed
/// `LlamaEosToks::Multiple` elsewhere via [`route_for`] — this panics on it
/// rather than silently picking one, so a future call site that skips that
/// check fails loudly instead of reintroducing the narrowing this module
/// exists to avoid.
fn single_eos_id(eos: Option<&LlamaEosToks>) -> Option<u32> {
    match eos {
        None => None,
        Some(LlamaEosToks::Single(id)) => Some(*id),
        Some(LlamaEosToks::Multiple(_)) => unreachable!(
            "single_eos_id called on a Multiple-EOS checkpoint — route it to \
             the serial path via has_multiple_eos first"
        ),
    }
}

enum LlamaSchedulerDriverInner<'m> {
    F32(FuelSchedulerDriver<'m, LlamaModel>),
    QuantizedGguf(FuelSchedulerDriver<'m, QuantizedLlama3Model>),
}

/// Enum-dispatch over the two concrete Llama `DecodeModel` types
/// `ModelRunner::start`'s fuel-engine arm can load, so `run_scheduled_jobs`
/// sees one `SchedulerDriver` regardless of checkpoint format.
///
/// Owns `context_length` so `try_admit` can apply
/// [`effective_generation_budget`] itself — the one piece of policy that is
/// NOT `FuelSchedulerDriver`'s concern (it has no notion of a server-wide
/// context ceiling), but IS this integration layer's, same role
/// `FuelEngineModel::step_one` played in the old serial path.
struct LlamaSchedulerDriver<'m> {
    inner: LlamaSchedulerDriverInner<'m>,
    context_length: usize,
}

impl<'m> SchedulerDriver for LlamaSchedulerDriver<'m> {
    type Id = fuel_inference::multi_session::SessionId;

    fn try_admit(
        &mut self,
        prompt: &[u32],
        max_new: usize,
        temperature: f64,
    ) -> Result<Self::Id, AdmitError> {
        let (effective_max_new, truncated) =
            effective_generation_budget(prompt.len(), max_new, self.context_length);
        if truncated {
            tracing::warn!(
                requested_max_new_tokens = max_new,
                truncated_max_new_tokens = effective_max_new,
                context_length = self.context_length,
                prompt_tokens = prompt.len(),
                "requested max_new_tokens does not fit in context_length; truncating \
                 the generation budget so this request returns the tokens it can \
                 generate instead of erroring mid-decode"
            );
        }
        match &mut self.inner {
            LlamaSchedulerDriverInner::F32(d) => {
                d.try_admit(prompt, effective_max_new, temperature)
            }
            LlamaSchedulerDriverInner::QuantizedGguf(d) => {
                d.try_admit(prompt, effective_max_new, temperature)
            }
        }
    }

    fn step(&mut self) -> Result<Vec<(Self::Id, u32)>, String> {
        match &mut self.inner {
            LlamaSchedulerDriverInner::F32(d) => d.step(),
            LlamaSchedulerDriverInner::QuantizedGguf(d) => d.step(),
        }
    }

    fn reap_finished(&mut self) -> Vec<(Self::Id, FinishOutcome)> {
        match &mut self.inner {
            LlamaSchedulerDriverInner::F32(d) => d.reap_finished(),
            LlamaSchedulerDriverInner::QuantizedGguf(d) => d.reap_finished(),
        }
    }

    fn has_active_sessions(&self) -> bool {
        match &self.inner {
            LlamaSchedulerDriverInner::F32(d) => d.has_active_sessions(),
            LlamaSchedulerDriverInner::QuantizedGguf(d) => d.has_active_sessions(),
        }
    }
}

/// A generous block size: large enough that `kv_blocks_required`'s ceiling
/// division wastes little, small enough that a short prompt does not reserve
/// a whole extra block it will never use. Matches the block size fuel's own
/// `multi_session.rs` test fixtures use.
const KV_BLOCK_SIZE: usize = 16;

/// The server-wide (not per-request) pieces of the batched path, bundled so
/// [`run_batched`] takes one parameter instead of three — `device`/`policy`/
/// `budget` are all `Copy`, so sharing this by reference costs nothing.
struct BatchedServeConfig {
    device: fuel::Device,
    policy: SchedulePolicy,
    budget: KvBudget,
}

impl BatchedServeConfig {
    fn new(max_batch_size: usize, context_length: usize) -> Self {
        let max_batch_size = max_batch_size.max(1);
        Self {
            device: super::device::select(),
            policy: SchedulePolicy::Batched {
                max_batch: max_batch_size,
            },
            budget: KvBudget {
                block_size: KV_BLOCK_SIZE,
                num_blocks: max_batch_size * context_length.div_ceil(KV_BLOCK_SIZE),
            },
        }
    }
}

/// Build the batched driver for one already-loaded model and drive it —
/// shared by both the `.gguf` and SafeTensors-directory branches of
/// [`run_fuel_engine`], which differ only in the CONCRETE `M` and which
/// `LlamaSchedulerDriverInner` variant wraps it (`wrap`, a tuple-variant
/// constructor used as a plain `fn`).
fn run_batched<'m, M: fuel_inference::multi_session::DecodeModel>(
    model: &'m M,
    config: BatchedServeConfig,
    eos_id: Option<u32>,
    context_length: usize,
    tokenizer: tokenizers::Tokenizer,
    rx: Receiver<InferenceJob>,
    wrap: fn(FuelSchedulerDriver<'m, M>) -> LlamaSchedulerDriverInner<'m>,
) {
    let scheduler = match FuelSchedulerDriver::new(
        model,
        config.device,
        fuel::DType::F32,
        config.policy,
        config.budget,
        eos_id,
    ) {
        Ok(s) => s,
        Err(e) => return drain_with_error(rx, &format!("building the scheduler: {e}")),
    };
    let driver = LlamaSchedulerDriver {
        inner: wrap(scheduler),
        context_length,
    };
    // `tokenizer.encode`/`.decode` each need their own `move` closure below,
    // and `Tokenizer` is not `Copy` — an `Arc` is what lets both closures
    // share it rather than one of them moving the only copy.
    let tokenizer = std::sync::Arc::new(tokenizer);
    let encode_tokenizer = tokenizer.clone();
    run_scheduled_jobs(
        driver,
        rx,
        move |text, add_special| {
            encode_tokenizer
                .encode(text, add_special)
                .map_err(|e| anyhow::anyhow!("tokenizing: {e}"))
                .map(|enc| enc.get_ids().to_vec())
        },
        move |tokens, skip_special| {
            tokenizer
                .decode(tokens, skip_special)
                .map_err(|e| anyhow::anyhow!("detokenizing: {e}"))
        },
    );
}

/// Load a checkpoint (dispatching on `.gguf` vs a SafeTensors directory, same
/// extension check `FuelEngineModel::load` makes) and drive it — through the
/// real `FuelSchedulerDriver`/`run_scheduled_jobs` batched path for a
/// `Single`/`None`-EOS checkpoint, or through the existing serial
/// `FuelEngineModel`/`run_jobs` path for a `Multiple`-EOS one (see this
/// module's "Routing" doc).
///
/// `max_batch_size` sizes the KV pool (`max_batch_size` concurrent sessions
/// of up to `context_length` tokens each) AND selects
/// `SchedulePolicy::Batched` on the batched path — a model that doesn't
/// implement the batched arm degrades to serial automatically
/// (`SessionScheduler`'s own documented fallback), so this is never a
/// correctness risk, only a speed one.
///
/// Never returns an `Err` to the caller: a load or construction failure is
/// handled HERE, by answering every already-queued and future job with an
/// error (`drain_with_error`, the same function both backends' `ModelRunner::
/// start` already use) — matching the old call site's own two-level match
/// exactly, since `rx` is consumed by whichever path runs and cannot be
/// handed back to a caller to drain a second time.
pub(crate) fn run_fuel_engine(
    model_path: &Path,
    max_batch_size: usize,
    context_length: usize,
    rx: Receiver<InferenceJob>,
) {
    let config = BatchedServeConfig::new(max_batch_size, context_length);
    let is_gguf = model_path
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"));
    if is_gguf {
        serve_gguf(model_path, context_length, config, rx);
    } else {
        serve_f32(model_path, context_length, config, rx);
    }
}

/// The `.gguf` branch of [`run_fuel_engine`] — split out so that function
/// stays short; see its own doc for the load/route/batch shape this repeats.
fn serve_gguf(
    model_path: &Path,
    context_length: usize,
    config: BatchedServeConfig,
    rx: Receiver<InferenceJob>,
) {
    let loaded = match super::loader_gguf::load_quantized_llama_gguf(model_path) {
        Ok(l) => l,
        Err(e) => return load_failed(model_path, e, rx),
    };
    println!("Fuel model loaded at {}", model_path.display());
    if route_for(loaded.eos.as_ref()) == EngineRoute::Serial {
        log_serial_fallback(model_path);
        let model =
            FuelEngineModel::from_loaded(LoadedModel::QuantizedGguf(loaded), context_length);
        return run_jobs(model, rx);
    }
    let eos_id = single_eos_id(loaded.eos.as_ref());
    run_batched(
        &loaded.model,
        config,
        eos_id,
        context_length,
        loaded.tokenizer,
        rx,
        LlamaSchedulerDriverInner::QuantizedGguf,
    );
}

/// The SafeTensors-directory branch of [`run_fuel_engine`] — see [`serve_gguf`].
fn serve_f32(
    model_path: &Path,
    context_length: usize,
    config: BatchedServeConfig,
    rx: Receiver<InferenceJob>,
) {
    let loaded = match super::loader_f32::load_llama_f32_from_dir(model_path) {
        Ok(l) => l,
        Err(e) => return load_failed(model_path, e, rx),
    };
    println!("Fuel model loaded at {}", model_path.display());
    if route_for(loaded.eos.as_ref()) == EngineRoute::Serial {
        log_serial_fallback(model_path);
        let model = FuelEngineModel::from_loaded(LoadedModel::F32(loaded), context_length);
        return run_jobs(model, rx);
    }
    let eos_id = single_eos_id(loaded.eos.as_ref());
    run_batched(
        &loaded.model,
        config,
        eos_id,
        context_length,
        loaded.tokenizer,
        rx,
        LlamaSchedulerDriverInner::F32,
    );
}

/// Report a load failure the same way both branches of [`run_fuel_engine`]
/// need to, then drain `rx` so no job is left waiting on a model that will
/// never load.
fn load_failed(model_path: &Path, e: anyhow::Error, rx: Receiver<InferenceJob>) {
    eprintln!(
        "Failed to load Fuel model at {}: {e:#}",
        model_path.display()
    );
    drain_with_error(rx, &format!("model load failed: {e:#}"));
}

/// Log the routing decision both branches of [`run_fuel_engine`] make the
/// same way when a checkpoint's EOS set forces the serial fallback.
fn log_serial_fallback(model_path: &Path) {
    tracing::info!(
        "{}: checkpoint declares multiple EOS tokens; serving on the serial \
         FuelEngineModel path (see scheduled_llama_engine's Routing doc)",
        model_path.display()
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_multi_eos_checkpoint_selects_the_serial_path() {
        // The PM's explicit ask after the multi-EOS finding: a checkpoint
        // declaring more than one stop token must NOT take the batched path
        // (which can only honor one), it must fall back to the serial one.
        assert_eq!(
            route_for(Some(&LlamaEosToks::Multiple(vec![7, 8, 9]))),
            EngineRoute::Serial,
            "a checkpoint with multiple EOS tokens must route to the serial \
             FuelEngineModel/run_jobs path, never the batched one — running \
             past an unlisted EOS token to the budget ceiling is a \
             user-visible regression, not an acceptable narrowing"
        );
    }

    #[test]
    fn a_single_or_no_eos_checkpoint_selects_the_batched_path() {
        assert_eq!(route_for(None), EngineRoute::Batched);
        assert_eq!(
            route_for(Some(&LlamaEosToks::Single(2))),
            EngineRoute::Batched
        );
    }

    #[test]
    fn single_eos_id_reads_through_none_and_single() {
        assert_eq!(single_eos_id(None), None);
        assert_eq!(single_eos_id(Some(&LlamaEosToks::Single(2))), Some(2));
    }

    #[test]
    #[should_panic(expected = "route it to the serial path")]
    fn single_eos_id_refuses_a_multiple_checkpoint_rather_than_picking_one() {
        let _ = single_eos_id(Some(&LlamaEosToks::Multiple(vec![7, 8, 9])));
    }
}
