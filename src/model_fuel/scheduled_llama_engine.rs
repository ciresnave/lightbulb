//! Wires `FuelSchedulerDriver` (`super::scheduler_driver`) into
//! `ModelRunner::start`'s `fuel-engine` arm (board item 97's actual remaining
//! deliverable: everything before this file only unblocked it).
//!
//! `ModelRunner::start`'s `fuel-engine` arm can load an eager SafeTensors
//! checkpoint (`LlamaModel`), a GGUF-quantized Llama one
//! (`QuantizedLlama3Model`), or — since `serve_gguf` reads the file's own
//! `general.architecture` via `crate::gguf::detect_architecture` — a
//! GGUF-quantized Qwen3 one (`QuantizedQwen3Model`). Three different
//! concrete `DecodeModel` types, so one `FuelSchedulerDriver<'m, M>` cannot
//! hold any two interchangeably. `ServingDriver` (below, `pub(crate)` — not
//! an intra-doc link, same reason as `scheduler_driver.rs`'s own module doc)
//! is the small enum-dispatch wrapper that lets `run_scheduled_jobs` stay
//! generic over ONE `SchedulerDriver` regardless of which checkpoint format
//! or architecture was loaded.
//!
//! The SafeTensors-directory path (`serve_f32`) stays Llama-only — there is
//! no eager Qwen3 loader in this codebase, and no architecture metadata to
//! read the way GGUF's `general.architecture` provides it; widening that
//! path is out of scope here, not an oversight.
//!
//! # Multi-token EOS checkpoints: no longer a special case
//!
//! Earlier revisions of this file routed `LlamaEosToks::Multiple` checkpoints
//! to the serial `FuelEngineModel`/`run_jobs` path instead of the batched
//! one, because fuel's `SessionState` tracked only ONE `eos_id: Option<u32>`
//! — a checkpoint with several stop tokens would have stopped on whichever
//! one was chosen and run PAST the others to the budget ceiling on the
//! batched path. **fuel#307 widened that to `eos_ids: Option<Vec<u32>>`**
//! (stop-check is "any configured id"), so every checkpoint — `None`,
//! `Single`, or `Multiple` — now takes the batched path uniformly via
//! `eos_ids_from` (private, not an intra-doc link here since this module
//! is `pub`).
#![allow(dead_code)]

use std::path::Path;
use std::sync::mpsc::Receiver;

use anyhow::Result;

use fuel_model_llama::LlamaModel;
use fuel_transformers::models::lazy_llama_full::LlamaEosToks;
use fuel_transformers::models::lazy_quantized_llama::QuantizedLlama3Model;
use fuel_transformers::models::lazy_quantized_qwen3::QuantizedQwen3Model;

use fuel_inference::multi_session::{KvBudget, SchedulePolicy};

use super::engine_model::effective_generation_budget;
use super::scheduler_driver::FuelSchedulerDriver;
use crate::engine::model_runner::{InferenceJob, drain_with_error};
use crate::engine::scheduled_runner::{
    AdmitError, FinishOutcome, SchedulerDriver, run_scheduled_jobs,
};
use crate::gguf::{GgufArchitecture, detect_architecture};

/// The checkpoint's full EOS set, in fuel's `eos_ids: Option<Vec<u32>>`
/// shape. `LlamaEosToks::Single`/`Multiple`/absent map onto it directly —
/// see this module's doc comment for why there is no longer a case this
/// cannot represent.
fn eos_ids_from(eos: Option<&LlamaEosToks>) -> Option<Vec<u32>> {
    match eos {
        None => None,
        Some(LlamaEosToks::Single(id)) => Some(vec![*id]),
        Some(LlamaEosToks::Multiple(ids)) => Some(ids.clone()),
    }
}

enum ServingDriverInner<'m> {
    F32(FuelSchedulerDriver<'m, LlamaModel>),
    QuantizedGguf(FuelSchedulerDriver<'m, QuantizedLlama3Model>),
    QuantizedGgufQwen3(FuelSchedulerDriver<'m, QuantizedQwen3Model>),
}

/// Enum-dispatch over the concrete `DecodeModel` types `ModelRunner::start`'s
/// fuel-engine arm can load (Llama eager/GGUF, and — since
/// `detect_architecture` lets `serve_gguf` pick a loader — Qwen3 GGUF), so
/// `run_scheduled_jobs` sees one `SchedulerDriver` regardless of checkpoint
/// format or architecture.
///
/// Owns `context_length` so `try_admit` can apply
/// [`effective_generation_budget`] itself — the one piece of policy that is
/// NOT `FuelSchedulerDriver`'s concern (it has no notion of a server-wide
/// context ceiling), but IS this integration layer's, same role
/// `FuelEngineModel::step_one` played in the old serial path.
struct ServingDriver<'m> {
    inner: ServingDriverInner<'m>,
    context_length: usize,
}

impl<'m> SchedulerDriver for ServingDriver<'m> {
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
            ServingDriverInner::F32(d) => d.try_admit(prompt, effective_max_new, temperature),
            ServingDriverInner::QuantizedGguf(d) => {
                d.try_admit(prompt, effective_max_new, temperature)
            }
            ServingDriverInner::QuantizedGgufQwen3(d) => {
                d.try_admit(prompt, effective_max_new, temperature)
            }
        }
    }

    fn step(&mut self) -> Result<Vec<(Self::Id, u32)>, String> {
        match &mut self.inner {
            ServingDriverInner::F32(d) => d.step(),
            ServingDriverInner::QuantizedGguf(d) => d.step(),
            ServingDriverInner::QuantizedGgufQwen3(d) => d.step(),
        }
    }

    fn reap_finished(&mut self) -> Vec<(Self::Id, FinishOutcome)> {
        match &mut self.inner {
            ServingDriverInner::F32(d) => d.reap_finished(),
            ServingDriverInner::QuantizedGguf(d) => d.reap_finished(),
            ServingDriverInner::QuantizedGgufQwen3(d) => d.reap_finished(),
        }
    }

    fn has_active_sessions(&self) -> bool {
        match &self.inner {
            ServingDriverInner::F32(d) => d.has_active_sessions(),
            ServingDriverInner::QuantizedGguf(d) => d.has_active_sessions(),
            ServingDriverInner::QuantizedGgufQwen3(d) => d.has_active_sessions(),
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
/// `ServingDriverInner` variant wraps it (`wrap`, a tuple-variant
/// constructor used as a plain `fn`).
fn run_batched<'m, M: fuel_inference::multi_session::DecodeModel>(
    model: &'m M,
    config: BatchedServeConfig,
    eos_ids: Option<Vec<u32>>,
    context_length: usize,
    tokenizer: tokenizers::Tokenizer,
    rx: Receiver<InferenceJob>,
    wrap: fn(FuelSchedulerDriver<'m, M>) -> ServingDriverInner<'m>,
) {
    let scheduler = match FuelSchedulerDriver::new(
        model,
        config.device,
        fuel::DType::F32,
        config.policy,
        config.budget,
        eos_ids,
    ) {
        Ok(s) => s,
        Err(e) => return drain_with_error(rx, &format!("building the scheduler: {e}")),
    };
    let driver = ServingDriver {
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

/// Load a checkpoint (dispatching on `.gguf` vs a SafeTensors directory) and
/// drive it through the real `FuelSchedulerDriver`/`run_scheduled_jobs`
/// batched path — uniformly, regardless of its EOS shape (see this module's
/// doc comment).
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

/// The `.gguf` branch of [`run_fuel_engine`]: reads the file's own
/// `general.architecture` declaration FIRST — through a throwaway
/// `crate::gguf::Content` read, same pattern the loaders' own "up to three
/// times" safety note already accepts — and picks the loader that actually
/// matches, instead of hardcoding Llama and discovering a Qwen3 checkpoint's
/// mismatch as a refusal deep inside `require_llama_architecture`.
fn serve_gguf(
    model_path: &Path,
    context_length: usize,
    config: BatchedServeConfig,
    rx: Receiver<InferenceJob>,
) {
    let content = match crate::gguf::Content::read(model_path) {
        Ok(c) => c,
        Err(e) => return load_failed(model_path, e, rx),
    };
    match detect_architecture(content.metadata()) {
        Ok(GgufArchitecture::Llama) => serve_gguf_llama(model_path, context_length, config, rx),
        Ok(GgufArchitecture::Qwen3) => serve_gguf_qwen3(model_path, context_length, config, rx),
        Err(e) => load_failed(model_path, e, rx),
    }
}

/// The Llama-architecture half of [`serve_gguf`] — split out so that
/// function stays short; see its own doc for the load/route/batch shape both
/// architecture branches repeat.
fn serve_gguf_llama(
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
    let eos_ids = eos_ids_from(loaded.eos.as_ref());
    run_batched(
        &loaded.model,
        config,
        eos_ids,
        context_length,
        loaded.tokenizer,
        rx,
        ServingDriverInner::QuantizedGguf,
    );
}

/// The Qwen3-architecture half of [`serve_gguf`] — see [`serve_gguf_llama`].
fn serve_gguf_qwen3(
    model_path: &Path,
    context_length: usize,
    config: BatchedServeConfig,
    rx: Receiver<InferenceJob>,
) {
    let loaded = match super::loader_gguf_qwen3::load_quantized_qwen3_gguf(model_path) {
        Ok(l) => l,
        Err(e) => return load_failed(model_path, e, rx),
    };
    println!("Fuel model loaded at {}", model_path.display());
    let eos_ids = eos_ids_from(loaded.eos.as_ref());
    run_batched(
        &loaded.model,
        config,
        eos_ids,
        context_length,
        loaded.tokenizer,
        rx,
        ServingDriverInner::QuantizedGgufQwen3,
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
    let eos_ids = eos_ids_from(loaded.eos.as_ref());
    run_batched(
        &loaded.model,
        config,
        eos_ids,
        context_length,
        loaded.tokenizer,
        rx,
        ServingDriverInner::F32,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eos_ids_from_reads_through_none_and_single() {
        assert_eq!(eos_ids_from(None), None);
        assert_eq!(eos_ids_from(Some(&LlamaEosToks::Single(2))), Some(vec![2]));
    }

    #[test]
    fn eos_ids_from_carries_the_full_multiple_set() {
        // fuel#307's whole point: a checkpoint with several stop tokens no
        // longer needs a different code path, it needs its full set carried
        // through unchanged.
        assert_eq!(
            eos_ids_from(Some(&LlamaEosToks::Multiple(vec![7, 8, 9]))),
            Some(vec![7, 8, 9])
        );
    }
}
