// SPDX-License-Identifier: MIT OR Apache-2.0
//! Wires `FuelPagedSchedulerDriver` (`super::scheduler_driver_paged`) into
//! `ModelRunner::start`'s `fuel-engine` arm (board item 97's actual remaining
//! deliverable: everything before this file only unblocked it). All three
//! arms were ported from the block-reservation `FuelSchedulerDriver` to the
//! paged one once Fuel 0.22.4 published `PagedDecodeModel` for all three
//! concrete types this file loads (paged-scheduler slice 2, 2026-10-23).
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

use fuel_inference::multi_session::{KvBudget, PagedSchedulePolicy};

use super::engine_model::effective_generation_budget;
use super::policies::BlockPrefixIndexFor;
use super::scheduler_driver_paged::{FuelPagedSchedulerDriver, SERVING_SCHEDULE_POLICY};
use crate::engine::model_runner::{
    ControlMsg, InferenceJob, SharedCacheStats, SharedScheduler, drain_with_error,
};
use crate::engine::scheduled_runner::{
    AdmitError, FinishOutcome, SchedulerDriver, run_scheduled_jobs,
};
use crate::engine::slot_pool::Request as SlotRequest;
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
    F32(FuelPagedSchedulerDriver<'m, LlamaModel>),
    QuantizedGguf(FuelPagedSchedulerDriver<'m, QuantizedLlama3Model>),
    QuantizedGgufQwen3(FuelPagedSchedulerDriver<'m, QuantizedQwen3Model>),
}

/// Enum-dispatch over the concrete `DecodeModel` types `ModelRunner::start`'s
/// fuel-engine arm can load (Llama eager/GGUF, and — since
/// `detect_architecture` lets `serve_gguf` pick a loader — Qwen3 GGUF), so
/// `run_scheduled_jobs` sees one `SchedulerDriver` regardless of checkpoint
/// format or architecture.
///
/// Owns `context_length` so `try_admit` can apply
/// [`effective_generation_budget`] itself — the one piece of policy that is
/// NOT `FuelPagedSchedulerDriver`'s concern (it has no notion of a
/// server-wide context ceiling), but IS this integration layer's, same role
/// `FuelEngineModel::step_one` played in the old serial path.
///
/// `scheduler`/`sessions` mirror every real admission/departure into the
/// shared `MemoryAwareScheduler` (board PM ruling, 2026-10-22, now that
/// Fuel 0.20.71 is published) — the same role `run_jobs`'s `admit_pending`/
/// `release_scheduler_slot` played on the deleted candlelight path (M4
/// slice 5.4a), rebuilt here against the live Fuel path instead. A passive
/// mirror, not a second authority: Fuel's own `try_admit` above already
/// decided admission — optimistically, as of the paged-scheduler port (see
/// `FuelPagedSchedulerDriver`'s own doc comment: there is no up-front
/// KV-block capacity check any more, only a later per-session failure if it
/// doesn't fit); this only records that decision so `scheduler_stats`/
/// `scheduler_queue` have real numbers to report. `sessions` maps Fuel's
/// `SessionId` to the synthetic request id (`"session-{id}"`) submitted to
/// the mirror, needed because `reap_finished` only gets the `SessionId`
/// back, not the original request's own metadata.
///
/// `prefix_index`/`cache_stats` give `cache_stats` real DETECTION numbers
/// (board PM task, 2026-10-22, ROADMAP research-track item 1 / FUEL-GAPS
/// entry #1) — `BlockPrefixIndexFor<SessionId>` (the policy-level chain-hash
/// index `src/model_fuel/policies.rs` already built and proved correct, now
/// generalized off `fuel::kv_block_pool::SessionHandle` so a caller with no
/// live `KvBlockPool` session can still use it) records every admitted
/// prompt's token-level prefix and looks up each new one against it. **This
/// is detection only, not exploitation**: a hit is NOT currently spliced
/// into the donor's KV blocks or used to skip prefill — this file now runs
/// on `PagedSessionScheduler` (via `FuelPagedSchedulerDriver`), which DOES
/// expose `add_session_sharing_prefix`/`register_prefix` for that, but
/// wiring a detected hit into an actual splice is a separate, not-yet-done
/// step. `hits`/`misses`/`insertions`/`entries` are therefore real; `evictions`
/// counts real `forget_donor` calls (on departure, below); nothing here
/// claims a `total_saved_tokens` figure — `CacheStatsResponse` never reads
/// that field, and claiming it without an actual skipped prefill would be
/// exactly the fabricated-number defect this project's conventions forbid.
struct ServingDriver<'m> {
    inner: ServingDriverInner<'m>,
    context_length: usize,
    scheduler: SharedScheduler,
    sessions: std::collections::HashMap<fuel_inference::multi_session::SessionId, String>,
    prefix_index: BlockPrefixIndexFor<fuel_inference::multi_session::SessionId>,
    cache_stats: SharedCacheStats,
    /// ROADMAP research-track item 1, slice 3: off-by-default switch for
    /// actually exploiting a `prefix_index` hit (splice + skip prefill),
    /// rather than only detecting it (`hits`/`misses` in `cache_stats`,
    /// which run regardless of this flag). `false` for every construction
    /// site today -- no operator-facing toggle exists yet; exposing one
    /// (env var/CLI) is a follow-up once this is trusted, not this slice's
    /// scope. Exists so `try_admit`'s off path and on path share one
    /// function instead of diverging into two.
    prefix_sharing_enabled: bool,
    /// Donors already registered as a shared [`fuel::kv_block_pool::PrefixId`]
    /// with the live `PagedSessionScheduler`, and how many blocks each
    /// registration covers -- a SEPARATE lifetime mechanism from
    /// `prefix_index`'s own `donor_refs` (that one is lightbulb's own
    /// bookkeeping for the old, never-live `KvBlockPool`/`splice_prefix`
    /// path; this one is the pool's real refcount on the paged path).
    /// Released when the donor departs (`mirror_departure`), which is safe
    /// even if a live sharer still references the blocks independently
    /// (`PagedSessionScheduler::release_prefix`'s own doc comment).
    ///
    /// Keyed by `(donor, blocks)` EXACTLY: a `PrefixId` splices every block
    /// it was registered over, so reusing a registration that covers MORE
    /// blocks than the current lookup matched would graft KV of tokens the
    /// new prompt does not contain.
    registered_prefixes: std::collections::HashMap<
        (fuel_inference::multi_session::SessionId, usize),
        fuel::kv_block_pool::PrefixId,
    >,
    /// Admissions that actually went through `add_session_sharing_prefix`
    /// (not the cold fallback). Lets a test tell a real splice from a
    /// silent fallback, which `registered_prefixes` alone cannot.
    prefix_shared_admissions: u64,
}

impl<'m> ServingDriver<'m> {
    /// Mirror a just-admitted session into the shared scheduler: submit a
    /// synthetic [`SlotRequest`] and immediately try to allocate it a slot.
    /// `top_p` has no real value here -- `SchedulerDriver::try_admit` never
    /// receives one -- so `1.0` (no nucleus truncation) is used as a
    /// placeholder; nothing downstream of the mirror reads it back.
    fn mirror_admission(
        &mut self,
        id: fuel_inference::multi_session::SessionId,
        prompt: &[u32],
        max_new: usize,
        temperature: f64,
    ) {
        let request_id = format!("session-{}", id.0);
        let request = SlotRequest {
            id: request_id.clone(),
            prompt_tokens: prompt.to_vec(),
            max_new_tokens: max_new,
            temperature,
            top_p: 1.0,
        };
        let mut scheduler = self.scheduler.lock().unwrap_or_else(|e| e.into_inner());
        scheduler.submit_request(request);
        scheduler.allocate_pending_requests();
        drop(scheduler);
        self.sessions.insert(id, request_id);
    }

    /// Mirror a just-reaped session's departure: free its slot in the
    /// shared scheduler, if the earlier mirrored admission ever got one
    /// (it may not have, if the mirror's own smaller/misconfigured budget
    /// queued it rather than allocating it -- freeing a slot that was
    /// never allocated would be a bug, not a no-op, so this checks first).
    fn mirror_departure(&mut self, id: fuel_inference::multi_session::SessionId) {
        // Bounds the index's growth and gives `evictions` a real count — not
        // a correctness requirement the way it is for `splice_prefix`'s
        // actual pool callers (this index never gates a `KvBlockPool::
        // discard`), just honest bookkeeping for a finished session.
        if self.prefix_index.forget_donor(id) > 0 {
            let entries = self.prefix_index.len();
            let mut stats = self.cache_stats.lock().unwrap_or_else(|e| e.into_inner());
            stats.evictions += 1;
            stats.entries = entries;
        }
        // Release this donor's pool-level prefix registration, if it had
        // one -- safe even while a live sharer still references the blocks
        // (see `registered_prefixes`'s own doc comment). A no-op whenever
        // `prefix_sharing_enabled` is false: nothing is ever inserted into
        // `registered_prefixes` on that path.
        let owned: Vec<(fuel_inference::multi_session::SessionId, usize)> = self
            .registered_prefixes
            .keys()
            .filter(|(donor, _)| *donor == id)
            .copied()
            .collect();
        for key in owned {
            if let Some(prefix) = self.registered_prefixes.remove(&key) {
                let _ = self.release_prefix_inner(prefix);
            }
        }
        let Some(request_id) = self.sessions.remove(&id) else {
            return;
        };
        let scheduler = self.scheduler.lock().unwrap_or_else(|e| e.into_inner());
        let slot_id = scheduler.slot_pool().get_request_slot(&request_id).copied();
        drop(scheduler);
        if let Some(slot_id) = slot_id {
            let mut scheduler = self.scheduler.lock().unwrap_or_else(|e| e.into_inner());
            let _ = scheduler.free_slot(slot_id);
        }
    }

    /// Index `prompt` itself under `id` (a just-admitted session) so a
    /// LATER admission can be detected against it, then mirror the real
    /// `BlockPrefixStats` into `SharedCacheStats` — the same immediate-write
    /// pattern `mirror_admission`/`mirror_departure` already use for the
    /// scheduler. The matching *lookup* for `prompt` itself happens earlier,
    /// in `try_admit`, before admission is even decided (needed there to
    /// pick the admission path when `prefix_sharing_enabled`); this is only
    /// the record-and-mirror tail, run unconditionally and identically on
    /// both the exploited and detection-only paths.
    fn record_prefix(&mut self, id: fuel_inference::multi_session::SessionId, prompt: &[u32]) {
        self.prefix_index.record(prompt, id);
        let block_stats = self.prefix_index.stats();
        let entries = self.prefix_index.len();
        let mut stats = self.cache_stats.lock().unwrap_or_else(|e| e.into_inner());
        stats.hits = block_stats.hits;
        stats.misses = block_stats.misses;
        stats.insertions = block_stats.blocks_indexed;
        stats.entries = entries;
    }

    /// The `prefix_sharing_enabled == false` admission path — unchanged
    /// from before slice 3: every arm's own cold `try_admit`, nothing else.
    fn plain_admit(
        &mut self,
        prompt: &[u32],
        max_new: usize,
        temperature: f64,
    ) -> Result<fuel_inference::multi_session::SessionId, AdmitError> {
        match &mut self.inner {
            ServingDriverInner::F32(d) => d.try_admit(prompt, max_new, temperature),
            ServingDriverInner::QuantizedGguf(d) => d.try_admit(prompt, max_new, temperature),
            ServingDriverInner::QuantizedGgufQwen3(d) => d.try_admit(prompt, max_new, temperature),
        }
    }

    /// On a lookup `hit`, try to actually splice the shared prefix instead
    /// of a cold prefill (`prefix_sharing_enabled`'s real exploitation).
    /// Falls back to [`Self::plain_admit`] -- never fails the request just
    /// because sharing didn't pan out -- on a miss, on a donor whose blocks
    /// are not (or no longer) fully resident up to the matched length (e.g.
    /// still mid-prefill, or reaped since the match was computed), or on any
    /// other Fuel-side refusal of the splice.
    fn try_admit_sharing_or_fallback(
        &mut self,
        hit: Option<super::policies::PrefixMatchFor<fuel_inference::multi_session::SessionId>>,
        prompt: &[u32],
        max_new: usize,
        temperature: f64,
    ) -> Result<fuel_inference::multi_session::SessionId, AdmitError> {
        let Some(hit) = hit else {
            return self.plain_admit(prompt, max_new, temperature);
        };
        // Fuel rejects a share that leaves no suffix (`prompt.len() <=
        // shared_tokens`): the first generated token needs logits from at
        // least one prefilled prompt token. A full-prompt match (the common
        // repeat-request case) therefore shares one block fewer.
        let blocks = hit
            .blocks
            .min(prompt.len().saturating_sub(1) / self.prefix_index.block_size());
        if blocks == 0 {
            return self.plain_admit(prompt, max_new, temperature);
        }
        let Some(prefix) = self.ensure_prefix_registered(hit.donor, blocks) else {
            return self.plain_admit(prompt, max_new, temperature);
        };
        let shared = match &mut self.inner {
            ServingDriverInner::F32(d) => {
                d.try_admit_sharing_prefix(prefix, prompt, max_new, temperature)
            }
            ServingDriverInner::QuantizedGguf(d) => {
                d.try_admit_sharing_prefix(prefix, prompt, max_new, temperature)
            }
            ServingDriverInner::QuantizedGgufQwen3(d) => {
                d.try_admit_sharing_prefix(prefix, prompt, max_new, temperature)
            }
        };
        match shared {
            Ok(id) => {
                self.prefix_shared_admissions += 1;
                Ok(id)
            }
            Err(_) => self.plain_admit(prompt, max_new, temperature),
        }
    }

    /// Register `donor`'s first `blocks` filled blocks as a shared prefix
    /// with the live `PagedSessionScheduler`, reusing a registration only
    /// if it covers EXACTLY this many blocks (see `registered_prefixes`).
    /// `None` on any Fuel-side refusal (e.g. the donor's blocks are not all
    /// filled yet).
    fn ensure_prefix_registered(
        &mut self,
        donor: fuel_inference::multi_session::SessionId,
        blocks: usize,
    ) -> Option<fuel::kv_block_pool::PrefixId> {
        if let Some(&prefix) = self.registered_prefixes.get(&(donor, blocks)) {
            return Some(prefix);
        }
        let registered = match &mut self.inner {
            ServingDriverInner::F32(d) => d.register_prefix(donor, blocks),
            ServingDriverInner::QuantizedGguf(d) => d.register_prefix(donor, blocks),
            ServingDriverInner::QuantizedGgufQwen3(d) => d.register_prefix(donor, blocks),
        };
        match registered {
            Ok(prefix) => {
                self.registered_prefixes.insert((donor, blocks), prefix);
                Some(prefix)
            }
            Err(_) => None,
        }
    }

    fn release_prefix_inner(&mut self, prefix: fuel::kv_block_pool::PrefixId) -> fuel::Result<()> {
        match &mut self.inner {
            ServingDriverInner::F32(d) => d.release_prefix(prefix),
            ServingDriverInner::QuantizedGguf(d) => d.release_prefix(prefix),
            ServingDriverInner::QuantizedGgufQwen3(d) => d.release_prefix(prefix),
        }
    }

    /// The real `cache_clear` admin endpoint's implementation (board PM
    /// task, 2026-10-22). Rebuilds `prefix_index` from scratch -- the
    /// simplest correct "clear": every future lookup is a fresh miss until
    /// new prompts are admitted and recorded again -- and zeroes the
    /// `SharedCacheStats` mirror to match. `sessions`/`scheduler`
    /// (admission bookkeeping) are untouched: clearing the prefix cache
    /// must never affect in-flight requests' admission state.
    fn clear_prefix_cache(&mut self) {
        let block_size = self.prefix_index.block_size();
        self.prefix_index = BlockPrefixIndexFor::new(block_size);
        let mut stats = self.cache_stats.lock().unwrap_or_else(|e| e.into_inner());
        *stats = crate::cache::PrefixCacheStats::default();
    }
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
        // Always run, regardless of `prefix_sharing_enabled` -- same
        // lookup `detect_and_record_prefix` used to do here directly,
        // giving `hits`/`misses` byte-identical behavior on both paths.
        // Doing it before deciding the admission path (rather than after,
        // as before) is required so the `true` path can act on the result;
        // the `false` path simply never looks at `hit`, which is the whole
        // reason the off path stays byte-identical to pre-slice-3 behavior.
        let hit = self.prefix_index.lookup(prompt);

        let admitted = if self.prefix_sharing_enabled {
            self.try_admit_sharing_or_fallback(hit, prompt, effective_max_new, temperature)
        } else {
            self.plain_admit(prompt, effective_max_new, temperature)
        };
        if let Ok(id) = admitted {
            self.mirror_admission(id, prompt, effective_max_new, temperature);
            self.record_prefix(id, prompt);
        }
        admitted
    }

    fn step(&mut self) -> Result<Vec<(Self::Id, u32)>, String> {
        match &mut self.inner {
            ServingDriverInner::F32(d) => d.step(),
            ServingDriverInner::QuantizedGguf(d) => d.step(),
            ServingDriverInner::QuantizedGgufQwen3(d) => d.step(),
        }
    }

    fn reap_finished(&mut self) -> Vec<(Self::Id, FinishOutcome)> {
        let reaped = match &mut self.inner {
            ServingDriverInner::F32(d) => d.reap_finished(),
            ServingDriverInner::QuantizedGguf(d) => d.reap_finished(),
            ServingDriverInner::QuantizedGgufQwen3(d) => d.reap_finished(),
        };
        for (id, _) in &reaped {
            self.mirror_departure(*id);
        }
        reaped
    }

    fn has_active_sessions(&self) -> bool {
        match &self.inner {
            ServingDriverInner::F32(d) => d.has_active_sessions(),
            ServingDriverInner::QuantizedGguf(d) => d.has_active_sessions(),
            ServingDriverInner::QuantizedGgufQwen3(d) => d.has_active_sessions(),
        }
    }

    fn handle_control(&mut self, msg: ControlMsg) {
        match msg {
            ControlMsg::ClearPrefixCache => self.clear_prefix_cache(),
        }
    }

    fn set_priority(&mut self, id: Self::Id, priority: i32) -> Result<(), String> {
        match &mut self.inner {
            ServingDriverInner::F32(d) => d.set_priority(id, priority),
            ServingDriverInner::QuantizedGguf(d) => d.set_priority(id, priority),
            ServingDriverInner::QuantizedGgufQwen3(d) => d.set_priority(id, priority),
        }
    }
}

/// A generous block size: large enough that `kv_blocks_required`'s ceiling
/// division wastes little, small enough that a short prompt does not reserve
/// a whole extra block it will never use. Matches the block size fuel's own
/// `multi_session.rs` test fixtures use.
const KV_BLOCK_SIZE: usize = 16;

/// The server-wide (not per-request) pieces of the batched path, bundled so
/// every function between [`run_fuel_engine`] and [`run_batched`] takes one
/// parameter instead of two or more — `device`/`policy`/`budget` are all
/// `Copy`; `scheduler` is an `Arc` clone away from being too, so bundling it
/// here too keeps every one of those functions at the same parameter count
/// it had before the 2026-10-22 scheduler-mirroring wiring added a new
/// cross-cutting dependency (`clippy::too_many_arguments`, not just style).
struct BatchedServeConfig {
    device: fuel::Device,
    /// Passed to `FuelPagedSchedulerDriver::step`'s `step_batched` every
    /// tick -- `PagedSchedulePolicy` (below) only controls ORDER (admission
    /// order vs. priority); batching itself is a separate knob on
    /// `PagedSessionScheduler::step_batched`, not a policy variant the way
    /// `SchedulePolicy::Batched` was on the non-paged scheduler.
    max_batch: usize,
    policy: PagedSchedulePolicy,
    budget: KvBudget,
    scheduler: SharedScheduler,
    cache_stats: SharedCacheStats,
    ctrl_rx: Receiver<ControlMsg>,
    /// ROADMAP research-track item 1, slice 3's off-by-default switch --
    /// see `ServingDriver::prefix_sharing_enabled`'s own doc comment.
    /// `false` from `new`; `run_fuel_engine` sets it from
    /// `EngineOptions::prefix_sharing`.
    prefix_sharing_enabled: bool,
}

impl BatchedServeConfig {
    fn new(
        max_batch_size: usize,
        context_length: usize,
        scheduler: SharedScheduler,
        cache_stats: SharedCacheStats,
        ctrl_rx: Receiver<ControlMsg>,
    ) -> Self {
        let max_batch_size = max_batch_size.max(1);
        Self {
            device: super::device::select(),
            max_batch: max_batch_size,
            policy: SERVING_SCHEDULE_POLICY,
            budget: KvBudget {
                block_size: KV_BLOCK_SIZE,
                num_blocks: max_batch_size * context_length.div_ceil(KV_BLOCK_SIZE),
            },
            scheduler,
            cache_stats,
            ctrl_rx,
            prefix_sharing_enabled: false,
        }
    }
}

/// Build the batched driver for one already-loaded model and drive it —
/// shared by both the `.gguf` and SafeTensors-directory branches of
/// [`run_fuel_engine`], which differ only in the CONCRETE `M` and which
/// `ServingDriverInner` variant wraps it (`wrap`, a tuple-variant
/// constructor used as a plain `fn`).
fn run_batched<'m, M: fuel_inference::multi_session::PagedDecodeModel>(
    model: &'m M,
    config: BatchedServeConfig,
    eos_ids: Option<Vec<u32>>,
    context_length: usize,
    tokenizer: tokenizers::Tokenizer,
    rx: Receiver<InferenceJob>,
    wrap: fn(FuelPagedSchedulerDriver<'m, M>) -> ServingDriverInner<'m>,
) {
    let fuel_scheduler = match FuelPagedSchedulerDriver::new(
        model,
        config.budget,
        fuel::DType::F32,
        &config.device,
        config.policy,
        config.max_batch,
        eos_ids,
    ) {
        Ok(s) => s,
        Err(e) => return drain_with_error(rx, &format!("building the scheduler: {e}")),
    };
    let ctrl_rx = config.ctrl_rx;
    let driver = ServingDriver {
        inner: wrap(fuel_scheduler),
        context_length,
        scheduler: config.scheduler,
        sessions: std::collections::HashMap::new(),
        prefix_index: BlockPrefixIndexFor::new(KV_BLOCK_SIZE),
        cache_stats: config.cache_stats,
        prefix_sharing_enabled: config.prefix_sharing_enabled,
        registered_prefixes: std::collections::HashMap::new(),
        prefix_shared_admissions: 0,
    };
    // `tokenizer.encode`/`.decode` each need their own `move` closure below,
    // and `Tokenizer` is not `Copy` — an `Arc` is what lets both closures
    // share it rather than one of them moving the only copy.
    let tokenizer = std::sync::Arc::new(tokenizer);
    let encode_tokenizer = tokenizer.clone();
    run_scheduled_jobs(
        driver,
        rx,
        ctrl_rx,
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
#[allow(clippy::too_many_arguments)] // `ModelRunner::start`'s seven plus `options`
pub(crate) fn run_fuel_engine(
    model_path: &Path,
    max_batch_size: usize,
    context_length: usize,
    rx: Receiver<InferenceJob>,
    ctrl_rx: Receiver<ControlMsg>,
    scheduler: SharedScheduler,
    cache_stats: SharedCacheStats,
    options: crate::engine::model_runner::EngineOptions,
) {
    let mut config = BatchedServeConfig::new(
        max_batch_size,
        context_length,
        scheduler,
        cache_stats,
        ctrl_rx,
    );
    config.prefix_sharing_enabled = options.prefix_sharing;
    if let Err(e) =
        super::preflight::enforce(model_path, &config.device, options.allow_oversize_model)
    {
        return load_failed(model_path, e, rx);
    }
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
        Ok(GgufArchitecture::Qwen35) => {
            load_failed(model_path, anyhow::anyhow!(QWEN35_UNSERVED), rx)
        }
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

/// Why a `qwen35` GGUF is recognised but not served: a specific refusal rather
/// than the generic unsupported-architecture one, so the reader is not sent
/// looking for a defect in a file that is fine.
const QWEN35_UNSERVED: &str = "this GGUF declares `general.architecture = \"qwen35\"` \
    (Qwen3.5/3.8: Gated DeltaNet layers interleaved with full attention). Lightbulb \
    recognises it and its tokenizer is supported, but serving it needs a hybrid decode \
    model with per-session recurrent state and K-quants kept quantized, which Fuel does \
    not provide yet (ROADMAP F10).";

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
    use crate::engine::{MemoryAwareConfig, MemoryAwareScheduler};
    use fuel::lazy::LayerWeights;
    use fuel_model_llama::{LlamaConfig, LlamaModel, LlamaWeights};
    use std::sync::{Arc, Mutex};

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

    // Mirrors `scheduler_driver.rs`'s own `tiny_cfg`/`tiny_weights`/
    // `tiny_model` byte-for-byte (same seeded PRNG) -- kept local rather
    // than shared, same reasoning that module gives for its own copy of
    // fuel's `#[cfg(test)]`-private fixture: this file needs a real
    // `ServingDriver` constructed with direct (private-field) access, which
    // only a `#[cfg(test)] mod tests` inside THIS module can do.
    fn tiny_cfg() -> LlamaConfig {
        LlamaConfig {
            vocab_size: 16,
            dim: 8,
            n_layers: 2,
            n_heads: 2,
            n_kv_heads: 2,
            head_dim: 4,
            ffn_dim: 16,
            norm_eps: 1e-5,
            rope_base: 10000.0,
        }
    }

    fn tiny_weights(cfg: &LlamaConfig, seed: u32) -> LlamaWeights {
        let mut s = seed;
        let mut next = || {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.1
        };
        let mut vec_of =
            |n: usize| -> Arc<[f32]> { Arc::from((0..n).map(|_| next()).collect::<Vec<_>>()) };
        let kv = cfg.n_kv_heads * cfg.head_dim;
        LlamaWeights {
            instance: fuel::decode_shape::ModelInstanceId::next(),
            token_embedding: vec_of(cfg.vocab_size * cfg.dim),
            layers: (0..cfg.n_layers)
                .map(|_| LayerWeights {
                    attn_q: vec_of(cfg.dim * cfg.dim).into(),
                    attn_q_bias: None,
                    attn_k: vec_of(cfg.dim * kv).into(),
                    attn_k_bias: None,
                    attn_v: vec_of(cfg.dim * kv).into(),
                    attn_v_bias: None,
                    attn_o: vec_of(cfg.dim * cfg.dim).into(),
                    ffn_gate: vec_of(cfg.dim * cfg.ffn_dim).into(),
                    ffn_up: vec_of(cfg.dim * cfg.ffn_dim).into(),
                    ffn_down: vec_of(cfg.ffn_dim * cfg.dim).into(),
                    attn_norm_gain: Arc::from(vec![1.0; cfg.dim]),
                    ffn_norm_gain: Arc::from(vec![1.0; cfg.dim]),
                })
                .collect(),
            final_norm_gain: Arc::from(vec![1.0; cfg.dim]),
            output: vec_of(cfg.dim * cfg.vocab_size).into(),
        }
    }

    fn tiny_model(seed: u32) -> LlamaModel {
        let cfg = tiny_cfg();
        LlamaModel {
            config: cfg.clone(),
            weights: tiny_weights(&cfg, seed),
        }
    }

    /// A real `ServingDriver` wrapping a real tiny `LlamaModel`, generous
    /// enough on KV capacity that admission never fails for these tests'
    /// short prompts -- the point is `try_admit`'s prefix-detection wiring,
    /// not capacity edge cases (those belong to `scheduler_driver_paged.rs`'s
    /// own tests).
    fn serving_driver(model: &LlamaModel, prefix_sharing_enabled: bool) -> ServingDriver<'_> {
        let fuel_scheduler = FuelPagedSchedulerDriver::new(
            model,
            KvBudget {
                block_size: 2,
                num_blocks: 4096,
            },
            fuel::DType::F32,
            &fuel::Device::cpu(),
            PagedSchedulePolicy::AdmissionOrder,
            1,
            None,
        )
        .expect("tiny model's uniform per-head KV must satisfy ModelDims::from_model");
        ServingDriver {
            inner: ServingDriverInner::F32(fuel_scheduler),
            context_length: 4096,
            scheduler: Arc::new(Mutex::new(MemoryAwareScheduler::new(
                MemoryAwareConfig::default(),
            ))),
            sessions: std::collections::HashMap::new(),
            prefix_index: BlockPrefixIndexFor::new(2),
            cache_stats: Arc::new(Mutex::new(crate::cache::PrefixCacheStats::default())),
            prefix_sharing_enabled,
            registered_prefixes: std::collections::HashMap::new(),
            prefix_shared_admissions: 0,
        }
    }

    /// The test `detect_and_record_prefix` exists for: admitting the SAME
    /// prompt twice must detect the second admission as a repeat (a real
    /// `BlockPrefixIndexFor::lookup` hit), while a genuinely different
    /// prompt must miss. If `try_admit`'s call into
    /// `detect_and_record_prefix` were ever deleted, `hits` would stay `0`
    /// and this test would fail -- the mutation this slice's board task
    /// asked to be provable against.
    #[test]
    fn repeated_prefix_is_detected_as_a_real_hit_distinct_prompt_is_a_real_miss() {
        let model = tiny_model(1);
        let mut d = serving_driver(&model, false);

        // Two whole blocks (block_size 2): long enough to be indexed.
        let shared_prompt = [1u32, 2, 3, 4];
        d.try_admit(&shared_prompt, 1, 0.0)
            .expect("first admission must fit in the generous budget");
        assert_eq!(
            d.cache_stats.lock().unwrap().hits,
            0,
            "the FIRST admission of a prompt can never be a hit -- nothing was \
             indexed yet when its own lookup ran"
        );

        // Same tokens again: must hit against the first admission's record.
        d.try_admit(&shared_prompt, 1, 0.0)
            .expect("second admission must also fit");
        let stats_after_repeat = d.cache_stats.lock().unwrap().clone();
        assert_eq!(
            stats_after_repeat.hits, 1,
            "repeating an already-admitted prompt must register a real hit"
        );
        assert!(
            stats_after_repeat.entries >= 1,
            "a real record must have been indexed: {stats_after_repeat:?}"
        );

        // A genuinely different prompt: must miss, not hit.
        let distinct_prompt = [9u32, 10, 11, 12];
        d.try_admit(&distinct_prompt, 1, 0.0)
            .expect("third admission must also fit");
        let stats_after_distinct = d.cache_stats.lock().unwrap().clone();
        assert_eq!(
            stats_after_distinct.hits, stats_after_repeat.hits,
            "a distinct prompt must not be counted as a hit"
        );
        assert!(
            stats_after_distinct.misses >= 1,
            "a distinct prompt must register a real miss: {stats_after_distinct:?}"
        );
    }

    /// The test `handle_control`/`clear_prefix_cache` exist for (board PM
    /// task, 2026-10-22): fill the index with a real hit, clear it via a
    /// real `ControlMsg`, then prove the clear actually took -- the SAME
    /// prefix that was a hit a moment ago must now be a miss, and the
    /// shared stats must read zero. If `handle_control`'s match arm (or the
    /// `ClearPrefixCache` message itself) were ever skipped, the repeated
    /// prompt below would still hit and this test would fail.
    #[test]
    fn handle_control_clear_prefix_cache_makes_the_next_identical_prefix_a_real_miss() {
        let model = tiny_model(1);
        let mut d = serving_driver(&model, false);

        let prompt = [1u32, 2, 3, 4];
        d.try_admit(&prompt, 1, 0.0)
            .expect("first admission must fit");
        d.try_admit(&prompt, 1, 0.0)
            .expect("second admission must also fit");
        let stats_before_clear = d.cache_stats.lock().unwrap().clone();
        assert_eq!(
            stats_before_clear.hits, 1,
            "sanity check: the index must actually be primed before testing the clear"
        );

        d.handle_control(ControlMsg::ClearPrefixCache);

        let stats_after_clear = d.cache_stats.lock().unwrap().clone();
        assert_eq!(stats_after_clear.hits, 0, "{stats_after_clear:?}");
        assert_eq!(stats_after_clear.misses, 0, "{stats_after_clear:?}");
        assert_eq!(stats_after_clear.entries, 0, "{stats_after_clear:?}");

        // The SAME prompt that was a hit before the clear must now miss --
        // proving the live index was actually rebuilt, not just the stats
        // mirror zeroed out from underneath it (which would desync the two
        // and make a later real hit under-report).
        d.try_admit(&prompt, 1, 0.0)
            .expect("third admission must also fit");
        let stats_after_repeat = d.cache_stats.lock().unwrap().clone();
        assert_eq!(
            stats_after_repeat.hits, 0,
            "the cleared index must not still recognize the pre-clear prefix: \
             {stats_after_repeat:?}"
        );
        assert_eq!(
            stats_after_repeat.misses, 1,
            "the post-clear admission must register as a genuine miss: \
             {stats_after_repeat:?}"
        );
    }

    /// Drives `id` to completion, returning its generated tokens in order.
    /// `step`/`reap_finished` are called on `d` directly -- the same
    /// `SchedulerDriver` trait methods `run_scheduled_jobs` itself uses --
    /// so this exercises the real admission-to-completion path, not a
    /// shortcut around it.
    fn drive_to_completion(
        d: &mut ServingDriver<'_>,
        id: fuel_inference::multi_session::SessionId,
    ) -> Vec<u32> {
        let mut tokens = Vec::new();
        loop {
            for (tok_id, tok) in d.step().expect("step must not error") {
                if tok_id == id {
                    tokens.push(tok);
                }
            }
            if d.reap_finished()
                .into_iter()
                .any(|(fin_id, _)| fin_id == id)
            {
                return tokens;
            }
        }
    }

    /// Greedy completion of `prompt` on a driver that never shared anything:
    /// the cold reference every shared completion below is compared to.
    fn cold_completion(model: &LlamaModel, prompt: &[u32], max_new: usize) -> Vec<u32> {
        let mut d = serving_driver(model, false);
        let id = d
            .try_admit(prompt, max_new, 0.0)
            .expect("cold admission must succeed");
        drive_to_completion(&mut d, id)
    }

    /// A donor (`donor_prompt`) admitted and prefilled on a sharing-enabled
    /// driver, kept alive (large `max_new`) so later admissions can splice
    /// from it. Block size is 2 (`serving_driver`).
    fn driver_with_live_donor<'m>(
        model: &'m LlamaModel,
        donor_prompt: &[u32],
    ) -> (ServingDriver<'m>, fuel_inference::multi_session::SessionId) {
        let mut d = serving_driver(model, true);
        let donor = d
            .try_admit(donor_prompt, 32, 0.0)
            .expect("donor must admit");
        // `prefill_pass` feeds a session's whole prompt in one `step()`.
        let _ = d.step().expect("donor's prefill tick must not error");
        (d, donor)
    }

    /// ROADMAP research-track item 1, slice 3's required test: a request
    /// sharing a prefix with a live donor produces the SAME completion as a
    /// cold admission AND really went through `add_session_sharing_prefix`.
    ///
    /// The second half is `prefix_shared_admissions`, not
    /// `registered_prefixes`: registration happens BEFORE Fuel decides
    /// whether to accept the share, so a registration alone cannot tell a
    /// splice from a silent cold fallback (which would also match the cold
    /// completion trivially). Prefill skipping itself is Fuel's contract
    /// (`add_session_sharing_prefix` prefills only `prompt[shared..]`);
    /// timing it would be flaky on CI.
    #[test]
    fn a_prefix_sharing_admission_really_splices_and_matches_the_cold_completion() {
        let model = tiny_model(1);
        // 5 tokens, block size 2: two whole blocks (4 tokens) shared, one
        // suffix token left to prefill.
        let prompt = [1u32, 2, 3, 4, 5];
        let (mut d, donor) = driver_with_live_donor(&model, &prompt);
        assert!(d.registered_prefixes.is_empty() && d.prefix_shared_admissions == 0);

        let sharer = d.try_admit(&prompt, 3, 0.0).expect("sharer must admit");
        assert_eq!(d.prefix_shared_admissions, 1, "must be a real splice");
        assert!(d.registered_prefixes.contains_key(&(donor, 2)));
        assert_eq!(
            drive_to_completion(&mut d, sharer),
            cold_completion(&model, &prompt, 3),
            "shared completion must be bit-identical to a cold admission"
        );
    }

    /// A repeat of a prompt that is exactly block-aligned would match ALL
    /// its blocks, which Fuel refuses (no suffix left to sample from). The
    /// driver must share one block fewer rather than fall back cold.
    #[test]
    fn a_fully_block_aligned_repeat_shares_one_block_fewer_instead_of_falling_back() {
        let model = tiny_model(1);
        let prompt = [1u32, 2, 3, 4];
        let (mut d, donor) = driver_with_live_donor(&model, &prompt);

        let sharer = d.try_admit(&prompt, 3, 0.0).expect("sharer must admit");
        assert_eq!(d.prefix_shared_admissions, 1, "must be a real splice");
        assert!(d.registered_prefixes.contains_key(&(donor, 1)));
        assert!(!d.registered_prefixes.contains_key(&(donor, 2)));
        assert_eq!(
            drive_to_completion(&mut d, sharer),
            cold_completion(&model, &prompt, 3)
        );
    }

    /// A donor's longer registration must never be reused for a request that
    /// matched fewer blocks: the splice would graft KV for tokens that
    /// request does not contain. `BlockPrefixIndexFor::record` is
    /// last-writer-wins, so a live index rarely leaves ONE donor holding both
    /// lengths -- the registration table is driven directly here to put the
    /// donor in exactly that state, then the real fallback path is run with
    /// a 2-block hit.
    #[test]
    fn a_shorter_match_never_reuses_the_donors_longer_registration() {
        let model = tiny_model(1);
        let donor_prompt = [1u32, 2, 3, 4, 5, 6];
        let (mut d, donor) = driver_with_live_donor(&model, &donor_prompt);

        let long = d
            .ensure_prefix_registered(donor, 3)
            .expect("donor holds 3 whole blocks");
        let hit = super::super::policies::PrefixMatchFor {
            donor,
            blocks: 2,
            shared_tokens: 4,
            remainder_tokens: 3,
        };
        let short_prompt = [1u32, 2, 3, 4, 9, 9, 9];
        let id = d
            .try_admit_sharing_or_fallback(Some(hit), &short_prompt, 3, 0.0)
            .expect("short sharer must admit");

        assert_eq!(d.prefix_shared_admissions, 1, "must be a real splice");
        assert_eq!(d.registered_prefixes.len(), 2);
        assert_ne!(
            d.registered_prefixes[&(donor, 2)],
            long,
            "the 2-block match needs its own registration, not the 3-block one"
        );
        assert_eq!(
            drive_to_completion(&mut d, id),
            cold_completion(&model, &short_prompt, 3),
            "the short sharer must see only its own 2 matched blocks"
        );
    }

    /// The companion half of the slice 3 requirement: with the switch OFF
    /// (the default for every construction site today), behavior must be
    /// byte-identical to before slice 3 existed. Runs the EXACT scenario
    /// that would trigger sharing if the flag were on (a donor, then an
    /// identical-prompt second admission) and asserts two things the
    /// on-path test above does NOT need to: `registered_prefixes` never
    /// gains an entry (the new pool-level machinery is never touched at
    /// all when the flag is off, not merely "touched but declines to
    /// share"), and the second admission's completion is identical to an
    /// admission of that prompt on a driver that never saw the donor at
    /// all -- proving the mere PRESENCE of a detected hit changes nothing
    /// observable when `prefix_sharing_enabled` is false.
    #[test]
    fn prefix_sharing_disabled_is_byte_identical_to_detection_only_behavior() {
        let model = tiny_model(1);
        let prompt = [1u32, 2, 3, 4];
        let max_new = 3;

        let mut off_driver = serving_driver(&model, false);
        let _donor_id = off_driver
            .try_admit(&prompt, max_new, 0.0)
            .expect("donor must admit");
        let _ = off_driver
            .step()
            .expect("donor's prefill tick must not error");

        let repeat_id = off_driver
            .try_admit(&prompt, max_new, 0.0)
            .expect("identical-prompt admission must still succeed");
        assert!(
            off_driver.registered_prefixes.is_empty(),
            "the off path must never call register_prefix at all, not even once -- the \
             pool-level machinery this slice added must stay completely untouched"
        );
        let repeat_tokens = drive_to_completion(&mut off_driver, repeat_id);

        let mut fresh_driver = serving_driver(&model, false);
        let fresh_id = fresh_driver
            .try_admit(&prompt, max_new, 0.0)
            .expect("a driver with no donor at all must admit the same prompt identically");
        let fresh_tokens = drive_to_completion(&mut fresh_driver, fresh_id);

        assert_eq!(
            repeat_tokens, fresh_tokens,
            "a detected-but-unexploited hit must not change the completion at all when \
             prefix sharing is disabled"
        );
    }
}
